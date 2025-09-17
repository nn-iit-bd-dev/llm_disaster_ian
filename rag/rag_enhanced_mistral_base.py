#!/usr/bin/env python3
"""
RAG-Enhanced Hurricane Forecasting using Mistral + evacuation knowledge base

This script combines:
1. Mistral 7B base model for forecasting
2. Evacuation knowledge base for contextual information
3. Comprehensive evaluation using custom metrics and reporting modules

Usage:
    python rag_enhanced_mistral.py \
        --knowledge-base evacuation_kb.json \
        --test-data prepared_data/tampa_2025_09_01/data/tampa_test.jsonl \
        --output-dir ./rag_results \
        --city Tampa \
        --generate-plots

Output Structure:
    rag_results/
    └── rag_mistral_results_Tampa/
        ├── csv_outputs/
        │   └── rag_predictions.csv
        ├── reports/
        │   ├── rag_Tampa_detailed_report.csv
        │   ├── rag_Tampa_summary.csv
        │   ├── rag_Tampa_overall_stats.csv
        │   └── rag_Tampa_metrics_summary.json
        ├── performance_plots/
        │   ├── best_performance/
        │   ├── worst_performance/
        │   └── summary_comparison.png
        └── rag_execution_summary.json
"""

# ---- silence TF and keep transformers on PyTorch only ----
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import re
import sys

# Import our custom modules
from compute_metric import compute_mae, compute_rmse, compute_smape, compute_rmsle
from summary_report import generate_all_reports
from visual_plot import plot_best_worst_pois

# -------------------- Evacuation KB --------------------
class EvacuationKB:
    """
    Knowledge Base for evacuation orders with location and temporal context
    
    Provides evacuation context based on FIPS codes and target dates,
    helping the model understand evacuation status during prediction windows.
    """
    
    def __init__(self, kb_json_path):
        kb_path = Path(kb_json_path)
        if not kb_path.exists():
            sys.exit(f"[ERROR] Knowledge base not found: {kb_path}")
        with open(kb_path, 'r', encoding='ascii', errors='ignore') as f:
            self.kb_data = json.load(f)
        self.entries = self.kb_data.get('entries', [])
        print(f"Knowledge Base loaded: {len(self.entries)} evacuation entries")

    def get_context_for_location(self, fips_code, target_date, max_context=3):
        """
        Retrieve relevant evacuation orders for a location and date
        
        Args:
            fips_code: County FIPS code
            target_date: Target prediction date
            max_context: Maximum number of orders to return
            
        Returns:
            List of relevant evacuation orders with timing information
        """
        if not fips_code:
            return []
        
        fips_orders = [e for e in self.entries if e.get('fips_code') == fips_code]
        target_dt = pd.to_datetime(target_date)

        relevant = []
        for order in fips_orders:
            announcement_date = order.get('announcement_date')
            if not announcement_date:
                continue
            announcement_dt = pd.to_datetime(announcement_date)
            if announcement_dt <= target_dt:
                days_diff = (target_dt - announcement_dt).days
                _o = dict(order)  # copy
                _o['days_since_announcement'] = days_diff

                effective_date = order.get('effective_date')
                if effective_date:
                    effective_dt = pd.to_datetime(effective_date)
                    _o['days_since_effective'] = (target_dt - effective_dt).days
                    _o['order_active'] = effective_dt <= target_dt
                else:
                    _o['days_since_effective'] = days_diff
                    _o['order_active'] = True
                relevant.append(_o)

        # Sort: lower priority first (e.g., 1 > 2), then more recent announcements first
        relevant.sort(key=lambda x: (x.get('priority', 9999), -x.get('days_since_announcement', 10_000)))
        return relevant[:max_context]

    def format_context_text(self, context_orders):
        """Format evacuation orders into readable context text"""
        if not context_orders:
            return "No evacuation orders found for this location."
        
        parts = []
        for o in context_orders:
            text = f"Evacuation: {o.get('order_type', 'Unknown')}"
            if o.get('days_since_announcement', 0) >= 0:
                text += f" (announced {o['days_since_announcement']} days ago"
                if 'days_since_effective' in o and o['days_since_effective'] != o['days_since_announcement']:
                    text += f", effective {o['days_since_effective']} days ago"
                text += ")"
            area = o.get('evacuation_area')
            if area and str(area).lower() != 'nan':
                text += f". Area: {area}"
            parts.append(text)
        return " | ".join(parts)

# -------------------- CLI --------------------
def parse_args():
    """Parse command line arguments with comprehensive options"""
    p = argparse.ArgumentParser(
        description="RAG-Enhanced Hurricane Forecasting with Mistral",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python rag_enhanced_mistral.py \\
        --knowledge-base evacuation_kb.json \\
        --test-data tampa_test.jsonl \\
        --output-dir ./results \\
        --city Tampa

    # With plots and custom model
    python rag_enhanced_mistral.py \\
        --base-model mistralai/Mistral-7B-v0.1.1 \\
        --knowledge-base evacuation_kb.json \\
        --test-data tampa_test.jsonl \\
        --output-dir ./results \\
        --city Tampa \\
        --generate-plots \\
        --use-quantization
        """
    )
    
    # Required arguments
    p.add_argument("--knowledge-base", required=True, 
                   help="Path to evacuation knowledge base JSON file")
    p.add_argument("--test-data", required=True, 
                   help="Test JSONL file with hurricane impact data")
    p.add_argument("--output-dir", required=True, 
                   help="Output directory for all results and reports")
    p.add_argument("--city", required=True, 
                   help="City name for this evaluation run")
    
    # Model configuration
    p.add_argument("--base-model", default="mistralai/Mistral-7B-v0.1",
                   help="Base model name or local path (default: Mistral 7B base)")
    p.add_argument("--use-quantization", action="store_true",
                   help="Use 4-bit quantization to reduce memory usage")
    p.add_argument("--local-files-only", action="store_true", 
                   help="Force offline model loading")
    
    # Generation parameters
    p.add_argument("--max-new-tokens", type=int, default=32,
                   help="Maximum tokens to generate (default: 32)")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Generation temperature (default: 0.1 for consistency)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size for inference (default: 8)")
    
    # Analysis options
    p.add_argument("--max-samples", type=int,
                   help="Optional limit on number of test samples for quick testing")
    p.add_argument("--generate-plots", action="store_true",
                   help="Generate performance visualization plots")
    p.add_argument("--num-plots", type=int, default=10,
                   help="Number of best/worst plots to generate (default: 10)")
    
    return p.parse_args()

# -------------------- Prompt Engineering --------------------
def create_rag_enhanced_prompt(record, evacuation_context):
    """
    Create RAG-enhanced prompt combining visit data with evacuation context
    
    This prompt engineering approach provides the model with:
    1. Historical visit patterns (13 days)
    2. Location and category information
    3. Hurricane timeline context
    4. Evacuation order context from knowledge base
    """
    values = ", ".join(f"{float(v):.1f}" for v in record['prev_13_values'])
    return (
        "You are an expert forecaster. Given 13 daily visit counts and evacuation context, predict day 14. "
        "Reply ONLY: 'PREDICTION: <number> CONFIDENCE: <0-100>%'.\n"
        f"Location: {record.get('location_name', '')}\n"
        f"Category: {record.get('top_category', '')}\n"
        f"City: {record.get('city', '')}\n"
        f"SeriesStart: {record.get('series_start_date', '')}\n"
        f"Landfall: {record.get('landfall_date', '')}\n"
        f"TargetDate: {record.get('target_date_d14', '')}\n"
        f"Evacuation Context: {evacuation_context}\n"
        f"Visits(1-13): {values}\n"
        "PREDICTION: "
    )

def parse_prediction(text):
    """
    Parse model output to extract prediction and confidence
    
    Implements multiple fallback strategies to handle various output formats
    """
    t = (text or "").strip()
    
    # Primary pattern: PREDICTION: X CONFIDENCE: Y
    m = re.search(r'PREDICTION:\s*([0-9]*\.?[0-9]+)', t, re.I)
    c = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', t, re.I)
    
    if m:
        pred = float(m.group(1))
        conf = float(c.group(1)) if c else 50.0
        pred = max(0.0, min(pred, 10000.0))
        conf = max(0.0, min(conf, 100.0)) / 100.0
        return pred, conf
    
    # Fallback: first number in the text
    nums = re.findall(r'([0-9]+\.?[0-9]*)', t)
    if nums:
        pred = max(0.0, min(float(nums[0]), 10000.0))
        return pred, 0.30  # Low confidence for fallback
    
    return None, None

# -------------------- Model Setup --------------------
def setup_model_and_pipeline(model_name, use_quantization=False, local_files_only=False, temperature=0.1, max_new_tokens=32):
    """
    Setup Mistral model with optional quantization and create generation pipeline
    
    Args:
        model_name: Model identifier or path
        use_quantization: Whether to use 4-bit quantization
        local_files_only: Whether to force offline loading
        temperature: Generation temperature
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Configured text generation pipeline
    """
    print(f"Setting up model: {model_name}")
    print(f"Quantization: {'Enabled' if use_quantization else 'Disabled'}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("[WARN] CUDA not available; running on CPU may be slow.")

    model_kwargs = {"local_files_only": local_files_only}
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Setup quantization if requested
    model_load_kwargs = dict(model_kwargs)
    if use_quantization and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_load_kwargs["quantization_config"] = quantization_config
        model_load_kwargs["device_map"] = "auto"
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_load_kwargs["torch_dtype"] = dtype
        if torch.cuda.is_available():
            model_load_kwargs["device_map"] = "auto"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)

    # Create pipeline
    pipeline_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "do_sample": True,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }
    
    if torch.cuda.is_available():
        pipeline_kwargs["device_map"] = "auto"

    pipe = pipeline("text-generation", **pipeline_kwargs)
    
    print("Model and pipeline setup complete")
    return pipe

# -------------------- Evaluation --------------------
def evaluate_rag_model(pipe, evac_kb, test_records, city_name, max_samples=None):
    """
    Evaluate RAG-enhanced model on test data
    
    Args:
        pipe: Text generation pipeline
        evac_kb: Evacuation knowledge base
        test_records: List of test samples
        city_name: City name for context
        max_samples: Optional limit on samples
        
    Returns:
        DataFrame with results and evaluation metrics
    """
    if max_samples:
        test_records = test_records[:max_samples]
    
    print(f"Processing {len(test_records)} test samples for {city_name}...")
    
    results = []
    failed_predictions = 0
    start_time = time.time()

    for idx, record in enumerate(test_records):
        try:
            # Get evacuation context
            fips_code = record.get('fips_county_code')
            target_date = record.get('target_date_d14', record.get('actual_target_date'))
            
            ctx_orders = evac_kb.get_context_for_location(fips_code, target_date)
            evac_text = evac_kb.format_context_text(ctx_orders)

            # Create RAG-enhanced prompt
            prompt = create_rag_enhanced_prompt(record, evac_text)

            # Generate prediction
            try:
                gen_result = pipe(prompt)
                if isinstance(gen_result, list) and gen_result:
                    generated_text = gen_result[0].get("generated_text", "").strip()
                elif isinstance(gen_result, dict):
                    generated_text = gen_result.get("generated_text", "").strip()
                else:
                    generated_text = str(gen_result).strip()
            except Exception as gen_error:
                print(f"[WARN] Generation failed for sample {idx}: {gen_error}")
                generated_text = ""

            # Parse prediction
            pred, conf = parse_prediction(generated_text)

            if pred is not None:
                y_true = float(record["y_true_d14"])
                abs_error = abs(y_true - pred)
                pct_error = (abs_error / max(y_true, 1)) * 100 if y_true != 0 else abs_error * 100

                # Calculate individual metrics using our compute_metric functions
                smape_mean, _ = compute_smape(np.array([y_true]), np.array([pred]))
                rmsle_val = compute_rmsle(np.array([y_true]), np.array([pred]))

                results.append({
                    "placekey": record.get("placekey", ""),
                    "location_name": record.get("location_name", ""),
                    "top_category": record.get("top_category", ""),
                    "city": record.get("city", ""),
                    "latitude": record.get("latitude", np.nan),
                    "longitude": record.get("longitude", np.nan),
                    "series_start_date": record.get("series_start_date", ""),
                    "landfall_date": record.get("landfall_date", ""),
                    "target_date_d14": record.get("target_date_d14", ""),
                    "actual_target_date": record.get("actual_target_date", record.get("target_date_d14", "")),
                    "target_days_after_landfall": record.get("target_days_after_landfall", None),
                    "time_periods_used": record.get("time_periods_used", ""),
                    "fips_code": fips_code,
                    "y_true_d14": y_true,
                    "y_pred_d14": float(pred),
                    "confidence_score": float(conf),
                    "absolute_error": float(abs_error),
                    "percent_error": float(pct_error),
                    "smape_individual": smape_mean,
                    "rmsle_individual": rmsle_val ** 2,
                    "prev_13_values": record["prev_13_values"],
                    "source_city": city_name,
                    "evacuation_context": evac_text,
                    "has_evacuation": len(ctx_orders) > 0,
                    "num_evacuation_orders": len(ctx_orders),
                    "llm_text": generated_text[:200]
                })
            else:
                failed_predictions += 1
                if idx < 5:  # Print first few failures
                    print(f"[WARN] Could not parse prediction for sample {idx}")
                    print(f"Generated: {generated_text[:100]}...")

            # Progress updates
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                success_rate = ((idx + 1 - failed_predictions) / (idx + 1)) * 100
                print(f"Progress: {idx + 1}/{len(test_records)} samples | "
                      f"Success rate: {success_rate:.1f}% | "
                      f"Time: {elapsed/60:.1f}min")

        except Exception as e:
            print(f"[ERROR] Processing sample {idx}: {e}")
            failed_predictions += 1
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate aggregate metrics
    evaluation_metrics = {}
    if not results_df.empty:
        valid_df = results_df.dropna(subset=["y_true_d14", "y_pred_d14"])
        
        if not valid_df.empty:
            y_true = valid_df["y_true_d14"].values
            y_pred = valid_df["y_pred_d14"].values
            
            # Use our standardized metric functions
            mae = compute_mae(y_true, y_pred)
            rmse = compute_rmse(y_true, y_pred)
            smape_mean, smape_median = compute_smape(y_true, y_pred)
            rmsle = compute_rmsle(y_true, y_pred)
            
            evaluation_metrics = {
                'mae': mae,
                'rmse': rmse,
                'smape_mean': smape_mean,
                'smape_median': smape_median,
                'rmsle': rmsle,
                'total_samples': len(test_records),
                'successful_predictions': len(results_df),
                'failed_predictions': failed_predictions,
                'success_rate': (len(results_df) / len(test_records)) * 100,
                'samples_with_evacuation': int(sum(1 for r in results if r['has_evacuation'])),
                'processing_time_minutes': (time.time() - start_time) / 60
            }

    return results_df, evaluation_metrics

# -------------------- Main Execution --------------------
def main():
    """Main execution function with comprehensive logging and error handling"""
    args = parse_args()
    
    # Setup output directory with organized structure
    output_dir = Path(args.output_dir)
    model_name_clean = args.base_model.split('/')[-1]
    results_dir = output_dir / f"rag_mistral_results_{args.city}"
    csv_dir = results_dir / "csv_outputs"
    reports_dir = results_dir / "reports"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("RAG-Enhanced Hurricane Forecasting with Mistral")
    print("=" * 80)
    print(f"City: {args.city}")
    print(f"Base model: {args.base_model}")
    print(f"Knowledge base: {args.knowledge_base}")
    print(f"Test data: {args.test_data}")
    print(f"Output directory: {results_dir}")
    print(f"Quantization: {'Enabled' if args.use_quantization else 'Disabled'}")
    
    execution_start = time.time()

    try:
        # Load evacuation knowledge base
        print(f"\n{'-'*50}")
        print("Loading Evacuation Knowledge Base")
        print(f"{'-'*50}")
        evac_kb = EvacuationKB(args.knowledge_base)

        # Setup model and pipeline
        print(f"\n{'-'*50}")
        print("Setting Up Model and Pipeline")
        print(f"{'-'*50}")
        pipe = setup_model_and_pipeline(
            args.base_model, 
            args.use_quantization, 
            args.local_files_only,
            args.temperature,
            args.max_new_tokens
        )

        # Load test data
        print(f"\n{'-'*50}")
        print("Loading Test Data")
        print(f"{'-'*50}")
        test_records = []
        with open(args.test_data, 'r', encoding='ascii', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_records.append(json.loads(line))
        
        print(f"Loaded {len(test_records)} test samples")
        if args.max_samples:
            print(f"Will process first {args.max_samples} samples for testing")

        # Run evaluation
        print(f"\n{'-'*50}")
        print("Running RAG-Enhanced Evaluation")
        print(f"{'-'*50}")
        results_df, metrics = evaluate_rag_model(
            pipe, evac_kb, test_records, args.city, args.max_samples
        )

        # Print results summary
        print(f"\n{'-'*50}")
        print("Evaluation Results Summary")
        print(f"{'-'*50}")
        if metrics:
            print(f"Total samples: {metrics['total_samples']}")
            print(f"Successful predictions: {metrics['successful_predictions']}")
            print(f"Failed predictions: {metrics['failed_predictions']}")
            print(f"Success rate: {metrics['success_rate']:.1f}%")
            print(f"Samples with evacuation context: {metrics['samples_with_evacuation']}")
            print(f"Processing time: {metrics['processing_time_minutes']:.1f} minutes")
            print()
            print("Performance Metrics:")
            print(f"  MAE: {metrics['mae']:.3f}")
            print(f"  RMSE: {metrics['rmse']:.3f}")
            print(f"  sMAPE (mean): {metrics['smape_mean']:.3f}%")
            print(f"  sMAPE (median): {metrics['smape_median']:.3f}%")
            print(f"  RMSLE: {metrics['rmsle']:.3f}")

        # Save results and generate reports
        print(f"\n{'-'*50}")
        print("Generating Reports and Visualizations")
        print(f"{'-'*50}")
        
        if not results_df.empty:
            # Save main predictions CSV
            main_output_path = csv_dir / "rag_predictions.csv"
            results_df.to_csv(main_output_path, index=False)
            print(f"Main predictions saved: {main_output_path}")

            # Generate comprehensive reports using summary_report module
            summary_results = generate_all_reports(results_df, reports_dir, f"rag_{args.city}")
            print(f"Comprehensive reports generated in: {reports_dir}")

            # Generate performance plots if requested
            if args.generate_plots:
                plots_dir = results_dir / "performance_plots"
                plots_dir.mkdir(exist_ok=True)
                plot_best_worst_pois(
                    csv_path=main_output_path,
                    output_dir=plots_dir,
                    num_plots=args.num_plots,
                    show_confidence=False
                )
                print(f"Performance plots saved: {plots_dir}")

        # Save execution summary
        execution_summary = {
            "script_info": {
                "script_name": "rag_enhanced_mistral.py",
                "execution_date": pd.Timestamp.now().isoformat(),
                "total_execution_time_minutes": (time.time() - execution_start) / 60
            },
            "configuration": {
                "city": args.city,
                "base_model": args.base_model,
                "knowledge_base": str(Path(args.knowledge_base).name),
                "test_data": str(Path(args.test_data).name),
                "use_8bit": args.use_8bit,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "max_samples": args.max_samples,
                "generate_plots": args.generate_plots
            },
            "results": metrics if metrics else {},
            "file_outputs": {
                "main_predictions": str(main_output_path.relative_to(results_dir)) if not results_df.empty else None,
                "reports_directory": str(reports_dir.relative_to(results_dir)),
                "plots_directory": str(Path("performance_plots")) if args.generate_plots else None
            }
        }
        
        summary_path = results_dir / "rag_execution_summary.json"
        with open(summary_path, "w") as f:
            json.dump(execution_summary, f, indent=2, default=str)

        # Final summary
        print(f"\n{'-'*50}")
        print("Execution Complete")
        print(f"{'-'*50}")
        print(f"All results saved to: {results_dir}")
        print(f"Execution summary: {summary_path}")
        print(f"Total execution time: {(time.time() - execution_start)/60:.1f} minutes")
        
        if not results_df.empty:
            print(f"\nKey outputs:")
            print(f"  - Predictions CSV: {main_output_path}")
            print(f"  - Detailed reports: {reports_dir}")
            if args.generate_plots:
                print(f"  - Performance plots: {results_dir / 'performance_plots'}")

    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
