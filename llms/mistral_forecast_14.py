#!/usr/bin/env python3
# Run Mistral 7B (quantized) on prepared 13->14 windows and save predictions.

import json, argparse, math, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# Import functions from external modules
from compute_metric import compute_mae, compute_rmse, compute_smape, compute_rmsle
from summary_report import generate_all_reports
from visual_plot import plot_best_worst_pois

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Prepared JSONL from data_prep.py")
    p.add_argument("--output", required=True, help="final_predictions.csv")
    p.add_argument("--city", required=True, help="City name for this test run")
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--use-8bit", action="store_true", help="Use 8-bit instead of 4-bit")
    p.add_argument("--generate-plots", action="store_true", help="Generate performance plots for best/worst POIs")
    p.add_argument("--num-plots", type=int, default=10, help="Number of best/worst plots to generate")
    return p.parse_args()

def create_prompt(prev13, loc, cat, series_start, landfall, target_date):
    values = ", ".join(f"{float(v):.1f}" for v in prev13)
    return (
        "You are an expert forecaster. Given 13 daily visit counts, predict day 14. "
        "Reply ONLY: 'PREDICTION: <number> CONFIDENCE: <0-100>%'.\n"
        f"Location: {loc}\nCategory: {cat}\n"
        f"SeriesStart: {series_start}\n"
        f"Landfall: {landfall}\n"
        f"TargetDate: {target_date}\n"
        f"Visits(1-13): {values}\n"
        "PREDICTION: "
    )

def parse_pred(text, fallback):
    m = re.search(r'PREDICTION:\s*([0-9]*\.?[0-9]+)', text, re.I)
    c = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', text, re.I)
    if m:
        pred = float(m.group(1))
        conf = float(c.group(1)) if c else 50.0
        pred = max(0.0, min(pred, 10000.0))
        return pred, conf/100.0
    # fallback: first number
    nums = re.findall(r'([0-9]*\.?[0-9]+)', text)
    if nums:
        pred = max(0.0, min(float(nums[0]), 10000.0))
        return pred, 0.5
    return float(fallback), 0.3

def calculate_individual_smape(y_true, y_pred):
    """Calculate sMAPE for a single prediction using imported function"""
    smape_mean, _ = compute_smape(np.array([y_true]), np.array([y_pred]))
    return smape_mean

def calculate_individual_rmsle(y_true, y_pred):
    """Calculate RMSLE for a single prediction using imported function"""
    rmsle = compute_rmsle(np.array([y_true]), np.array([y_pred]))
    return rmsle ** 2

def load_prepared(jsonl_path):
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    df = load_prepared(args.input)
    if df.empty:
        print("[ERROR] No records in prepared JSONL.")
        sys.exit(1)

    print(f"Processing city: {args.city}")
    print(f"Records: {len(df)}")

    # Quantized load
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    bnb = BitsAndBytesConfig(load_in_8bit=args.use_8bit) if args.use_8bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True
    )
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", trust_remote_code=True, quantization_config=bnb)

    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        device_map="auto",
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )

    # Warm-up
    _ = gen("Warm up. PREDICTION:", max_new_tokens=4)

    # Build prompts
    prompts, meta = [], []
    for i, r in df.iterrows():
        prompts.append(create_prompt(
            r["prev_13_values"], r.get("location_name", ""), r.get("top_category",""),
            r["series_start_date"], r["landfall_date"], r["target_date_d14"]
        ))
        meta.append((r["placekey"], r.get("city",""), r.get("latitude", np.nan), r.get("longitude", np.nan),
                     r.get("location_name",""), r.get("top_category",""), r["y_true_d14"],
                     r["prev_13_values"], r["series_start_date"], r["landfall_date"], r["target_date_d14"],
                     r.get("actual_target_date", r["target_date_d14"]), r.get("target_days_after_landfall", None),
                     r.get("time_periods_used","")))

    # Batched inference
    out_rows = []
    bs = max(1, args.batch_size)
    t0 = time.time()
    for s in range(0, len(prompts), bs):
        batch = prompts[s:s+bs]
        outs = gen(batch, return_full_text=False)
        for (pk, city, lat, lon, name, cat, y_true, prev13, s_start, lfall, tgt_d, act_tgt, days_after, tp_used), o in zip(meta[s:s+bs], outs):
            text = (o[0]["generated_text"] if isinstance(o, list) else o["generated_text"]).strip()
            pred, conf = parse_pred(text, prev13[-1] if prev13 else 0.0)
            abs_err = abs(y_true - pred)
            pct_err = (abs_err / max(y_true, 1)) * 100 if y_true != 0 else np.nan
            
            # Calculate individual sMAPE and RMSLE for each prediction
            smape_individual = calculate_individual_smape(y_true, pred)
            rmsle_individual = calculate_individual_rmsle(y_true, pred)
            
            row_data = {
                "placekey": pk,
                "city": city,
                "latitude": lat,
                "longitude": lon,
                "location_name": name,
                "top_category": cat,
                "series_start_date": s_start,
                "landfall_date": lfall,
                "target_date_d14": tgt_d,
                "actual_target_date": act_tgt,
                "target_days_after_landfall": days_after,
                "time_periods_used": tp_used,
                "y_true_d14": y_true,
                "y_pred_d14": pred,
                "confidence_score": conf,
                "absolute_error": abs_err,
                "percent_error": pct_err,
                "smape_individual": smape_individual,
                "rmsle_individual": rmsle_individual,
                "prev_13_values": prev13,
                "source_city": args.city,
                "llm_text": text[:200]
            }
            out_rows.append(row_data)

    elapsed = time.time() - t0
    pred_df = pd.DataFrame(out_rows)

    # Create organized output directory structure with city name
    output_path = Path(args.output)
    base_output_dir = output_path.parent
    model_name = args.model.split('/')[-1]
    
    # Create city-specific results directory
    results_dir = base_output_dir / f"mistral_results_{model_name}" / f"city_{args.city}"
    csv_dir = results_dir / "csv_outputs"
    reports_dir = results_dir / "reports"
    
    csv_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    # Save main predictions CSV (same name for all cities)
    main_output_path = csv_dir / output_path.name
    pred_df.to_csv(main_output_path, index=False)
    
    # Generate all summary reports
    results = generate_all_reports(pred_df, reports_dir, f"{args.city}_{output_path.stem}")
    
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
    
    # Print results
    print(f"\nAll files saved to: {results_dir}")
    print(f"Main predictions: {main_output_path}")
    
    if results:
        metrics = results['metrics']
        file_paths = results['file_paths']
        
        print(f"Reports directory: {reports_dir}")
        print(f"  - Detailed report: {file_paths['detailed_report'].name}")
        print(f"  - Best/Worst summary: {file_paths['summary'].name}")
        print(f"  - Overall statistics: {file_paths['overall_stats'].name}")
        print(f"  - Metrics JSON: {file_paths['metrics_json'].name}")
        
        if args.generate_plots:
            print(f"Plots directory: {plots_dir}")
        
        print(f"\nPerformance Summary for {args.city}:")
        print(f"Records: {len(pred_df)} total | Time: {elapsed/60:.2f} min")
        print(f"MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f}")
        print(f"sMAPE (mean): {metrics['smape_mean']:.3f}% | sMAPE (median): {metrics['smape_median']:.3f}%")
        print(f"RMSLE: {metrics['rmsle']:.3f}")
    else:
        print("No valid predictions for summary reports.")
        print(f"Records: {len(pred_df)} | Time: {elapsed/60:.2f} min")

if __name__ == "__main__":
    main()
