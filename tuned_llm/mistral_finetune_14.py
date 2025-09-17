#!/usr/bin/env python3
# Fine-tune Mistral 7B for hurricane impact forecasting using PEFT/LoRA

import json, argparse, math, re, sys, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Import our custom modules
from compute_metric import compute_mae, compute_rmse, compute_smape, compute_rmsle
from summary_report import generate_all_reports
from visual_plot import plot_best_worst_pois

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-input", required=True, help="Training JSONL file")
    p.add_argument("--test-input", required=True, help="Test JSONL file") 
    p.add_argument("--outdir", required=True, help="Output directory for fine-tuned model")
    p.add_argument("--city", required=True, help="City name for this fine-tuning run")
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    p.add_argument("--eval-batch-size", type=int, default=16, help="Evaluation batch size")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    p.add_argument("--limit", type=int, help="Limit number of samples for testing")
    p.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    p.add_argument("--resume-from", help="Resume training from checkpoint")
    p.add_argument("--evaluate-only", action="store_true", help="Only evaluate existing model")
    p.add_argument("--generate-plots", action="store_true", help="Generate performance plots")
    p.add_argument("--num-plots", type=int, default=10, help="Number of best/worst plots to generate")
    return p.parse_args()

def create_training_prompt(prev13, loc, cat, series_start, landfall, target_date, y_true):
    """Create training prompt with ground truth"""
    values = ", ".join(f"{float(v):.1f}" for v in prev13)
    return (
        "You are an expert forecaster. Given 13 daily visit counts, predict day 14. "
        "Reply ONLY: 'PREDICTION: <number> CONFIDENCE: <0-100>%'.\n"
        f"Location: {loc}\nCategory: {cat}\n"
        f"SeriesStart: {series_start}\n"
        f"Landfall: {landfall}\n"
        f"TargetDate: {target_date}\n"
        f"Visits(1-13): {values}\n"
        f"PREDICTION: {y_true:.1f} CONFIDENCE: 90%"
    )

def create_inference_prompt(prev13, loc, cat, series_start, landfall, target_date):
    """Create inference prompt without ground truth"""
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

def load_jsonl_data(jsonl_path, limit=None):
    """Load data from a single JSONL file"""
    rows = []
    with open(jsonl_path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No records in JSONL file: {jsonl_path}")
    
    # Apply limit if specified
    if limit and limit > 0:
        df = df.head(limit)
        print(f"Using subset: {len(df)} samples (limited from {len(rows)})")
    
    return df

def load_and_prepare_data(train_jsonl_path, test_jsonl_path, limit=None):
    """Load pre-split training and test data"""
    print("Loading pre-split training and test data...")
    
    # Load training data
    train_df = load_jsonl_data(train_jsonl_path, limit)
    print(f"Loaded {len(train_df)} training samples from {train_jsonl_path}")
    
    # Load test data
    test_df = load_jsonl_data(test_jsonl_path)
    print(f"Loaded {len(test_df)} test samples from {test_jsonl_path}")
    
    # Create training prompts from training data
    train_texts = []
    for _, row in train_df.iterrows():
        prompt = create_training_prompt(
            row["prev_13_values"], 
            row.get("location_name", ""), 
            row.get("top_category", ""),
            row["series_start_date"], 
            row["landfall_date"], 
            row["target_date_d14"],
            row["y_true_d14"]
        )
        train_texts.append(prompt)
    
    # Create test prompts from test data (for validation during training)
    test_texts = []
    for _, row in test_df.iterrows():
        prompt = create_training_prompt(
            row["prev_13_values"], 
            row.get("location_name", ""), 
            row.get("top_category", ""),
            row["series_start_date"], 
            row["landfall_date"], 
            row["target_date_d14"],
            row["y_true_d14"]
        )
        test_texts.append(prompt)
    
    print(f"Prepared {len(train_texts)} training prompts")
    print(f"Prepared {len(test_texts)} validation prompts")
    
    return train_texts, test_texts, test_df

def create_datasets(train_texts, test_texts, tokenizer, max_length):
    """Create tokenized datasets for training"""
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    test_dataset = Dataset.from_dict({"text": test_texts})
    
    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, test_dataset

def setup_model_and_tokenizer(model_name, use_8bit=False):
    """Setup quantized model and tokenizer with LoRA"""
    
    # Quantization config
    if use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    return model, tokenizer

def setup_lora(model, lora_r=8, lora_alpha=16, lora_dropout=0.1):  # More conservative settings
    """Setup LoRA configuration for Mistral"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def parse_prediction(text):
    """Parse prediction from model output with fallback handling"""
    # Clean the text
    text = text.strip()
    
    # Primary pattern: PREDICTION: number
    m = re.search(r'PREDICTION:\s*([0-9]*\.?[0-9]+)', text, re.I)
    c = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', text, re.I)
    
    if m:
        pred = float(m.group(1))
        conf = float(c.group(1)) if c else 50.0
        pred = max(0.0, min(pred, 10000.0))
        return pred, conf/100.0
    
    # Fallback patterns
    pred_match = re.search(r'PREDICTION[:\s]*([0-9]*\.?[0-9]+)', text, re.I)
    if pred_match:
        pred = max(0.0, min(float(pred_match.group(1)), 10000.0))
        conf = float(c.group(1)) if c else 50.0
        return pred, conf/100.0
    
    # Extract first number
    nums = re.findall(r'([0-9]+\.?[0-9]*)', text)
    if nums:
        pred = max(0.0, min(float(nums[0]), 10000.0))
        return pred, 0.3
    
    print(f"Warning: Could not parse prediction from: '{text[:100]}'")
    return None, None

def evaluate_model(model, tokenizer, test_df, outdir, city_name, generate_plots=False, num_plots=10):
    """Evaluate fine-tuned model on test set using our custom modules"""
    from transformers import pipeline
    
    print(f"Evaluating on {len(test_df)} test samples...")
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        do_sample=False,
        max_new_tokens=30,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    results = []
    failed_predictions = 0
    
    for idx, row in test_df.iterrows():
        prompt = create_inference_prompt(
            row["prev_13_values"], 
            row.get("location_name", ""), 
            row.get("top_category", ""),
            row["series_start_date"], 
            row["landfall_date"], 
            row["target_date_d14"]
        )
        
        try:
            output = pipe(prompt, return_full_text=False)
            generated_text = output[0]["generated_text"].strip()
            
            # Debug: print first few examples
            if idx < 3:
                print(f"Sample {idx+1} generated: '{generated_text}'")
            
            pred, conf = parse_prediction(generated_text)
            
            if pred is not None:
                y_true = row["y_true_d14"]
                abs_error = abs(y_true - pred)
                pct_error = (abs_error / max(y_true, 1)) * 100 if y_true != 0 else np.nan
                
                # Use our compute_metric functions for individual calculations
                smape_mean, _ = compute_smape(np.array([y_true]), np.array([pred]))
                rmsle_val = compute_rmsle(np.array([y_true]), np.array([pred]))
                
                results.append({
                    "placekey": row["placekey"],
                    "location_name": row.get("location_name", ""),
                    "top_category": row.get("top_category", ""),
                    "city": row.get("city", ""),
                    "latitude": row.get("latitude", np.nan),
                    "longitude": row.get("longitude", np.nan),
                    "series_start_date": row["series_start_date"],
                    "landfall_date": row["landfall_date"],
                    "target_date_d14": row["target_date_d14"],
                    "actual_target_date": row.get("actual_target_date", row["target_date_d14"]),
                    "target_days_after_landfall": row.get("target_days_after_landfall", None),
                    "time_periods_used": row.get("time_periods_used", ""),
                    "y_true_d14": y_true,
                    "y_pred_d14": pred,
                    "confidence_score": conf,
                    "absolute_error": abs_error,
                    "percent_error": pct_error,
                    "smape_individual": smape_mean,
                    "rmsle_individual": rmsle_val ** 2,
                    "prev_13_values": row["prev_13_values"],
                    "source_city": city_name,
                    "llm_text": generated_text[:200]
                })
            else:
                failed_predictions += 1
                if idx < 5:
                    print(f"Failed to parse prediction {idx+1}: '{generated_text}'")
                    
        except Exception as e:
            print(f"Error processing sample {idx+1}: {e}")
            failed_predictions += 1
            continue
    
    print(f"Successfully parsed {len(results)}/{len(test_df)} predictions")
    if failed_predictions > 0:
        print(f"Failed to parse {failed_predictions} predictions")
    
    results_df = pd.DataFrame(results)
    
    # Calculate aggregate metrics using our compute_metric functions
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
            
            overall_metrics = {
                'mae': mae,
                'rmse': rmse,
                'smape_mean': smape_mean,
                'smape_median': smape_median,
                'rmsle': rmsle,
                'count': len(valid_df)
            }
            
            print(f"\n=== Fine-tuned Mistral Evaluation ===")
            print(f"Test samples: {len(results_df)}")
            print(f"MAE: {overall_metrics['mae']:.3f}")
            print(f"RMSE: {overall_metrics['rmse']:.3f}")
            print(f"sMAPE (mean): {overall_metrics['smape_mean']:.3f}%")
            print(f"sMAPE (median): {overall_metrics['smape_median']:.3f}%")
            print(f"RMSLE: {overall_metrics['rmsle']:.3f}")
            
            # Create organized output structure
            model_name = "Mistral-7B-v0.1-finetuned"
            results_dir = outdir / f"evaluation_results_{city_name}"
            csv_dir = results_dir / "csv_outputs"
            reports_dir = results_dir / "reports"
            
            csv_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(exist_ok=True)
            
            # Save main predictions CSV
            main_output_path = csv_dir / "finetuned_predictions.csv"
            results_df.to_csv(main_output_path, index=False)
            
            # Generate comprehensive reports using summary_report module
            summary_results = generate_all_reports(results_df, reports_dir, f"finetuned_{city_name}")
            
            # Generate performance plots if requested
            if generate_plots:
                plots_dir = results_dir / "performance_plots"
                plots_dir.mkdir(exist_ok=True)
                plot_best_worst_pois(
                    csv_path=main_output_path,
                    output_dir=plots_dir,
                    num_plots=num_plots,
                    show_confidence=False
                )
                print(f"Performance plots saved to: {plots_dir}")
            
            print(f"Evaluation results saved to: {results_dir}")
            print(f"Main predictions: {main_output_path}")
            print(f"Reports directory: {reports_dir}")
            
            return overall_metrics
        else:
            print("No valid predictions found")
            return {}
    else:
        print("No predictions generated")
        return {}

def main():
    # Disable problematic integrations
    import os
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    
    args = parse_args()
    
    # Setup output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Hurricane Forecasting Fine-tuning with Mistral ===")
    print(f"Model: {args.model}")
    print(f"City: {args.city}")
    print(f"Output: {outdir}")
    
    # Load and prepare data
    print(f"\n=== Loading Data ===")
    print(f"Training: {args.train_input}")
    print(f"Test: {args.test_input}")
    train_texts, test_texts, test_df = load_and_prepare_data(
        args.train_input, args.test_input, args.limit
    )
    
    if args.evaluate_only:
        print("\n=== Evaluation Only Mode ===")
        # Load existing fine-tuned model
        model_path = outdir / "final_model"
        if not model_path.exists():
            print(f"Error: No model found at {model_path}")
            sys.exit(1)
            
        base_model, tokenizer = setup_model_and_tokenizer(args.model, args.use_8bit)
        model = PeftModel.from_pretrained(base_model, model_path)
        evaluate_model(model, tokenizer, test_df, outdir, args.city, args.generate_plots, args.num_plots)
        return
    
    # Setup model and tokenizer
    print("\n=== Setting up Mistral Model ===")
    model, tokenizer = setup_model_and_tokenizer(args.model, args.use_8bit)
    
    # Apply LoRA
    print("\n=== Applying LoRA ===")
    model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # Create datasets
    print("\n=== Creating Datasets ===")
    train_dataset, test_dataset = create_datasets(train_texts, test_texts, tokenizer, args.max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=outdir / "checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        logging_dir=outdir / "logs",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        max_grad_norm=0.3,
        resume_from_checkpoint=args.resume_from,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n=== Starting Training ({args.epochs} epochs) ===")
    start_time = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from)
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = outdir / "final_model"
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n=== Training Complete ===")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Model saved to: {final_model_path}")
    
    # Evaluate using our custom modules
    print(f"\n=== Evaluating Fine-tuned Mistral ===")
    
    # Run full evaluation for all cities
    print(f"Generating full evaluation results for {args.city}...")
    evaluation_metrics = evaluate_model(
        trainer.model, tokenizer, test_df, outdir, args.city, 
        args.generate_plots, args.num_plots
    )
    
    # Save training summary
    summary = {
        "model": args.model,
        "city": args.city,
        "training_samples": len(train_texts),
        "test_samples": len(test_texts),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "training_time_minutes": training_time / 60,
        "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", "N/A"),
        "evaluation_metrics": evaluation_metrics
    }
    
    with open(outdir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {outdir / 'training_summary.json'}")

if __name__ == "__main__":
    main()
