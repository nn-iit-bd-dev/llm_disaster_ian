#!/usr/bin/env python3
# Fine-tune Llama 3 8B for hurricane impact forecasting using PEFT/LoRA

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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-input", required=True, help="Training JSONL file")
    p.add_argument("--test-input", required=True, help="Test JSONL file")
    p.add_argument("--outdir", required=True, help="Output directory for fine-tuned model")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    p.add_argument("--eval-batch-size", type=int, default=16, help="Evaluation batch size")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    p.add_argument("--limit", type=int, help="Limit number of training samples")
    p.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    p.add_argument("--resume-from", help="Resume training from checkpoint")
    p.add_argument("--evaluate-only", action="store_true", help="Only evaluate existing model")
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
    """Load JSONL file and return DataFrame"""
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No records in {jsonl_path}")
    
    if limit and limit > 0:
        df = df.head(limit)
        print(f"Using subset: {len(df)} samples from {jsonl_path}")
    
    return df

def prepare_training_data(train_df):
    """Convert training DataFrame to training prompts"""
    training_texts = []
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
        training_texts.append(prompt)
    
    print(f"Prepared {len(training_texts)} training examples")
    return training_texts

def create_datasets(train_texts, tokenizer, max_length):
    """Create tokenized datasets for training"""
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    train_dataset = Dataset.from_dict({"text": train_texts})
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    
    return train_dataset

def setup_model_and_tokenizer(model_name, use_8bit=False):
    """Setup quantized model and tokenizer with LoRA"""
    
    if use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    return model, tokenizer

def setup_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.1):
    """Setup LoRA configuration"""
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
    text = text.strip()
    
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
        pred = max(0.0, min(float(pred_match.group(1)), 1000.0))
        conf = float(c.group(1)) if c else 50.0
        return pred, conf/100.0
    
    # Extract first number
    nums = re.findall(r'([0-9]+\.?[0-9]*)', text)
    if nums:
        pred = max(0.0, min(float(nums[0]), 10000.0))
        return pred, 0.3
    
    print(f"Warning: Could not parse prediction from: '{text[:100]}'")
    return None, None

def evaluate_model(model, tokenizer, test_df, outdir):
    """Evaluate fine-tuned model on test set"""
    from transformers import pipeline
    
    print(f"Evaluating on {len(test_df)} test samples...")
    
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
    y_true_list = []
    y_pred_list = []
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
            
            if idx < 3:
                print(f"Sample {idx+1} generated: '{generated_text}'")
            
            pred, conf = parse_prediction(generated_text)
            
            if pred is not None:
                y_true = row["y_true_d14"]
                y_true_list.append(y_true)
                y_pred_list.append(pred)
                
                abs_error = abs(y_true - pred)
                pct_error = (abs_error / max(y_true, 1)) * 100 if y_true != 0 else np.nan
                
                # Individual sMAPE
                denominator = (abs(y_true) + abs(pred)) / 2
                smape_ind = (abs(y_true - pred) / denominator) * 100 if denominator != 0 else np.nan
                
                # Individual RMSLE
                epsilon = 1e-8
                rmsle_ind = np.sqrt((np.log(max(y_true, 0) + epsilon) - np.log(max(pred, 0) + epsilon)) ** 2)
                
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
                    "y_true_d14": y_true,
                    "y_pred_d14": pred,
                    "confidence_score": conf,
                    "absolute_error": abs_error,
                    "percent_error": pct_error,
                    "smape_individual": smape_ind,
                    "rmsle_individual": rmsle_ind,
                    "generated_text": generated_text[:200]
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
    
    # Calculate aggregate metrics
    if not results_df.empty and len(y_true_list) > 0:
        y_true_arr = np.array(y_true_list)
        y_pred_arr = np.array(y_pred_list)
        
        # Basic metrics
        mae = np.mean(np.abs(y_true_arr - y_pred_arr))
        rmse = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
        
        # sMAPE calculation
        denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
        mask = denominator != 0
        smape_values = np.full_like(y_true_arr, np.nan, dtype=float)
        smape_values[mask] = (np.abs(y_true_arr[mask] - y_pred_arr[mask]) / denominator[mask]) * 100
        
        smape_mean = np.nanmean(smape_values)
        smape_median = np.nanmedian(smape_values)
        
        # RMSLE calculation
        epsilon = 1e-8
        y_true_log = np.log(np.maximum(y_true_arr, 0) + epsilon)
        y_pred_log = np.log(np.maximum(y_pred_arr, 0) + epsilon)
        rmsle = np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))
        
        overall_metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'smape_mean': float(smape_mean),
            'smape_median': float(smape_median),
            'rmsle': float(rmsle),
            'count': len(y_true_list)
        }
        
        print(f"\n=== Fine-tuned Model Evaluation ===")
        print(f"Test samples: {len(results_df)}")
        print(f"MAE: {overall_metrics['mae']:.3f}")
        print(f"RMSE: {overall_metrics['rmse']:.3f}")
        print(f"sMAPE (mean): {overall_metrics['smape_mean']:.3f}%")
        print(f"sMAPE (median): {overall_metrics['smape_median']:.3f}%")
        print(f"RMSLE: {overall_metrics['rmsle']:.3f}")
        
        # Save results
        results_df.to_csv(outdir / "finetuned_predictions.csv", index=False)
        
        # Category metrics
        if "top_category" in results_df.columns and len(results_df) > 0:
            try:
                category_metrics = results_df.groupby("top_category").agg({
                    "y_true_d14": "count",
                    "absolute_error": ["mean", "std"],
                    "smape_individual": "mean",
                    "rmsle_individual": "mean",
                    "confidence_score": "mean"
                }).round(3)
                
                category_metrics.columns = ["count", "mean_ae", "std_ae", "mean_smape", "mean_rmsle", "mean_conf"]
                category_metrics = category_metrics.reset_index()
                category_metrics.to_csv(outdir / "category_metrics.csv", index=False)
            except Exception as e:
                print(f"Warning: Could not generate category metrics: {e}")
        
        return overall_metrics
    else:
        print("No valid predictions generated")
        return {}

def main():
    # Disable problematic integrations
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    
    args = parse_args()
    
    # Setup output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Hurricane Forecasting Fine-tuning ===")
    print(f"Model: {args.model}")
    print(f"Output: {outdir}")
    print(f"Train data: {args.train_input}")
    print(f"Test data: {args.test_input}")
    
    # Load train and test data separately
    print("\n=== Loading Data ===")
    train_df = load_jsonl_data(args.train_input, args.limit)
    test_df = load_jsonl_data(args.test_input)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    if args.evaluate_only:
        print("\n=== Evaluation Only Mode ===")
        model_path = outdir / "final_model"
        if not model_path.exists():
            print(f"Error: No model found at {model_path}")
            sys.exit(1)
            
        base_model, tokenizer = setup_model_and_tokenizer(args.model, args.use_8bit)
        model = PeftModel.from_pretrained(base_model, model_path)
        evaluate_model(model, tokenizer, test_df, outdir)
        return
    
    # Prepare training data
    train_texts = prepare_training_data(train_df)
    
    # Setup model and tokenizer
    print("\n=== Setting up Model ===")
    model, tokenizer = setup_model_and_tokenizer(args.model, args.use_8bit)
    
    # Apply LoRA
    print("\n=== Applying LoRA ===")
    model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # Create datasets
    print("\n=== Creating Datasets ===")
    train_dataset = create_datasets(train_texts, tokenizer, args.max_length)
    
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
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to=[],
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
    
    # Evaluate on test set
    print(f"\n=== Evaluating Fine-tuned Model ===")
    evaluation_metrics = evaluate_model(trainer.model, tokenizer, test_df, outdir)
    
    # Save training summary
    summary = {
        "model": args.model,
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "training_time_minutes": training_time / 60,
        "evaluation_metrics": evaluation_metrics
    }
    
    with open(outdir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {outdir / 'training_summary.json'}")

if __name__ == "__main__":
    main()
