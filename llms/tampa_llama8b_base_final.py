#!/usr/bin/env python3
# Run Llama 3 8B (quantized) on prepared 13->14 windows and save predictions.

import json, argparse, math, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Prepared JSONL from data_prep.py")
    p.add_argument("--output", required=True, help="final_predictions.csv")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--use-8bit", action="store_true", help="Use 8-bit instead of 4-bit")
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
        pred = max(0.0, min(pred, 1000.0))
        return pred, conf/100.0
    # fallback: first number
    nums = re.findall(r'([0-9]*\.?[0-9]+)', text)
    if nums:
        pred = max(0.0, min(float(nums[0]), 1000.0))
        return pred, 0.5
    return float(fallback), 0.3

def calculate_individual_smape(y_true, y_pred):
    """Calculate sMAPE for a single prediction"""
    denominator = (abs(y_true) + abs(y_pred)) / 2
    if denominator != 0:
        return (abs(y_true - y_pred) / denominator) * 100
    else:
        return np.nan

def calculate_individual_rmsle(y_true, y_pred):
    """Calculate RMSLE for a single prediction"""
    epsilon = 1e-8
    y_true_log = np.log(max(y_true, 0) + epsilon)
    y_pred_log = np.log(max(y_pred, 0) + epsilon)
    return (y_true_log - y_pred_log) ** 2

def calculate_aggregate_metrics(y_true_series, y_pred_series):
    """Calculate aggregate metrics including sMAPE mean/median and RMSLE"""
    y_true = np.array(y_true_series)
    y_pred = np.array(y_pred_series)
    
    # Basic metrics
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    
    # sMAPE calculation for all predictions
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    smape_values = np.full_like(y_true, np.nan, dtype=float)
    smape_values[mask] = (np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    smape_mean = float(np.nanmean(smape_values))
    smape_median = float(np.nanmedian(smape_values))
    
    # RMSLE calculation
    epsilon = 1e-8
    y_true_log = np.log(np.maximum(y_true, 0) + epsilon)
    y_pred_log = np.log(np.maximum(y_pred, 0) + epsilon)
    rmsle = float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))
    
    return mae, rmse, smape_mean, smape_median, rmsle

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
            
            out_rows.append({
                "placekey": pk, "city": city, "latitude": lat, "longitude": lon,
                "location_name": name, "top_category": cat,
                "series_start_date": s_start, "landfall_date": lfall,
                "target_date_d14": tgt_d, "actual_target_date": act_tgt,
                "target_days_after_landfall": days_after, "time_periods_used": tp_used,
                "y_true_d14": y_true, "y_pred_d14": pred, "confidence_score": conf,
                "absolute_error": abs_err, "percent_error": pct_err,
                "smape_individual": smape_individual, "rmsle_individual": rmsle_individual,
                "prev_13_values": prev13,
                "llm_text": text[:200]
            })

    elapsed = time.time() - t0
    pred_df = pd.DataFrame(out_rows)

    # Calculate comprehensive metrics
    valid = pred_df.dropna(subset=["y_true_d14","y_pred_d14"])
    if not valid.empty:
        mae, rmse, smape_mean, smape_median, rmsle = calculate_aggregate_metrics(
            valid["y_true_d14"].values, valid["y_pred_d14"].values
        )
    else:
        mae = rmse = smape_mean = smape_median = rmsle = float("nan")

    # Create output directory
    output_path = Path(args.output)
    output_dir = output_path.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Define all output file paths in the output directory
    main_output_path = output_dir / output_path.name
    detailed_report_path = output_dir / f"{output_path.stem}_detailed_report.csv"
    summary_path = output_dir / f"{output_path.stem}_summary.csv"
    stats_path = output_dir / f"{output_path.stem}_overall_stats.csv"
    
    # Save main predictions CSV
    pred_df.to_csv(main_output_path, index=False)
    
    # Detailed report with expanded input series
    detailed_rows = []
    for _, row in pred_df.iterrows():
        # Parse the prev_13_values if it's stored as string
        prev13 = row.get('prev_13_values', [])
        if isinstance(prev13, str):
            try:
                prev13 = eval(prev13)
            except:
                prev13 = []
        
        # Create columns for each day in the input series
        detailed_row = {
            'placekey': row['placekey'],
            'location_name': row['location_name'],
            'top_category': row['top_category'],
            'city': row['city'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'series_start_date': row['series_start_date'],
            'landfall_date': row['landfall_date'],
            'target_date_d14': row['target_date_d14'],
            'actual_target_date': row['actual_target_date'],
            'target_days_after_landfall': row['target_days_after_landfall']
        }
        
        # Add input series (days 1-13)
        for i in range(13):
            day_val = prev13[i] if i < len(prev13) else np.nan
            detailed_row[f'input_day_{i+1:02d}'] = day_val
        
        # Add prediction results
        detailed_row.update({
            'y_true_d14': row['y_true_d14'],
            'y_pred_d14': row['y_pred_d14'],
            'confidence_score': row['confidence_score'],
            'absolute_error': row['absolute_error'],
            'percent_error': row['percent_error'],
            'smape_individual': row['smape_individual'],
            'rmsle_individual': row['rmsle_individual'],
            'llm_text': row['llm_text']
        })
        
        detailed_rows.append(detailed_row)
    
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(detailed_report_path, index=False)
    
    # Create summary report with top 10 best and worst
    if not valid.empty:
        # Sort by absolute error for best/worst performance
        valid_sorted = valid.sort_values('absolute_error')
        
        # Top 10 best (lowest absolute error)
        top_10_best = valid_sorted.head(10).copy()
        top_10_best['rank'] = range(1, min(11, len(top_10_best) + 1))
        top_10_best['performance'] = 'Best'
        
        # Top 10 worst (highest absolute error)
        top_10_worst = valid_sorted.tail(10).copy()
        top_10_worst = top_10_worst.sort_values('absolute_error', ascending=False)
        top_10_worst['rank'] = range(1, min(11, len(top_10_worst) + 1))
        top_10_worst['performance'] = 'Worst'
        
        # Combine best and worst
        summary_df = pd.concat([top_10_best, top_10_worst], ignore_index=True)
        
        # Select key columns for summary
        summary_columns = [
            'performance', 'rank', 'placekey', 'location_name', 'top_category', 
            'city', 'y_true_d14', 'y_pred_d14', 'absolute_error', 'percent_error',
            'smape_individual', 'rmsle_individual', 'confidence_score', 'target_days_after_landfall'
        ]
        
        summary_df = summary_df[summary_columns]
        summary_df.to_csv(summary_path, index=False)
        
        # Overall statistics with enhanced metrics
        overall_stats = {
            'metric': ['Total Records', 'Valid Records', 'MAE', 'RMSE', 
                      'sMAPE Mean (%)', 'sMAPE Median (%)', 'RMSLE',
                      'Mean Confidence', 'Median Absolute Error', 
                      'Best Absolute Error', 'Worst Absolute Error'],
            'value': [
                len(pred_df),
                len(valid),
                f"{mae:.3f}",
                f"{rmse:.3f}",
                f"{smape_mean:.3f}",
                f"{smape_median:.3f}",
                f"{rmsle:.3f}",
                f"{valid['confidence_score'].mean():.3f}",
                f"{valid['absolute_error'].median():.3f}",
                f"{valid['absolute_error'].min():.3f}",
                f"{valid['absolute_error'].max():.3f}"
            ]
        }
        
        # Add category-wise performance
        category_stats = valid.groupby('top_category').agg({
            'absolute_error': ['count', 'mean', 'median'],
            'smape_individual': 'mean',
            'rmsle_individual': 'mean'
        }).round(3)
        
        category_stats.columns = ['Count', 'Mean_AE', 'Median_AE', 'Mean_sMAPE', 'Mean_RMSLE']
        category_stats = category_stats.reset_index()
        
        # Save category stats
        with open(stats_path, 'w') as f:
            f.write("OVERALL STATISTICS\n")
            pd.DataFrame(overall_stats).to_csv(f, index=False)
            f.write("\n\nCATEGORY-WISE PERFORMANCE\n")
            category_stats.to_csv(f, index=False)
    else:
        print("No valid predictions to create summary report.")
    
    print(f"\nAll files saved to: {output_dir}")
    print(f"Main predictions: {main_output_path.name}")
    print(f"Detailed report: {detailed_report_path.name}")
    print(f"Summary report: {summary_path.name}")
    if not valid.empty:
        print(f"Overall stats: {stats_path.name}")
    print(f"Records: {len(pred_df)} | Time: {elapsed/60:.2f} min")
    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f}")
    print(f"sMAPE (mean): {smape_mean:.3f}% | sMAPE (median): {smape_median:.3f}%")
    print(f"RMSLE: {rmsle:.3f}")

if __name__ == "__main__":
    main()
