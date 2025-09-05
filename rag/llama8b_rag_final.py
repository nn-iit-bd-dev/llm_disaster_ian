#!/usr/bin/env python3
"""
RAG-Enhanced Hurricane Forecasting using fine-tuned LoRA adapter + evacuation knowledge base
"""

# ---- silence TF and keep transformers on PyTorch only ----
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import re
import sys

# -------------------- Evacuation KB --------------------
class EvacuationKB:
    def __init__(self, kb_json_path):
        kb_path = Path(kb_json_path)
        if not kb_path.exists():
            sys.exit(f"[ERROR] Knowledge base not found: {kb_path}")
        with open(kb_path, 'r', encoding='ascii', errors='ignore') as f:
            self.kb_data = json.load(f)
        self.entries = self.kb_data.get('entries', [])

    def get_context_for_location(self, fips_code, target_date, max_context=3):
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

        # sort: lower priority first (e.g., 1 > 2), then more recent announcements first
        relevant.sort(key=lambda x: (x.get('priority', 9999), -x.get('days_since_announcement', 10_000)))
        return relevant[:max_context]

    def format_context_text(self, context_orders):
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
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Local LoRA adapter directory (contains adapter_model.safetensors)")
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B", help="Base model name or local path")
    p.add_argument("--knowledge-base", required=True, help="Path to evacuation knowledge base JSON")
    p.add_argument("--test-data", required=True, help="Test JSONL file")
    p.add_argument("--output-dir", required=True, help="Output directory for results")
    p.add_argument("--max-samples", type=int, help="Optional cap on number of test samples")
    p.add_argument("--local-files-only", action="store_true", help="Force offline loads")
    return p.parse_args()

# -------------------- Prompt / Parse --------------------
def create_rag_enhanced_prompt(record, evacuation_context):
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

_num_pat = re.compile(r'([0-9]+\.?[0-9]*)')
def parse_prediction(text):
    t = (text or "").strip()
    m = re.search(r'PREDICTION:\s*([0-9]*\.?[0-9]+)', t, re.I)
    c = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', t, re.I)
    if m:
        pred = float(m.group(1))
        conf = float(c.group(1)) if c else 50.0
        pred = max(0.0, min(pred, 10000.0))
        return pred, conf / 100.0
    # fallback: first number
    nums = _num_pat.findall(t)
    if nums:
        pred = max(0.0, min(float(nums[0]), 10000.0))
        return pred, 0.30
    return None, None

# -------------------- Main --------------------
def main():
    args = parse_args()

    # paths & output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"[ERROR] --model-dir '{model_dir}' does not exist. "
                 f"Pass a valid local LoRA adapter folder (with adapter_model.safetensors).")

    print("=== RAG-Enhanced Hurricane Forecasting ===")
    print(f"Base model: {args.base_model}")
    print(f"Adapter dir: {model_dir}")
    print(f"Knowledge base: {args.knowledge_base}")
    print(f"Test data: {args.test_data}")

    # Load KB
    print("\nLoading evacuation knowledge base...")
    evac_kb = EvacuationKB(args.knowledge_base)
    print(f"Loaded {len(evac_kb.entries)} evacuation entries")

    # Hardware guard
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available; running on CPU may be slow. You can add --local-files-only if fully offline.")

    # Load tokenizer from base model
    print("\nLoading tokenizer & models...")
    tok_kwargs = {"use_fast": True, "local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **tok_kwargs)

    # Ensure pad/eos tokens are defined
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model then apply LoRA adapter
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=args.local_files_only
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(model_dir),
        local_files_only=True  # adapter is local
    )

    # Pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        do_sample=False,
        max_new_tokens=50,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Load test data
    print("\nLoading test data...")
    test_records = []
    with open(args.test_data, 'r', encoding='ascii', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                test_records.append(json.loads(line))
    if args.max_samples:
        test_records = test_records[:args.max_samples]
    print(f"Processing {len(test_records)} test samples")

    results = []
    y_true_list, y_pred_list = [], []

    for idx, record in enumerate(test_records):
        try:
            fips_code = record.get('fips_county_code')
            target_date = record.get('target_date_d14', record.get('actual_target_date'))

            ctx_orders = evac_kb.get_context_for_location(fips_code, target_date)
            evac_text = evac_kb.format_context_text(ctx_orders)

            prompt = create_rag_enhanced_prompt(record, evac_text)

            gen = pipe(prompt, return_full_text=False)
            # robust read of pipeline output
            generated_text = ""
            if isinstance(gen, list) and gen:
                generated_text = gen[0].get("generated_text", "").strip()
            elif isinstance(gen, dict):
                generated_text = gen.get("generated_text", "").strip()

            pred, conf = parse_prediction(generated_text)

            if pred is not None:
                y_true = float(record["y_true_d14"])
                y_true_list.append(y_true)
                y_pred_list.append(pred)

                abs_error = abs(y_true - pred)
                pct_error = (abs_error / max(y_true, 1)) * 100 if y_true != 0 else 0.0

                results.append({
                    "placekey": record.get("placekey", ""),
                    "location_name": record.get("location_name", ""),
                    "city": record.get("city", ""),
                    "fips_code": fips_code,
                    "target_date": str(target_date),
                    "y_true_d14": y_true,
                    "y_pred_d14": float(pred),
                    "confidence": float(conf),
                    "absolute_error": float(abs_error),
                    "percent_error": float(pct_error),
                    "evacuation_context": evac_text,
                    "has_evacuation": len(ctx_orders) > 0,
                    "generated_text": generated_text[:200],
                })

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(test_records)} samples")

        except Exception as e:
            print(f"[ERROR] sample {idx}: {e}")
            continue

    # -------------------- Metrics & save --------------------
    if y_true_list and y_pred_list:
        y_true_arr = np.array(y_true_list, dtype=float)
        y_pred_arr = np.array(y_pred_list, dtype=float)

        # MAE / RMSE
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
        rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

        # sMAPE per-sample, then mean & median (ignore denominator=0)
        denom = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0
        smape_vals = np.full_like(y_true_arr, np.nan, dtype=float)
        mask = denom != 0
        smape_vals[mask] = (np.abs(y_true_arr[mask] - y_pred_arr[mask]) / denom[mask]) * 100.0
        smape_mean = float(np.nanmean(smape_vals))
        smape_median = float(np.nanmedian(smape_vals))

        # RMSLE (log1p to handle zeros; clip negatives to 0)
        y_true_clip = np.clip(y_true_arr, 0, None)
        y_pred_clip = np.clip(y_pred_arr, 0, None)
        rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_clip) - np.log1p(y_true_clip)) ** 2)))

        print("\n=== RAG-Enhanced Results ===")
        print(f"Samples processed: {len(results)}")
        print(f"With evacuation context: {sum(1 for r in results if r['has_evacuation'])}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"sMAPE (mean): {smape_mean:.3f}%")
        print(f"sMAPE (median): {smape_median:.3f}%")
        print(f"RMSLE: {rmsle:.3f}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "rag_enhanced_results.csv", index=False)

        summary = {
            "model_type": "RAG_enhanced_LoRA",
            "base_model": args.base_model,
            "adapter_dir": str(model_dir),
            "knowledge_base": args.knowledge_base,
            "total_samples": len(results),
            "samples_with_evacuation": int(sum(1 for r in results if r['has_evacuation'])),
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "smape_mean": smape_mean,
                "smape_median": smape_median,
                "rmsle": rmsle
            }
        }
        with open(output_dir / "rag_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to: {output_dir}")
    else:
        print("[WARN] No predictions parsed; nothing to save.")

if __name__ == "__main__":
    main()

