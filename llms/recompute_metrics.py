#!/usr/bin/env python3
"""
Recompute evaluation metrics (MAE, RMSE, sMAPE mean/median, RMSLE)
from an existing prediction CSV and save summary JSON in the same folder.
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path

def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_smape(y_true, y_pred):
    """Return both mean and median sMAPE (%)"""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    smape_vals = np.full_like(y_true, np.nan, dtype=float)
    smape_vals[mask] = (np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0
    return float(np.nanmean(smape_vals)), float(np.nanmedian(smape_vals))

def compute_rmsle(y_true, y_pred):
    """Standard RMSLE with log1p and clipping negatives"""
    y_true_clip = np.clip(y_true, 0, None)
    y_pred_clip = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred_clip) - np.log1p(y_true_clip)) ** 2)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with y_true_d14 and y_pred_d14 columns")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)

    if "y_true_d14" not in df.columns or "y_pred_d14" not in df.columns:
        raise ValueError("CSV must contain 'y_true_d14' and 'y_pred_d14' columns")

    y_true = df["y_true_d14"].to_numpy(dtype=float)
    y_pred = df["y_pred_d14"].to_numpy(dtype=float)

    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    smape_mean, smape_median = compute_smape(y_true, y_pred)
    rmsle = compute_rmsle(y_true, y_pred)

    metrics = {
        "samples": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "smape_mean": smape_mean,
        "smape_median": smape_median,
        "rmsle": rmsle,
    }

    # Print to console
    print("\n=== Recomputed Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

    # Save to JSON in the same folder as input file
    output_file = input_path.with_name(f"{input_path.stem}_metrics_summary.json")
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_file}")

if __name__ == "__main__":
    main()
