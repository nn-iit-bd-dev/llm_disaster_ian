#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_prep_dl.py

Build D14 forecasting samples (13-in -> Day-14 out) directly from your
split CSVs (each already includes weekly arrays + metadata).
"""

import argparse, ast, json
from pathlib import Path
import pandas as pd
import numpy as np

PERIOD_ORDER = {"before": 1, "landfall": 2, "after": 3}

# ---------- JSON Serialization Fix ----------
def convert_numpy_types(obj):
    """Convert NumPy data types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Prepare D14 dataset (13->1) from split CSVs")
    ap.add_argument("--train-csv", required=True, help="Train split CSV (weekly rows)")
    ap.add_argument("--test-csv",  required=True, help="Test split CSV (weekly rows)")
    ap.add_argument("--outdir",    required=True, help="Output directory")
    ap.add_argument("--city", default=None, help="Optional city filter (case-insensitive)")
    ap.add_argument("--required-start", default=None, help="Require BEFORE start = YYYY-MM-DD")
    ap.add_argument("--strict-start", action="store_true",
                    help="If set, skip placekeys whose BEFORE start != required-start")
    ap.add_argument("--max-rows-per-split", type=int, default=0,
                    help="Limit number of placekeys per split (0=all)")
    return ap.parse_args()

# ---------- helpers ----------
def to_naive(ts):
    s = pd.to_datetime(ts, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            return s.dt.tz_localize(None)
    except Exception:
        pass
    return s

def parse_vbd(v):
    """visits_by_day_json -> list[float] length 7, else None"""
    if isinstance(v, list):
        out = v
    elif isinstance(v, str):
            t = v.strip()
            try:
                out = ast.literal_eval(t) if t.startswith("[") else [float(x) for x in t.split(",")]
            except Exception:
                return None
    else:
        return None
    try:
        out = [float(x) for x in out]
    except Exception:
        return None
    return out if len(out) == 7 else None

def expand_weekly_to_daily(group: pd.DataFrame) -> pd.DataFrame:
    """Sort (period_order, start) and expand 7-day arrays to daily rows."""
    g = group.copy()
    g["time_period"] = g["time_period"].astype(str).str.lower()
    g["date_range_start"] = to_naive(g["date_range_start"])
    g["period_order"] = g["time_period"].map(PERIOD_ORDER).fillna(999).astype(int)
    g = g.sort_values(["period_order", "date_range_start"])

    rows = []
    for _, r in g.iterrows():
        vbd = parse_vbd(r.get("visits_by_day_json"))
        if vbd is None or pd.isna(r["date_range_start"]):
            continue
        start = r["date_range_start"]
        for i, val in enumerate(vbd):
            rows.append({
                "placekey": r["placekey"],
                "date": start + pd.Timedelta(days=i),
                "visits": float(val),
                "time_period": r["time_period"]
            })
    return pd.DataFrame(rows)

def first_before_start(group: pd.DataFrame) -> pd.Timestamp:
    bf = group[group["time_period"].str.lower() == "before"].sort_values("date_range_start")
    return bf["date_range_start"].iloc[0] if len(bf) else pd.NaT

def detect_landfall_date(group: pd.DataFrame) -> pd.Timestamp:
    # Hardcoded Hurricane Ian landfall date: September 28, 2022
    return pd.to_datetime("2022-09-28")

def build_one_sample(group: pd.DataFrame,
                     required_start: pd.Timestamp | None,
                     strict_start: bool) -> dict | None:
    """One sample per placekey: take 13 days from BEFORE start -> predict Day-14."""
    start = first_before_start(group)
    if pd.isna(start):
        return None

    if required_start is not None and strict_start and start != required_start:
        return None

    daily = expand_weekly_to_daily(group)
    if daily.empty:
        return None

    daily = daily.sort_values("date")
    window = daily[(daily["date"] >= start) & (daily["date"] < start + pd.Timedelta(days=14))]
    if len(window) != 14:
        return None

    prev_13 = window.iloc[:13]["visits"].tolist()
    y14 = float(window.iloc[13]["visits"])
    target_date = start + pd.Timedelta(days=13)

    # pick a meta row
    g0 = group.sort_values(["date_range_start", "time_period"]).iloc[0]
    landfall_dt = detect_landfall_date(group)

    city_val = g0.get("city", "")
    if not city_val:
        city_val = g0.get("city_norm", "")

    rec = {
        "placekey": str(g0.get("placekey", "")),
        "city": str(city_val),
        "location_name": str(g0.get("location_name", "")),
        "top_category": str(g0.get("top_category", "")),
        "latitude": float(g0.get("latitude", np.nan)) if pd.notna(g0.get("latitude", np.nan)) else None,
        "longitude": float(g0.get("longitude", np.nan)) if pd.notna(g0.get("longitude", np.nan)) else None,
        "series_start_date": start.strftime("%Y-%m-%d"),
        "landfall_date": landfall_dt.strftime("%Y-%m-%d"),
        "target_date_d14": target_date.strftime("%Y-%m-%d"),
        "actual_target_date": target_date.strftime("%Y-%m-%d"),
        "time_periods_used": ";".join(group["time_period"].dropna().astype(str).str.lower().unique()),
        "prev_13_values": [float(x) for x in prev_13],
        "y_true_d14": float(y14)
    }
    return rec

def prep_from_split(split_df: pd.DataFrame,
                    city: str | None,
                    required_start: pd.Timestamp | None,
                    strict_start: bool,
                    max_rows: int) -> list[dict]:
    df = split_df.copy()
    df["time_period"] = df["time_period"].astype(str).str.lower()
    df["date_range_start"] = to_naive(df["date_range_start"])

    # Apply city filter if provided
    if city:
        city_upper = city.upper()
        if "city" in df.columns:
            df = df[df["city"].astype(str).str.upper() == city_upper]
        elif "city_norm" in df.columns:
            df = df[df["city_norm"].astype(str).str.upper() == city_upper]

    records = []
    for pk, g in df.groupby("placekey"):
        if max_rows and len(records) >= max_rows:
            break
        if g["time_period"].str.lower().eq("before").sum() == 0:
            continue
        sample = build_one_sample(g, required_start, strict_start)
        if sample:
            records.append(sample)
    return records

def save_data(data, output_path):
    """Save data to JSONL format with proper NumPy type handling"""
    with open(output_path, 'w') as f:
        for item in data:
            clean_item = convert_numpy_types(item)
            f.write(json.dumps(clean_item) + '\n')

def create_summary_stats(train_data, test_data, output_dir):
    """Create summary statistics with proper JSON serialization"""
    stats = {
        "dataset_info": {
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "total_samples": len(train_data) + len(test_data)
        }
    }
    
    # Convert any remaining NumPy types to native Python types
    stats = convert_numpy_types(stats)
    
    # Save statistics
    stats_file = Path(output_dir) / "dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_file}")
    return stats

# ---------- Main ----------
def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Hurricane Impact Data Preparation for Deep Learning (D14)")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    
    # Parse required_start if provided
    required_start = None
    if args.required_start:
        try:
            required_start = pd.to_datetime(args.required_start)
            print(f"Using required start date: {required_start.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"Warning: Could not parse required_start '{args.required_start}': {e}")
    
    # Load CSVs
    print(f"Loading train data from: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    print(f"Loading test data from: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    
    print(f"Train CSV shape: {train_df.shape}")
    print(f"Test CSV shape: {test_df.shape}")
    
    # Process train data
    print("\nProcessing train data...")
    train_data = prep_from_split(
        train_df, 
        args.city, 
        required_start, 
        args.strict_start, 
        args.max_rows_per_split
    )
    
    # Process test data
    print("Processing test data...")
    test_data = prep_from_split(
        test_df, 
        args.city, 
        required_start, 
        args.strict_start, 
        args.max_rows_per_split
    )
    
    # Save JSONLs
    train_output = output_dir / "prepared_d14_train.jsonl"
    test_output = output_dir / "prepared_d14_test.jsonl"
    
    print(f"\nSaving train data to: {train_output}")
    save_data(train_data, train_output)
    
    print(f"Saving test data to: {test_output}")
    save_data(test_data, test_output)
    
    print(f"\nDataset summary:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Total samples: {len(train_data) + len(test_data)}")
    
    # Create summary statistics
    if train_data or test_data:
        try:
            print("\nCreating summary statistics...")
            stats = create_summary_stats(train_data, test_data, output_dir)
            print("Summary statistics created successfully!")
        except Exception as e:
            print(f"Warning: Could not create summary statistics: {e}")

    print("\nData preparation complete!")
    print(f"Now all samples will have landfall_date: '2022-09-28'")

if __name__ == "__main__":
    main()