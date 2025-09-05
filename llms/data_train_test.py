#!/usr/bin/env python3
# data_prep.py
# Prepare 13->14 daily windows for LLM forecasting with train/test split.

import argparse, json, ast, sys, random, os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

# -------------------------- CLI --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prepare D14 forecasting data (13 input days + D14 target).")
    p.add_argument("--input", default="all_4cities_vbd3w.csv", help="Raw weekly CSV (SafeGraph-style).")
    p.add_argument("--city", default="Tampa", help="City name to filter.")
    p.add_argument("--output-dir", default="prepared_data", help="Output directory for organized files.")
    p.add_argument("--train-split", type=float, default=0.8, help="Training split ratio.")
    p.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible splits.")
    p.add_argument("--landfall-date", default="2022-09-28", help="Landfall date (YYYY-MM-DD).")
    p.add_argument("--start-policy", choices=["strict", "first_before", "nearest"], default="strict",
                   help="How to anchor the 14-day series start.")
    p.add_argument("--required-start", default="2022-09-19", help="Required start date (YYYY-MM-DD).")
    p.add_argument("--tolerance-days", type=int, default=7, help="Max distance in days from required-start.")
    p.add_argument("--max-rows", type=int, default=None, help="Read only first N input rows.")
    p.add_argument("--print-samples", type=int, default=3, help="Print first N prepared records.")
    return p.parse_args()

# -------------------------- helpers --------------------------

def norm_city(s: str) -> str:
    return str(s).strip().upper().replace("  ", " ")

def parse_vbd(v) -> Optional[List[float]]:
    """Parse visits_by_day into a list[float] of length 7, tolerant of JSON/py-literal/comma-string."""
    if isinstance(v, (list, tuple)):
        return list(map(float, v))
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(s)
                if isinstance(obj, (list, tuple)):
                    return list(map(float, obj))
            except Exception:
                pass
        try:
            return [float(x) for x in s.split(",")]
        except Exception:
            return None
    return None

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def to_naive_date(ts):
    """Return a pure date() from tz-aware or tz-naive input; None if invalid."""
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return None
    try:
        return t.tz_convert(None).date()
    except Exception:
        try:
            return t.tz_localize(None).date()
        except Exception:
            return t.date()

def to_naive_ts(ts):
    """Return tz-naive midnight Timestamp from tz-aware/naive input."""
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    try:
        return t.tz_convert(None).normalize()
    except Exception:
        try:
            return t.tz_localize(None).normalize()
        except Exception:
            return pd.Timestamp(pd.to_datetime(t).date())

def first_before_start(g: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Earliest 'before' date_range_start (tz preserved)."""
    if "time_period" not in g.columns:
        return None
    mask = g["time_period"].astype(str).str.lower() == "before"
    if not mask.any():
        return None
    return pd.to_datetime(g.loc[mask, "date_range_start"]).min()

def expand_week_to_daily(row: pd.Series) -> List[Dict]:
    """Expand a weekly record (7-day visits_by_day) to 7 daily rows starting at its date_range_start."""
    start = pd.to_datetime(row["date_range_start"], errors="coerce")
    if pd.isna(start):
        return []
    vlist = row["vbd_list"] or []
    out = []
    for i, v in enumerate(vlist):
        out.append({
            "date": start + pd.Timedelta(days=i),
            "visits": float(v),
            "time_period": row.get("time_period", np.nan),
        })
    return out

def split_train_test(prepared_data: List[Dict], train_ratio: float, random_seed: int):
    """Split prepared data into train/test sets with reproducible randomization."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    shuffled_data = prepared_data.copy()
    random.shuffle(shuffled_data)
    
    n_total = len(shuffled_data)
    n_train = int(n_total * train_ratio)
    
    train_data = shuffled_data[:n_train]
    test_data = shuffled_data[n_train:]
    
    return train_data, test_data

def create_organized_output(output_dir: Path, city: str, timestamp: str):
    """Create organized directory structure for outputs."""
    # Main output directory: prepared_data/tampa_2025_01_15/
    city_dir = output_dir / f"{city.lower()}_{timestamp}"
    city_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    data_dir = city_dir / "data"
    logs_dir = city_dir / "logs" 
    stats_dir = city_dir / "stats"
    
    for d in [data_dir, logs_dir, stats_dir]:
        d.mkdir(exist_ok=True)
    
    return {
        "base": city_dir,
        "data": data_dir,
        "logs": logs_dir,
        "stats": stats_dir
    }

def write_jsonl(data: List[Dict], filepath: Path):
    """Write data to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        for rec in data:
            f.write(json.dumps(rec) + "\n")

def write_stats_report(stats: Dict, dirs: Dict, args, train_count: int, test_count: int):
    """Write detailed statistics report."""
    stats_file = dirs["stats"] / "preparation_stats.txt"
    
    with open(stats_file, "w") as f:
        f.write("HURRICANE FORECASTING DATA PREPARATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"Input file: {args.input}\n")
        f.write(f"City: {args.city}\n")
        f.write(f"Landfall date: {args.landfall_date}\n")
        f.write(f"Required start: {args.required_start}\n")
        f.write(f"Start policy: {args.start_policy}\n")
        f.write(f"Train split: {args.train_split:.1%}\n")
        f.write(f"Random seed: {args.random_seed}\n\n")
        
        f.write("PROCESSING STATISTICS:\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nTrain records: {train_count} ({args.train_split:.1%})\n")
        f.write(f"Test records: {test_count} ({1-args.train_split:.1%})\n")
        f.write(f"Total prepared: {train_count + test_count}\n")
        
        if stats["placekeys_total"] > 0:
            success_rate = stats["placekeys_prepared"] / stats["placekeys_total"] * 100
            f.write(f"Success rate: {success_rate:.1f}%\n")

# -------------------------- main --------------------------

def main():
    args = parse_args()
    
    # Create timestamp for organized output
    timestamp = datetime.now().strftime("%Y_%m_%d")
    
    # Create organized directory structure
    output_dir = Path(args.output_dir)
    dirs = create_organized_output(output_dir, args.city, timestamp)
    
    # Input validation
    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERROR] Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    landfall_date = pd.to_datetime(args.landfall_date, errors="coerce")
    if pd.isna(landfall_date):
        print(f"[ERROR] Invalid --landfall-date: {args.landfall_date}", file=sys.stderr)
        sys.exit(1)

    required_start_ts = pd.to_datetime(args.required_start, errors="coerce")
    if pd.isna(required_start_ts):
        print(f"[ERROR] Invalid --required-start: {args.required_start}", file=sys.stderr)
        sys.exit(1)
    required_day = required_start_ts.date()
    tolerance = pd.Timedelta(days=args.tolerance_days)

    # Create log file
    log_file = dirs["logs"] / f"preparation_{timestamp}.log"
    
    # Read and process input
    df = pd.read_csv(inp, nrows=args.max_rows)
    ensure_columns(df, ["placekey", "visits_by_day", "date_range_start"])

    # Parse visits_by_day and date_range_start
    df["vbd_list"] = df["visits_by_day"].apply(parse_vbd)
    df = df[df["vbd_list"].apply(lambda x: isinstance(x, list) and len(x) == 7)].copy()
    df["date_range_start"] = pd.to_datetime(df["date_range_start"], errors="coerce")

    # City filtering
    city_col = "city_norm" if "city_norm" in df.columns else ("city" if "city" in df.columns else None)
    if city_col is None:
        print("[WARN] No 'city' or 'city_norm' column found; proceeding without city filter.")
        df["_CITYN"] = ""
    else:
        df["_CITYN"] = df[city_col].map(norm_city)

    target_city = norm_city(args.city)
    if city_col is not None:
        df = df[df["_CITYN"] == target_city].copy()

    if df.empty:
        print(f"[WARN] No rows for city='{args.city}'. Creating empty output files.")
        train_path = dirs["data"] / f"{args.city.lower()}_train.jsonl"
        test_path = dirs["data"] / f"{args.city.lower()}_test.jsonl"
        write_jsonl([], train_path)
        write_jsonl([], test_path)
        sys.exit(0)

    # Add optional columns
    for c in ["time_period", "location_name", "top_category", "latitude", "longitude"]:
        if c not in df.columns:
            df[c] = np.nan

    # Sort data
    if "time_period" in df.columns:
        order = {"before": 1, "landfall": 2, "after": 3}
        df["_period_order"] = df["time_period"].astype(str).str.lower().map(order).fillna(999)
        df = df.sort_values(["placekey", "_period_order", "date_range_start"])
    else:
        df = df.sort_values(["placekey", "date_range_start"])

    # Process each placekey
    prepared = []
    stats = dict(
        placekeys_total=0,
        placekeys_skipped_no_before=0,
        placekeys_skipped_wrong_before_start=0,
        placekeys_insufficient_days=0,
        placekeys_prepared=0,
        weekly_rows=0,
        daily_rows=0
    )

    policy = args.start_policy

    for pk, g in df.groupby("placekey"):
        stats["placekeys_total"] += 1
        g = g.copy()

        # Find 'before' baseline
        before_start = first_before_start(g)
        if before_start is None:
            stats["placekeys_skipped_no_before"] += 1
            continue

        # Decide series_start
        series_start_day = None
        if policy == "strict":
            if to_naive_date(before_start) != required_day:
                stats["placekeys_skipped_wrong_before_start"] += 1
                continue
            series_start_day = required_day
        elif policy == "first_before":
            series_start_day = to_naive_date(before_start)
        elif policy == "nearest":
            bstarts = g.loc[g["time_period"].astype(str).str.lower() == "before", "date_range_start"].dropna()
            if bstarts.empty:
                stats["placekeys_skipped_no_before"] += 1
                continue
            candidates = [pd.Timestamp(to_naive_date(d)) for d in bstarts]
            target = pd.Timestamp(required_day)
            candidate = min(candidates, key=lambda d: abs(d - target))
            if abs(candidate - target) > tolerance:
                stats["placekeys_skipped_wrong_before_start"] += 1
                continue
            series_start_day = candidate.date()

        # Expand weekly to daily
        daily_records = []
        for _, row in g.iterrows():
            stats["weekly_rows"] += 1
            daily_records.extend(expand_week_to_daily(row))
        if not daily_records:
            stats["placekeys_insufficient_days"] += 1
            continue

        daily_df = pd.DataFrame(daily_records).sort_values("date")
        daily_df["date_naive"] = daily_df["date"].apply(to_naive_ts)

        # Build 14-day window
        start = to_naive_ts(series_start_day)
        end = start + pd.Timedelta(days=13)
        window = daily_df[(daily_df["date_naive"] >= start) & (daily_df["date_naive"] <= end)]

        if len(window) < 14:
            stats["placekeys_insufficient_days"] += 1
            continue

        visits = window["visits"].to_numpy()
        prev_13 = visits[:13].tolist()
        y14 = float(visits[13])

        # Create record
        r0 = g.iloc[0]
        target_ts = window.iloc[13]["date_naive"]
        prepared.append({
            "placekey": pk,
            "city": r0.get("city_norm", r0.get("city", args.city)),
            "location_name": r0.get("location_name", np.nan),
            "top_category": r0.get("top_category", np.nan),
            "latitude": r0.get("latitude", np.nan),
            "longitude": r0.get("longitude", np.nan),
            "series_start_date": str(start.date()),
            "landfall_date": str(to_naive_ts(landfall_date).date()),
            "target_date_d14": str(end.date()),
            "actual_target_date": str(target_ts.date()),
            "time_periods_used": ";".join(sorted(set(map(str, g["time_period"].dropna().astype(str))))),
            "target_days_after_landfall": int((target_ts - to_naive_ts(landfall_date)).days),
            "prev_13_values": prev_13,
            "y_true_d14": y14
        })
        stats["placekeys_prepared"] += 1

    if not prepared:
        print("[WARN] No records prepared. Creating empty output files.")
        train_path = dirs["data"] / f"{args.city.lower()}_train.jsonl"
        test_path = dirs["data"] / f"{args.city.lower()}_test.jsonl"
        write_jsonl([], train_path)
        write_jsonl([], test_path)
        sys.exit(0)

    # Split train/test and write files
    train_data, test_data = split_train_test(prepared, args.train_split, args.random_seed)
    
    train_path = dirs["data"] / f"{args.city.lower()}_train.jsonl"
    test_path = dirs["data"] / f"{args.city.lower()}_test.jsonl"
    
    write_jsonl(train_data, train_path)
    write_jsonl(test_data, test_path)

    # Write statistics report
    write_stats_report(stats, dirs, args, len(train_data), len(test_data))

    # Create summary file with key paths
    summary_file = dirs["base"] / "README.md"
    with open(summary_file, "w") as f:
        f.write(f"# Hurricane Forecasting Data - {args.city.title()}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Files:\n")
        f.write(f"- Training data: `data/{train_path.name}` ({len(train_data)} records)\n")
        f.write(f"- Test data: `data/{test_path.name}` ({len(test_data)} records)\n")
        f.write(f"- Statistics: `stats/preparation_stats.txt`\n")
        f.write(f"- Logs: `logs/preparation_{timestamp}.log`\n\n")
        f.write("## Configuration:\n")
        f.write(f"- City: {args.city}\n")
        f.write(f"- Train/Test Split: {args.train_split:.0%}/{1-args.train_split:.0%}\n")
        f.write(f"- Random Seed: {args.random_seed}\n")
        f.write(f"- Landfall Date: {args.landfall_date}\n")

    # Console output
    print(f"\n=== Data Preparation Complete ===")
    print(f"Output directory: {dirs['base']}")
    print(f"City: {args.city}")
    print(f"Train records: {len(train_data)} ({args.train_split:.1%})")
    print(f"Test records: {len(test_data)} ({1-args.train_split:.1%})")
    print(f"Success rate: {stats['placekeys_prepared']}/{stats['placekeys_total']} placekeys")
    
    # Show file paths
    print(f"\nFiles created:")
    print(f"  Training: {train_path}")
    print(f"  Testing: {test_path}")
    print(f"  Stats: {dirs['stats']}/preparation_stats.txt")
    print(f"  README: {summary_file}")

    # Show samples
    if train_data and args.print_samples > 0:
        print(f"\n=== Sample Training Records ===")
        for rec in train_data[:args.print_samples]:
            print(json.dumps(rec, indent=2))

if __name__ == "__main__":
    main()

# -------------------------- helpers --------------------------

def norm_city(s: str) -> str:
    return str(s).strip().upper().replace("  ", " ")

def parse_vbd(v) -> Optional[List[float]]:
    """Parse visits_by_day into a list[float] of length 7, tolerant of JSON/py-literal/comma-string."""
    if isinstance(v, (list, tuple)):
        return list(map(float, v))
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(s)
                if isinstance(obj, (list, tuple)):
                    return list(map(float, obj))
            except Exception:
                pass
        try:
            return [float(x) for x in s.split(",")]
        except Exception:
            return None
    return None

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def to_naive_date(ts):
    """Return a pure date() from tz-aware or tz-naive input; None if invalid."""
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return None
    try:
        return t.tz_convert(None).date()   # tz-aware -> naive date
    except Exception:
        try:
            return t.tz_localize(None).date()  # already naive but ensure date
        except Exception:
            return t.date()

def to_naive_ts(ts):
    """Return tz-naive midnight Timestamp from tz-aware/naive input."""
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    try:
        return t.tz_convert(None).normalize()   # tz-aware -> naive midnight
    except Exception:
        try:
            return t.tz_localize(None).normalize()
        except Exception:
            return pd.Timestamp(pd.to_datetime(t).date())

def first_before_start(g: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Earliest 'before' date_range_start (tz preserved)."""
    if "time_period" not in g.columns:
        return None
    mask = g["time_period"].astype(str).str.lower() == "before"
    if not mask.any():
        return None
    return pd.to_datetime(g.loc[mask, "date_range_start"]).min()

def expand_week_to_daily(row: pd.Series) -> List[Dict]:
    """Expand a weekly record (7-day visits_by_day) to 7 daily rows starting at its date_range_start."""
    start = pd.to_datetime(row["date_range_start"], errors="coerce")
    if pd.isna(start):
        return []
    vlist = row["vbd_list"] or []
    out = []
    for i, v in enumerate(vlist):
        out.append({
            "date": start + pd.Timedelta(days=i),
            "visits": float(v),
            "time_period": row.get("time_period", np.nan),
        })
    return out

def split_train_test(prepared_data: List[Dict], train_ratio: float, random_seed: int):
    """Split prepared data into train/test sets with reproducible randomization."""
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle the data
    shuffled_data = prepared_data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split point
    n_total = len(shuffled_data)
    n_train = int(n_total * train_ratio)
    
    train_data = shuffled_data[:n_train]
    test_data = shuffled_data[n_train:]
    
    return train_data, test_data

def write_jsonl(data: List[Dict], filepath: Path):
    """Write data to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        for rec in data:
            f.write(json.dumps(rec) + "\n")

# -------------------------- main --------------------------

def main():
    args = parse_args()
    inp = Path(args.input)
    
    # Create train/test output paths
    output_path = Path(args.output)
    train_path = output_path.parent / f"{output_path.stem}_train.jsonl"
    test_path = output_path.parent / f"{output_path.stem}_test.jsonl"
    
    if not inp.exists():
        print(f"[ERROR] Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    landfall_date = pd.to_datetime(args.landfall_date, errors="coerce")
    if pd.isna(landfall_date):
        print(f"[ERROR] Invalid --landfall-date: {args.landfall_date}", file=sys.stderr)
        sys.exit(1)

    required_start_ts = pd.to_datetime(args.required_start, errors="coerce")
    if pd.isna(required_start_ts):
        print(f"[ERROR] Invalid --required-start: {args.required_start}", file=sys.stderr)
        sys.exit(1)
    required_day = required_start_ts.date()
    tolerance = pd.Timedelta(days=args.tolerance_days)

    # Read input
    df = pd.read_csv(inp, nrows=args.max_rows)

    # Required columns (others optional)
    ensure_columns(df, ["placekey", "visits_by_day", "date_range_start"])

    # Parse visits_by_day and date_range_start
    df["vbd_list"] = df["visits_by_day"].apply(parse_vbd)
    df = df[df["vbd_list"].apply(lambda x: isinstance(x, list) and len(x) == 7)].copy()
    df["date_range_start"] = pd.to_datetime(df["date_range_start"], errors="coerce")

    # Choose city column and filter
    city_col = "city_norm" if "city_norm" in df.columns else ("city" if "city" in df.columns else None)
    if city_col is None:
        print("[WARN] No 'city' or 'city_norm' column found; proceeding without city filter.")
        df["_CITYN"] = ""
    else:
        df["_CITYN"] = df[city_col].map(norm_city)

    target_city = norm_city(args.city)
    if city_col is not None:
        df = df[df["_CITYN"] == target_city].copy()

    if df.empty:
        print(f"[WARN] No rows for city='{args.city}'. Exiting.")
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_path.write_text("")  # empty file
        test_path.write_text("")   # empty file
        sys.exit(0)

    # Optional fields
    for c in ["time_period", "location_name", "top_category", "latitude", "longitude"]:
        if c not in df.columns:
            df[c] = np.nan

    # Sort rows: by placekey, time_period order, then date
    if "time_period" in df.columns:
        order = {"before": 1, "landfall": 2, "after": 3}
        df["_period_order"] = df["time_period"].astype(str).str.lower().map(order).fillna(999)
        df = df.sort_values(["placekey", "_period_order", "date_range_start"])
    else:
        df = df.sort_values(["placekey", "date_range_start"])

    # Build outputs
    prepared = []
    stats = dict(
        placekeys_total=0,
        placekeys_skipped_no_before=0,
        placekeys_skipped_wrong_before_start=0,
        placekeys_insufficient_days=0,
        placekeys_prepared=0,
        weekly_rows=0,
        daily_rows=0
    )

    policy = args.start_policy

    for pk, g in df.groupby("placekey"):
        stats["placekeys_total"] += 1
        g = g.copy()

        # Find 'before' baseline
        before_start = first_before_start(g)
        if before_start is None:
            stats["placekeys_skipped_no_before"] += 1
            continue

        # Decide series_start (timezone-agnostic, compare by calendar day)
        series_start_day = None
        if policy == "strict":
            if to_naive_date(before_start) != required_day:
                stats["placekeys_skipped_wrong_before_start"] += 1
                continue
            series_start_day = required_day
        elif policy == "first_before":
            series_start_day = to_naive_date(before_start)
        elif policy == "nearest":
            bstarts = g.loc[g["time_period"].astype(str).str.lower() == "before", "date_range_start"].dropna()
            if bstarts.empty:
                stats["placekeys_skipped_no_before"] += 1
                continue
            candidates = [pd.Timestamp(to_naive_date(d)) for d in bstarts]
            target = pd.Timestamp(required_day)
            candidate = min(candidates, key=lambda d: abs(d - target))
            if abs(candidate - target) > tolerance:
                stats["placekeys_skipped_wrong_before_start"] += 1
                continue
            series_start_day = candidate.date()
        else:
            print(f"[ERROR] Unknown start-policy: {policy}", file=sys.stderr)
            sys.exit(1)

        # Expand weekly -> daily
        daily_records = []
        for _, row in g.iterrows():
            stats["weekly_rows"] += 1
            daily_records.extend(expand_week_to_daily(row))
        if not daily_records:
            stats["placekeys_insufficient_days"] += 1
            continue

        daily_df = pd.DataFrame(daily_records).sort_values("date")
        # normalize dates to tz-naive for window filtering
        daily_df["date_naive"] = daily_df["date"].apply(to_naive_ts)

        # Build strict 14-day window from series_start_day (tz-agnostic)
        start = to_naive_ts(series_start_day)
        end = start + pd.Timedelta(days=13)
        window = daily_df[(daily_df["date_naive"] >= start) & (daily_df["date_naive"] <= end)]

        if len(window) < 14:
            stats["placekeys_insufficient_days"] += 1
            continue

        visits = window["visits"].to_numpy()
        prev_13 = visits[:13].tolist()
        y14 = float(visits[13])

        # Representative static fields
        r0 = g.iloc[0]
        target_ts = window.iloc[13]["date_naive"]
        prepared.append({
            "placekey": pk,
            "city": r0.get("city_norm", r0.get("city", args.city)),
            "location_name": r0.get("location_name", np.nan),
            "top_category": r0.get("top_category", np.nan),
            "latitude": r0.get("latitude", np.nan),
            "longitude": r0.get("longitude", np.nan),
            "series_start_date": str(start.date()),
            "landfall_date": str(to_naive_ts(landfall_date).date()),
            "target_date_d14": str(end.date()),
            "actual_target_date": str(target_ts.date()),
            "time_periods_used": ";".join(sorted(set(map(str, g["time_period"].dropna().astype(str))))),
            "target_days_after_landfall": int((target_ts - to_naive_ts(landfall_date)).days),
            "prev_13_values": prev_13,
            "y_true_d14": y14
        })
        stats["placekeys_prepared"] += 1

    if not prepared:
        print("[WARN] No records prepared. Creating empty output files.")
        write_jsonl([], train_path)
        write_jsonl([], test_path)
        sys.exit(0)

    # Split into train/test sets
    train_data, test_data = split_train_test(prepared, args.train_split, args.random_seed)
    
    # Write train and test files
    write_jsonl(train_data, train_path)
    write_jsonl(test_data, test_path)

    # Update statistics
    stats["train_records"] = len(train_data)
    stats["test_records"] = len(test_data)
    stats["total_prepared"] = len(prepared)

    # Summary
    print("=== Preparation Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"Train file: {train_path} ({len(train_data)} records, {args.train_split:.1%})")
    print(f"Test file: {test_path} ({len(test_data)} records, {1-args.train_split:.1%})")
    print(f"Random seed: {args.random_seed}")

    # Show samples from train set
    if train_data and args.print_samples > 0:
        print("\n=== Sample training records ===")
        for rec in train_data[:args.print_samples]:
            print(json.dumps(rec, indent=2))

if __name__ == "__main__":
    main()
