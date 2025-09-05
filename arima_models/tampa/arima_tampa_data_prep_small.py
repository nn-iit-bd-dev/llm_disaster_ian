#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# daily_panel_prep.py
# Prepare daily panel data from Tampa hurricane CSV for ARIMA/Prophet forecasting
# Creates clean daily time series with complete 21-day coverage

import os, ast, json, argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Hurricane Ian timeline (3 weeks)
WIN_START = pd.Timestamp("2022-09-19")  # Week before hurricane
WIN_END = pd.Timestamp("2022-10-09")    # Week after hurricane
EXPECTED_DATES = pd.date_range(WIN_START, WIN_END, freq="D")  # 21 days total
EXPECTED_PERIODS = {"before", "landfall", "after"}

def parse_args():
    ap = argparse.ArgumentParser(description="Prepare daily panel for Tampa hurricane forecasting")
    ap.add_argument("--input", 
                    default="Tampa_vbd3w_for_test.csv",
                    help="Input CSV with weekly hurricane data")
    ap.add_argument("--city", default="Tampa", help="City to process")
    ap.add_argument("--outdir", default="ts_daily_panel_small", help="Output directory")
    return ap.parse_args()

def parse_visits_by_day(v):
    """Parse visits_by_day from various formats (JSON, list, string)"""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        try:
            # Try JSON first
            return json.loads(s)
        except:
            try:
                # Try literal_eval
                return ast.literal_eval(s)
            except:
                try:
                    # Try comma-separated values
                    return [float(x.strip()) for x in s.split(",")]
                except:
                    return None
    return None

def expand_weekly_to_daily(df_week: pd.DataFrame) -> pd.DataFrame:
    """
    Expand weekly data to daily time series
    Each row has 7 days of visits -> expand to 7 daily rows
    """
    rows = []
    
    for _, row in df_week.iterrows():
        # Parse visits array
        visits_7day = parse_visits_by_day(row.get("visits_by_day"))
        if not (isinstance(visits_7day, list) and len(visits_7day) == 7):
            continue
            
        # Parse start date
        try:
            start_date = pd.to_datetime(row["date_range_start"])
            # Ensure it's at midnight local time
            start_date = start_date.normalize()
        except:
            continue
        
        # Expand 7 days
        for day_idx, visits in enumerate(visits_7day):
            daily_date = start_date + pd.Timedelta(days=day_idx)
            rows.append({
                "placekey": row["placekey"],
                "date": daily_date,
                "visits": int(visits) if pd.notna(visits) else 0,
                "location_name": row.get("location_name", ""),
                "top_category": row.get("top_category", ""),
                "latitude": row.get("latitude", np.nan),
                "longitude": row.get("longitude", np.nan),
                "time_period": row.get("time_period", ""),
            })
    
    daily_df = pd.DataFrame(rows)
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.floor("D")
    
    return daily_df

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"=== Daily Panel Preparation for {args.city} ===")
    print(f"Reading: {args.input}")
    print(f"Building individual 21-day timelines per placekey (starting from 'before' period)")
    
    # --- Load and filter data ---
    # Check required columns
    df_sample = pd.read_csv(args.input, nrows=0)
    required_cols = ["placekey", "time_period", "date_range_start", "visits_by_day"]
    missing_cols = [col for col in required_cols if col not in df_sample.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Load full data
    print("Loading CSV data...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Total rows loaded: {len(df):,}")
    
    # Filter by city
    if "city_norm" in df.columns:
        city_df = df[df["city_norm"].str.upper() == args.city.upper()].copy()
    elif "city" in df.columns:
        city_df = df[df["city"].str.upper() == args.city.upper()].copy()
    else:
        raise ValueError("No 'city' or 'city_norm' column found")
    
    print(f"{args.city} rows: {len(city_df):,}")
    
    # Normalize time periods
    city_df["time_period"] = city_df["time_period"].astype(str).str.strip().str.lower()
    city_df["time_period"] = city_df["time_period"].replace({"during": "landfall"})
    city_df = city_df[city_df["time_period"].isin(EXPECTED_PERIODS)].copy()
    
    print(f"Valid time periods: {dict(city_df['time_period'].value_counts())}")
    
    # Handle duplicates - keep latest date_range_start per (placekey, time_period)
    city_df["_start_dt"] = pd.to_datetime(city_df["date_range_start"], errors="coerce")
    city_df = (city_df.sort_values(["placekey", "time_period", "_start_dt"])
                     .drop_duplicates(["placekey", "time_period"], keep="last")
                     .drop(columns=["_start_dt"]))
    
    print(f"After deduplication: {len(city_df):,} rows")
    
    # --- Build placekey-specific 21-day timelines ---
    print("Building individual 21-day timelines per placekey...")
    
    complete_panels = []
    placekey_stats = {
        'has_all_periods': 0,
        'missing_before': 0, 
        'missing_landfall': 0,
        'missing_after': 0,
        'complete_21days': 0,
        'timeline_issues': 0
    }
    
    for placekey, placekey_group in city_df.groupby("placekey"):
        # Check if placekey has all three time periods
        periods_present = set(placekey_group['time_period'])
        if periods_present != EXPECTED_PERIODS:
            missing = EXPECTED_PERIODS - periods_present
            for period in missing:
                placekey_stats[f'missing_{period}'] += 1
            continue
        
        placekey_stats['has_all_periods'] += 1
        
        # Sort by time period order for chronological sequence
        period_order = {"before": 1, "landfall": 2, "after": 3}
        placekey_group = placekey_group.copy()
        placekey_group["period_order"] = placekey_group["time_period"].map(period_order)
        placekey_group = placekey_group.sort_values(["period_order", "date_range_start"])
        
        # Build chronological daily sequence
        daily_rows = []
        current_date = None
        
        for _, row in placekey_group.iterrows():
            # Parse visits array
            visits_7day = parse_visits_by_day(row.get("visits_by_day"))
            if not (isinstance(visits_7day, list) and len(visits_7day) == 7):
                continue
                
            # Parse start date for this week
            try:
                week_start = pd.to_datetime(row["date_range_start"]).normalize()
            except:
                continue
            
            # For the first period ("before"), this sets our timeline start
            if current_date is None:
                current_date = week_start
                timeline_start = current_date
            else:
                # For subsequent periods, use the next expected date
                # (should be 7 days after the last week)
                pass  # current_date continues from previous week
            
            # Expand this week's 7 days
            for day_idx, visits in enumerate(visits_7day):
                daily_rows.append({
                    "placekey": placekey,
                    "date": current_date,
                    "visits": int(visits) if pd.notna(visits) else 0,
                    "day_idx": len(daily_rows) + 1,  # Sequential day numbering
                    "location_name": row.get("location_name", ""),
                    "top_category": row.get("top_category", ""),
                    "latitude": row.get("latitude", np.nan),
                    "longitude": row.get("longitude", np.nan),
                    "time_period": row.get("time_period", ""),
                    "timeline_start": timeline_start
                })
                current_date += pd.Timedelta(days=1)
        
        # Check if we got exactly 21 days
        if len(daily_rows) != 21:
            placekey_stats['timeline_issues'] += 1
            continue
            
        placekey_stats['complete_21days'] += 1
        
        # Create DataFrame for this placekey
        placekey_df = pd.DataFrame(daily_rows)
        complete_panels.append(placekey_df)
    
    print(f"\nPlacekey processing statistics:")
    print(f"  Has all 3 periods: {placekey_stats['has_all_periods']:,}")
    print(f"  Missing 'before': {placekey_stats['missing_before']:,}")
    print(f"  Missing 'landfall': {placekey_stats['missing_landfall']:,}")
    print(f"  Missing 'after': {placekey_stats['missing_after']:,}")
    print(f"  Complete 21-day timelines: {placekey_stats['complete_21days']:,}")
    print(f"  Timeline issues: {placekey_stats['timeline_issues']:,}")
    
    if not complete_panels:
        print("\nERROR: No placekey has complete 21-day timeline!")
        print("This could be due to:")
        print("1. Missing time periods (need before + landfall + after)")
        print("2. Invalid visits_by_day arrays (need exactly 7 values each)")
        print("3. Date parsing issues")
        
        # Show sample data for debugging
        if len(city_df) > 0:
            print("\nSample data for debugging:")
            sample = city_df.head(5)[['placekey', 'time_period', 'date_range_start', 'visits_by_day']]
            for _, row in sample.iterrows():
                visits = parse_visits_by_day(row['visits_by_day'])
                visits_info = f"{len(visits)} values" if isinstance(visits, list) else "invalid"
                print(f"  {row['placekey'][:20]} | {row['time_period']:<8} | {row['date_range_start']} | visits: {visits_info}")
        
        return
    
    # Combine all placekey panels
    panel_df = pd.concat(complete_panels, ignore_index=True)
    n_placekeys = panel_df["placekey"].nunique()
    
    print(f"\nSuccess! Built complete timelines:")
    print(f"  Placekeys: {n_placekeys:,}")
    print(f"  Total daily records: {len(panel_df):,}")
    print(f"  Days per placekey: {len(panel_df) / n_placekeys:.1f}")
    
    # Show sample timelines
    print(f"\nSample timeline (first placekey):")
    sample_pk = panel_df['placekey'].iloc[0]
    sample_timeline = panel_df[panel_df['placekey'] == sample_pk].head(10)
    for _, row in sample_timeline.iterrows():
        print(f"  Day {row['day_idx']:2d}: {row['date'].date()} | {row['visits']:3d} visits | {row['time_period']}")
    
    # --- Create output datasets ---
    
    # 1. Main daily panel 
    main_cols = ["placekey", "date", "visits", "day_idx", "location_name", 
                 "top_category", "latitude", "longitude", "timeline_start"]
    panel_clean = panel_df[main_cols].copy()
    
    # 2. Prophet format
    prophet_df = panel_clean.rename(columns={"date": "ds", "visits": "y"})
    
    # 3. Targets for D14 and D21 forecasting
    targets_df = prophet_df[prophet_df["day_idx"].isin([14, 21])].copy()
    targets_df = targets_df.rename(columns={
        "ds": "target_date", 
        "y": "target_visits"
    })[["placekey", "target_date", "day_idx", "target_visits", "location_name", "top_category", "timeline_start"]]
    
    # Combine all complete panels
    panel_df = pd.concat(complete_panels, ignore_index=True)
    n_placekeys = panel_df["placekey"].nunique()
    
    print(f"Complete 21-day panels: {n_placekeys:,} placekeys")
    print(f"Total daily records: {len(panel_df):,}")
    print(f"Days per placekey: {len(panel_df) / n_placekeys:.1f}")
    
    # --- Create output datasets ---
    
    # 1. Main daily panel (for ARIMA/general use)
    main_cols = ["placekey", "date", "visits", "day_idx", "location_name", 
                 "top_category", "latitude", "longitude"]
    panel_clean = panel_df[main_cols].copy()
    
    # 2. Prophet format (rename columns)
    prophet_df = panel_clean.rename(columns={"date": "ds", "visits": "y"})
    
    # 3. Targets for D14 and D21 forecasting
    targets_df = prophet_df[prophet_df["day_idx"].isin([14, 21])].copy()
    targets_df = targets_df.rename(columns={
        "ds": "target_date", 
        "y": "target_visits"
    })[["placekey", "target_date", "day_idx", "target_visits", "location_name", "top_category"]]
    
    # --- Save outputs ---
    outdir = Path(args.outdir)
    
    # Main panel (CSV + Parquet)
    panel_csv = outdir / f"{args.city.lower()}_daily_panel.csv"
    panel_parquet = outdir / f"{args.city.lower()}_daily_panel.parquet"
    
    panel_clean.to_csv(panel_csv, index=False)
    panel_clean.to_parquet(panel_parquet, index=False)
    
    # Prophet format
    prophet_csv = outdir / f"{args.city.lower()}_prophet_format.csv"
    prophet_parquet = outdir / f"{args.city.lower()}_prophet_format.parquet"
    
    prophet_df.to_csv(prophet_csv, index=False)
    prophet_df.to_parquet(prophet_parquet, index=False)
    
    # Targets
    targets_csv = outdir / f"{args.city.lower()}_targets_d14_d21.csv"
    targets_df.to_csv(targets_csv, index=False)
    
    # Summary stats
    summary_txt = outdir / f"{args.city.lower()}_panel_summary.txt"
    with open(summary_txt, "w") as f:
        f.write(f"Daily Panel Summary - {args.city}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Source: {args.input}\n")
        f.write(f"Date range: {WIN_START.date()} to {WIN_END.date()}\n")
        f.write(f"Total days: {len(EXPECTED_DATES)}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"- Complete placekeys: {n_placekeys:,}\n")
        f.write(f"- Total daily records: {len(panel_df):,}\n")
        f.write(f"- Visit statistics:\n")
        f.write(f"  - Mean: {panel_clean['visits'].mean():.2f}\n")
        f.write(f"  - Median: {panel_clean['visits'].median():.2f}\n")
        f.write(f"  - Min: {panel_clean['visits'].min()}\n")
        f.write(f"  - Max: {panel_clean['visits'].max()}\n")
        f.write(f"  - Std: {panel_clean['visits'].std():.2f}\n\n")
        
        f.write(f"Top categories:\n")
        top_cats = panel_clean['top_category'].value_counts().head(10)
        for cat, count in top_cats.items():
            f.write(f"  {cat}: {count:,}\n")
    
    print(f"\n=== Output Files Created ===")
    print(f"Main panel: {panel_csv}")
    print(f"Main panel: {panel_parquet}")
    print(f"Prophet format: {prophet_csv}")  
    print(f"Prophet format: {prophet_parquet}")
    print(f"Targets: {targets_csv}")
    print(f"Summary: {summary_txt}")
    
    print(f"\n=== Usage Instructions ===")
    print(f"For D14 forecasting:")
    print(f"  - Use day_idx 1-13 as training data")
    print(f"  - Predict day_idx 14")
    print(f"For D21 forecasting:")
    print(f"  - Use day_idx 1-20 as training data")
    print(f"  - Predict day_idx 21")
    
    print(f"\nSample ARIMA usage:")
    print(f"  df = pd.read_parquet('{panel_parquet}')")
    print(f"  for placekey, group in df.groupby('placekey'):")
    print(f"      train = group[group['day_idx'] <= 13]['visits']")
    print(f"      # Fit ARIMA on train, predict day 14")

if __name__ == "__main__":
    main()