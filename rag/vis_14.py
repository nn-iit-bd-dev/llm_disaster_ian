#!/usr/bin/env python3
# Select 10 worst/best cases and plot per-placekey time series with D14 markers.

import argparse, ast, os, re, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NEEDED_COLS = [
    "placekey","city","location_name","top_category","latitude","longitude",
    "first_date","target_date_d14","actual_target_date","y_true_d14","y_pred_d14",
    "model_type","confidence_lower","confidence_upper","fallback_reason","n_days",
    "time_periods_used","prev_13_values","absolute_error","percent_error","prev_13_values_parsed"
]

def parse_args():
    p = argparse.ArgumentParser(description="Top-10 best/worst selector + plots")
    p.add_argument("--results", required=True, help="Path to results CSV (ARIMA or RAG).")
    p.add_argument("--test-jsonl", required=True, help="Test JSONL used to generate predictions (for merge).")
    p.add_argument("--outdir", required=True, help="Output folder for CSVs and plots.")
    p.add_argument("--landfall", default="2022-09-28", help="Landfall date (YYYY-MM-DD).")
    p.add_argument("--city", help="Optional: filter to a single city (e.g., Tampa).")
    p.add_argument("--topk", type=int, default=10, help="How many best/worst to keep.")
    return p.parse_args()

def safe_parse_list(x):
    if isinstance(x, list):
        return [float(v) for v in x]
    if pd.isna(x):
        return []
    s = str(x).strip()
    # Handle JSON/py list strings
    try:
        lst = ast.literal_eval(s)
        if isinstance(lst, (list, tuple)):
            return [float(v) for v in lst]
    except Exception:
        pass
    # Fallback: split by comma
    try:
        return [float(v) for v in re.findall(r"-?\d+\.?\d*", s)]
    except Exception:
        return []

def compact_list_str(lst, max_items=13):
    if not lst:
        return "[]"
    if len(lst) <= max_items:
        return "[" + ", ".join(f"{v:g}" for v in lst) + "]"
    head = ", ".join(f"{v:g}" for v in lst[:max_items])
    return f"[{head}, ...]"

def ensure_columns(df):
    for c in NEEDED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Load results ---
    res = pd.read_csv(args.results)
    # Standardize some column names from RAG output if present
    if "target_date" in res.columns and "target_date_d14" not in res.columns:
        res = res.rename(columns={"target_date": "target_date_d14"})
    if "confidence" in res.columns and "confidence_lower" not in res.columns:
        res["confidence_lower"] = np.nan
        res["confidence_upper"] = np.nan
    if "model_type" not in res.columns:
        # Tag with something sensible if missing
        res["model_type"] = "model"

    # --- Load test JSONL for enrichment ---
    test_rows = []
    with open(args.test_jsonl, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                test_rows.append(ast.literal_eval(line) if line.startswith("{") else None)
    # Fallback robust JSON loader
    if any(r is None for r in test_rows):
        test_rows = []
        with open(args.test_jsonl, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    test_rows.append(pd.read_json(pd.io.common.StringIO(line), typ="series").to_dict())

    test = pd.DataFrame(test_rows)
    # Some common fields across your datasets
    # (If any are missing, they'll stay NaN and the script still works.)
    keep_from_test = [
        "placekey","city","location_name","top_category","latitude","longitude",
        "series_start_date","first_date","target_date_d14","actual_target_date",
        "prev_13_values","time_periods_used","n_days","landfall_date"
    ]
    for k in keep_from_test:
        if k not in test.columns:
            test[k] = np.nan

    # Prefer joining on (placekey, target_date_d14) when available
    join_keys = [k for k in ["placekey","target_date_d14"] if k in res.columns and k in test.columns]
    if not join_keys:
        join_keys = ["placekey"]

    merged = res.merge(
        test[keep_from_test],
        on=join_keys,
        how="left",
        suffixes=("", "_test")
    )

    # If first_date missing, try series_start_date
    merged["first_date"] = merged["first_date"].fillna(merged.get("series_start_date"))

    # Compute/clean errors
    if "absolute_error" not in merged.columns:
        merged["absolute_error"] = (merged["y_true_d14"] - merged["y_pred_d14"]).abs()
    if "percent_error" not in merged.columns:
        merged["percent_error"] = merged.apply(
            lambda r: 0.0 if pd.isna(r["y_true_d14"]) or r["y_true_d14"] == 0
            else 100.0 * abs(r["y_true_d14"] - r["y_pred_d14"]) / abs(r["y_true_d14"]), axis=1
        )

    # Parse prev_13_values
    merged["prev_13_values_parsed"] = merged["prev_13_values"].apply(safe_parse_list)
    merged["prev_13_values"] = merged["prev_13_values_parsed"].apply(compact_list_str)

    # Optional city filter
    if args.city:
        merged = merged[merged["city"].astype(str).str.lower() == args.city.lower()].copy()
        if merged.empty:
            sys.exit(f"No rows after city filter = {args.city}")

    # Keep just the needed columns (ensure exists)
    merged = ensure_columns(merged)

    # --- Top-K worst/best by absolute_error ---
    worst = merged.sort_values("absolute_error", ascending=False).head(args.topk).copy()
    best  = merged.sort_values("absolute_error", ascending=True ).head(args.topk).copy()

    worst_out = outdir / f"worst_top{args.topk}.csv"
    best_out  = outdir / f"best_top{args.topk}.csv"

    worst[NEEDED_COLS].to_csv(worst_out, index=False)
    best[NEEDED_COLS].to_csv(best_out, index=False)

    print(f"Saved: {worst_out}")
    print(f"Saved: {best_out}")

    # --- Plotting helper ---
    def sanitize(s):
        s = str(s)
        return re.sub(r"[^A-Za-z0-9_.@-]+", "_", s)[:80]

    def plot_row(row):
        try:
            # Dates
            landfall = pd.to_datetime(row.get("landfall_date") or args.landfall)
            start = pd.to_datetime(row.get("first_date") or row.get("series_start_date"))
            if pd.isna(start):
                # fallback: 13 days before target
                target = pd.to_datetime(row["target_date_d14"])
                start = target - pd.Timedelta(days=13)
            dates_1_13 = pd.date_range(start=start, periods=13, freq="D")
            target_d14 = pd.to_datetime(row["target_date_d14"])

            y13 = row["prev_13_values_parsed"]
            if not y13:
                return  # skip if missing history

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(dates_1_13, y13, marker="o", label="Observed (Days 1-13)")

            # D14 actual and predicted
            if not pd.isna(row["y_true_d14"]):
                plt.scatter([target_d14], [row["y_true_d14"]], s=80, marker="o", label=f"Actual D14 = {row['y_true_d14']:.1f}")
            if not pd.isna(row["y_pred_d14"]):
                plt.scatter([target_d14], [row["y_pred_d14"]], s=80, marker="s", label=f"Pred D14 = {row['y_pred_d14']:.1f}")

            # Landfall line
            if not pd.isna(landfall):
                plt.axvline(landfall, linestyle="--", label="Ian landfall")

            # Titles
            loc = row.get("location_name", "")
            city = row.get("city", "")
            abserr = row.get("absolute_error", np.nan)
            pk = row.get("placekey", "")
            lat = row.get("latitude", np.nan)
            lon = row.get("longitude", np.nan)

            plt.title(f"{loc}\n{city} | abs err: {abserr:.2f}\nplacekey: {pk} | lat: {lat:.4f}, lon: {lon:.4f}")
            plt.xlabel("Date")
            plt.ylabel("Visits")
            plt.legend(loc="best")
            plt.tight_layout()

            fname = f"{sanitize(city)}_{sanitize(loc)}_{sanitize(pk)}.png"
            plt.savefig(plots_dir / fname, dpi=150)
            plt.close()
        except Exception as e:
            print(f"[plot] skip row (placekey={row.get('placekey')}): {e}")

    # Make plots for worst and best sets
    for _, r in pd.concat([worst, best], axis=0).iterrows():
        plot_row(r)

    print(f"Per-row plots saved to: {plots_dir}")

if __name__ == "__main__":
    main()

