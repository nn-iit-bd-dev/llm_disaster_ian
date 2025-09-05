#!/usr/bin/env python3
# data_prep.py
# Prepare 13->14 daily windows for LLM forecasting with timezone-agnostic start-date handling.

import argparse, json, ast, sys
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

# -------------------------- CLI --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prepare D14 forecasting data (13 input days + D14 target).")
    p.add_argument("--input", required=True, help="Raw weekly CSV (SafeGraph-style).")
    p.add_argument("--city", required=True, help="City name to filter (e.g., 'Tampa', 'Cape Coral').")
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument("--landfall-date", default="2022-09-28",
                   help="Landfall date (YYYY-MM-DD). Default: 2022-09-28.")
    p.add_argument("--start-policy", choices=["strict", "first_before", "nearest"], default="strict",
                   help=("How to anchor the 14-day series start:\n"
                         "  strict       -> requires --required-start EXACT calendar day\n"
                         "  first_before -> earliest 'before' period start per placekey\n"
                         "  nearest      -> 'before' start closest to --required-start within --tolerance-days"))
    p.add_argument("--required-start", default="2022-09-19",
                   help="Required start date (YYYY-MM-DD) for 'strict' and anchor for 'nearest'.")
    p.add_argument("--tolerance-days", type=int, default=7,
                   help="Max distance in days allowed from required-start when using 'nearest'.")
    p.add_argument("--max-rows", type=int, default=None, help="Read only first N input rows (speedy trial).")
    p.add_argument("--print-samples", type=int, default=3, help="Print first N prepared records to stdout.")
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

# -------------------------- main --------------------------

def main():
    args = parse_args()
    inp = Path(args.input)
    outp = Path(args.output)
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
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("")  # empty file
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

    # Write JSONL
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for rec in prepared:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    print("=== Preparation Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"Output file: {outp} (records: {len(prepared)})")

    # Show samples
    if prepared and args.print_samples > 0:
        print("\n=== Sample records ===")
        for rec in prepared[:args.print_samples]:
            print(json.dumps(rec, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
