#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dl_daily_panel_prep.py

Create deep-learning train/test datasets for a SINGLE CITY using global
train/test placekey lists (one placekey per line, no headers).

- Expands weekly SafeGraph rows (visits_by_day) into 21-day timelines
  for each placekey (requires periods: before, landfall, after).
- Builds supervised samples:
    d14: X=days 1..13, y=day 14
    d21: X=days 1..20, y=day 21
- Writes per-city outputs: <city>_<task>_{train,test}.{jsonl,csv,(npz)}

Usage example:
  python dl_daily_panel_prep.py \
    --input all_4cities_vbd3w.csv \
    --city "Tampa" \
    --placekeys-train placekeys/train.txt \
    --placekeys-test  placekeys/test.txt \
    --outdir dl_out \
    --task d14 \
    --save-npz
"""

import os, ast, json, argparse
from pathlib import Path
from typing import Set, List, Dict, Tuple

import pandas as pd
import numpy as np

EXPECTED_PERIODS = {"before", "landfall", "after"}
PERIOD_ORDER = {"before": 1, "landfall": 2, "after": 3}

# -------------------- I/O helpers --------------------

def load_placekeys(file_path: Path) -> Set[str]:
    if not file_path.exists():
        raise FileNotFoundError("Placekeys file not found: %s" % file_path)
    s: Set[str] = set()
    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            t = line.strip()
            if t and t.lower() != "placekey":
                s.add(t)
    if not s:
        raise ValueError("No placekeys loaded from file: %s" % file_path)
    return s

def parse_visits_by_day(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return j
        except Exception:
            pass
        try:
            j = ast.literal_eval(s)
            if isinstance(j, list):
                return j
        except Exception:
            pass
        try:
            return [float(x.strip()) for x in s.split(",")]
        except Exception:
            return None
    return None

def expand_week_to_daily(row):
    visits_7 = parse_visits_by_day(row.get("visits_by_day"))
    if not (isinstance(visits_7, list) and len(visits_7) == 7):
        return []
    try:
        start_date = pd.to_datetime(row["date_range_start"]).normalize()
    except Exception:
        return []
    out = []
    for i, v in enumerate(visits_7):
        out.append({
            "placekey": row["placekey"],
            "date": start_date + pd.Timedelta(days=i),
            "visits": int(v) if pd.notna(v) else 0,
            "location_name": row.get("location_name", ""),
            "top_category": row.get("top_category", ""),
            "latitude": row.get("latitude", np.nan),
            "longitude": row.get("longitude", np.nan),
            "time_period": row.get("time_period", "")
        })
    return out

def build_21day_panels(city_df: pd.DataFrame):
    stats = {"has_all_periods": 0, "missing_before": 0, "missing_landfall": 0,
             "missing_after": 0, "complete_21days": 0, "timeline_issues": 0}
    panels = []
    for pk, grp in city_df.groupby("placekey"):
        periods = set(grp["time_period"])
        if periods != EXPECTED_PERIODS:
            for m in EXPECTED_PERIODS - periods:
                stats[f"missing_{m}"] += 1
            continue
        stats["has_all_periods"] += 1
        g = grp.copy()
        g["period_order"] = g["time_period"].map(PERIOD_ORDER)
        g = g.sort_values(["period_order", "date_range_start"])
        daily = []
        for _, row in g.iterrows():
            daily.extend(expand_week_to_daily(row))
        if len(daily) != 21:
            stats["timeline_issues"] += 1
            continue
        df21 = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
        df21["day_idx"] = np.arange(1, 22, dtype=int)
        df21["timeline_start"] = df21["date"].iloc[0]
        panels.append(df21)
        stats["complete_21days"] += 1
    return panels, stats

def build_samples(panel_df: pd.DataFrame, city: str, task: str):
    if task == "d14":
        x_lo, x_hi, y_day, tag = 1, 13, 14, "d14"
    else:
        x_lo, x_hi, y_day, tag = 1, 20, 21, "d21"

    samples = []
    for pk, g in panel_df.groupby("placekey"):
        g = g.sort_values("day_idx")
        if g["day_idx"].min() > x_lo or g["day_idx"].max() < y_day:
            continue
        X = g[g["day_idx"].between(x_lo, x_hi)]["visits"].astype(float).to_list()
        if len(X) != (x_hi - x_lo + 1):
            continue
        y = float(g[g["day_idx"] == y_day]["visits"].iloc[0])
        first = g.iloc[0]
        meta = {
            "placekey": pk,
            "city": city,
            "location_name": first.get("location_name", ""),
            "top_category": first.get("top_category", ""),
            "latitude": float(first.get("latitude", np.nan)) if pd.notna(first.get("latitude", np.nan)) else None,
            "longitude": float(first.get("longitude", np.nan)) if pd.notna(first.get("longitude", np.nan)) else None,
            "timeline_start": str(pd.to_datetime(first["timeline_start"]).date()),
            "x_start_date": str(pd.to_datetime(g[g["day_idx"] == x_lo]["date"].iloc[0]).date()),
            "x_end_date": str(pd.to_datetime(g[g["day_idx"] == x_hi]["date"].iloc[0]).date()),
            "y_date": str(pd.to_datetime(g[g["day_idx"] == y_day]["date"].iloc[0]).date()),
        }
        samples.append({"placekey": pk, "task": tag, "X": X, "y": y, "meta": meta})
    return samples, y_day

def write_outputs(city: str, task: str, split: str, samples: List[dict], outdir: Path, save_npz: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{city.lower().replace(' ', '_')}_{task}_{split}"

    # JSONL
    jsonl_path = outdir / f"{tag}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # CSV
    if samples:
        L = len(samples[0]["X"])
    else:
        L = 0
    cols = [f"x_{i}" for i in range(1, L+1)] + ["y","placekey","city","location_name","top_category","latitude","longitude","x_start_date","x_end_date","y_date"]
    rows = []
    for s in samples:
        row = {f"x_{i}": v for i, v in enumerate(s["X"], start=1)}
        m = s["meta"]
        row.update({
            "y": s["y"],
            "placekey": s["placekey"],
            "city": m.get("city",""),
            "location_name": m.get("location_name",""),
            "top_category": m.get("top_category",""),
            "latitude": m.get("latitude",""),
            "longitude": m.get("longitude",""),
            "x_start_date": m.get("x_start_date",""),
            "x_end_date": m.get("x_end_date",""),
            "y_date": m.get("y_date",""),
        })
        rows.append(row)
    csv_df = pd.DataFrame(rows, columns=cols)
    csv_path = outdir / f"{tag}.csv"
    csv_df.to_csv(csv_path, index=False)

    # NPZ (optional)
    npz_path = None
    if save_npz and samples:
        X = np.asarray([s["X"] for s in samples], dtype=np.float32)
        y = np.asarray([s["y"] for s in samples], dtype=np.float32)
        pks = np.asarray([s["placekey"] for s in samples], dtype=object)
        npz_path = outdir / f"{tag}.npz"
        np.savez_compressed(npz_path, X=X, y=y, placekeys=pks)

    return jsonl_path, csv_path, npz_path

# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="DL prep for a single city using global train/test placekeys")
    ap.add_argument("--input", required=True, help="Weekly hurricane CSV (all cities)")
    ap.add_argument("--city", required=True, help="City name to process (must match CSV 'city' or 'city_norm')")
    ap.add_argument("--placekeys-train", required=True, help="File with TRAIN placekeys (one per line)")
    ap.add_argument("--placekeys-test",  required=True, help="File with TEST placekeys (one per line)")
    ap.add_argument("--outdir", required=True, help="Output directory (per-city files will be placed here)")
    ap.add_argument("--task", choices=["d14","d21"], default="d14", help="Supervised task")
    ap.add_argument("--save-npz", action="store_true", help="Also save NPZ tensors")
    return ap.parse_args()

def main():
    args = parse_args()
    city = args.city
    outdir = Path(args.outdir)

    # Load placekey splits
    train_pk = load_placekeys(Path(args.placekeys_train))
    test_pk  = load_placekeys(Path(args.placekeys_test))
    inter = train_pk & test_pk
    if inter:
        print("[WARN] %d placekeys appear in BOTH train and test; they will be treated as TEST." % len(inter))
        train_pk = train_pk - inter  # ensure disjoint

    print("City:", city)
    print("Train placekeys:", len(train_pk))
    print("Test placekeys :", len(test_pk))

    # Load CSV once
    df = pd.read_csv(args.input, low_memory=False)

    # Normalize city column
    if "city_norm" in df.columns:
        df["city_used"] = df["city_norm"].astype(str)
    elif "city" in df.columns:
        df["city_used"] = df["city"].astype(str)
    else:
        raise ValueError("No 'city' or 'city_norm' column found.")
    df["city_used_up"] = df["city_used"].str.upper()

    # Filter by this city
    city_df = df[df["city_used_up"] == city.upper()].copy()
    if city_df.empty:
        raise SystemExit("[ERROR] No rows found for city: %s" % city)

    # Normalize time_period and keep expected ones only
    city_df["time_period"] = city_df["time_period"].astype(str).str.strip().str.lower().replace({"during":"landfall"})
    city_df = city_df[city_df["time_period"].isin(EXPECTED_PERIODS)].copy()

    # Deduplicate per (placekey, time_period) by latest date_range_start
    city_df["_start_dt"] = pd.to_datetime(city_df["date_range_start"], errors="coerce")
    city_df = (city_df.sort_values(["placekey","time_period","_start_dt"])
                     .drop_duplicates(["placekey","time_period"], keep="last")
                     .drop(columns=["_start_dt"]))

    # Keep only placekeys that are in train or test (to speed up)
    keep_pks = train_pk | test_pk
    city_df = city_df[city_df["placekey"].isin(keep_pks)].copy()
    if city_df.empty:
        raise SystemExit("[ERROR] None of your provided placekeys are present in city '%s'." % city)

    # Build panels
    panels, stats = build_21day_panels(city_df)
    print("Panels: has_all=%d, complete=%d, missing(before=%d,landfall=%d,after=%d), timeline_issues=%d" %
          (stats["has_all_periods"], stats["complete_21days"], stats["missing_before"],
           stats["missing_landfall"], stats["missing_after"], stats["timeline_issues"]))

    if not panels:
        raise SystemExit("[ERROR] No complete 21-day panels built for the selected placekeys in this city.")

    panel_df = pd.concat(panels, ignore_index=True)
    panel_df["city"] = city

    # Split by placekeys (TEST takes precedence for any overlap)
    test_panel  = panel_df[panel_df["placekey"].isin(test_pk)].copy()
    train_panel = panel_df[panel_df["placekey"].isin(train_pk)].copy()

    # Build DL samples
    train_samples, y_day = build_samples(train_panel, city, args.task)
    test_samples,  _     = build_samples(test_panel,  city, args.task)

    print("Samples: train=%d, test=%d (target day=%d)" % (len(train_samples), len(test_samples), y_day))
    print("Distinct placekeys in samples: train=%d, test=%d" %
          (len({s['placekey'] for s in train_samples}), len({s['placekey'] for s in test_samples})))

    # Write
    city_dir = outdir / city.lower().replace(" ", "_")
    write_outputs(city, args.task, "train", train_samples, city_dir, args.save_npz)
    write_outputs(city, args.task, "test",  test_samples,  city_dir, args.save_npz)

    print("Done. Outputs written under:", city_dir.resolve())

if __name__ == "__main__":
    main()
