#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prophet D14 per-POI Visualization (ASCII-only, includes placekey/lat/lon in figures)

Inputs:
    - out_<RUN_DATE>/prophet_d14_forecast/all_predictions.csv   (preferred)
      or per-city files:
    - out_<RUN_DATE>/prophet_d14_forecast/<city>/<city>_d14_predictions.csv

Outputs:
    out_<RUN_DATE>/prophet_d14_forecast/<city>/viz/
        - top10_best.csv / top10_worst.csv
        - best/*.png and worst/*.png per-POI charts
        - top10_best_grid.png / top10_worst_grid.png
"""

import os, json, ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============== CONFIG ==============
RUN_DATE = os.getenv("RUN_DATE", "2025-08-28")
BASE_DIR  = Path(f"prophet_d14_forecast")
ALL_PRED  = BASE_DIR / "all_predictions.csv"
CITIES    = ["Tampa", "Miami", "Orlando", "Cape Coral"]
LANDFALL  = pd.to_datetime("2022-09-28")
TOPN      = 10
FALLBACK_PER_CITY_FILES = True  # search per-city CSVs if ALL_PRED does not exist

# ============== HELPERS ==============
def _parse_prev13(x):
    """Parse prev_13_values into a list[float] with length >= 1."""
    if isinstance(x, (list, tuple)):
        return list(map(float, x))
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(s)
                if isinstance(v, (list, tuple)) and len(v) >= 1:
                    return [float(y) for y in v]
            except Exception:
                pass
        try:
            return [float(y) for y in s.split(",")]
        except Exception:
            return None
    return None

def _first_existing(series, names, default=np.nan):
    """Return the first existing non-null column from names in a Series/Mapping."""
    for n in names:
        if n in series and pd.notna(series[n]):
            return series[n]
    return default

def _load_predictions_for_city(city: str) -> pd.DataFrame:
    """Load Prophet predictions for a city."""
    if ALL_PRED.exists():
        df = pd.read_csv(ALL_PRED)
        if "city" not in df.columns:
            raise ValueError("Expected a 'city' column in all_predictions.csv")
        df = df[df["city"].astype(str).str.upper() == city.upper()].copy()
        return df

    if not FALLBACK_PER_CITY_FILES:
        raise FileNotFoundError(f"Missing {ALL_PRED} and per-city fallback disabled.")

    cdir = BASE_DIR / city.lower().replace(" ", "_")
    guess = cdir / f"{city.lower().replace(' ', '_')}_d14_predictions.csv"
    if guess.exists():
        return pd.read_csv(guess)

    cand = list(cdir.glob("*_d14_predictions.csv"))
    if not cand:
        raise FileNotFoundError(f"No predictions found for {city} in {cdir}")
    return pd.read_csv(cand[0])

def _build_dates(row) -> pd.DatetimeIndex:
    """Build 14 dates (13 train + D14) using available date columns."""
    d14 = _first_existing(row, ["actual_target_date", "target_date_d14"], default=np.nan)
    if pd.notna(d14):
        d14 = pd.to_datetime(d14)
        start = d14 - pd.Timedelta(days=13)
        return pd.date_range(start, d14, freq="D")

    fd = row.get("first_date", np.nan)
    if pd.notna(fd):
        fd = pd.to_datetime(fd)
        return pd.date_range(fd, fd + pd.Timedelta(days=13), freq="D")

    # Last resort: fabricate a 14-day window relative to landfall
    base = LANDFALL - pd.Timedelta(days=6)
    return pd.date_range(base, periods=14, freq="D")

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and are typed properly."""
    # Parse prev_13_values
    if "prev_13_values" in df.columns:
        df["prev_13_values_parsed"] = df["prev_13_values"].apply(_parse_prev13)
    else:
        has_days = all((f"day_{i}_visits" in df.columns) for i in range(1, 14))
        if has_days:
            df["prev_13_values_parsed"] = (
                df[[f"day_{i}_visits" for i in range(1, 14)]].astype(float).values.tolist()
            )
        else:
            df["prev_13_values_parsed"] = None

    # y_true / y_pred flexible mapping
    if "y_true_d14" not in df.columns and "target" in df.columns:
        df["y_true_d14"] = df["target"]
    if "y_pred_d14" not in df.columns and "yhat" in df.columns:
        df["y_pred_d14"] = df["yhat"]

    # Absolute error if missing
    if "absolute_error" not in df.columns and "y_true_d14" in df.columns and "y_pred_d14" in df.columns:
        df["absolute_error"] = (df["y_true_d14"].astype(float) - df["y_pred_d14"].astype(float)).abs()

    # Coerce date columns
    for col in ["actual_target_date", "target_date_d14", "first_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df

def _plot_one_poi(row, out_dir: Path, landfall: pd.Timestamp):
    """
    Per-POI figure:
      - Observed Days 1-13 line
      - Actual D14 (green dot)
      - Prophet D14 (orange square)
      - 80% CI at D14 if available
      - Vertical landfall line
      - Title includes name, city, error, placekey, lat, lon
    """
    prev13 = row["prev_13_values_parsed"]
    if not isinstance(prev13, (list, tuple)) or len(prev13) < 13:
        return

    y_true = float(row.get("y_true_d14", np.nan))
    y_pred = float(row.get("y_pred_d14", np.nan))
    dts = _build_dates(row)
    if len(dts) != 14:
        return

    actual_series = list(map(float, prev13[:13])) + [y_true]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Observed 1-13
    ax.plot(dts[:13], actual_series[:13], marker="o", linewidth=2, label="Observed (Days 1-13)")

    # 80% CI band if available
    lo = row.get("confidence_lower", np.nan)
    hi = row.get("confidence_upper", np.nan)
    if pd.notna(lo) and pd.notna(hi):
        ax.fill_between([dts[13]], [lo], [hi], alpha=0.25, label="80% CI")

    # Actual & Predicted D14
    if pd.notna(y_true):
        ax.plot(dts[13], y_true, "o", markersize=9, label=f"Actual D14 = {y_true:.1f}", color="tab:green")
    if pd.notna(y_pred):
        ax.scatter([dts[13]], [y_pred], marker="s", s=70, label=f"Prophet D14 = {y_pred:.1f}", color="tab:orange")

    # Landfall vline
    if pd.notna(landfall):
        ax.axvline(landfall, linestyle="--", linewidth=1.5, color="gray", alpha=0.9, label="Ian landfall")

    # Title with placekey, lat, lon
    name  = str(row.get("location_name", "") or row.get("placekey", "POI"))
    city  = str(row.get("city", ""))
    pk    = str(row.get("placekey", ""))
    # Cast to float safely to allow formatting; if NaN, show as NaN
    lat   = row.get("latitude", np.nan)
    lon   = row.get("longitude", np.nan)
    try:
        lat_val = float(lat)
    except Exception:
        lat_val = np.nan
    try:
        lon_val = float(lon)
    except Exception:
        lon_val = np.nan
    err   = float(row.get("absolute_error", np.nan))

    ax.set_title(
        f"{name}\n{city} | abs err: {err:.2f}\nplacekey: {pk} | lat: {lat_val:.4f}, lon: {lon_val:.4f}",
        fontsize=10,
        weight="bold"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Visits")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.xticks(rotation=45, ha="right")

    # File name
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
    fname = f"{safe_name[:60]}_{pk}.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()

def _grid10(city_df: pd.DataFrame, rows: int, cols: int, title: str, out_path: Path, landfall: pd.Timestamp):
    """
    2x5 composite grid of Top-10 items for quick inspection.
    Subplot titles include placekey, lat, lon.
    """
    n = len(city_df)
    if n == 0:
        return
    fig, axes = plt.subplots(rows, cols, figsize=(26, 14))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(title, fontsize=18, fontweight="bold")

    for i in range(rows * cols):
        ax = axes[i]
        if i >= n:
            ax.axis("off")
            continue

        row = city_df.iloc[i]
        prev13 = row["prev_13_values_parsed"]
        if not isinstance(prev13, (list, tuple)) or len(prev13) < 13:
            ax.axis("off")
            continue

        dts = _build_dates(row)
        if len(dts) != 14:
            ax.axis("off")
            continue

        y_true = float(row.get("y_true_d14", np.nan))
        y_pred = float(row.get("y_pred_d14", np.nan))
        actual_series = list(map(float, prev13[:13])) + [y_true]

        # Observed
        ax.plot(dts[:13], actual_series[:13], marker="o", linewidth=2, label="Observed (1-13)")

        # D14 points
        if pd.notna(y_true):
            ax.plot(dts[13], y_true, "o", color="tab:green", label="Actual D14")
        if pd.notna(y_pred):
            ax.scatter([dts[13]], [y_pred], marker="s", s=60, color="tab:orange", label="Prophet D14")

        # Landfall
        if pd.notna(landfall):
            ax.axvline(landfall, linestyle="--", linewidth=1.2, color="gray", alpha=0.9)

        name = str(row.get("location_name", "") or row.get("placekey", "POI"))
        city = str(row.get("city", ""))
        pk   = str(row.get("placekey", ""))
        lat  = row.get("latitude", np.nan)
        lon  = row.get("longitude", np.nan)
        try:
            lat_val = float(lat)
        except Exception:
            lat_val = np.nan
        try:
            lon_val = float(lon)
        except Exception:
            lon_val = np.nan
        err  = float(row.get("absolute_error", np.nan))

        ax.set_title(
            f"#{i+1} {name[:22]}\n{city} | err: {err:.2f}\npk: {pk} | lat: {lat_val:.3f}, lon: {lon_val:.3f}",
            fontsize=9,
            weight="bold"
        )

        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

        if i == 0:
            ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============== MAIN ==============
def main():
    print(f"[INFO] Prophet per-POI visualization (RUN_DATE={RUN_DATE})")
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base output dir not found: {BASE_DIR}")

    for city in CITIES:
        print(f"\n=== {city} ===")
        pdf = _load_predictions_for_city(city)
        if pdf.empty:
            print(f"[WARN] No predictions for {city}")
            continue

        pdf = _ensure_columns(pdf)

        # Need rows with truth, pred, and history
        keep = pdf.dropna(subset=["y_true_d14", "y_pred_d14", "prev_13_values_parsed"])
        if keep.empty:
            print(f"[WARN] No valid rows for {city}")
            continue

        # Rank by absolute error
        keep = keep.sort_values("absolute_error", ascending=True).reset_index(drop=True)
        best = keep.head(min(TOPN, len(keep))).copy()
        worst = keep.tail(min(TOPN, len(keep))).copy()

        city_dir = BASE_DIR / city.lower().replace(" ", "_") / "viz"
        city_dir.mkdir(parents=True, exist_ok=True)

        # Save lists
        best.to_csv(city_dir / f"top{TOPN}_best.csv", index=False)
        worst.to_csv(city_dir / f"top{TOPN}_worst.csv", index=False)

        # Per-POI charts
        best_dir = city_dir / "best"
        worst_dir = city_dir / "worst"
        for _, r in best.iterrows():
            _plot_one_poi(r, best_dir, LANDFALL)
        for _, r in worst.iterrows():
            _plot_one_poi(r, worst_dir, LANDFALL)

        # Composite grids
        _grid10(best, 2, 5,
                title="TOP 10 BEST PROPHET FORECASTS (Lowest Absolute Error)",
                out_path=city_dir / "top10_best_grid.png",
                landfall=LANDFALL)

        _grid10(worst, 2, 5,
                title="TOP 10 WORST PROPHET FORECASTS (Highest Absolute Error)",
                out_path=city_dir / "top10_worst_grid.png",
                landfall=LANDFALL)

        print(f"[OK] {city}: wrote per-POI charts and grids -> {city_dir}")

    print("\n[DONE] Prophet visualization complete.")

if __name__ == "__main__":
    main()
