#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# arima_forecast_clean.py
# ARIMA D14/D21 forecasts with integerized outputs, tz-naive dates,
# overview charts, per-placekey plots, and Top-10 best/worst exports (LLM-style plots).

import argparse, warnings, math, ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore', category=RuntimeWarning, module='statsmodels')
warnings.filterwarnings('ignore', message='.*divide by zero.*')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    raise ImportError("Install statsmodels: pip install statsmodels")

IAN_LANDFALL = pd.to_datetime("2022-09-28")  # tz-naive

# ----------------------------- Helpers -----------------------------

def _to_naive(dt_series: pd.Series) -> pd.Series:
    s = pd.to_datetime(dt_series, errors='coerce')
    try:
        if getattr(s.dt, "tz", None) is not None:
            return s.dt.tz_localize(None)
    except Exception:
        pass
    return s

def _safe_list(obj, length=13):
    if isinstance(obj, list): return obj
    if isinstance(obj, str):
        try:
            val = ast.literal_eval(obj)
            return val if isinstance(val, list) else [0]*length
        except Exception:
            return [0]*length
    return [0]*length

def parse_args():
    ap = argparse.ArgumentParser(description="ARIMA forecasting on daily panel data")
    ap.add_argument("--input", default="ts_daily_panel/cape_daily_panel.parquet",
                    help="Input daily panel (CSV or Parquet)")
    ap.add_argument("--city", default="Cape Coral", help="City name for outputs")
    ap.add_argument("--target", default="d14", choices=["d14","d21"],
                    help="Forecast target: day 14 or 21")
    ap.add_argument("--outdir", default="arima_forecasts", help="Output dir")
    ap.add_argument("--parallel", type=int, default=0,
                    help="Workers (0=auto, -1=disable -> sequential)")
    ap.add_argument("--sample", type=int, default=0,
                    help="Random sample N placekeys for a quick run (0=all)")
    # per-placekey plots
    ap.add_argument("--plot-each", type=int, default=0,
                    help="Make N per-placekey plots (random subset). 0=skip")
    ap.add_argument("--plot-topk", type=int, default=10,
                    help="Also plot Top-K best (lowest error). 0=skip")
    ap.add_argument("--plot-worstk", type=int, default=10,
                    help="Also plot Worst-K (highest error). 0=skip")
    return ap.parse_args()

ARIMA_CANDIDATES = [
    (0,0,0),(1,0,0),(0,0,1),(1,0,1),(2,0,0),(0,0,2),(2,0,1),(1,0,2),
    (0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,0),(0,1,2),(2,1,1),(1,1,2),
]

def diagnose_series(y):
    y = np.array(y, dtype=float)
    d = {'length': len(y),
         'mean': float(np.mean(y)) if len(y) else 0.0,
         'std': float(np.std(y)) if len(y) else 0.0,
         'min': float(np.min(y)) if len(y) else 0.0,
         'max': float(np.max(y)) if len(y) else 0.0,
         'zeros': int(np.sum(y == 0)) if len(y) else 0,
         'constant': np.allclose(y, y[0]) if len(y) > 0 else True,
         'negative': int(np.sum(y < 0)) if len(y) else 0,
         'trend': 'unknown', 'stationarity': 'unknown'}
    if len(y) > 2:
        fh, sh = np.mean(y[:len(y)//2]), np.mean(y[len(y)//2:])
        d['trend'] = 'increasing' if sh > fh*1.2 else ('decreasing' if sh < fh*0.8 else 'stable')
    if len(y) >= 10 and d['std'] > 0:
        try:
            pval = adfuller(y, autolag='AIC')[1]
            d['stationarity'] = 'stationary' if pval < 0.05 else 'non_stationary'
            d['adf_pvalue'] = float(pval)
        except Exception:
            pass
    return d

def fit_best_arima(y_train, max_models=8):
    y = np.array(y_train, dtype=float)
    diag = diagnose_series(y)
    if len(y) < 5 or diag['constant'] or diag['std'] <= 1e-10:
        return None, (0,0,0), np.inf, diag
    if diag['std'] / max(abs(diag['mean']), 1) < 1e-6:
        return None, (0,0,0), np.inf, diag
    best_model, best_order, best_aic = None, None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in ARIMA_CANDIDATES[:max_models]:
            try:
                m = ARIMA(y, order=order, enforce_stationarity=False,
                          enforce_invertibility=False, concentrate_scale=True)
                fit = m.fit(method_kwargs={"warn_convergence": False})
                aic = fit.aic if np.isfinite(getattr(fit, "aic", np.inf)) else np.inf
                if aic < best_aic:
                    best_model, best_order, best_aic = fit, order, aic
            except Exception:
                continue
    return best_model, best_order, best_aic, diag

# ----------------------------- Forecast core -----------------------------

def forecast_placekey_d14(placekey_data):
    placekey = placekey_data['placekey']
    series_data = placekey_data['series']
    metadata = placekey_data['metadata']
    try:
        target_day = 14 if metadata.get('target','d14') == 'd14' else 21
        train_max  = target_day - 1
        train_data = series_data[series_data['day_idx'] <= train_max].sort_values('day_idx')

        if len(train_data) != train_max:
            return create_result(placekey, metadata, np.nan, np.nan, "error", "", "insufficient_training_data")

        y_train = train_data['visits'].values
        y_act_s = series_data[series_data['day_idx'] == target_day]['visits']
        if len(y_act_s) == 0:
            return create_result(placekey, metadata, np.nan, np.nan, "error", "", "missing_actual")
        y_actual = float(y_act_s.iloc[0])

        model, order, aic, diagnostics = fit_best_arima(y_train)
        if model is None:
            raw_pred = float(y_train[-1])
            pred_int = max(0, int(round(max(0.0, min(raw_pred, 10000.0)))))
            return create_result(placekey, metadata, pred_int, raw_pred, "naive", "",
                                 "arima_failed", y_actual, diagnostics, aic)

        try:
            raw_pred = float(model.forecast(steps=1)[0])
            if not np.isfinite(raw_pred):
                raw_pred = float(y_train[-1])
                clamped  = max(0.0, min(raw_pred, 10000.0))
                pred_int = max(0, int(round(clamped)))
                mtype, reason = "naive", "invalid_forecast"
            else:
                clamped  = max(0.0, min(raw_pred, 10000.0))
                pred_int = max(0, int(round(clamped)))
                mtype, reason = "arima", ""
            return create_result(placekey, metadata, pred_int, raw_pred, mtype, str(order),
                                 reason, y_actual, diagnostics, aic)
        except Exception as e:
            raw_pred = float(y_train[-1])
            pred_int = max(0, int(round(max(0.0, min(raw_pred, 10000.0)))))
            return create_result(placekey, metadata, pred_int, raw_pred, "naive", "",
                                 f"forecast_error:{str(e)[:50]}", y_actual, diagnostics, aic)
    except Exception as e:
        return create_result(placekey, metadata, np.nan, np.nan, "error", "",
                             f"general_error:{str(e)[:50]}")

def create_result(placekey, metadata, prediction_int, prediction_raw, model_type, order, reason,
                  actual=np.nan, diagnostics=None, aic=np.nan):
    abs_error = (abs(actual - prediction_int)
                 if not (np.isnan(actual) or np.isnan(prediction_int)) else np.nan)
    rel_error_pct = ((abs_error / max(actual, 1)) * 100
                     if not np.isnan(abs_error) and actual > 0 else np.nan)

    denom = (abs(actual) + abs(prediction_int)) / 2 + 1e-9
    smape_val = (abs(prediction_int - actual) / denom * 100
                 if not (np.isnan(actual) or np.isnan(prediction_int)) else np.nan)

    rmsle_sq = ( (np.log(max(actual,0)+1) - np.log(max(prediction_int,0)+1))**2
                 if not (np.isnan(actual) or np.isnan(prediction_int)) else np.nan )

    result = {
        'placekey': placekey,
        'city': metadata.get('city',''),
        'location_name': metadata.get('location_name',''),
        'top_category': metadata.get('top_category',''),
        'latitude': metadata.get('latitude', np.nan),
        'longitude': metadata.get('longitude', np.nan),
        'first_date': metadata.get('first_date', pd.NaT),
        'target_date_d14': metadata.get('target_date_d14', pd.NaT),
        'actual_target_date': metadata.get('actual_target_date', pd.NaT),
        'y_true_d14': actual,
        'y_pred_d14': prediction_int,
        'y_pred_d14_raw': prediction_raw,
        'model_type': model_type,
        'confidence_lower': np.nan,
        'confidence_upper': np.nan,
        'fallback_reason': reason,
        'n_days': 13,
        'time_periods_used': metadata.get('time_periods_used',''),
        'prev_13_values': metadata.get('prev_13_values', []),
        'absolute_error': abs_error,
        'percent_error': rel_error_pct,
        'smape': smape_val,
        'rmsle': rmsle_sq,
        'arima_order': order,
        'aic': aic
    }
    if diagnostics:
        result.update({f'diag_{k}': v for k,v in diagnostics.items()})
    return result

def calculate_performance_metrics(results_df):
    df = results_df.dropna(subset=['y_true_d14','y_pred_d14']).copy()
    if len(df) == 0: return {}
    resid = df['y_pred_d14'] - df['y_true_d14']
    def smape_mean(a,p):
        a, p = np.array(a), np.array(p)
        denom = (np.abs(a)+np.abs(p))/2 + 1e-9
        return (np.abs(p-a)/denom).mean()*100
    def smape_median(a,p):
        a, p = np.array(a), np.array(p)
        denom = (np.abs(a)+np.abs(p))/2 + 1e-9
        return np.median(np.abs(p-a)/denom)*100
    def rmsle(a,p):
        a = np.maximum(np.array(a),0); p = np.maximum(np.array(p),0)
        return np.sqrt(((np.log1p(p)-np.log1p(a))**2).mean())
    return {
        'n_total': len(results_df),
        'n_valid': len(df),
        'n_arima': int((df['model_type']=='arima').sum()),
        'n_naive': int((df['model_type']=='naive').sum()),
        'pct_arima_success': float((df['model_type']=='arima').mean()*100),
        'mae': float(np.abs(resid).mean()),
        'rmse': float(np.sqrt((resid**2).mean())),
        'smape': float(smape_mean(df['y_true_d14'], df['y_pred_d14'])),
        'smape_median': float(smape_median(df['y_true_d14'], df['y_pred_d14'])),
        'rmsle': float(rmsle(df['y_true_d14'], df['y_pred_d14'])),
        'median_error': float(np.abs(resid).median()),
        'prediction_mean': float(df['y_pred_d14'].mean()),
        'actual_mean': float(df['y_true_d14'].mean()),
    }

# ----------------------------- Plotting (LLM style) -----------------------------

def _plot_llm_style(row, out_path: Path):
    """Make a figure matching the LLM style:
       - Observed (Days 1–13) line+dots
       - Actual D14 (green circle)
       - ARIMA D14 (orange square)
       - 80% CI slab centered on D14 (if available from ARIMA refit)
       - Vertical dashed Ian landfall
       - 3-line title (name; 'city | abs err'; 'placekey | lat, lon')
    """
    name = str(row.get("location_name",""))
    city = str(row.get("city",""))
    pk   = str(row.get("placekey",""))
    lat  = row.get("latitude", np.nan)
    lon  = row.get("longitude", np.nan)

    prev = row.get("prev_13_values", [])
    if isinstance(prev, str):
        try: prev = ast.literal_eval(prev)
        except Exception: prev = []
    if not isinstance(prev, list): prev = []
    if len(prev) < 13: prev = (prev + [0]*13)[:13]

    start_date = pd.to_datetime(row.get("first_date"), errors='coerce')
    try:
        if getattr(start_date,'tzinfo',None) is not None:
            start_date = start_date.tz_localize(None)
    except Exception:
        pass
    if pd.isna(start_date): start_date = pd.to_datetime("2022-09-19")
    dates_train = pd.date_range(start=start_date, periods=13, freq='D')
    d14_date = start_date + pd.Timedelta(days=13)

    y_true = float(row.get("y_true_d14", np.nan)) if pd.notna(row.get("y_true_d14", np.nan)) else np.nan
    y_pred = float(row.get("y_pred_d14", np.nan)) if pd.notna(row.get("y_pred_d14", np.nan)) else np.nan

    # 80% CI via ARIMA refit using stored order if available
    ci_low = ci_high = None
    try:
        y = np.asarray(prev, dtype=float)
        order = (0,0,2)
        stored = row.get("arima_order")
        if isinstance(stored, str) and stored.startswith("("):
            stored = ast.literal_eval(stored)
        if isinstance(stored,(list,tuple)) and len(stored)==3:
            order = tuple(int(v) for v in stored)
        m = ARIMA(y, order=order, enforce_stationarity=False,
                  enforce_invertibility=False, concentrate_scale=True)
        fit = m.fit(method_kwargs={"warn_convergence": False})
        ci = fit.get_forecast(steps=1).conf_int(alpha=0.2)  # 80%
        ci_low, ci_high = float(ci.iloc[0,0]), float(ci.iloc[0,1])
    except Exception:
        pass

    plt.figure(figsize=(12,7), dpi=150)
    # Observed 1–13
    plt.plot(dates_train, prev, marker="o", linewidth=2.0, label="Observed (Days 1-13)")

    # CI slab around D14 if available
    if ci_low is not None and ci_high is not None:
        left = d14_date - pd.Timedelta(hours=12)
        right = d14_date + pd.Timedelta(hours=12)
        plt.fill_between([left, right], [ci_low, ci_low], [ci_high, ci_high], alpha=0.15, label="80% CI")

    # Actual D14 (green circle)
    if np.isfinite(y_true):
        plt.scatter(d14_date, y_true, s=120, marker="o", label=f"Actual D14 = {y_true:.1f}", zorder=3)

    # ARIMA D14 (orange square)
    if np.isfinite(y_pred):
        plt.scatter(d14_date, y_pred, s=120, marker="s", label=f"ARIMA D14 = {y_pred:.1f}", zorder=3)

    # Landfall line
    if dates_train.min() <= IAN_LANDFALL <= d14_date:
        plt.axvline(IAN_LANDFALL, linestyle="--", linewidth=1.5, label="Ian landfall")

    # Title
    abs_err = (abs(y_true - y_pred) if (np.isfinite(y_true) and np.isfinite(y_pred)) else np.nan)
    title1 = f"{name}"
    title2 = f"{city} | abs err: {abs_err:.2f}" if np.isfinite(abs_err) else f"{city}"
    title3 = f"placekey: {pk} | lat: {lat:.4f}, lon: {lon:.4f}"
    plt.title(f"{title1}\n{title2}\n{title3}", fontsize=14, fontweight="bold")

    plt.xlabel("Date"); plt.ylabel("Visits"); plt.legend(loc="lower left")

    # y-lims padding
    yvals = [*prev]
    if np.isfinite(y_true): yvals.append(y_true)
    if np.isfinite(y_pred): yvals.append(y_pred)
    if ci_low is not None:  yvals.append(ci_low)
    if ci_high is not None: yvals.append(ci_high)
    if len(yvals) == 0: yvals = [0,1]
    y_min = max(0.0, min(yvals) - 0.1 * (max(yvals) - min(yvals) + 1))
    y_max = max(yvals) + 0.15 * (max(yvals) - min(yvals) + 1)
    if y_min == y_max: y_max = y_min + 1.0
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path

# ----------------------------- Main -----------------------------

def create_visualizations(results_df, outdir, city):
    try:
        valid = results_df.dropna(subset=['absolute_error']).copy()
        if len(valid) == 0:
            print("No valid results for visualization"); return
        valid = valid.sort_values('absolute_error', na_position='last')

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(20, 16), constrained_layout=True
        )
        fig.suptitle(f'ARIMA D14 Forecast Performance - {city}\nHurricane Ian Analysis',
                     fontsize=18, fontweight='bold')

        best_10 = valid.head(10); worst_10 = valid.tail(10)

        ax1.barh(range(len(best_10)), best_10['absolute_error'], alpha=0.7)
        ax1.set_yticks(range(len(best_10)))
        ax1.set_yticklabels([str(n)[:28]+("..." if len(str(n))>28 else "") for n in best_10['location_name']], fontsize=9)
        ax1.set_xlabel('Absolute Error'); ax1.set_title('Top 10 Best (Lowest MAE)'); ax1.grid(True, alpha=0.3)

        ax2.barh(range(len(worst_10)), worst_10['absolute_error'], alpha=0.7)
        ax2.set_yticks(range(len(worst_10)))
        ax2.set_yticklabels([str(n)[:28]+("..." if len(str(n))>28 else "") for n in worst_10['location_name']], fontsize=9)
        ax2.set_xlabel('Absolute Error'); ax2.set_title('Top 10 Worst (Highest MAE)'); ax2.grid(True, alpha=0.3)

        ax3.hist(valid['absolute_error'], bins=50, alpha=0.7, edgecolor='black')
        mean_err = valid['absolute_error'].mean(); med_err = valid['absolute_error'].median()
        ax3.axvline(mean_err, linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f}')
        ax3.axvline(med_err,  linestyle='--', linewidth=2, label=f'Median: {med_err:.2f}')
        ax3.set_xlabel('Absolute Error'); ax3.set_ylabel('Frequency'); ax3.set_title('Error Distribution')
        ax3.legend(); ax3.grid(True, alpha=0.3)

        if 'top_category' in valid.columns:
            cat_err = valid.groupby('top_category')['absolute_error'].mean().sort_values().head(15)
            ax4.barh(range(len(cat_err)), cat_err.values, alpha=0.7)
            ax4.set_yticks(range(len(cat_err)))
            ax4.set_yticklabels([c[:30]+("..." if len(c)>30 else "") for c in cat_err.index], fontsize=8)
            ax4.set_xlabel('Mean Absolute Error'); ax4.set_title('Performance by Category'); ax4.grid(True, alpha=0.3)

        viz_path = outdir / f'{city.lower()}_arima_performance_overview.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved main visualization: {viz_path}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback; traceback.print_exc()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    plots_best = outdir / "plots_top_best"
    plots_worst = outdir / "plots_top_worst"
    plots_best.mkdir(exist_ok=True); plots_worst.mkdir(exist_ok=True)

    print(f"=== ARIMA {args.target.upper()} Forecasting - {args.city} ===")
    print(f"Input: {args.input}"); print(f"Target: Day {args.target[1:]}")

    # Load
    if args.input.endswith('.parquet'):
        panel_df = pd.read_parquet(args.input)
    else:
        panel_df = pd.read_csv(args.input)

    # Normalize datetimes to tz-naive
    if 'date' in panel_df.columns:
        panel_df['date'] = _to_naive(panel_df['date'])
    if 'timeline_start' in panel_df.columns:
        panel_df['timeline_start'] = _to_naive(panel_df['timeline_start'])

    print(f"Loaded panel: {len(panel_df):,} rows, {panel_df['placekey'].nunique():,} placekeys")

    # Random sample
    if args.sample > 0:
        all_pk = panel_df['placekey'].dropna().unique()
        rng = np.random.default_rng(42)
        pick = rng.choice(all_pk, size=min(args.sample, len(all_pk)), replace=False)
        panel_df = panel_df[panel_df['placekey'].isin(pick)]
        print(f"Sampling {len(pick)} random placekeys for testing")

    # Prepare forecasting payloads
    forecast_data = []
    target_day = 14 if args.target == "d14" else 21
    train_max_day = target_day - 1

    for placekey, group in panel_df.groupby('placekey'):
        if target_day not in set(group['day_idx'].values):
            continue
        meta_row = group.iloc[0]
        train_data = group[group['day_idx'] <= train_max_day].sort_values('day_idx')
        prev_13_values = train_data['visits'].tolist() if len(train_data) == train_max_day else []
        time_periods = ';'.join(group['time_period'].dropna().unique()) if 'time_period' in group.columns else ''

        first_dt = pd.to_datetime(meta_row.get('timeline_start', meta_row.get('date', pd.NaT)), errors='coerce')
        try:
            if getattr(first_dt,'tzinfo',None) is not None:
                first_dt = first_dt.tz_localize(None)
        except Exception:
            pass
        target_d14 = first_dt + pd.Timedelta(days=13) if pd.notna(first_dt) else pd.NaT
        actual_dt_series = group.loc[group['day_idx'] == target_day, 'date']
        actual_dt = actual_dt_series.iloc[0] if len(actual_dt_series) else pd.NaT

        metadata = {
            'city': args.city,
            'location_name': meta_row.get('location_name',''),
            'top_category': meta_row.get('top_category',''),
            'latitude': meta_row.get('latitude', np.nan),
            'longitude': meta_row.get('longitude', np.nan),
            'first_date': first_dt,
            'target_date_d14': target_d14,
            'actual_target_date': actual_dt,
            'time_periods_used': time_periods,
            'prev_13_values': prev_13_values,
            'target': args.target
        }
        forecast_data.append({'placekey': placekey, 'series': group.copy(), 'metadata': metadata})

    print(f"Prepared {len(forecast_data):,} placekeys for {args.target} forecasting")

    # Workers
    n_workers = args.parallel
    if n_workers == 0: n_workers = max(1, min(mp.cpu_count()-1, 8))
    elif n_workers == -1: n_workers = 1
    print(f"Using {n_workers} parallel workers")

    # Run
    print("Running ARIMA forecasting...")
    if n_workers == 1:
        results = []
        for i, data in enumerate(forecast_data):
            if i % 1000 == 0:
                print(f"  Progress: {i:,}/{len(forecast_data):,} ({i/len(forecast_data)*100:.1f}%)")
            results.append(forecast_placekey_d14(data))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(forecast_placekey_d14, d) for d in forecast_data]
            results = []
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 1000 == 0 or i == len(futs):
                    print(f"  Parallel progress: {i:,}/{len(futs):,} ({i/len(futs)*100:.1f}%)")

    results_df = pd.DataFrame(results)
    print(f"Completed forecasting: {len(results_df):,} results")

    # Metrics
    metrics = calculate_performance_metrics(results_df)
    print(f"\n=== Performance Summary ===")
    print(f"Total placekeys: {metrics.get('n_total', 0):,}")
    print(f"Valid forecasts: {metrics.get('n_valid', 0):,}")
    print(f"ARIMA models: {metrics.get('n_arima', 0):,} ({metrics.get('pct_arima_success', 0):.1f}%)")
    print(f"Naive fallbacks: {metrics.get('n_naive', 0):,}")
    print(f"MAE: {metrics.get('mae', np.nan):.2f}")
    print(f"RMSE: {metrics.get('rmse', np.nan):.2f}")
    print(f"sMAPE (mean): {metrics.get('smape', np.nan):.2f}%")
    print(f"sMAPE (median): {metrics.get('smape_median', np.nan):.2f}%")
    print(f"RMSLE: {metrics.get('rmsle', np.nan):.3f}")
    print(f"Median Error: {metrics.get('median_error', np.nan):.2f}")

    # Expand prev values to columns (audit)
    results_df['prev_13_values_parsed'] = results_df['prev_13_values'].apply(lambda x: _safe_list(x, length=train_max_day))
    for i in range(train_max_day):
        results_df[f'day_{i+1}_visits'] = results_df['prev_13_values_parsed'].apply(lambda x, j=i: x[j] if len(x)>j else np.nan)

    # Save detailed + summary
    for col in ['location_name','top_category','fallback_reason']:
        if col in results_df.columns:
            results_df[col] = results_df[col].astype(str).apply(lambda s: s.encode('ascii', errors='replace').decode('ascii'))

    detailed_cols = [
        'placekey','city','location_name','top_category','latitude','longitude',
        'first_date','target_date_d14','actual_target_date',
        'y_true_d14','y_pred_d14','y_pred_d14_raw','model_type',
        'confidence_lower','confidence_upper',
        'fallback_reason','n_days','time_periods_used','prev_13_values'
    ] + [f'day_{i+1}_visits' for i in range(train_max_day)] + [
        'absolute_error','percent_error','smape','rmsle',
        'prev_13_values_parsed','arima_order','aic'
    ]
    avail = [c for c in detailed_cols if c in results_df.columns]
    detailed = results_df[avail].copy().sort_values('absolute_error', na_position='last')

    results_csv = outdir / f'{args.city.lower()}_arima_{args.target}_results.csv'
    summary_csv = outdir / f'{args.city.lower()}_arima_{args.target}_summary.csv'
    detailed.to_csv(results_csv, index=False, encoding='ascii', errors='replace')
    pd.DataFrame([{ 'city': args.city, 'target_day': target_day, 'model_type': 'ARIMA', **metrics }]).to_csv(
        summary_csv, index=False, encoding='ascii', errors='replace'
    )

    # Overview plots
    print("Creating visualizations...")
    create_visualizations(results_df, outdir, args.city)

    # Top-10 best/worst CSVs + LLM-style PNGs
    valid = results_df.dropna(subset=['absolute_error']).copy()
    if len(valid) > 0:
        ranked = valid.sort_values('absolute_error', na_position='last')
        best10  = ranked.head(10)
        worst10 = ranked.tail(10)

        best_csv  = outdir / f"{args.city.lower()}_top10_best.csv"
        worst_csv = outdir / f"{args.city.lower()}_top10_worst.csv"
        best10.to_csv(best_csv, index=False, encoding="utf-8")
        worst10.to_csv(worst_csv, index=False, encoding="utf-8")
        print(f"Saved Top 10 best:  {best_csv}")
        print(f"Saved Top 10 worst: {worst_csv}")

        # Console preview
        print("\n=== Top 10 Best (Lowest Error) ===")
        for i, (_, r) in enumerate(best10.iterrows(), 1):
            print(f"{i:2d}. {r['placekey']} | {str(r['location_name'])[:35]} | Err={r['absolute_error']:.2f} | Pred={r['y_pred_d14']:.1f} | True={r['y_true_d14']:.1f}")

        print("\n=== Top 10 Worst (Highest Error) ===")
        for i, (_, r) in enumerate(worst10.iterrows(), 1):
            print(f"{i:2d}. {r['placekey']} | {str(r['location_name'])[:35]} | Err={r['absolute_error']:.2f} | Pred={r['y_pred_d14']:.1f} | True={r['y_true_d14']:.1f}")

        # LLM-style Top-K plots to subfolders
        if args.plot_topk > 0:
            k = min(args.plot_topk, len(best10))
            print(f"Plotting Top-{k} best (LLM style)...")
            for i, (_, row) in enumerate(best10.head(k).iterrows(), 1):
                p = _plot_llm_style(row, plots_best / f"{row['placekey'].replace('/','_')}.png")
                if i % 20 == 0 or i == k: print(f"  {i}/{k}: {p.name}")

        if args.plot_worstk > 0:
            k = min(args.plot_worstk, len(worst10))
            print(f"Plotting Worst-{k} (LLM style)...")
            for i, (_, row) in enumerate(worst10.tail(k).iterrows(), 1):
                p = _plot_llm_style(row, plots_worst / f"{row['placekey'].replace('/','_')}.png")
                if i % 20 == 0 or i == k: print(f"  {i}/{k}: {p.name}")
    else:
        print("No valid results to rank.")

    # Random per-placekey plots (optional, not LLM-style)
    if args.plot_each > 0 and len(results_df) > 0:
        rng = np.random.default_rng(123)
        idx = rng.choice(results_df.index, size=min(args.plot_each, len(results_df)), replace=False)
        print(f"Making {len(idx)} per-placekey plots (random subset, LLM style)...")
        for i, ridx in enumerate(idx, 1):
            p = _plot_llm_style(results_df.loc[ridx], outdir / "plots_random" / f"{results_df.loc[ridx,'placekey'].replace('/','_')}.png")
            if i % 20 == 0 or i == len(idx):
                print(f"  Plotted {i}/{len(idx)} (last: {p.name})")

    print(f"\n=== Output Files ===")
    print(f"Detailed results: {results_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Overview visualization: {outdir / f'{args.city.lower()}_arima_performance_overview.png'}")

if __name__ == "__main__":
    main()

