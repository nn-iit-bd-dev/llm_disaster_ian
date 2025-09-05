# Prophet D14 forecast per placekey with fallback, aggregate by city
import os, json, ast, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======== REQUIRE Prophet =========
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
except Exception as e:
    raise SystemExit("Prophet is required. Install it: pip install prophet. Details: " + str(e))

# =================== CONFIG ===================
RUN_DATE = os.getenv("RUN_DATE", "2025-08-28")
INPUT_CSV = Path("all_4cities_vbd3w_test_set.csv")
CITIES = ["Tampa", "Miami", "Orlando", "Cape Coral"]
OUT_ROOT = Path(f"out_{RUN_DATE}/prophet_d14_forecast_all_metrics_test_set_only")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# =================== HELPERS ===================
def parse_vbd(v):
    """Parse visits_by_day from various formats"""
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                try:
                    return [float(x) for x in s.split(",")]
                except Exception:
                    return None
    return None

def prophet_forecast_d14(y_train, dates_train):
    """
    Fit Prophet on 13 days; forecast day 14.
    Returns: (prediction, model_type, confidence_interval, fallback_reason)
    """
    y = np.asarray(y_train, dtype=float)
    
    if len(y) != 13:
        return float("nan"), "error", None, "wrong_length"
    
    # Check for constant series
    if np.allclose(y, y[0]):
        return float(y[-1]), "naive", None, "constant_series"
    
    # Check for all zeros (Prophet doesn't handle well)
    if np.all(y == 0):
        return 0.0, "naive", None, "all_zeros"
    
    try:
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(dates_train),
            'y': y
        })
        
        # Initialize Prophet with minimal parameters for short series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            model = Prophet(
                yearly_seasonality=False,    # Only 13 days, no yearly pattern
                weekly_seasonality=False,    # Too short for weekly pattern
                daily_seasonality=False,     # No hourly data
                changepoint_prior_scale=0.1, # Less flexible for short series
                seasonality_prior_scale=0.1,
                interval_width=0.8,          # 80% confidence interval
                uncertainty_samples=100      # Faster computation
            )
            
            # Fit the model
            model.fit(prophet_df)
            
            # Create future dataframe for 1 day ahead
            future = model.make_future_dataframe(periods=1, freq='D')
            
            # Make prediction
            forecast = model.predict(future)
            
            # Extract day 14 prediction
            prediction = forecast.iloc[-1]['yhat']
            confidence = (forecast.iloc[-1]['yhat_lower'], forecast.iloc[-1]['yhat_upper'])
            
            # Ensure non-negative prediction (visits can't be negative)
            prediction = max(0, prediction)
            
            return float(prediction), "prophet", confidence, ""
            
    except Exception as e:
        # Fallback to last value if Prophet fails
        return float(y[-1]), "naive", None, f"prophet_failed:{str(e)[:50]}"

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, sMAPE (mean & median), and RMSLE for valid predictions"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not valid.any():
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]
    
    # Basic metrics
    diff = y_true_valid - y_pred_valid
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff**2).mean()))
    
    # sMAPE calculation - individual values for mean and median
    denominator = (np.abs(y_true_valid) + np.abs(y_pred_valid)) / 2
    mask = denominator != 0
    smape_values = np.full_like(y_true_valid, np.nan, dtype=float)
    smape_values[mask] = (np.abs(y_true_valid[mask] - y_pred_valid[mask]) / denominator[mask]) * 100
    
    smape_mean = float(np.nanmean(smape_values))
    smape_median = float(np.nanmedian(smape_values))
    
    # RMSLE calculation
    epsilon = 1e-8
    y_true_log = np.log(np.maximum(y_true_valid, 0) + epsilon)
    y_pred_log = np.log(np.maximum(y_pred_valid, 0) + epsilon)
    rmsle = float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))
    
    return mae, rmse, smape_mean, smape_median, rmsle

def calculate_individual_metrics(y_true, y_pred):
    """Calculate individual sMAPE and RMSLE for each prediction"""
    if np.isnan(y_true) or np.isnan(y_pred):
        return np.nan, np.nan
    
    # Individual sMAPE
    denominator = (abs(y_true) + abs(y_pred)) / 2
    if denominator != 0:
        smape_ind = (abs(y_true - y_pred) / denominator) * 100
    else:
        smape_ind = np.nan
    
    # Individual RMSLE
    epsilon = 1e-8
    y_true_log = np.log(max(y_true, 0) + epsilon)
    y_pred_log = np.log(max(y_pred, 0) + epsilon)
    rmsle_ind = (y_true_log - y_pred_log) ** 2
    
    return float(smape_ind), float(rmsle_ind)

def create_prophet_performance_visualizations(series_data, best_performers, worst_performers, out_dir, all_valid):
    """Create Prophet-specific visualizations for best and worst performing forecasts"""
    
    plt.style.use('default')
    
    # 1. Enhanced performance dashboard with sMAPE and RMSLE
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ENHANCED PROPHET PERFORMANCE ANALYSIS', fontsize=16, fontweight='bold')
    
    # Error distribution
    ax1 = axes[0, 0]
    ax1.hist(all_valid['absolute_error'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Absolute Error Distribution')
    ax1.axvline(all_valid['absolute_error'].mean(), color='red', linestyle='--', 
               label=f'Mean: {all_valid["absolute_error"].mean():.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # sMAPE distribution
    ax2 = axes[0, 1]
    ax2.hist(all_valid['smape_individual'].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('sMAPE (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('sMAPE Distribution')
    ax2.axvline(all_valid['smape_individual'].mean(), color='red', linestyle='--', 
               label=f'Mean: {all_valid["smape_individual"].mean():.1f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # RMSLE distribution
    ax3 = axes[0, 2]
    ax3.hist(all_valid['rmsle_individual'].dropna(), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.set_xlabel('RMSLE')
    ax3.set_ylabel('Frequency')
    ax3.set_title('RMSLE Distribution')
    ax3.axvline(all_valid['rmsle_individual'].mean(), color='red', linestyle='--', 
               label=f'Mean: {all_valid["rmsle_individual"].mean():.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # City comparison - MAE
    ax4 = axes[1, 0]
    city_mae = all_valid.groupby('city')['absolute_error'].mean()
    bars = ax4.bar(city_mae.index, city_mae.values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    ax4.set_title('Mean Absolute Error by City')
    ax4.set_ylabel('MAE')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, city_mae.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # City comparison - sMAPE
    ax5 = axes[1, 1]
    city_smape = all_valid.groupby('city')['smape_individual'].mean()
    bars = ax5.bar(city_smape.index, city_smape.values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    ax5.set_title('Mean sMAPE by City')
    ax5.set_ylabel('sMAPE (%)')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, city_smape.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Metrics summary
    metrics_text = f"""Overall Metrics:
MAE: {all_valid['absolute_error'].mean():.3f}
RMSE: {np.sqrt((all_valid['absolute_error'] ** 2).mean()):.3f}
sMAPE (mean): {all_valid['smape_individual'].mean():.1f}%
sMAPE (median): {all_valid['smape_individual'].median():.1f}%
RMSLE: {all_valid['rmsle_individual'].mean():.3f}

Prophet Success: {(all_valid['model_type'] == 'prophet').mean()*100:.1f}%
Total Locations: {len(all_valid)}"""
    
    ax6 = axes[1, 2]
    ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, 
             fontsize=12, verticalalignment='center', fontfamily='monospace')
    ax6.set_title('Performance Summary')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_dir / "enhanced_performance_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Prophet visualizations saved to {out_dir}:")
    print(f"  [OK] enhanced_performance_dashboard.png")

# =================== LOAD & VALIDATE DATA ===================
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

print(f"Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# Use visits_by_day field
visits_col = "visits_by_day"
if visits_col not in df.columns:
    raise ValueError(f"'{visits_col}' column not found in CSV")

print(f"Using column: {visits_col}")

# Parse and filter visits data
df["vbd_list"] = df[visits_col].apply(parse_vbd)
valid_visits = df["vbd_list"].apply(lambda x: isinstance(x, list) and len(x) == 7)
df = df[valid_visits].copy()

# Parse dates and time periods
df["start_date"] = pd.to_datetime(df["date_range_start"])

# Show data structure
if "time_period" in df.columns:
    time_periods = df["time_period"].value_counts()
    print(f"Time periods found: {dict(time_periods)}")
    
    # Define time period order for chronological sorting
    period_order = {"before": 1, "landfall": 2, "after": 3}
    df["period_order"] = df["time_period"].map(period_order).fillna(999)
    
    # Sort by time period and date for chronological order
    df = df.sort_values(["period_order", "start_date"]).reset_index(drop=True)

print(f"Loaded {len(df)} valid weekly records")

# =================== PROCESS EACH CITY ===================
all_summaries, all_predictions = [], []
all_series_data = {}  # Store time series data for visualization

for city in CITIES:
    print(f"\n=== Processing {city} ===")
    
    # Filter by city - handle potential case variations
    city_data = df[df["city_norm"].str.upper() == city.upper()].copy()
    if city_data.empty:
        # Try alternate city column if city_norm doesn't work
        if "city" in df.columns:
            city_data = df[df["city"].str.upper() == city.upper()].copy()
        
        if city_data.empty:
            print(f"No data for {city}")
            continue
    
    # Build chronological daily time series per placekey
    daily_rows = []
    for placekey, placekey_group in city_data.groupby("placekey"):
        # Sort this placekey's data chronologically
        placekey_group = placekey_group.sort_values(["period_order", "start_date"])
        
        # Get the absolute first date for this placekey (for D14 reference)
        first_start_date = placekey_group["start_date"].iloc[0]
        
        # Build continuous daily series from all time periods
        current_date = first_start_date
        for _, row in placekey_group.iterrows():
            for day_idx, visits in enumerate(row["vbd_list"]):
                daily_rows.append({
                    "placekey": placekey,
                    "location_name": row.get("location_name", ""),
                    "top_category": row.get("top_category", ""),
                    "time_period": row.get("time_period", ""),
                    "date": current_date,
                    "visits": float(visits),
                    "first_date": first_start_date  # Track original start for D14 calculation
                })
                current_date += pd.Timedelta(days=1)
    
    daily_df = pd.DataFrame(daily_rows)
    # Average any potential duplicates (same placekey + date)
    daily_df = daily_df.groupby(["placekey", "location_name", "top_category", "date", "first_date"], 
                               as_index=False).agg({
                                   "visits": "mean",
                                   "time_period": "first"  # Keep time period info
                               })
    
    # Forecast D14 for each placekey
    results = []
    placekey_series = {}  # Store the 13-day series for each placekey
    
    for placekey, group in daily_df.groupby("placekey"):
        group = group.sort_values("date")
        dates, visits = group["date"].values, group["visits"].values
        first_date = group["first_date"].iloc[0]  # Original start date
        
        # Store latitude and longitude for this placekey
        placekey_lat = city_data[city_data["placekey"] == placekey]["latitude"].iloc[0] if len(city_data[city_data["placekey"] == placekey]) > 0 else np.nan
        placekey_lon = city_data[city_data["placekey"] == placekey]["longitude"].iloc[0] if len(city_data[city_data["placekey"] == placekey]) > 0 else np.nan
        
        if len(visits) < 14:
            results.append({
                "placekey": placekey, "city": city,
                "location_name": group["location_name"].iloc[0],
                "top_category": group["top_category"].iloc[0],
                "latitude": placekey_lat, "longitude": placekey_lon,
                "first_date": pd.to_datetime(first_date),
                "target_date_d14": pd.to_datetime(first_date) + pd.Timedelta(days=13),
                "actual_target_date": pd.to_datetime(dates[13]) if len(dates) > 13 else pd.NaT,
                "y_true_d14": float(visits[13]) if len(visits) > 13 else np.nan,
                "y_pred_d14": np.nan,
                "model_type": "skipped", "confidence_lower": np.nan, "confidence_upper": np.nan,
                "fallback_reason": "insufficient_data",
                "n_days": len(visits),
                "time_periods_used": ";".join(group["time_period"].unique()),
                "prev_13_values": visits[:min(13, len(visits))].tolist(),
                "absolute_error": np.nan,
                "percent_error": np.nan,
                "smape_individual": np.nan,
                "rmsle_individual": np.nan
            })
            continue
        
        # Train on first 13 days, predict day 14
        y_train = visits[:13]
        dates_train = dates[:13]
        y_true = float(visits[13])
        
        y_pred, model_type, confidence, reason = prophet_forecast_d14(y_train, dates_train)
        
        # Extract confidence intervals
        conf_lower = confidence[0] if confidence else np.nan
        conf_upper = confidence[1] if confidence else np.nan
        
        # Calculate performance metrics
        abs_error = abs(y_true - y_pred) if not np.isnan(y_pred) else np.nan
        pct_error = (abs_error / max(y_true, 1)) * 100 if not np.isnan(abs_error) and y_true != 0 else np.nan
        
        # Calculate individual sMAPE and RMSLE
        smape_ind, rmsle_ind = calculate_individual_metrics(y_true, y_pred)
        
        results.append({
            "placekey": placekey, "city": city,
            "location_name": group["location_name"].iloc[0],
            "top_category": group["top_category"].iloc[0],
            "latitude": placekey_lat, "longitude": placekey_lon,
            "first_date": pd.to_datetime(first_date),
            "target_date_d14": pd.to_datetime(first_date) + pd.Timedelta(days=13),
            "actual_target_date": pd.to_datetime(dates[13]),
            "y_true_d14": y_true, "y_pred_d14": y_pred,
            "model_type": model_type, "confidence_lower": conf_lower, "confidence_upper": conf_upper,
            "fallback_reason": reason, "n_days": 13,
            "time_periods_used": ";".join(group["time_period"].unique()),
            "prev_13_values": y_train.tolist(),
            "absolute_error": abs_error,
            "percent_error": pct_error,
            "smape_individual": smape_ind,
            "rmsle_individual": rmsle_ind
        })
        
        # Store series data for visualization
        placekey_series[placekey] = {
            "dates": [first_date + pd.Timedelta(days=i) for i in range(14)],
            "actual": list(visits[:14]),
            "predicted": list(y_train) + [y_pred],
            "confidence_lower": list(y_train) + [conf_lower] if not np.isnan(conf_lower) else None,
            "confidence_upper": list(y_train) + [conf_upper] if not np.isnan(conf_upper) else None,
            "location_name": group["location_name"].iloc[0],
            "city": city,
            "abs_error": abs_error
        }
    
    # Create results dataframe
    city_results = pd.DataFrame(results)
    
    # Calculate metrics for valid predictions
    valid_preds = city_results.dropna(subset=["y_true_d14", "y_pred_d14"])
    n_total = len(city_results)
    n_scored = len(valid_preds)
    n_prophet = (valid_preds["model_type"] == "prophet").sum()
    
    mae_score, rmse_score, smape_mean_score, smape_median_score, rmsle_score = calculate_metrics(
        valid_preds["y_true_d14"], valid_preds["y_pred_d14"]
    )
    
    # City summary
    summary = pd.DataFrame([{
        "city": city,
        "n_placekeys_total": n_total,
        "n_placekeys_scored": n_scored,
        "n_prophet_models": int(n_prophet),
        "n_naive_models": int((valid_preds["model_type"] == "naive").sum()),
        "pct_prophet_success": (n_prophet / n_scored * 100) if n_scored > 0 else 0,
        "mae_d14": mae_score,
        "rmse_d14": rmse_score,
        "smape_mean_d14": smape_mean_score,
        "smape_median_d14": smape_median_score,
        "rmsle_d14": rmsle_score
    }])
    
    # Save city outputs
    city_dir = OUT_ROOT / city.lower().replace(" ", "_")
    city_dir.mkdir(exist_ok=True)
    
    pred_file = city_dir / f"{city.lower().replace(' ', '_')}_d14_predictions.csv"
    summ_file = city_dir / f"{city.lower().replace(' ', '_')}_d14_summary.csv"
    
    city_results.to_csv(pred_file, index=False)
    summary.to_csv(summ_file, index=False)
    
    print(f"Results: {n_scored}/{n_total} scored, {n_prophet} Prophet")
    print(f"MAE={mae_score:.2f}, sMAPE(mean)={smape_mean_score:.1f}%, sMAPE(median)={smape_median_score:.1f}%")
    
    # Store results for later visualization
    all_predictions.append(valid_preds)
    all_summaries.append(summary)
    
    # Store series data for this city
    if city not in all_series_data:
        all_series_data[city] = {}
    all_series_data[city].update(placekey_series)

# =================== OVERALL SUMMARY ===================
if all_summaries:
    # Combined city summaries
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(OUT_ROOT / "cities_summary.csv", index=False)
    
    # Overall metrics across all cities
    if all_predictions:
        all_valid = pd.concat(all_predictions, ignore_index=True)
        overall_mae, overall_rmse, overall_smape_mean, overall_smape_median, overall_rmsle = calculate_metrics(
            all_valid["y_true_d14"], all_valid["y_pred_d14"]
        )
        
        overall_summary = pd.DataFrame([{
            "cities": ", ".join(CITIES),
            "total_placekeys": len(all_valid),
            "prophet_models": int((all_valid["model_type"] == "prophet").sum()),
            "pct_prophet_success": (all_valid["model_type"] == "prophet").mean() * 100,
            "overall_mae": overall_mae,
            "overall_rmse": overall_rmse,
            "overall_smape_mean": overall_smape_mean,
            "overall_smape_median": overall_smape_median,
            "overall_rmsle": overall_rmsle
        }])
        
        overall_summary.to_csv(OUT_ROOT / "overall_summary.csv", index=False)
        all_valid.to_csv(OUT_ROOT / "all_predictions.csv", index=False)
        
        print(f"\n=== OVERALL RESULTS ===")
        print(f"Total placekeys: {len(all_valid)}")
        print(f"Prophet success: {(all_valid['model_type'] == 'prophet').mean()*100:.1f}%")
        print(f"MAE: {overall_mae:.2f}, RMSE: {overall_rmse:.2f}")
        print(f"sMAPE (mean): {overall_smape_mean:.1f}%, sMAPE (median): {overall_smape_median:.1f}%")
        print(f"RMSLE: {overall_rmsle:.3f}")
        
        # =================== PERFORMANCE ANALYSIS & VISUALIZATION ===================
        print(f"\n=== PERFORMANCE ANALYSIS ===")
        
        # Create performance-sorted CSV
        performance_df = all_valid[all_valid["absolute_error"].notna()].copy()
        performance_df = performance_df.sort_values("absolute_error", ascending=True)
        
        # Create detailed performance CSV with expanded columns for prev 13 days
        detailed_csv = performance_df[[
            "placekey", "city", "latitude", "longitude", "location_name", "top_category",
            "y_pred_d14", "y_true_d14", "confidence_lower", "confidence_upper", 
            "absolute_error", "percent_error", "smape_individual", "rmsle_individual", "model_type"
        ]].copy()
        
        # Expand prev_13_values into separate columns
        prev_13_data = pd.DataFrame(performance_df["prev_13_values"].tolist(), 
                                  columns=[f"day_{i+1}_visits" for i in range(13)])
        
        # Combine with main data
        detailed_csv = pd.concat([detailed_csv, prev_13_data], axis=1)
        
        # Reorder columns for better readability
        column_order = [
            "placekey", "city", "latitude", "longitude", "location_name", "top_category"
        ] + [f"day_{i+1}_visits" for i in range(13)] + [
            "y_pred_d14", "y_true_d14", "confidence_lower", "confidence_upper",
            "absolute_error", "percent_error", "smape_individual", "rmsle_individual", "model_type"
        ]
        
        detailed_csv = detailed_csv[column_order]
        detailed_csv.to_csv(OUT_ROOT / "detailed_performance_analysis.csv", index=False)
        
        # Get best and worst performers
        n_show = min(10, len(performance_df))
        best_performers = performance_df.head(n_show)
        worst_performers = performance_df.tail(n_show)
        
        print(f"\nTOP {n_show} BEST PERFORMERS (Lowest Absolute Error):")
        for i, (_, row) in enumerate(best_performers.iterrows(), 1):
            print(f"  {i:2d}. {row['location_name'][:28]:<28} | {row['city']:<12} | Error: {row['absolute_error']:.2f} | sMAPE: {row['smape_individual']:.1f}%")
        
        print(f"\nTOP {n_show} WORST PERFORMERS (Highest Absolute Error):")
        for i, (_, row) in enumerate(worst_performers.iterrows(), 1):
            print(f"  {i:2d}. {row['location_name'][:28]:<28} | {row['city']:<12} | Error: {row['absolute_error']:.2f} | sMAPE: {row['smape_individual']:.1f}%")
        
        # Create visualizations
        create_prophet_performance_visualizations(all_series_data, best_performers, worst_performers, OUT_ROOT, all_valid)

print(f"\nAll outputs saved to: {OUT_ROOT}")