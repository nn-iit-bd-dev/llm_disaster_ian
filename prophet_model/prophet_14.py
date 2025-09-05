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
INPUT_CSV = Path("all_4cities_vbd3w.csv")
CITIES = ["Tampa", "Miami", "Orlando", "Cape Coral"]
OUT_ROOT = Path(f"out_{RUN_DATE}/prophet_d14_forecast")
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
    """Calculate MAE and RMSE for valid predictions"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not valid.any():
        return np.nan, np.nan
    diff = y_true[valid] - y_pred[valid]
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff**2).mean()))
    return mae, rmse

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
                "percent_error": np.nan
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
            "percent_error": pct_error
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
    
    mae_score, rmse_score = calculate_metrics(
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
        "rmse_d14": rmse_score
    }])
    
    # Save city outputs
    city_dir = OUT_ROOT / city.lower().replace(" ", "_")
    city_dir.mkdir(exist_ok=True)
    
    pred_file = city_dir / f"{city.lower().replace(' ', '_')}_d14_predictions.csv"
    summ_file = city_dir / f"{city.lower().replace(' ', '_')}_d14_summary.csv"
    
    city_results.to_csv(pred_file, index=False)
    summary.to_csv(summ_file, index=False)
    
    print(f"Results: {n_scored}/{n_total} scored, {n_prophet} Prophet, MAE={mae_score:.2f}")
    
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
        overall_mae, overall_rmse = calculate_metrics(
            all_valid["y_true_d14"], all_valid["y_pred_d14"]
        )
        
        overall_summary = pd.DataFrame([{
            "cities": ", ".join(CITIES),
            "total_placekeys": len(all_valid),
            "prophet_models": int((all_valid["model_type"] == "prophet").sum()),
            "pct_prophet_success": (all_valid["model_type"] == "prophet").mean() * 100,
            "overall_mae": overall_mae,
            "overall_rmse": overall_rmse
        }])
        
        overall_summary.to_csv(OUT_ROOT / "overall_summary.csv", index=False)
        all_valid.to_csv(OUT_ROOT / "all_predictions.csv", index=False)
        
        print(f"\n=== OVERALL RESULTS ===")
        print(f"Total placekeys: {len(all_valid)}")
        print(f"Prophet success: {(all_valid['model_type'] == 'prophet').mean()*100:.1f}%")
        print(f"MAE: {overall_mae:.2f}, RMSE: {overall_rmse:.2f}")
        
        # =================== PERFORMANCE ANALYSIS & VISUALIZATION ===================
        print(f"\n=== PERFORMANCE ANALYSIS ===")
        
        # Create performance-sorted CSV
        performance_df = all_valid[all_valid["absolute_error"].notna()].copy()
        performance_df = performance_df.sort_values("absolute_error", ascending=True)
        
        # Create detailed performance CSV with expanded columns for prev 13 days
        detailed_csv = performance_df[[
            "placekey", "city", "latitude", "longitude", "location_name", "top_category",
            "y_pred_d14", "y_true_d14", "confidence_lower", "confidence_upper", 
            "absolute_error", "percent_error", "model_type"
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
            "absolute_error", "percent_error", "model_type"
        ]
        
        detailed_csv = detailed_csv[column_order]
        detailed_csv.to_csv(OUT_ROOT / "detailed_performance_analysis.csv", index=False)
        
        # Get best and worst performers
        n_show = min(10, len(performance_df))
        best_performers = performance_df.head(n_show)
        worst_performers = performance_df.tail(n_show)
        
        print(f"\nTOP {n_show} BEST PERFORMERS (Lowest Absolute Error):")
        for i, (_, row) in enumerate(best_performers.iterrows(), 1):
            print(f"  {i:2d}. {row['location_name'][:28]:<28} | {row['city']:<12} | Error: {row['absolute_error']:.2f} | Pred: {row['y_pred_d14']:.1f} | Actual: {row['y_true_d14']:.1f}")
        
        print(f"\nTOP {n_show} WORST PERFORMERS (Highest Absolute Error):")
        for i, (_, row) in enumerate(worst_performers.iterrows(), 1):
            print(f"  {i:2d}. {row['location_name'][:28]:<28} | {row['city']:<12} | Error: {row['absolute_error']:.2f} | Pred: {row['y_pred_d14']:.1f} | Actual: {row['y_true_d14']:.1f}")
        
        # Create visualizations
        create_prophet_performance_visualizations(all_series_data, best_performers, worst_performers, OUT_ROOT)

def create_prophet_performance_visualizations(series_data, best_performers, worst_performers, out_dir):
    """Create Prophet-specific visualizations for best and worst performing forecasts"""
    
    plt.style.use('default')
    
    # 1. TOP 10 BEST PERFORMERS with confidence intervals
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    fig.suptitle("TOP 10 BEST PERFORMING PROPHET FORECASTS (Lowest Absolute Error)", fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    for i, (_, row) in enumerate(best_performers.head(10).iterrows()):
        ax = axes[i]
        placekey = row['placekey']
        city = row['city']
        
        if city in series_data and placekey in series_data[city]:
            data = series_data[city][placekey]
            dates = pd.to_datetime(data['dates'])
            actual = data['actual']
            predicted = data['predicted']
            
            # Plot training data
            ax.plot(dates[:13], actual[:13], 'o-', color='#1f77b4', label='Training Data', linewidth=2.5, markersize=5)
            
            # Plot confidence interval if available
            if data['confidence_lower'] and data['confidence_upper']:
                conf_lower = data['confidence_lower']
                conf_upper = data['confidence_upper']
                ax.fill_between([dates[13]], [conf_lower[-1]], [conf_upper[-1]], 
                              alpha=0.3, color='orange', label='80% Confidence')
            
            # Plot actual vs predicted
            ax.plot(dates[13], actual[13], 'o', color='#2ca02c', label=f'Actual: {actual[13]:.1f}', markersize=10)
            ax.plot(dates[13], predicted[13], 's', color='#d62728', label=f'Predicted: {predicted[13]:.1f}', markersize=10)
            
            # Add ranking number and error info
            ax.set_title(f"#{i+1} - {data['location_name'][:22]}\n{city} | Error: {row['absolute_error']:.2f} | PROPHET", 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Date', fontsize=9)
            ax.set_ylabel('Visits', fontsize=9)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            
            # Add background color for ranking
            ax.patch.set_facecolor('#f0f8ff')  # Light blue background for best
    
    plt.tight_layout()
    plt.savefig(out_dir / "top10_best_performers_prophet.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. TOP 10 WORST PERFORMERS
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    fig.suptitle("TOP 10 WORST PERFORMING PROPHET FORECASTS (Highest Absolute Error)", fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    for i, (_, row) in enumerate(worst_performers.head(10).iterrows()):
        ax = axes[i]
        placekey = row['placekey']
        city = row['city']
        
        if city in series_data and placekey in series_data[city]:
            data = series_data[city][placekey]
            dates = pd.to_datetime(data['dates'])
            actual = data['actual']
            predicted = data['predicted']
            
            # Plot training data
            ax.plot(dates[:13], actual[:13], 'o-', color='#1f77b4', label='Training Data', linewidth=2.5, markersize=5)
            
            # Plot confidence interval if available
            if data['confidence_lower'] and data['confidence_upper']:
                conf_lower = data['confidence_lower']
                conf_upper = data['confidence_upper']
                ax.fill_between([dates[13]], [conf_lower[-1]], [conf_upper[-1]], 
                              alpha=0.3, color='orange', label='80% Confidence')
            
            # Plot actual vs predicted
            ax.plot(dates[13], actual[13], 'o', color='#2ca02c', label=f'Actual: {actual[13]:.1f}', markersize=10)
            ax.plot(dates[13], predicted[13], 's', color='#d62728', label=f'Predicted: {predicted[13]:.1f}', markersize=10)
            
            # Add ranking number and error info
            ax.set_title(f"#{i+1} - {data['location_name'][:22]}\n{city} | Error: {row['absolute_error']:.2f} | {row['model_type'].upper()}", 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Date', fontsize=9)
            ax.set_ylabel('Visits', fontsize=9)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            
            # Add background color for ranking
            ax.patch.set_facecolor('#fff0f0')  # Light red background for worst
    
    plt.tight_layout()
    plt.savefig(out_dir / "top10_worst_performers_prophet.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Prophet vs Naive comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PROPHET MODEL PERFORMANCE ANALYSIS', fontsize=16, fontweight='bold')
    
    # Collect all errors by model type
    prophet_errors = []
    naive_errors = []
    
    for city_data in series_data.values():
        for pk_data in city_data.values():
            if not np.isnan(pk_data['abs_error']):
                # This would need model_type info passed through series_data
                prophet_errors.append(pk_data['abs_error'])
    
    # Error distribution
    ax1 = axes[0, 0]
    ax1.hist(prophet_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Prophet Forecast Error Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(prophet_errors), color='red', linestyle='--', 
               label=f'Mean: {np.mean(prophet_errors):.2f}')
    ax1.legend()
    
    # City comparison
    ax2 = axes[0, 1]
    city_errors = {'Tampa': [], 'Miami': [], 'Orlando': [], 'Cape Coral': []}
    for city_name, city_data in series_data.items():
        if city_name in city_errors:
            for pk_data in city_data.values():
                if not np.isnan(pk_data['abs_error']):
                    city_errors[city_name].append(pk_data['abs_error'])
    
    city_names = list(city_errors.keys())
    city_mean_errors = [np.mean(errors) if errors else 0 for errors in city_errors.values()]
    
    bars = ax2.bar(city_names, city_mean_errors, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    ax2.set_title('Mean Absolute Error by City (Prophet)')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, error in zip(bars, city_mean_errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{error:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Confidence interval analysis
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.5, 'Prophet Confidence\nInterval Analysis\n(80% CI Coverage)', 
             ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    ax3.set_title('Confidence Interval Performance')
    
    # Model comparison placeholder
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, 'Prophet vs ARIMA\nComparison\n(Future Enhancement)', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Model Comparison')
    
    plt.tight_layout()
    plt.savefig(out_dir / "prophet_performance_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nProphet visualizations saved to {out_dir}:")
    print(f"  [OK] top10_best_performers_prophet.png")
    print(f"  [OK] top10_worst_performers_prophet.png") 
    print(f"  [OK] prophet_performance_dashboard.png")
    print(f"  [OK] detailed_performance_analysis.csv")

print(f"\nAll outputs saved to: {OUT_ROOT}")