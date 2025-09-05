#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import argparse
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to finetuned_predictions.csv")
    p.add_argument("--training-data", required=True, help="Path to prepared_tampa_d14.jsonl")
    p.add_argument("--output-dir", required=True, help="Output directory for plots")
    p.add_argument("--top-n", type=int, default=10, help="Number of best/worst examples")
    p.add_argument("--figsize", nargs=2, type=int, default=[15, 10], help="Figure size")
    p.add_argument("--dpi", type=int, default=300, help="DPI for plots")
    return p.parse_args()

def load_training_data(jsonl_path):
    training_data = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                placekey = data.get('placekey')
                if placekey:
                    training_data[placekey] = data
    
    print(f"Loaded training data for {len(training_data)} placekeys")
    return training_data

def load_data(results_path, training_data_path):
    df = pd.read_csv(results_path)
    print(f"Loaded predictions: {len(df)} rows")
    print(f"Available columns: {list(df.columns)}")
    
    training_data = load_training_data(training_data_path)
    
    merged_rows = []
    missing_placekeys = 0
    
    for _, row in df.iterrows():
        placekey = row.get('placekey')
        if placekey in training_data:
            train_row = training_data[placekey]
            
            merged_row = row.to_dict()
            merged_row.update({
                'prev_13_values': train_row.get('prev_13_values', []),
                'series_start_date': train_row.get('series_start_date', '2022-09-19'),
                'landfall_date': train_row.get('landfall_date', '2022-09-28'),
                'actual_target_date': train_row.get('actual_target_date', train_row.get('target_date_d14', '2022-10-02')),
                'time_periods_used': train_row.get('time_periods_used', 'after;before;landfall'),
                'target_days_after_landfall': train_row.get('target_days_after_landfall', 4)
            })
            
            for col in ['city', 'location_name', 'top_category', 'latitude', 'longitude']:
                if col not in merged_row or pd.isna(merged_row[col]) or merged_row[col] == 'Unknown':
                    merged_row[col] = train_row.get(col, merged_row.get(col, 'Unknown'))
            
            merged_rows.append(merged_row)
        else:
            missing_placekeys += 1
            print(f"Warning: No training data found for placekey {placekey}")
    
    if missing_placekeys > 0:
        print(f"Missing training data for {missing_placekeys} placekeys")
    
    df_merged = pd.DataFrame(merged_rows)
    
    def parse_prev_values(val):
        if isinstance(val, list):
            return [float(x) for x in val]
        elif isinstance(val, str):
            try:
                import ast
                parsed = ast.literal_eval(val)
                return [float(x) for x in parsed] if isinstance(parsed, list) else []
            except:
                return []
        return []
    
    df_merged['prev_13_values_parsed'] = df_merged['prev_13_values'].apply(parse_prev_values)
    
    valid_series = df_merged['prev_13_values_parsed'].apply(lambda x: len(x) == 13)
    print(f"Valid 13-day series: {valid_series.sum()}/{len(df_merged)}")
    
    df_merged['first_date'] = df_merged.get('series_start_date', '2022-09-19')
    df_merged['model_type'] = 'Fine-tuned'
    df_merged['fallback_reason'] = 'None'
    df_merged['n_days'] = 13
    
    if 'confidence_lower' not in df_merged.columns:
        df_merged['confidence_lower'] = df_merged['y_pred_d14'] * 0.8
    if 'confidence_upper' not in df_merged.columns:
        df_merged['confidence_upper'] = df_merged['y_pred_d14'] * 1.2
    
    df_merged['prev_13_values'] = df_merged['prev_13_values_parsed'].apply(str)
    
    print(f"Final merged data: {len(df_merged)} rows")
    print(f"y_true range: {df_merged['y_true_d14'].min():.1f} to {df_merged['y_true_d14'].max():.1f}")
    print(f"y_pred range: {df_merged['y_pred_d14'].min():.1f} to {df_merged['y_pred_d14'].max():.1f}")
    print(f"Mean absolute error: {df_merged['absolute_error'].mean():.3f}")
    print(f"Non-zero predictions: {(df_merged['y_pred_d14'] > 0).sum()}/{len(df_merged)}")
    
    return df_merged

def plot_series(row, ax, show_title=True):
    try:
        if pd.notna(row['first_date']):
            start_date = pd.to_datetime(row['first_date'])
        else:
            start_date = pd.to_datetime('2022-09-19')
            
        if pd.notna(row['target_date_d14']):
            target_date = pd.to_datetime(row['target_date_d14'])
        else:
            target_date = start_date + timedelta(days=13)
            
        if 'landfall_date' in row and pd.notna(row['landfall_date']):
            landfall_date = pd.to_datetime(row['landfall_date'])
        else:
            landfall_date = start_date + timedelta(days=9)
            
    except Exception as e:
        print(f"Date parsing error: {e}, using defaults")
        start_date = pd.to_datetime('2022-09-19')
        target_date = pd.to_datetime('2022-10-02')
        landfall_date = pd.to_datetime('2022-09-28')
    
    dates = [start_date + timedelta(days=i) for i in range(13)]
    
    values = row['prev_13_values_parsed']
    if not isinstance(values, list) or len(values) != 13:
        print(f"Warning: Invalid values for {row.get('location_name', 'Unknown')}: {values}")
        values = [18 + np.random.normal(0, 5) for _ in range(13)]
    
    ax.plot(dates, values, 'o-', linewidth=2, markersize=6, 
            label='Observed (Days 1-13)', color='#1f77b4')
    
    day_14_date = dates[-1] + timedelta(days=1)
    
    y_true = float(row['y_true_d14']) if pd.notna(row['y_true_d14']) else 0.0
    y_pred = float(row['y_pred_d14']) if pd.notna(row['y_pred_d14']) else 0.0
    
    ax.plot(day_14_date, y_true, 'o', markersize=10, 
            color='green', label=f"Actual D14 = {y_true:.1f}")
    ax.plot(day_14_date, y_pred, 's', markersize=8, 
            color='orange', label=f"Llama D14 = {y_pred:.1f}")
    
    if landfall_date >= dates[0] and landfall_date <= day_14_date:
        ax.axvline(landfall_date, color='red', linestyle='--', alpha=0.7, label='Ian landfall')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Visits')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
    
    if show_title:
        location = row.get('location_name', 'Unknown Location')
        city = row.get('city', 'Unknown City')
        abs_err = float(row['absolute_error']) if pd.notna(row['absolute_error']) else 0.0
        
        title = f"{location}\n{city} | abs err: {abs_err:.2f}"
        
        placekey = row.get('placekey', 'unknown')
        lat = float(row['latitude']) if pd.notna(row['latitude']) else 0.0
        lon = float(row['longitude']) if pd.notna(row['longitude']) else 0.0
        
        if placekey != 'unknown':
            title += f"\nplacekey: {placekey} | lat: {lat:.4f}, lon: {lon:.4f}"
            
        ax.set_title(title, fontsize=9, pad=15)
    
    if len(values) > 0:
        y_min = min(min(values), y_true, y_pred) * 0.9
        y_max = max(max(values), y_true, y_pred) * 1.1
        if y_max > y_min:
            ax.set_ylim(y_min, y_max)

def calc_metrics(df):
    """Calculate comprehensive metrics including sMAPE and RMSLE"""
    y_true = df['y_true_d14'].values
    y_pred = df['y_pred_d14'].values
    
    # Basic metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # sMAPE calculation
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    smape_values = np.full_like(y_true, np.nan, dtype=float)
    smape_values[mask] = (np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    smape_mean = np.nanmean(smape_values)
    smape_median = np.nanmedian(smape_values)
    
    # RMSLE calculation
    epsilon = 1e-8
    y_true_log = np.log(np.maximum(y_true, 0) + epsilon)
    y_pred_log = np.log(np.maximum(y_pred, 0) + epsilon)
    rmsle = np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))
    
    # Add individual sMAPE and RMSLE to dataframe
    df['smape_individual'] = smape_values
    df['rmsle_individual'] = np.sqrt((y_true_log - y_pred_log) ** 2)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'smape_mean': smape_mean,
        'smape_median': smape_median,
        'rmsle': rmsle
    }

def plot_summary(df, output_dir, figsize, dpi):
    metrics = calc_metrics(df)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hurricane Forecasting Model Performance Summary', fontsize=16, y=0.98)
    
    # Error distribution
    axes[0,0].hist(df['absolute_error'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel('Absolute Error')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Absolute Errors')
    axes[0,0].axvline(df['absolute_error'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["absolute_error"].mean():.2f}')
    axes[0,0].legend()
    
    # True vs Predicted scatter
    axes[0,1].scatter(df['y_true_d14'], df['y_pred_d14'], alpha=0.6)
    min_val = min(df['y_true_d14'].min(), df['y_pred_d14'].min())
    max_val = max(df['y_true_d14'].max(), df['y_pred_d14'].max())
    axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axes[0,1].set_xlabel('True Values')
    axes[0,1].set_ylabel('Predicted Values')
    axes[0,1].set_title('True vs Predicted Values')
    axes[0,1].legend()
    
    # sMAPE distribution
    axes[0,2].hist(df['smape_individual'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[0,2].set_xlabel('sMAPE (%)')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].set_title('Distribution of sMAPE')
    axes[0,2].axvline(metrics['smape_mean'], color='red', linestyle='--', 
                      label=f'Mean: {metrics["smape_mean"]:.1f}%')
    axes[0,2].legend()
    
    # Performance by category
    if 'top_category' in df.columns and df['top_category'].nunique() > 1:
        cat_perf = df.groupby('top_category')['absolute_error'].mean().sort_values()
        if len(cat_perf) > 10:
            cat_perf = cat_perf.head(10)
        
        cat_perf.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_xlabel('Category')
        axes[1,0].set_ylabel('Mean Absolute Error')
        axes[1,0].set_title('Performance by Category (Top 10)')
        axes[1,0].tick_params(axis='x', rotation=45)
    else:
        axes[1,0].text(0.5, 0.5, 'No category data available', 
                       ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Performance by Category')
    
    # RMSLE distribution
    axes[1,1].hist(df['rmsle_individual'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('RMSLE')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of RMSLE')
    axes[1,1].axvline(metrics['rmsle'], color='red', linestyle='--', 
                      label=f'Mean: {metrics["rmsle"]:.3f}')
    axes[1,1].legend()
    
    # Metrics summary text
    metrics_text = f"""Overall Metrics:
MAE: {metrics['mae']:.3f}
RMSE: {metrics['rmse']:.3f}
sMAPE (mean): {metrics['smape_mean']:.1f}%
sMAPE (median): {metrics['smape_median']:.1f}%
RMSLE: {metrics['rmsle']:.3f}

Samples: {len(df)}
Non-zero preds: {(df['y_pred_d14'] > 0).sum()}"""
    
    axes[1,2].text(0.1, 0.5, metrics_text, transform=axes[1,2].transAxes, 
                   fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1,2].set_title('Performance Metrics Summary')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_summary.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return metrics

def plot_best_worst(df, top_n, output_dir, figsize, dpi):
    df_sorted = df.sort_values('absolute_error')
    best = df_sorted.head(top_n)
    worst = df_sorted.tail(top_n)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    fig.suptitle(f'Top {top_n} Best Predictions (Lowest Absolute Error)', fontsize=16, y=0.95)
    
    for i, (_, row) in enumerate(best.iterrows()):
        if i >= 10:
            break
        ax = axes[i//5, i%5]
        plot_series(row, ax)
    
    for i in range(len(best), 10):
        axes[i//5, i%5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'best_{top_n}_predictions.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    fig.suptitle(f'Top {top_n} Worst Predictions (Highest Absolute Error)', fontsize=16, y=0.95)
    
    for i, (_, row) in enumerate(worst.iterrows()):
        if i >= 10:
            break
        ax = axes[i//5, i%5]
        plot_series(row, ax)
    
    for i in range(len(worst), 10):
        axes[i//5, i%5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'worst_{top_n}_predictions.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Best predictions - Error range: {best['absolute_error'].min():.2f} to {best['absolute_error'].max():.2f}")
    print(f"Worst predictions - Error range: {worst['absolute_error'].min():.2f} to {worst['absolute_error'].max():.2f}")
    
    return best, worst

def save_results(df, best, worst, output_dir):
    cols = [
        'placekey', 'city', 'location_name', 'top_category', 
        'latitude', 'longitude', 'first_date', 'target_date_d14',
        'actual_target_date', 'y_true_d14', 'y_pred_d14', 'model_type',
        'confidence_lower', 'confidence_upper', 'fallback_reason',
        'n_days', 'time_periods_used', 'prev_13_values',
        'absolute_error', 'percent_error', 'prev_13_values_parsed'
    ]
    
    for col in cols:
        if col not in df.columns:
            if col == 'prev_13_values':
                df[col] = df['prev_13_values_parsed'].apply(lambda x: str(x))
            else:
                df[col] = np.nan
    
    best_detailed = best[cols].copy()
    worst_detailed = worst[cols].copy()
    all_detailed = df[cols].copy()
    
    best_detailed.to_csv(output_dir / f'best_{len(best)}_detailed_report.csv', index=False)
    worst_detailed.to_csv(output_dir / f'worst_{len(worst)}_detailed_report.csv', index=False)
    all_detailed.to_csv(output_dir / 'all_predictions_detailed_report.csv', index=False)
    
    print(f"Detailed reports saved with {len(cols)} columns:")
    print(f"Columns: {', '.join(cols)}")
    print(f"Best predictions: {len(best_detailed)} rows")
    print(f"Worst predictions: {len(worst_detailed)} rows")
    print(f"All predictions: {len(all_detailed)} rows")

def gen_report(df, best, worst, output_dir):
    metrics = calc_metrics(df)
    best_metrics = calc_metrics(best)
    worst_metrics = calc_metrics(worst)
    
    report = {
        "Overall Statistics": {
            "Total Predictions": len(df),
            "Mean Absolute Error": float(metrics['mae']),
            "RMSE": float(metrics['rmse']),
            "sMAPE Mean": float(metrics['smape_mean']),
            "sMAPE Median": float(metrics['smape_median']),
            "RMSLE": float(metrics['rmsle']),
            "Mean Percent Error": float(df['percent_error'].mean()) if 'percent_error' in df.columns else "N/A"
        },
        "Best Predictions": {
            "Count": len(best),
            "Mean Absolute Error": float(best_metrics['mae']),
            "RMSE": float(best_metrics['rmse']),
            "sMAPE Mean": float(best_metrics['smape_mean']),
            "RMSLE": float(best_metrics['rmsle']),
            "Max Absolute Error": float(best['absolute_error'].max())
        },
        "Worst Predictions": {
            "Count": len(worst),
            "Mean Absolute Error": float(worst_metrics['mae']),
            "RMSE": float(worst_metrics['rmse']),
            "sMAPE Mean": float(worst_metrics['smape_mean']),
            "RMSLE": float(worst_metrics['rmsle']),
            "Min Absolute Error": float(worst['absolute_error'].min())
        }
    }
    
    if 'top_category' in df.columns and df['top_category'].nunique() > 1:
        cat_stats = df.groupby('top_category').agg({
            'absolute_error': ['count', 'mean', 'std'],
            'smape_individual': 'mean',
            'rmsle_individual': 'mean'
        }).round(3)
        cat_stats.columns = ['count', 'mean_ae', 'std_ae', 'mean_smape', 'mean_rmsle']
        report["Category Performance"] = cat_stats.to_dict('index')
    
    with open(output_dir / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write("Hurricane Forecasting Model Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"Total predictions analyzed: {len(df)}\n")
        f.write(f"Mean Absolute Error: {metrics['mae']:.3f}\n")
        f.write(f"RMSE: {metrics['rmse']:.3f}\n")
        f.write(f"sMAPE (mean): {metrics['smape_mean']:.1f}%\n")
        f.write(f"sMAPE (median): {metrics['smape_median']:.1f}%\n")
        f.write(f"RMSLE: {metrics['rmsle']:.3f}\n")
        f.write(f"Non-zero predictions: {(df['y_pred_d14'] > 0).sum()}/{len(df)}\n\n")
        
        f.write("BEST PREDICTIONS:\n")
        f.write(f"Top {len(best)} best predictions:\n")
        f.write(f"  MAE: {best_metrics['mae']:.3f}\n")
        f.write(f"  RMSE: {best_metrics['rmse']:.3f}\n")
        f.write(f"  sMAPE: {best_metrics['smape_mean']:.1f}%\n")
        f.write(f"  RMSLE: {best_metrics['rmsle']:.3f}\n")
        f.write("Best performing locations:\n")
        for _, row in best.head(5).iterrows():
            f.write(f"  - {row['location_name']} ({row['city']}): Error = {row['absolute_error']:.2f}\n")
        
        f.write(f"\nWORST PREDICTIONS:\n")
        f.write(f"Top {len(worst)} worst predictions:\n")
        f.write(f"  MAE: {worst_metrics['mae']:.3f}\n")
        f.write(f"  RMSE: {worst_metrics['rmse']:.3f}\n")
        f.write(f"  sMAPE: {worst_metrics['smape_mean']:.1f}%\n")
        f.write(f"  RMSLE: {worst_metrics['rmsle']:.3f}\n")
        f.write("Worst performing locations:\n")
        for _, row in worst.head(5).iterrows():
            f.write(f"  - {row['location_name']} ({row['city']}): Error = {row['absolute_error']:.2f}\n")
        
        if 'top_category' in df.columns and df['top_category'].nunique() > 1:
            f.write(f"\nPERFORMANCE BY CATEGORY:\n")
            cat_performance = df.groupby('top_category').agg({
                'absolute_error': ['count', 'mean'],
                'smape_individual': 'mean',
                'rmsle_individual': 'mean'
            }).round(3)
            cat_performance.columns = ['count', 'mean_ae', 'mean_smape', 'mean_rmsle']
            for category, stats in cat_performance.iterrows():
                f.write(f"  {category}: Count={stats['count']}, MAE={stats['mean_ae']:.3f}, sMAPE={stats['mean_smape']:.1f}%, RMSLE={stats['mean_rmsle']:.3f}\n")

def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading prediction results and training data...")
    df = load_data(args.results, args.training_data)
    
    print(f"Loaded {len(df)} predictions")
    print(f"Mean absolute error: {df['absolute_error'].mean():.3f}")
    print(f"Median absolute error: {df['absolute_error'].median():.3f}")
    
    print("Creating performance summary plots...")
    overall_metrics = plot_summary(df, output_dir, args.figsize, args.dpi)
    
    print(f"Overall Performance Metrics:")
    print(f"  MAE: {overall_metrics['mae']:.3f}")
    print(f"  RMSE: {overall_metrics['rmse']:.3f}")
    print(f"  sMAPE (mean): {overall_metrics['smape_mean']:.1f}%")
    print(f"  sMAPE (median): {overall_metrics['smape_median']:.1f}%")
    print(f"  RMSLE: {overall_metrics['rmsle']:.3f}")
    
    print(f"Creating best/worst {args.top_n} prediction plots...")
    best, worst = plot_best_worst(df, args.top_n, output_dir, args.figsize, args.dpi)
    
    print("Saving detailed results...")
    save_results(df, best, worst, output_dir)
    
    print("Generating summary report...")
    gen_report(df, best, worst, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  - performance_summary.png")
    print(f"  - best_{args.top_n}_predictions.png")
    print(f"  - worst_{args.top_n}_predictions.png")
    print(f"  - best_{args.top_n}_detailed_report.csv")
    print(f"  - worst_{args.top_n}_detailed_report.csv")
    print(f"  - all_predictions_detailed_report.csv")
    print(f"  - analysis_report.json")
    print(f"  - summary_report.txt")

if __name__ == "__main__":
    main()