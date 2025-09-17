#!/usr/bin/env python3
"""
Generate comprehensive summary reports from prediction CSV files.
Creates detailed reports, best/worst summaries, overall statistics, and metrics JSON.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from compute_metric import compute_mae, compute_rmse, compute_smape, compute_rmsle

def create_detailed_report(pred_df, output_path):
    """Create detailed report with expanded input series"""
    detailed_rows = []
    for _, row in pred_df.iterrows():
        # Parse the prev_13_values if it's stored as string
        prev13 = row.get('prev_13_values', [])
        if isinstance(prev13, str):
            try:
                prev13 = eval(prev13)
            except:
                prev13 = []
        
        # Create columns for each day in the input series
        detailed_row = {
            'placekey': row['placekey'],
            'location_name': row['location_name'],
            'top_category': row['top_category'],
            'city': row['city'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'series_start_date': row['series_start_date'],
            'landfall_date': row['landfall_date'],
            'target_date_d14': row['target_date_d14'],
            'actual_target_date': row['actual_target_date'],
            'target_days_after_landfall': row['target_days_after_landfall']
        }
        
        # Add input series (days 1-13)
        for i in range(13):
            day_val = prev13[i] if i < len(prev13) else np.nan
            detailed_row[f'input_day_{i+1:02d}'] = day_val
        
        # Add prediction results
        detailed_row.update({
            'y_true_d14': row['y_true_d14'],
            'y_pred_d14': row['y_pred_d14'],
            'confidence_score': row['confidence_score'],
            'absolute_error': row['absolute_error'],
            'percent_error': row['percent_error'],
            'smape_individual': row['smape_individual'],
            'rmsle_individual': row['rmsle_individual'],
            'llm_text': row['llm_text']
        })
        
        detailed_rows.append(detailed_row)
    
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(output_path, index=False)
    return detailed_df

def create_best_worst_summary(valid_df, output_path):
    """Create summary report with top 10 best and worst predictions"""
    # Sort by absolute error for best/worst performance
    valid_sorted = valid_df.sort_values('absolute_error')
    
    # Top 10 best (lowest absolute error)
    top_10_best = valid_sorted.head(10).copy()
    top_10_best['rank'] = range(1, min(11, len(top_10_best) + 1))
    top_10_best['performance'] = 'Best'
    
    # Top 10 worst (highest absolute error)
    top_10_worst = valid_sorted.tail(10).copy()
    top_10_worst = top_10_worst.sort_values('absolute_error', ascending=False)
    top_10_worst['rank'] = range(1, min(11, len(top_10_worst) + 1))
    top_10_worst['performance'] = 'Worst'
    
    # Combine best and worst
    summary_df = pd.concat([top_10_best, top_10_worst], ignore_index=True)
    
    # Select key columns for summary
    summary_columns = [
        'performance', 'rank', 'placekey', 'location_name', 'top_category', 
        'city', 'y_true_d14', 'y_pred_d14', 'absolute_error', 'percent_error',
        'smape_individual', 'rmsle_individual', 'confidence_score', 'target_days_after_landfall'
    ]
    
    summary_df = summary_df[summary_columns]
    summary_df.to_csv(output_path, index=False)
    return summary_df

def create_overall_stats(pred_df, valid_df, mae, rmse, smape_mean, smape_median, rmsle, output_path):
    """Create overall statistics and category-wise performance report"""
    # Overall statistics with enhanced metrics
    overall_stats = {
        'metric': ['Total Records', 'Valid Records', 'MAE', 'RMSE', 
                  'sMAPE Mean (%)', 'sMAPE Median (%)', 'RMSLE',
                  'Mean Confidence', 'Median Absolute Error', 
                  'Best Absolute Error', 'Worst Absolute Error'],
        'value': [
            len(pred_df),
            len(valid_df),
            f"{mae:.3f}",
            f"{rmse:.3f}",
            f"{smape_mean:.3f}",
            f"{smape_median:.3f}",
            f"{rmsle:.3f}",
            f"{valid_df['confidence_score'].mean():.3f}",
            f"{valid_df['absolute_error'].median():.3f}",
            f"{valid_df['absolute_error'].min():.3f}",
            f"{valid_df['absolute_error'].max():.3f}"
        ]
    }
    
    # Add category-wise performance
    category_stats = valid_df.groupby('top_category').agg({
        'absolute_error': ['count', 'mean', 'median'],
        'smape_individual': 'mean',
        'rmsle_individual': 'mean'
    }).round(3)
    
    category_stats.columns = ['Count', 'Mean_AE', 'Median_AE', 'Mean_sMAPE', 'Mean_RMSLE']
    category_stats = category_stats.reset_index()
    
    # Save category stats
    with open(output_path, 'w') as f:
        f.write("OVERALL STATISTICS\n")
        pd.DataFrame(overall_stats).to_csv(f, index=False)
        f.write("\n\nCATEGORY-WISE PERFORMANCE\n")
        category_stats.to_csv(f, index=False)
    
    return overall_stats, category_stats

def create_metrics_json(valid_df, mae, rmse, smape_mean, smape_median, rmsle, output_path):
    """Create metrics summary JSON"""
    metrics_summary = {
        "samples": len(valid_df),
        "mae": mae,
        "rmse": rmse,
        "smape_mean": smape_mean,
        "smape_median": smape_median,
        "rmsle": rmsle,
    }
    
    with open(output_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    return metrics_summary

def generate_all_reports(pred_df, output_dir, base_name):
    """
    Generate all summary reports from prediction DataFrame
    
    Args:
        pred_df: DataFrame with predictions
        output_dir: Directory to save reports
        base_name: Base name for output files
    
    Returns:
        dict: Dictionary containing all metrics and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Calculate comprehensive metrics using imported functions
    valid = pred_df.dropna(subset=["y_true_d14","y_pred_d14"])
    
    if valid.empty:
        print("No valid predictions to create summary reports.")
        return None
    
    y_true = valid["y_true_d14"].values
    y_pred = valid["y_pred_d14"].values
    
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    smape_mean, smape_median = compute_smape(y_true, y_pred)
    rmsle = compute_rmsle(y_true, y_pred)
    
    # Define output file paths
    detailed_report_path = output_dir / f"{base_name}_detailed_report.csv"
    summary_path = output_dir / f"{base_name}_summary.csv"
    stats_path = output_dir / f"{base_name}_overall_stats.csv"
    metrics_json_path = output_dir / f"{base_name}_metrics_summary.json"
    
    # Generate all reports
    detailed_df = create_detailed_report(pred_df, detailed_report_path)
    summary_df = create_best_worst_summary(valid, summary_path)
    overall_stats, category_stats = create_overall_stats(pred_df, valid, mae, rmse, smape_mean, smape_median, rmsle, stats_path)
    metrics_summary = create_metrics_json(valid, mae, rmse, smape_mean, smape_median, rmsle, metrics_json_path)
    
    # Return results
    results = {
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'smape_mean': smape_mean,
            'smape_median': smape_median,
            'rmsle': rmsle,
            'total_records': len(pred_df),
            'valid_records': len(valid)
        },
        'file_paths': {
            'detailed_report': detailed_report_path,
            'summary': summary_path,
            'overall_stats': stats_path,
            'metrics_json': metrics_json_path
        },
        'dataframes': {
            'detailed': detailed_df,
            'summary': summary_df,
            'category_stats': category_stats
        }
    }
    
    return results
