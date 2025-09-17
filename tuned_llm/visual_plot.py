#!/usr/bin/env python3
"""
Create visualization plots for best and worst performing POIs from prediction results.
Generates time series plots similar to Prophet model format with confidence intervals.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import seaborn as sns

def setup_plot_style():
    """Set up matplotlib style for clean plots"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def parse_date_series(series_start_date, num_days=14):
    """Generate date range from series start date"""
    start_date = pd.to_datetime(series_start_date)
    return pd.date_range(start=start_date, periods=num_days, freq='D')

def calculate_confidence_interval(values, confidence=0.8):
    """Calculate confidence interval for time series"""
    values = np.array(values)
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Calculate confidence interval
    z_score = 1.28 if confidence == 0.8 else 1.96  # 80% or 95% CI
    margin = z_score * std_val / np.sqrt(len(values))
    
    lower_bound = mean_val - margin
    upper_bound = mean_val + margin
    
    return lower_bound, upper_bound

def create_single_poi_plot(row, landfall_date=None, save_path=None, show_confidence=True):
    """Create a single POI time series plot"""
    # Parse the prev_13_values
    prev13 = row.get('prev_13_values', [])
    if isinstance(prev13, str):
        try:
            prev13 = eval(prev13)
        except:
            prev13 = []
    
    # Generate dates
    dates = parse_date_series(row['series_start_date'], 14)
    
    # Prepare observed data (days 1-13)
    observed_values = prev13[:13] if len(prev13) >= 13 else prev13 + [np.nan] * (13 - len(prev13))
    observed_dates = dates[:13]
    
    # Day 14 values
    actual_d14 = row['y_true_d14']
    pred_d14 = row['y_pred_d14']
    d14_date = dates[13]
    
    # Parse landfall date
    landfall_dt = None
    if landfall_date and pd.notna(landfall_date):
        try:
            landfall_dt = pd.to_datetime(landfall_date)
        except:
            pass
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot observed time series (days 1-13)
    ax.plot(observed_dates, observed_values, 'o-', 
            color='#1f77b4', linewidth=2, markersize=6, 
            label='Observed (Days 1-13)', alpha=0.8)
    
    # Plot actual D14
    ax.plot(d14_date, actual_d14, 'o', 
            color='green', markersize=12, 
            label=f'Actual D14 = {actual_d14:.1f}', markeredgecolor='black', markeredgewidth=1)
    
    # Plot predicted D14 - offset slightly if values are the same
    if abs(actual_d14 - pred_d14) < 0.01:  # If values are very close/same
        # Offset predicted point slightly to the right to make it visible
        offset_date = d14_date + pd.Timedelta(hours=6)
        ax.plot(offset_date, pred_d14, 's', 
                color='orange', markersize=10, 
                label=f'Predicted D14 = {pred_d14:.1f}', markeredgecolor='black', markeredgewidth=1)
    else:
        ax.plot(d14_date, pred_d14, 's', 
                color='orange', markersize=10, 
                label=f'Predicted D14 = {pred_d14:.1f}', markeredgecolor='black', markeredgewidth=1)
    
    # Add landfall line if available
    if landfall_dt and landfall_dt >= observed_dates[0] and landfall_dt <= d14_date:
        ax.axvline(x=landfall_dt, color='gray', linestyle='--', 
                  alpha=0.7, label='Landfall')
    
    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Visits')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Title with key information
    abs_error = abs(actual_d14 - pred_d14)
    title = f"{row.get('location_name', 'Unknown Location')}\n"
    title += f"{row.get('city', 'Unknown City')} | abs err: {abs_error:.2f}\n"
    title += f"placekey: {row['placekey']} | lat: {row.get('latitude', 'N/A'):.4f}, lon: {row.get('longitude', 'N/A'):.4f}"
    
    ax.set_title(title, fontsize=12, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_best_worst_pois(csv_path, output_dir=None, num_plots=10, show_confidence=True):
    """
    Generate plots for best and worst performing POIs
    
    Args:
        csv_path: Path to predictions CSV file
        output_dir: Directory to save plots
        num_plots: Number of best/worst plots to generate
        show_confidence: Whether to show confidence intervals
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Calculate absolute error if not present
    if 'absolute_error' not in df.columns:
        df['absolute_error'] = abs(df['y_true_d14'] - df['y_pred_d14'])
    
    # Filter valid predictions
    valid_df = df.dropna(subset=['y_true_d14', 'y_pred_d14']).copy()
    
    if valid_df.empty:
        print("No valid predictions found in the CSV file.")
        return
    
    # Sort by absolute error
    valid_df = valid_df.sort_values('absolute_error')
    
    # Get best and worst performers
    best_performers = valid_df.head(num_plots)
    worst_performers = valid_df.tail(num_plots).sort_values('absolute_error', ascending=False)
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent / "performance_plots"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Setup plot style
    setup_plot_style()
    
    # Generate best performance plots
    print(f"Generating {len(best_performers)} best performance plots...")
    best_dir = output_dir / "best_performance"
    best_dir.mkdir(exist_ok=True)
    
    for idx, (_, row) in enumerate(best_performers.iterrows(), 1):
        filename = f"best_{idx:02d}_{row['placekey'][:10]}_ae_{row['absolute_error']:.2f}.png"
        save_path = best_dir / filename
        
        try:
            create_single_poi_plot(
                row, 
                landfall_date=row.get('landfall_date'),
                save_path=save_path,
                show_confidence=show_confidence
            )
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Error creating plot for {row['placekey']}: {e}")
    
    # Generate worst performance plots
    print(f"Generating {len(worst_performers)} worst performance plots...")
    worst_dir = output_dir / "worst_performance"
    worst_dir.mkdir(exist_ok=True)
    
    for idx, (_, row) in enumerate(worst_performers.iterrows(), 1):
        filename = f"worst_{idx:02d}_{row['placekey'][:10]}_ae_{row['absolute_error']:.2f}.png"
        save_path = worst_dir / filename
        
        try:
            create_single_poi_plot(
                row, 
                landfall_date=row.get('landfall_date'),
                save_path=save_path,
                show_confidence=show_confidence
            )
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Error creating plot for {row['placekey']}: {e}")
    
    # Create summary plot with both best and worst
    create_summary_plot(best_performers, worst_performers, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print(f"Best performance plots: {best_dir}")
    print(f"Worst performance plots: {worst_dir}")

def create_summary_plot(best_performers, worst_performers, output_dir):
    """Create a summary comparison plot"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle('Performance Comparison: Best vs Worst POI Predictions', fontsize=16)
    
    # Plot best performers (top row)
    for idx, (_, row) in enumerate(best_performers.head(5).iterrows()):
        ax = axes[0, idx]
        plot_mini_timeseries(row, ax, f"Best #{idx+1}")
    
    # Plot worst performers (bottom row)
    for idx, (_, row) in enumerate(worst_performers.head(5).iterrows()):
        ax = axes[1, idx]
        plot_mini_timeseries(row, ax, f"Worst #{idx+1}")
    
    plt.tight_layout()
    summary_path = output_dir / "summary_comparison.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary comparison saved: {summary_path.name}")

def plot_mini_timeseries(row, ax, title_prefix):
    """Create a mini time series plot for summary view"""
    # Parse data
    prev13 = row.get('prev_13_values', [])
    if isinstance(prev13, str):
        try:
            prev13 = eval(prev13)
        except:
            prev13 = []
    
    # Generate simple x-axis (days 1-14)
    days = list(range(1, 14))
    observed_values = prev13[:13] if len(prev13) >= 13 else prev13 + [0] * (13 - len(prev13))
    
    # Plot observed data
    ax.plot(days, observed_values, 'o-', color='blue', alpha=0.7, markersize=4)
    
    # Plot D14 predictions
    ax.plot(14, row['y_true_d14'], 'o', color='green', markersize=8, label='Actual')
    ax.plot(14, row['y_pred_d14'], 's', color='orange', markersize=8, label='Predicted')
    
    # Formatting
    ax.set_title(f"{title_prefix}\nAE: {row['absolute_error']:.2f}", fontsize=10)
    ax.set_xlabel('Day')
    ax.set_ylabel('Visits')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
