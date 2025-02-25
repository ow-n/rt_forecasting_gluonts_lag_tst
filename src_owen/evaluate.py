"""
src_owen/evaluate.py

This script evaluates the model's predictions by comparing them to the actual test data.
Supports both timestep and timestamp visualization with configurable y-axis bounds.

⚡To run:
    - python src_owen/evaluate.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime


def main():
    # Plot 1: Default with timestamps and auto y-axis
    evaluate_predictions(
        use_timestamps=True,
        y_min=None,
        y_max=None,
        plot_suffix="auto_scale"
    )
    
    # Plot 2: Timesteps with fixed y-axis
    evaluate_predictions(
        use_timestamps=False,
        y_min=200000,
        y_max=210000,
        plot_suffix="fixed_scale_timesteps"
    )
    
    # Plot 3: Timestamps with fixed y-axis
    evaluate_predictions(
        use_timestamps=True,
        y_min=200000,
        y_max=210000,
        plot_suffix="fixed_scale_timestamps"
    )


def format_timestamp(ts):
    """Convert timestamp to SS.mmm format, showing minute only when it changes"""
    dt = pd.to_datetime(ts)
    return f"{dt.second}.{dt.microsecond//100:02d}"


def evaluate_predictions(use_timestamps=True, y_min=None, y_max=None, plot_suffix=""):
    # Load predictions and actual test data
    root_dir = Path(__file__).parent.parent
    pred_path = root_dir / 'model_inputs_outputs/outputs/predictions/predictions.csv'
    test_path = root_dir / 'model_inputs_outputs/inputs/data/testing/test.csv'
    predictions = pd.read_csv(pred_path)
    test_data = pd.read_csv(test_path)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Prepare x-axis values
    if use_timestamps:
        # Convert timestamps and get formatted labels
        timestamps = pd.to_datetime(test_data['Timestamp'])
        x_labels = [format_timestamp(ts) for ts in timestamps]
        x_values = range(len(timestamps))
        
        # Set x-ticks at every nth position to avoid overcrowding
        n = max(1, len(x_values) // 10)  # Show ~10 ticks
        plt.xticks(x_values[::n], x_labels[::n], rotation=45, ha='right')
    else:
        x_values = range(len(test_data))
    
    # Plot data
    plt.plot(x_values, test_data['V_M_1'].values, label='Actual', marker='o', markersize=4)
    plt.plot(x_values, predictions['prediction'].values, label='Predicted', marker='x', markersize=4)
    
    # Set axis limits and labels
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
        y_range_text = f" (y-range: {y_min:,}-{y_max:,})"
    else:
        y_range_text = " (auto y-scale)"
    
    time_type = "Timestamps" if use_timestamps else "Timesteps"
    plt.title(f'Voltage Predictions vs Actual Values using {time_type}{y_range_text}')
    plt.xlabel('Time (SS.mmm)' if use_timestamps else 'Time Steps (samples)')
    plt.ylabel('Voltage')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with descriptive filename
    output_dir = Path('model_inputs_outputs/outputs/plots')
    output_dir.mkdir(exist_ok=True)
    plot_name = f'predictions_plot_{plot_suffix}.png'
    plt.savefig(output_dir / plot_name, dpi=300)
    plt.close()
    
    # Print descriptive message about the plot
    print(f"\nGenerated plot '{output_dir / plot_name}':")
    print(f"- Using {'timestamps' if use_timestamps else 'timesteps'}")
    print(f"- Y-axis range: {'auto-scaled' if y_min is None else f'{y_min:,} to {y_max:,}'}")
    

if __name__ == "__main__":
    main()