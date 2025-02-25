"""
plot_synthetic_data.py

⚡To run:
    > python src_owen/plot_synthetic_data.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import os

# Constants for file paths and column names
DATA_FILE = "model_inputs_outputs/inputs/data/training/synthetic/dataset4.csv"
TIME_COLUMN = "Time"
FEATURE_COLUMN = "D"
OUTPUT_DIR = "src_owen/"  # Directory to save the plots

def calculate_frequency(time_series: pd.Series) -> float:
    """
    Calculate the frequency of the time series data.
    
    Args:
        time_series (pd.Series): Time values from the dataset
        
    Returns:
        float: Frequency in Hz
    """
    # Calculate time differences
    time_diff = np.diff(time_series)
    
    # Calculate average time step
    avg_time_step = np.mean(time_diff)
    
    # Calculate frequency (1/period)
    frequency = 1 / avg_time_step
    
    return frequency

def plot_time_series(file_path: str, time_col: str, feature_col: str) -> Tuple[plt.Figure, float]:
    """
    Plot time series data and calculate its frequency.
    
    Args:
        file_path (str): Path to the CSV file
        time_col (str): Name of the time column
        feature_col (str): Name of the feature column to plot
        
    Returns:
        Tuple[plt.Figure, float]: Figure object and calculated frequency
    """
    # Read the data
    df = pd.read_csv(file_path)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[time_col], df[feature_col], 'b-', linewidth=2)
    
    # Customize the plot
    ax.set_xlabel('Time')
    ax.set_ylabel(feature_col)
    ax.set_title(f'Time Series Plot of {feature_col}')
    ax.grid(True)
    
    # Calculate frequency
    freq = calculate_frequency(df[time_col])
    
    return fig, freq

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get the dataset name from the file path
    dataset_name = os.path.splitext(os.path.basename(DATA_FILE))[0]
    
    # Create the plot and get the frequency
    fig, frequency = plot_time_series(DATA_FILE, TIME_COLUMN, FEATURE_COLUMN)
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_{FEATURE_COLUMN}_plot.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    
    # Print the frequency and completion message
    print(f"Data frequency: {frequency:.2f} Hz")
    print(f"Plot saved to: {output_path}")
    print("Program finished successfully!")

if __name__ == "__main__":
    main()