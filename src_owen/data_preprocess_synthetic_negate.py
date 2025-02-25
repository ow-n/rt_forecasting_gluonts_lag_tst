"""
data_preprocess.py
Creates train and test datasets using a subtractive approach:
1. Duplicates full dataset to train/test locations
2. Removes unwanted columns
3. Removes rows outside desired time windows

Data format:
- Input file has Time and multiple feature columns (A-L)
- We only keep Time and feature D
- Training data: 2.5s to 7.5s
- Test data: 100 samples after 7.5s

⚡To run:
    > python src_owen/data_preprocess_synthetic_negate.py
"""
import logging
import pandas as pd
import shutil
from pathlib import Path

# Constants
DATASET_FILE = "model_inputs_outputs/inputs/data/training/synthetic/dataset4.csv"  # Input dataset
TIME_COL = "Time"  # Time column name
FEATURE_COL = "D"  # Feature column to keep
TRAIN_START = 2.5  # Training data start time (seconds)
TRAIN_END = 7.5    # Training data end time (seconds)
TEST_SAMPLES = 100 # Number of test samples to use

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('tmp/preprocessing.log', mode='a'),
        logging.StreamHandler()
    ]
)

class DataPreprocessor:
    def __init__(self):
        self.root_dir = Path.cwd()
        self.dataset_path = self.root_dir / DATASET_FILE
        self.train_output = self.root_dir / 'model_inputs_outputs/inputs/data/training/train.csv'
        self.test_output = self.root_dir / 'model_inputs_outputs/inputs/data/testing/test.csv'
        
    def duplicate_dataset(self):
        """Create copies of the original dataset in train and test directories."""
        # Create output directories if they don't exist
        self.train_output.parent.mkdir(parents=True, exist_ok=True)
        self.test_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the original file to both locations
        shutil.copy2(self.dataset_path, self.train_output)
        shutil.copy2(self.dataset_path, self.test_output)
        logging.info(f"Duplicated {DATASET_FILE} to train and test locations")

    def process_train_data(self):
        """Process training data by removing unwanted columns and rows."""
        # Read the duplicated file
        df = pd.read_csv(self.train_output, dtype={TIME_COL: str})
        original_shape = df.shape
        
        # Remove all columns except Time and feature D
        columns_to_keep = [TIME_COL, FEATURE_COL]
        columns_to_remove = [col for col in df.columns if col not in columns_to_keep]
        df.drop(columns=columns_to_remove, inplace=True)
        logging.info(f"Removed {len(columns_to_remove)} columns from training data")
        
        # Create numeric time column for filtering
        df['Time_numeric'] = df[TIME_COL].astype(float)
        
        # Remove rows outside the training window
        original_rows = len(df)
        df = df[
            (df['Time_numeric'] >= TRAIN_START) & 
            (df['Time_numeric'] <= TRAIN_END)
        ]
        rows_removed = original_rows - len(df)
        logging.info(f"Removed {rows_removed} rows outside training window")
        
        # Drop the numeric helper column and save
        df = df.drop(columns=['Time_numeric'])
        df.to_csv(self.train_output, index=False, float_format="%.5f")
        logging.info(f"Training data shape: {original_shape} -> {df.shape}")

    def process_test_data(self):
        """Process test data by removing unwanted columns and rows."""
        # Read the duplicated file
        df = pd.read_csv(self.test_output, dtype={TIME_COL: str})
        original_shape = df.shape
        
        # Remove all columns except Time and feature D
        columns_to_keep = [TIME_COL, FEATURE_COL]
        columns_to_remove = [col for col in df.columns if col not in columns_to_keep]
        df.drop(columns=columns_to_remove, inplace=True)
        logging.info(f"Removed {len(columns_to_remove)} columns from test data")
        
        # Create numeric time column for filtering
        df['Time_numeric'] = df[TIME_COL].astype(float)
        
        # Keep only first TEST_SAMPLES rows after training window
        original_rows = len(df)
        df = df[df['Time_numeric'] > TRAIN_END].head(TEST_SAMPLES)
        rows_removed = original_rows - len(df)
        logging.info(f"Removed {rows_removed} rows to keep only {TEST_SAMPLES} test samples")
        
        # Drop the numeric helper column and save
        df = df.drop(columns=['Time_numeric'])
        df.to_csv(self.test_output, index=False, float_format="%.5f")
        logging.info(f"Test data shape: {original_shape} -> {df.shape}")

    def add_termid_column(self):
        """Add TermID column to both train and test files."""
        for file_path in [self.train_output, self.test_output]:
            df = pd.read_csv(file_path)
            df['TermID'] = 0
            df.to_csv(file_path, index=False, float_format="%.5f")
            logging.info(f"Added TermID column to {file_path}")

    def rename_time_column(self):
        """Rename Time column to Timestamp in both files."""
        for file_path in [self.train_output, self.test_output]:
            df = pd.read_csv(file_path)
            df = df.rename(columns={TIME_COL: 'Timestamp'})
            df.to_csv(file_path, index=False, float_format="%.5f")
            logging.info(f"Renamed Time column to Timestamp in {file_path}")

def main():
    preprocessor = DataPreprocessor()
    
    # First duplicate the dataset to both locations
    preprocessor.duplicate_dataset()
    
    # Process train and test data separately
    preprocessor.process_train_data()
    preprocessor.process_test_data()
    
    # Add required columns and rename as needed
    preprocessor.add_termid_column()
    preprocessor.rename_time_column()

if __name__ == "__main__":
    main()