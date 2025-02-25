"""
data_preprocess.py
Creates csv files into /data/training and testing for model training/testing

Data format:
- Input file has Time and multiple feature columns (A-L)
- We only use Time and feature D
- Training data: 2.5s to 7.5s
- Test data: 100 samples after 7.5s

⚡To run:
    > python src_owen/data_preprocess_synthetic.py
"""
import logging
import pandas as pd
from pathlib import Path
# pylint: disable=W1203   

# Constants
DATASET_FILE = "model_inputs_outputs/inputs/data/training/synthetic/dataset3.csv"  # Input dataset filename
TIME_COL = "Time"  # Time column name
FEATURE_COL = "D"  # Feature column to use
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

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and process data
    data = preprocessor.load_data()
    data = preprocessor.filter_features(data)
    train_data, val_data, test_data = preprocessor.split_data(data)
    
    # Save data in format required for LagTST model
    preprocessor.prepare_for_lagtst(train_data, is_train=True)
    #preprocessor.prepare_for_lagtst(val_data, is_train=True)
    preprocessor.prepare_for_lagtst(test_data, is_train=False)

    # Modify CSV to fit Oklahoma schema
    add_termid_column()
    rename_time_column()

class DataPreprocessor:
    def __init__(self):
        # Get the root directory (where the script is run from)
        self.root_dir = Path.cwd()
        self.dataset_path = self.root_dir / DATASET_FILE
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file.
        
        Read the Time column as a string (to preserve exactly 5 decimal places)
        and also create a numeric copy for filtering.
        """
        # Read Time as string so we preserve the original formatting,
        # and read FEATURE_COL normally.
        df = pd.read_csv(self.dataset_path, dtype={TIME_COL: str}, usecols=[TIME_COL, FEATURE_COL])
        # Create a numeric column for filtering/comparisons.
        df["Time_numeric"] = df[TIME_COL].astype(float)
        logging.info(f"Loaded data from {DATASET_FILE}")
        logging.info(f"Data shape: {df.shape}\n")
        return df
    
    def filter_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data to only include the time and feature columns.
        
        (The Time column is still preserved as a string.)
        """
        return data[[TIME_COL, "Time_numeric", FEATURE_COL]]
    
    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets based on time ranges.
        
        Uses the numeric copy of Time for filtering.
        
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        # TRAIN SET: 2.5s to 7.5s (using numeric time)
        train_data = data[(data["Time_numeric"] >= TRAIN_START) & (data["Time_numeric"] <= TRAIN_END)].copy()
        
        # VAL SET: Last TEST_SAMPLES rows before the midpoint (for example)
        val_mid_time = (TRAIN_START + TRAIN_END) / 2
        val_data = data[data["Time_numeric"] <= val_mid_time].tail(TEST_SAMPLES).copy()
        
        # TEST SET: 100 samples after 7.5s
        test_data = data[data["Time_numeric"] > TRAIN_END].head(TEST_SAMPLES).copy()
        
        logging.info(f"Training data shape: {train_data.shape}")
        logging.info(f"Validation data shape: {val_data.shape}")
        logging.info(f"Test data shape: {test_data.shape}\n")
        
        return train_data, val_data, test_data

    def prepare_for_lagtst(self, df: pd.DataFrame, is_train: bool) -> None:
        """
        Save training and testing data in format required for LagTST model.
        
        Drops the extra numeric column so that the original string representation of Time is preserved.
        FEATURE_COL (D) is written with 5 decimal places.
        
        Args:
            df: DataFrame containing data
            is_train: Boolean indicating if this is training data
        """
        # Save to appropriate directories using root directory
        output_base = self.root_dir / 'model_inputs_outputs/inputs/data'
        if is_train:
            output_dir = output_base / 'training'
            filename = 'train.csv'
        else:
            output_dir = output_base / 'testing'
            filename = 'test.csv'
            
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Drop the numeric helper column so that the original Time string is preserved.
        df = df.drop(columns=["Time_numeric"])
        # Optionally, sort by time if needed:
        # df = df.sort_values(by=TIME_COL)
        
        # Save file.
        # Note: the Time column is now a string and will be written as-is,
        # while FEATURE_COL will be formatted with 5 decimals.
        output_path = output_dir / filename
        df.to_csv(output_path, index=False, float_format="%.5f")
        logging.info(f"Saved {len(df)} rows to {output_path}")

def add_termid_column():
    """
    Add a 'TermID' column filled with zeros to both training and testing CSV files.
    This function should be called after the main data preprocessing is complete.
    """
    root_dir = Path.cwd()
    base_path = root_dir / 'model_inputs_outputs/inputs/data'
    
    # File paths
    train_path = base_path / 'training/train.csv'
    test_path = base_path / 'testing/test.csv'
    
    for file_path in [train_path, test_path]:
        try:
            df = pd.read_csv(file_path)
            df['TermID'] = 0
            df.to_csv(file_path, index=False, float_format="%.5f")
            logging.info(f"Added TermID column to {file_path}")
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

def rename_time_column():
    """
    Rename the 'Time' column to 'Timestamp' in both training and testing CSV files.
    This function should be called after the main data preprocessing is complete.
    """
    root_dir = Path.cwd()
    base_path = root_dir / 'model_inputs_outputs/inputs/data'
    
    train_path = base_path / 'training/train.csv'
    test_path = base_path / 'testing/test.csv'
    
    for file_path in [train_path, test_path]:
        try:
            df = pd.read_csv(file_path)
            df = df.rename(columns={TIME_COL: 'Timestamp'})
            df.to_csv(file_path, index=False, float_format="%.5f")
            logging.info(f"Renamed '{TIME_COL}' column to 'Timestamp' in {file_path}")
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    main()
