"""
data_preprocess.py
Creates csv files into /data/training and testing for model training/testing

⚠️ Things to edit:
1) The Train/Val/Test terminal IDs
2) the file range
3) TODO: figure out what to do with Val set, current disabled

Edit the main function to specify the range of files to load data from
Edit the TRAIN_TERMINAL_ID, VAL_TERMINAL_ID, TEST_TERMINAL_ID to specify terminal IDs

Data files info:
- files 1-2 are normal
- files 3-25 (22.44 to 22.52) are oscillatory
- there's a timeskip 
- files 26-27 (22.53.37 to 22.54.37) are post-event when plant returned to normal conditions

⚡To run:
    > python src_owen/data_preprocess_oklahoma.py
"""
# pylint: disable=W1203   
import logging
import pandas as pd
from pathlib import Path

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
    """
    ID 0: Arcadia-Northwest
    ID 1: Arcadia-Seminole
    ID 2: Conoco-Continental Empire
    """
    TRAIN_TERMINAL_ID, VAL_TERMINAL_ID, TEST_TERMINAL_ID = 0, 1, 1  # <-- 0,1,2 changed to 0,1,1
    FIRST_FILE, LAST_FILE = 1, 2  # Select file range to load data from | 1 to 27
    
    # Load dataset from files (Range=[1,10])
    preprocessor = DataPreprocessor()
    data = preprocessor.get_data_for_file_range(FIRST_FILE, LAST_FILE)  # inclusive
    
    # Get data for specific terminal IDs
    train_data = preprocessor.get_terminal_data(TRAIN_TERMINAL_ID, data)
    val_data = preprocessor.get_terminal_data(VAL_TERMINAL_ID, data)
    test_data = preprocessor.get_terminal_data(TEST_TERMINAL_ID, data)
    logging.info(f"Training data shape: {train_data.shape}")
    logging.info(f"Validation data shape: {val_data.shape}")
    logging.info(f"Test data shape: {test_data.shape}\n")

    # Save data in format required for LagTST model
    preprocessor.prepare_for_lagtst(train_data, is_train=True)
    #preprocessor.prepare_for_lagtst(val_data, is_train=True)
    preprocessor.prepare_for_lagtst(test_data, is_train=False)

class DataPreprocessor:
    FILE_MAPPING = {
        1: "2017-04-20 20.42.37..csv",
        2: "2017-04-20 20.43.37..csv",
        3: "2017-04-20 20.44.37..csv",
        4: "2017-04-20 20.45.37..csv",
        5: "2017-04-20 20.46.37..csv",
        6: "2017-04-20 20.47.37..csv",
        7: "2017-04-20 20.48.37..csv",
        8: "2017-04-20 20.49.37..csv",
        9: "2017-04-20 20.50.37..csv",
        10: "2017-04-20 20.51.37..csv",
        11: "2017-04-20 20.52.37..csv",
        12: "2017-04-20 20.56.37..csv",
        13: "2017-04-20 20.57.37..csv",
        14: "2017-04-20 20.58.37..csv",
        15: "2017-04-20 20.59.37..csv",
        16: "2017-04-20 21.00.37..csv",
        17: "2017-04-20 21.01.37..csv",
        18: "2017-04-20 21.37.37..csv",
        19: "2017-04-20 21.38.37..csv",
        20: "2017-04-20 21.39.37..csv",
        21: "2017-04-20 21.40.37..csv",
        22: "2017-04-20 22.49.37..csv",
        23: "2017-04-20 22.50.37..csv",
        24: "2017-04-20 22.51.37..csv",
        25: "2017-04-20 22.52.37..csv",
        26: "2017-04-20 22.53.37..csv",
        27: "2017-04-20 22.54.37..csv"
    }
    DATASET_DIR = Path(__file__).parent.parent / "model_inputs_outputs/inputs/data/training/2017-04-20 Redbud Plant_Oklahoma"

    def __init__(self):
        pass
    
    def get_data_for_file_range(self, start_file: int, end_file: int) -> pd.DataFrame:
        """Load and combine data from a range of files."""
        if start_file < 1 or end_file > len(self.FILE_MAPPING):
            raise ValueError(f"File range must be between 1 and {len(self.FILE_MAPPING)}")
        
        all_data = []
        for file_num in range(start_file, end_file + 1):
            file_name = self.FILE_MAPPING[file_num]
            file_path = self.DATASET_DIR / file_name
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Read CSV with TermID column instead of Terminal
            df = pd.read_csv(file_path, usecols=['Timestamp', 'TermID', 'V_M_1'])
            df = df.iloc[:-382]  # Drop the last 382 rows for each file (382 terminals, so last val for each terminal)
            all_data.append(df)
            logging.info(f"Processed file {file_num}: {file_name} with ")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logging.info(f"Loaded data for files {start_file} to {end_file}")
        logging.info(f"Combined data shape: {combined_df.shape}\n")
        return combined_df
    
    def get_terminal_data(self, term_id: int, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get data for a specific terminal ID.
        
        Args:
            term_id (int): Terminal ID to filter by
            data (pd.DataFrame): DataFrame containing all data
            
        Returns:
            pd.DataFrame: Filtered DataFrame for specified terminal
        """
        terminal_data = data[data['TermID'] == term_id].copy()
        if terminal_data.empty:
            raise ValueError(f"No data found for terminal ID: {term_id}")
        logging.info(f"Retrieved data for terminal ID: {term_id}")
        return terminal_data

    def prepare_for_lagtst(self, df: pd.DataFrame, is_train: bool) -> None:
        """
        Save training and testing data in format required for LagTST model
        
        Args:
            train_data: DataFrame containing training data from one terminal
            test_data: DataFrame containing testing data from different terminal
        """
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])  #  Ensure timestamp is in correct format
        
        # Save to appropriate directories
        output_base = Path('model_inputs_outputs/inputs/data')
        if is_train:
            train_dir = output_base / 'training'
            df.to_csv(train_dir / 'train.csv', index=False)
            logging.info(f"Saved training data ({len(df)} rows) to {train_dir / 'train.csv'}")
        else:
            test_dir = output_base / 'testing'
            processed_df = df.iloc[-100:]  # only keep last 100 samples (forecastLength from schema)
            processed_df.to_csv(test_dir / 'test.csv', index=False)  # save file
            logging.info(f"Saved testing data ({len(processed_df)} rows) to {test_dir / 'test.csv'}")
        

if __name__ == "__main__":
    main()