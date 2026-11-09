import pandas as pd
from .config import Config
from pathlib import Path

def load_sensor_data():
    """Loads the main sensor CSV file and ensures timestamp is handled."""
    df = pd.read_csv(Config.RAW_SENSORS_CSV)
    # Case-insensitive timestamp search
    cols = {col.lower(): col for col in df.columns}
    if 'timestamp' in cols:
        df = df.rename(columns={cols['timestamp']: 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def load_failure_labels():
    """Loads the failure windows CSV file."""
    return pd.read_csv(Config.FAILURE_WINDOWS_CSV)

def save_processed_data(df, filename):
    """Saves a dataframe as a parquet file in the processed directory."""
    path = Config.DATA_PROCESSED_PATH / filename
    try:
        df.to_parquet(path)
    except ImportError:
        csv_path = path.with_suffix(".csv") if path.suffix != ".csv" else path
        df.to_csv(csv_path, index=False)

def load_processed_data(filename):
    """Loads a processed data file and resets index for cloud compatibility."""
    path = Config.DATA_PROCESSED_PATH / filename

    if path.suffix == ".parquet" or not path.suffix:
        try:
            df = pd.read_parquet(path)
        except (ImportError, FileNotFoundError):
            df = None
    else:
        df = None

    if df is None:
        csv_path = path.with_suffix(".csv") if path.suffix != ".csv" else path
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Processed data not found at {path} or {csv_path}")

    return df.reset_index(drop=True)
