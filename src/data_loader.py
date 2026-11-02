import pandas as pd
from .config import Config
from pathlib import Path

def load_sensor_data():
    """Loads the main sensor CSV file."""
    return pd.read_csv(Config.RAW_SENSORS_CSV)

def load_failure_labels():
    """Loads the failure windows CSV file."""
    return pd.read_csv(Config.FAILURE_WINDOWS_CSV)

def save_processed_data(df, filename):
    """Saves a dataframe as a parquet file in the processed directory."""
    path = Config.DATA_PROCESSED_PATH / filename
    # Try parquet, fall back to csv if pyarrow is missing
    try:
        df.to_parquet(path)
    except ImportError:
        # Fallback to csv if parquet engine is not installed
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

def load_processed_data(filename):
    """Loads a processed data file from the processed directory."""
    path = Config.DATA_PROCESSED_PATH / filename

    # Try loading as parquet first
    if path.suffix == ".parquet" or not path.suffix:
        try:
            return pd.read_parquet(path)
        except (ImportError, FileNotFoundError):
            pass

    # Fallback to CSV
    csv_path = path.with_suffix(".csv") if path.suffix != ".csv" else path
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Processed data not found at {path} or {csv_path}")
