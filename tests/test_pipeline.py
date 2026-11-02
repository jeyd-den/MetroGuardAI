import pytest
from src.data_loader import load_sensor_data, save_processed_data
from src.preprocessor import clean_data, scale_features, create_windows
from src.config import Config

def test_end_to_end_pipeline():
    # Load
    df = load_sensor_data()
    # Clean
    df = clean_data(df)
    # Scale
    scaled_df, scaler = scale_features(df)
    # Window
    X = create_windows(scaled_df, 30)

    assert X.ndim == 3
    assert X.shape[2] == len(scaled_df.columns)
