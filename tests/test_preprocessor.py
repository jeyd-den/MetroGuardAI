import pytest
import pandas as pd
import numpy as np
from src.preprocessor import clean_data, scale_features, create_windows

def test_clean_data():
    df = pd.DataFrame({
        'timestamp': ['2026-01-01', '2026-01-02', '2026-01-02'], # duplicate
        'sensor_1': [1.0, np.nan, 3.0]
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert cleaned['sensor_1'].isna().sum() == 0

def test_scale_features():
    df = pd.DataFrame({'s1': [1, 2, 3], 's2': [10, 20, 30]})
    scaled, scaler = scale_features(df)
    assert scaled['s1'].mean() == pytest.approx(0, abs=1e-7)
    assert scaled['s1'].std() > 0

def test_create_windows():
    data = np.arange(100).reshape(10, 10)
    window_size = 3
    stride = 1
    windows = create_windows(data, window_size, stride)
    # (10 - 3 + 1) / 1 = 8
    assert windows.shape == (8, 3, 10)
