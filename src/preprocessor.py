import pandas as pd
import numpy as np
import pandera.pandas as pa
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Optional, Any
from .config import Config

# Define a type alias for DataFrame for better compatibility
DataFrame = pd.DataFrame

# Define the schema for sensor data validation
sensor_schema = pa.DataFrameSchema({
    "timestamp": pa.Column(pd.Timestamp),
    "TP2": pa.Column(float, nullable=True),
    "TP3": pa.Column(float, nullable=True),
    "H1": pa.Column(float, nullable=True),
    "DV_pressure": pa.Column(float, nullable=True),
    "Reservoirs": pa.Column(float, nullable=True),
    "Oil_temperature": pa.Column(float, nullable=True),
    "Motor_current": pa.Column(float, nullable=True),
})

def clean_data(df: DataFrame) -> DataFrame:
    """
    Performs professional data cleaning on sensor data.

    Includes:
    - Duplicate removal
    - Datetime normalization
    - Time-feature engineering (hour, day, weekend)
    - Linear interpolation for missing values
    - Strict schema validation via Pandera
    """
    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Ensure timestamp is datetime and sorted
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # 3. Feature Engineering: Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)

    # 4. Handle missing values via linear interpolation for sensors
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').bfill().ffill()

    # 5. Strict Validation
    try:
        sensor_cols = [col for col in sensor_schema.columns if col in df.columns]
        sensor_schema.validate(df[sensor_cols])
    except pa.errors.SchemaError as e:
        print(f"Data Validation Warning: {e}")

    return df

def scale_features(df: DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[DataFrame, StandardScaler]:
    """
    Scales features using StandardScaler with strict feature alignment.

    Args:
        df: The input DataFrame.
        scaler: An existing StandardScaler instance. If None, a new one is fitted.

    Returns:
        A tuple of (scaled_dataframe, scaler_instance).
    """
    if scaler is not None:
        if hasattr(scaler, "feature_names_in_"):
            expected_features = scaler.feature_names_in_

            # Filter and reorder based on expected features, filling missing with 0.0
            df_scaled_input = pd.DataFrame(index=df.index)
            for col in expected_features:
                df_scaled_input[col] = df[col] if col in df.columns else 0.0

            scaled_values = scaler.transform(df_scaled_input)
            numeric_cols = expected_features
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            scaled_values = scaler.transform(df[numeric_cols])
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[numeric_cols])

    scaled_df = pd.DataFrame(scaled_values, columns=numeric_cols, index=df.index)
    return scaled_df, scaler

def create_windows(data: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Transforms 2D data into 3D windows for LSTM input.

    Args:
        data: Input array of shape (samples, features).
        window_size: The length of each sequence.
        stride: The step size between windows.

    Returns:
        NumPy array of shape (num_windows, window_size, features).
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    num_samples = data.shape[0]

    windows = [data[i : i + window_size]
               for i in range(0, num_samples - window_size + 1, stride)]

    return np.array(windows)

def save_scaler(scaler: StandardScaler, filename: str = "scaler.joblib") -> None:
    """Saves the scaler to the models directory."""
    path = Config.MODEL_PATH / filename
    joblib.dump(scaler, path)

def load_scaler(filename: str = "scaler.joblib") -> StandardScaler:
    """Loads the scaler from the models directory."""
    path = Config.MODEL_PATH / filename
    return joblib.load(path)
