import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import json
from src.config import Config
from src.data_loader import load_sensor_data
from src.preprocessor import clean_data, scale_features, create_windows, save_scaler

def build_lstm_autoencoder(window_size, num_features):
    model = models.Sequential([
        # Encoder
        layers.LSTM(32, activation='relu', input_shape=(window_size, num_features), return_sequences=False),
        layers.RepeatVector(window_size),
        # Decoder
        layers.LSTM(32, activation='relu', return_sequences=True),
        layers.TimeDistributed(layers.Dense(num_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train():
    # 1. Load Config
    model_cfg = Config.load_model_config()
    window_size = model_cfg.get("lstm_autoencoder", {}).get("window_size", 30)
    epochs = model_cfg.get("lstm_autoencoder", {}).get("epochs", 50)
    batch_size = model_cfg.get("lstm_autoencoder", {}).get("batch_size", 32)

    # 2. Load and Preprocess
    print("Loading and preprocessing data...")
    df = load_sensor_data()
    df = clean_data(df)

    # Scale
    scaled_df, scaler = scale_features(df)
    save_scaler(scaler)

    # Window
    X = create_windows(scaled_df, window_size)
    print(f"Windowed data shape: {X.shape}")

    # 3. Model Training
    model = build_lstm_autoencoder(window_size, X.shape[2])
    print("Training LSTM Autoencoder...")
    model.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    # 4. Calibrate Threshold
    # Reconstruction error (MSE per sample)
    predictions = model.predict(X)
    mse = np.mean(np.power(X - predictions, 2), axis=(1, 2))
    threshold = np.percentile(mse, 95)
    print(f"Calibrated Anomaly Threshold (95th percentile): {threshold:.6f}")

    # 5. Save Model and Threshold
    model.save(Config.MODEL_PATH / "lstm_autoencoder.keras")

    # Update anomaly_threshold.json
    threshold_data = {}
    try:
        with open(Config.MODEL_PATH / "anomaly_threshold.json", "r") as f:
            threshold_data = json.load(f)
    except FileNotFoundError:
        pass

    threshold_data["lstm_threshold"] = float(threshold)
    with open(Config.MODEL_PATH / "anomaly_threshold.json", "w") as f:
        json.dump(threshold_data, f, indent=2)

    print("Training complete. Model and threshold saved.")

if __name__ == "__main__":
    train()
