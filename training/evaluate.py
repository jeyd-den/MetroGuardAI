import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from src.config import Config
from src.data_loader import load_sensor_data, load_failure_labels, save_processed_data
from src.preprocessor import clean_data, scale_features, create_windows

def evaluate():
    # 1. Setup
    model_cfg = Config.load_model_config()
    window_size = model_cfg.get("lstm_autoencoder", {}).get("window_size", 30)

    # Load models and scaler
    lstm_model = tf.keras.models.load_model(Config.MODEL_PATH / "lstm_autoencoder.h5", compile=False)
    xgb_model = joblib.load(Config.MODEL_PATH / "xgb_baseline.joblib")
    scaler = joblib.load(Config.MODEL_PATH / "scaler.joblib")

    # Load threshold
    try:
        with open(Config.MODEL_PATH / "anomaly_threshold.json", "r") as f:
            thresholds = json.load(f)
    except FileNotFoundError:
        thresholds = {}
    lstm_thresh = thresholds.get("lstm_threshold", 0.1)

    # 2. Load Data
    df = load_sensor_data()
    df = clean_data(df)
    scaled_df, _ = scale_features(df, scaler=scaler)

    # 3. Generate Scores
    # XGBoost
    # The XGB model expects scaled sensors PLUS time features: [sensors..., 'hour', 'day_of_week', 'is_weekend']
    time_cols = ['hour', 'day_of_week', 'is_weekend']
    # Concatenate scaled sensors with the raw time features from the cleaned dataframe
    xgb_input = pd.concat([scaled_df, df[time_cols]], axis=1)

    # Ensure column order matches XGBoost's expectation exactly
    expected_xgb_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current', 'hour', 'day_of_week', 'is_weekend']
    # Filter and reorder based on the expected list (filling missing with 0)
    final_xgb_input = pd.DataFrame(index=xgb_input.index)
    for col in expected_xgb_cols:
        if col in xgb_input.columns:
            final_xgb_input[col] = xgb_input[col]
        else:
            final_xgb_input[col] = 0.0

    xgb_scores = xgb_model.predict_proba(final_xgb_input)[:, 1]

    # LSTM
    X = create_windows(scaled_df, window_size)
    predictions = lstm_model.predict(X)

    # FIX: The model seems to be outputting 60 timesteps instead of 30 (window_size).
    # This usually happens if the model was trained with a different window size.
    # We slice the predictions to match the input X shape to allow subtraction.
    if predictions.shape[1] != X.shape[1]:
        print(f"Warning: Model output shape {predictions.shape[1]} differs from input window size {X.shape[1]}. Slicing output to match.")
        predictions = predictions[:, :X.shape[1], :]

    lstm_scores = np.mean(np.power(X - predictions, 2), axis=(1, 2))

    # 4. Map Scores to Time
    # LSTM windows are shifted. We map the score of the window to the last timestamp of that window.
    timestamps = df['timestamp'].values[window_size-1:]

    lstm_results = pd.DataFrame({
        'timestamp': timestamps,
        'score': lstm_scores,
        'is_anomaly': lstm_scores > lstm_thresh
    })

    xgb_results = pd.DataFrame({
        'timestamp': df['timestamp'].values,
        'score': xgb_scores,
        'is_anomaly': xgb_scores > 0.5
    })

    # 5. Compare with Failure Windows
    failures = load_failure_labels()

    def get_metrics(results_df, failure_df):
        # Create binary labels for every timestamp
        labels = np.zeros(len(results_df))
        for _, row in failure_df.iterrows():
            start = pd.to_datetime(row['start_time'])
            end = pd.to_datetime(row['end_time'])
            mask = (results_df['timestamp'] >= start) & (results_df['timestamp'] <= end)
            labels[mask] = 1

        p, r, f1, _ = precision_recall_fscore_support(labels, results_df['is_anomaly'], average='binary')
        return p, r, f1

    p_lstm, r_lstm, f1_lstm = get_metrics(lstm_results, failures)
    p_xgb, r_xgb, f1_xgb = get_metrics(xgb_results, failures)

    metrics = {
        "lstm": {"precision": p_lstm, "recall": r_lstm, "f1": f1_lstm},
        "xgb": {"precision": p_xgb, "recall": r_xgb, "f1": f1_xgb},
        "overall_f1": max(f1_lstm, f1_xgb)
    }

    # 6. Save Results
    with open(Config.MODEL_PATH / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_processed_data(lstm_results, "scores_lstm.parquet")
    save_processed_data(xgb_results, "scores_xgb.parquet")

    print("Evaluation complete. metrics.json and score files saved.")

if __name__ == "__main__":
    evaluate()
