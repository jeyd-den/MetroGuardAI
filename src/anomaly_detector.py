import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import json
from typing import Tuple, List
from .config import Config
from .model_loader import load_model_cached

class AnomalyDetector:
    """
    Handles anomaly detection using a hybrid approach with LSTM Autoencoders
    and XGBoost classifiers.
    """
    def __init__(self) -> None:
        self.config = Config.load_model_config()
        self.thresholds = self._load_thresholds()
        self.lstm_model = load_model_cached(Config.MODEL_PATH / "lstm_autoencoder.h5")
        self.xgb_model = joblib.load(Config.MODEL_PATH / "xgb_baseline.joblib")

    def _load_thresholds(self) -> dict:
        """Loads threshold configurations from JSON."""
        try:
            with open(Config.MODEL_PATH / "anomaly_threshold.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"lstm_threshold": 0.1}

    def detect_lstm_anomaly(self, windowed_data: np.ndarray) -> Tuple[float, bool]:
        """
        Scores a window and checks against the calibrated threshold.

        Args:
            windowed_data: Input window of shape (1, window_size, features).

        Returns:
            A tuple of (mse_score, is_anomaly).
        """
        prediction = self.lstm_model.predict(windowed_data, verbose=0)
        mse = float(np.mean(np.power(windowed_data - prediction, 2)))
        threshold = self.thresholds.get("lstm_threshold", 0.1)
        return mse, mse > threshold

    def detect_xgb_anomaly(self, scaled_data: np.ndarray) -> Tuple[float, bool]:
        """
        Scores a sample using the XGBoost classifier.

        Args:
            scaled_data: Input sample of shape (1, features).

        Returns:
            A tuple of (probability_score, is_anomaly).
        """
        # predict_proba returns [prob_class_0, prob_class_1]
        score = float(self.xgb_model.predict_proba(scaled_data)[0][1])
        return score, score > 0.5

    def apply_persistence_rule(self, scores: List[float], window: int = 3) -> np.ndarray:
        """
        Filters anomalies using a persistence rule to reduce false alarms.
        An anomaly is only valid if detected for 'window' consecutive steps.

        Args:
            scores: List of anomaly scores over time.
            window: Required consecutive anomalies to trigger.

        Returns:
            Boolean array of validated anomalies.
        """
        threshold = self.thresholds.get("lstm_threshold", 0.1)
        anomalies = np.array(scores) > threshold
        valid_anomalies = np.convolve(anomalies, np.ones(window), mode='valid') >= window
        # Pad the start to maintain original array length
        return np.concatenate([np.zeros(window-1), valid_anomalies])

    def get_sensor_contributions(self, windowed_data: np.ndarray) -> np.ndarray:
        """
        Identifies which sensor contributed most to the reconstruction error.

        Args:
            windowed_data: Input window of shape (1, window_size, features).

        Returns:
            An array of errors per feature.
        """
        prediction = self.lstm_model.predict(windowed_data, verbose=0)
        # MSE per feature over the window
        errors = np.mean(np.power(windowed_data - prediction, 2), axis=1)
        # Sum errors across the time window to get total contribution per feature
        total_error_per_feature = np.sum(errors, axis=0)
        return total_error_per_feature
