import numpy as np
import pandas as pd
from typing import List, Tuple
from .config import Config

class FailurePredictor:
    """
    Predicts system health and risk levels based on anomaly trends.
    """
    def __init__(self) -> None:
        self.config = Config.load_model_config()

    def calculate_health_score(self, recent_scores: List[float], threshold: float) -> float:
        """
        Calculates a health score from 0 to 100.
        Formula: Health = 100 * exp(- (current_score / (threshold * 2)))

        Args:
            recent_scores: List of recent anomaly scores.
            threshold: The calibrated anomaly threshold.
        """
        if not recent_scores:
            return 100.0

        current_score = np.mean(recent_scores)
        health = 100 * np.exp(-(current_score / (threshold * 2)))
        return float(np.clip(health, 0, 100))

    def predict_risk_level(self, health_score: float, trend: float) -> str:
        """
        Determines risk level based on current health and trend.

        Args:
            health_score: Current health (0-100).
            trend: Linear slope of health (positive = improving).
        """
        if health_score < 40 or (health_score < 60 and trend < -0.05):
            return "High"
        elif health_score < 75 or trend < -0.02:
            return "Medium"
        else:
            return "Low"

    def analyze_trend(self, scores: List[float]) -> float:
        """
        Analyzes the linear trend of the anomaly scores.

        Args:
            scores: List of anomaly scores over time.

        Returns:
            The slope of the health trend (negative = declining).
        """
        if len(scores) < 2:
            return 0.0

        x = np.arange(len(scores))
        # Using polyfit to find the slope of the error
        slope, _ = np.polyfit(x, scores, 1)

        # If error slope is positive, health is declining (negative trend)
        return float(-slope)
