import pytest
import numpy as np
from src.anomaly_detector import AnomalyDetector
from src.config import Config
import json

def test_persistence_rule():
    detector = AnomalyDetector()
    # Threshold is usually 0.1.
    scores = [0.05, 0.2, 0.2, 0.2, 0.05] # 3 consecutive anomalies
    # window=3
    result = detector.apply_persistence_rule(scores, window=3)
    assert result[3] == True
    assert result[0] == False
