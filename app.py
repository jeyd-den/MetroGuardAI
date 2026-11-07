import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import Config
from src.data_loader import load_processed_data
from src.failure_predictor import FailurePredictor

# 1. CONFIGURATION & ARTIFACTS
ARTIFACTS = {
    "models/lstm_autoencoder.h5": "https://drive.google.com/uc?id=17Bjut3pqMogUxGX1rQX8eQ18yjqSfGqP",
    "models/scaler.joblib": "https://drive.google.com/uc?id=1EGic5Gu0gt3J77gIHNpGwARUmuEvouee",
    "data/raw/MetroPT.csv": "https://drive.google.com/uc?id=1Wm5ijAzbDBI4GuJuOQkqBrsKHQ8YB51X",
    "data/raw/failure_windows.csv": "https://drive.google.com/uc?id=1ooJx3p5u7_K5OaTG0f4sweVZMlhmyApl"
}

def download_file(url, destination):
    """Downloads a file from a URL to a local path."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

def bootstrap_artifacts():
    """Checks for necessary files and downloads them if missing."""
    missing = False
    for path, url in ARTIFACTS.items():
        if not os.path.exists(path):
            # st.write(f"Downloading {path}...")
            if not download_file(url, path):
                st.error(f"Failed to download {path} from cloud storage.")
                missing = True
    return not missing

# --- Initialization ---
st.set_page_config(page_title="MetroGuard AI", layout="wide")

# Bootstrap the app
with st.spinner("Initializing AI Engine and Loading Artifacts..."):
    if not bootstrap_artifacts():
        st.error("Critical error: Could not download required models or data. Please check links.")
        st.stop()

st.title("🚇 MetroGuard AI: Transit Infrastructure Monitoring")
st.markdown("""
Welcome to **MetroGuard AI**, an intelligent anomaly detection and failure prediction system for transit infrastructure.
Use the sidebar to navigate through different monitoring and analysis pages.
""")

try:
    # Load data for the overview
    scores = load_processed_data("scores_lstm.parquet")
    predictor = FailurePredictor()

    # Calculate current state
    recent_scores = scores['score'].tail(50).values
    threshold = 0.1 # Default threshold

    health = predictor.calculate_health_score(recent_scores, threshold)
    trend = predictor.analyze_trend(recent_scores)
    risk = predictor.predict_risk_level(health, trend)

    # UI Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("System Health", f"{health:.1f}%", delta=f"{trend:.3f}")

    with col2:
        risk_color = "red" if risk == "High" else "orange" if risk == "Medium" else "green"
        st.markdown(f"### Risk Level: :{risk_color}[{risk}]")

    with col3:
        active_anomalies = scores['is_anomaly'].tail(100).sum()
        st.metric("Recent Anomalies", int(active_anomalies))

    st.divider()

    st.subheader("Fleet Status")
    # Mock fleet table
    fleet_data = pd.DataFrame({
        "Unit ID": ["Unit-01", "Unit-02", "Unit-03", "Unit-04"],
        "Health": [98, 85, 42, 91],
        "Status": ["Healthy", "Healthy", "Warning", "Healthy"],
        "Last Check": ["10m ago", "12m ago", "5m ago", "11m ago"]
    })
    st.table(fleet_data)

except Exception as e:
    st.error(f"Error loading overview data: {e}")
    st.info("The data files are being downloaded. Please refresh the page in a few seconds.")
