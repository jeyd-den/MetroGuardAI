import streamlit as st
import pandas as pd
import numpy as np
from src.config import Config
from src.data_loader import load_processed_data
from src.failure_predictor import FailurePredictor

st.set_page_config(page_title="Overview", layout="wide")

st.title("🚇 System Overview")

# Load data
try:
    scores = load_processed_data("scores_lstm.parquet")
    predictor = FailurePredictor()

    # Calculate current state
    recent_scores = scores['score'].tail(50).values
    threshold = 0.1 # Default or load from json

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
    st.info("Please run the training and evaluation pipeline first to generate scores.")
