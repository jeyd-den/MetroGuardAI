import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_processed_data, load_failure_labels
from src.failure_predictor import FailurePredictor
from src.visualization import plot_health_trend

st.set_page_config(page_title="Failure Analysis", layout="wide")

st.title("📉 Failure Analysis")

try:
    scores_df = load_processed_data("scores_lstm.parquet")
    failures_df = load_failure_labels()
    predictor = FailurePredictor()

    # Health Trend
    st.subheader("System Health Timeline")
    # Compute health for the whole series
    threshold = 0.1
    health_series = [predictor.calculate_health_score([s], threshold) for s in scores_df['score']]

    fig_health = plot_health_trend(health_series)
    st.plotly_chart(fig_health, use_container_width=True)

    # Failure Overlays
    st.divider()
    st.subheader("Failure Window Analysis")
    st.table(failures_df)

    # Risk Analysis
    current_health = health_series[-1]
    trend = predictor.analyze_trend(scores_df['score'].tail(50).values)
    risk = predictor.predict_risk_level(current_health, trend)

    st.markdown(f"### Current Predicted Risk: **{risk}**")

except Exception as e:
    st.error(f"Error loading failure analysis: {e}")
