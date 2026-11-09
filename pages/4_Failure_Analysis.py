import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_processed_data, load_failure_labels
from src.failure_predictor import FailurePredictor
from src.visualization import plot_health_trend

st.set_page_config(page_title="Failure Analysis", layout="wide")

st.title("📉 Failure Analysis")

# Initialize predictor
predictor = FailurePredictor()

# 1. Load Data - isolated to prevent total page crash
try:
    scores_df = load_processed_data("scores_lstm.parquet")
    failures_df = load_failure_labels()
except Exception as e:
    st.error(f"Critical Data Loading Error: {e}")
    st.stop()

# 2. Health Trend Section
st.subheader("System Health Timeline")
try:
    threshold = 0.1
    if 'score' in scores_df.columns:
        scores_values = scores_df['score'].values.flatten()
        health_series = [predictor.calculate_health_score([s], threshold) for s in scores_values]

        if health_series:
            fig_health = plot_health_trend(health_series)
            st.plotly_chart(fig_health, use_container_width=True)
        else:
            st.warning("Health series is empty.")
    else:
        st.error("Column 'score' not found in processed data.")
except Exception as e:
    st.error(f"Error generating health trend: {e}")
    health_series = []

# 3. Failure Window Analysis
st.divider()
st.subheader("Failure Window Analysis")
try:
    if not failures_df.empty:
        st.table(failures_df)
    else:
        st.info("No failure windows found in the dataset.")
except Exception as e:
    st.error(f"Error loading failure table: {e}")

# 4. Risk Analysis Section
st.divider()
st.subheader("Current Risk Assessment")
try:
    # Only run if we have a valid health series
    if 'health_series' in locals() and len(health_series) > 0:
        current_health = health_series[-1]

        # Calculate trend using the last 50 points
        recent_scores = scores_df['score'].tail(50).values if 'score' in scores_df.columns else []
        trend = predictor.analyze_trend(recent_scores)
        risk = predictor.predict_risk_level(current_health, trend)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Health Score", f"{current_health:.1f}%")
        with col2:
            risk_color = "red" if risk == "High" else "orange" if risk == "Medium" else "green"
            st.markdown(f"### Predicted Risk Level: :{risk_color}[{risk}]")
    else:
        st.warning("Cannot calculate risk: Health data is missing.")
except Exception as e:
    st.error(f"Error calculating risk level: {e}")
