import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_processed_data, load_sensor_data
from src.preprocessor import clean_data, scale_features
from src.anomaly_detector import AnomalyDetector
from src.visualization import plot_anomaly_score, plot_sensor_contributions
from src.config import Config

st.set_page_config(page_title="Anomaly Detection", layout="wide")

st.title("🚨 Anomaly Detection")

try:
    # Load data and model
    scores_df = load_processed_data("scores_lstm.parquet")
    detector = AnomalyDetector()
    threshold = detector.thresholds.get("lstm_threshold", 0.1)

    # Plot scores
    fig_scores = plot_anomaly_score(scores_df, 'score', threshold)
    st.plotly_chart(fig_scores, use_container_width=True)

    # Analysis of a specific anomaly
    st.divider()
    st.subheader("Deep Dive: Anomaly Contribution")

    anomaly_idx = st.number_input("Select anomaly index to analyze",
                                  min_value=0,
                                  max_value=len(scores_df)-1,
                                  value=scores_df[scores_df['is_anomaly']].index[0] if any(scores_df['is_anomaly']) else 0)

    # To get contributions, we need the actual windowed data
    df = load_sensor_data()
    df = clean_data(df)
    scaled_df, _ = scale_features(df)

    from src.preprocessor import create_windows
    window_size = Config.load_model_config().get("lstm_autoencoder", {}).get("window_size", 30)
    X = create_windows(scaled_df, window_size)

    if anomaly_idx < len(X):
        window_data = X[anomaly_idx].reshape(1, window_size, -1)
        contributions = detector.get_sensor_contributions(window_data)[0]

        # Map to sensor names
        sensor_names = scaled_df.columns.tolist()
        contrib_dict = dict(zip(sensor_names, contributions))

        fig_contrib = plot_sensor_contributions(contrib_dict)
        st.plotly_chart(fig_contrib, use_container_width=True)
    else:
        st.error("Index out of range.")

except Exception as e:
    st.error(f"Error loading anomaly data: {e}")
