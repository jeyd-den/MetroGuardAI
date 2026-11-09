import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_processed_data, load_sensor_data
from src.preprocessor import clean_data, scale_features, create_windows
from src.anomaly_detector import AnomalyDetector
from src.visualization import plot_anomaly_score, plot_sensor_contributions
from src.config import Config

st.set_page_config(page_title="Anomaly Detection", layout="wide")

st.title("🚨 Anomaly Detection")

try:
    # 1. Load data and model
    scores_df = load_processed_data("scores_lstm.parquet")
    detector = AnomalyDetector()
    threshold = detector.thresholds.get("lstm_threshold", 0.1)

    # 2. Global Anomaly Score Chart
    st.subheader("System-Wide Anomaly Scores")
    fig_scores = plot_anomaly_score(scores_df, 'score', threshold)
    st.plotly_chart(fig_scores, use_container_width=True)

    # 3. Deep Dive Section
    st.divider()
    st.subheader("🔍 Deep Dive: Anomaly Contribution")
    st.markdown("Select a specific window index to see which sensors contributed most to the reconstruction error.")

    # Use a column layout for the input and the result
    col_input, col_output = st.columns([1, 3])

    with col_input:
        # Default index: the first detected anomaly
        if any(scores_df['is_anomaly']):
            default_idx = int(scores_df[scores_df['is_anomaly']].index[0])
        else:
            default_idx = 0

        anomaly_idx = st.number_input(
            "Select Window Index",
            min_value=0,
            max_value=len(scores_df)-1,
            value=default_idx
        )

    with col_output:
        # To get contributions, we need the actual windowed data
        df = load_sensor_data()
        df = clean_data(df)
        scaled_df, _ = scale_features(df)

        window_size = Config.load_model_config().get("lstm_autoencoder", {}).get("window_size", 60)
        X = create_windows(scaled_df, window_size)

        if anomaly_idx < len(X):
            # Extract the specific window
            window_data = X[anomaly_idx].reshape(1, window_size, -1)

            # Calculate contributions
            contributions = detector.get_sensor_contributions(window_data)

            # Map to sensor names
            sensor_names = scaled_df.columns.tolist()
            contrib_dict = dict(zip(sensor_names, contributions))

            # Sort contributions for better visualization
            sorted_contribs = dict(sorted(contrib_dict.items(), key=lambda item: item[1], reverse=True))

            # 1. Display the Chart
            fig_contrib = plot_sensor_contributions(sorted_contribs)
            st.plotly_chart(fig_contrib, use_container_width=True)

            # 2. Display a detailed table for verification
            with st.expander("View Exact Error Values"):
                contrib_df = pd.DataFrame({
                    "Sensor": list(sorted_contribs.keys()),
                    "Reconstruction Error": list(sorted_contribs.values())
                })
                st.table(contrib_df)
        else:
            st.error(f"Index {anomaly_idx} is out of range for the windowed dataset.")

except Exception as e:
    st.error(f"Error loading anomaly data: {e}")
    st.exception(e)
