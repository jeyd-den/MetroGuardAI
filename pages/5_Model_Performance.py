import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from src.config import Config

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("🏆 Model Performance")

try:
    with open(Config.MODEL_PATH / "metrics.json", "r") as f:
        metrics = json.load(f)

    # Comparison Table
    st.subheader("Metric Comparison")
    comp_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score"],
        "LSTM Autoencoder": [
            metrics["lstm"]["precision"],
            metrics["lstm"]["recall"],
            metrics["lstm"]["f1"]
        ],
        "Isolation Forest": [
            metrics["iforest"]["precision"],
            metrics["iforest"]["recall"],
            metrics["iforest"]["f1"]
        ]
    })
    st.table(comp_df)

    # Performance Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp_df["Metric"],
        y=comp_df["LSTM Autoencoder"],
        name="LSTM"
    ))
    fig.add_trace(go.Bar(
        x=comp_df["Metric"],
        y=comp_df["Isolation Forest"],
        name="iForest"
    ))
    fig.update_layout(barmode='group', template="plotly_white", title="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown(f"**Overall Best F1 Score:** `{metrics['overall_f1']:.3f}`")

except Exception as e:
    st.error(f"Error loading metrics: {e}")
