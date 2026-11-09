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

    # Get available models from metrics keys
    # This makes the page dynamic: it will show whatever models exist in metrics.json
    models = [k for k in metrics.keys() if k != 'overall_f1']

    comp_data = {
        "Metric": ["Precision", "Recall", "F1-Score"]
    }

    for m in models:
        comp_data[m] = [
            metrics[m].get("precision", 0),
            metrics[m].get("recall", 0),
            metrics[m].get("f1", 0)
        ]

    comp_df = pd.DataFrame(comp_data)
    st.table(comp_df)

    # Performance Chart
    fig = go.Figure()
    for m in models:
        fig.add_trace(go.Bar(
            x=comp_df["Metric"],
            y=comp_df[m],
            name=m
        ))
    fig.update_layout(barmode='group', template="plotly_white", title="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown(f"**Overall Best F1 Score:** `{metrics['overall_f1']:.3f}`")

except Exception as e:
    st.error(f"Error loading metrics: {e}")
