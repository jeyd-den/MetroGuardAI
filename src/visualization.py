import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_sensor_timeseries(df, sensor_cols, anomaly_indices=None):
    """
    Creates a multi-line chart for sensor readings with optional anomaly shading.
    """
    fig = go.Figure()

    for col in sensor_cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode='lines'))

    if anomaly_indices is not None:
        # Add red vertical lines or shading for anomalies
        for idx in anomaly_indices:
            fig.add_vline(x=idx, line_width=1, line_dash="dash", line_color="red")

    fig.update_layout(
        title="Sensor Readings Over Time",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_anomaly_score(df, score_col, threshold):
    """
    Plots the anomaly score vs threshold.
    """
    fig = px.line(df, x=df.index, y=score_col, title="Anomaly Score vs Threshold")
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig.update_layout(template="plotly_white")
    return fig

def plot_health_trend(health_history):
    """
    Plots the health score trend.
    """
    fig = px.area(x=np.arange(len(health_history)), y=health_history,
                  title="System Health Trend", labels={'x': 'Time', 'y': 'Health Score'})
    fig.update_layout(yaxis_range=[0, 100], template="plotly_white")
    return fig

def plot_sensor_contributions(contributions):
    """
    Bar chart showing which sensor contributed most to the error.
    """
    fig = px.bar(x=list(contributions.keys()), y=list(contributions.values()),
                 title="Sensor Contribution to Anomaly", labels={'x': 'Sensor', 'y': 'Error Contribution'})
    fig.update_layout(template="plotly_white")
    return fig
