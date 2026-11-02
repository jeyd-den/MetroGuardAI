import streamlit as st
import pandas as pd
from src.data_loader import load_sensor_data
from src.visualization import plot_sensor_timeseries
from src.preprocessor import clean_data

st.set_page_config(page_title="Sensor Monitoring", layout="wide")

st.title("📊 Sensor Monitoring")

try:
    df = load_sensor_data()
    df = clean_data(df)
    df = df.set_index('timestamp')

    sensor_cols = df.select_dtypes(include=['number']).columns.tolist()
    selected_sensors = st.multiselect("Select sensors to monitor", sensor_cols, default=sensor_cols[:3])

    if selected_sensors:
        # Time range filter
        start_date = st.date_input("Start Date", df.index.min())
        end_date = st.date_input("End Date", df.index.max())

        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        filtered_df = df.loc[mask]

        fig = plot_sensor_timeseries(filtered_df, selected_sensors)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one sensor.")

except Exception as e:
    st.error(f"Error loading sensors: {e}")
