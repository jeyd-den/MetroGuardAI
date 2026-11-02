import streamlit as st
import joblib
import tensorflow as tf
from pathlib import Path

@st.cache_resource
def load_model_cached(path):
    """Loads a model and caches it in the Streamlit session."""
    path = Path(path)
    if path.suffix in [".keras", ".h5"]:
        return tf.keras.models.load_model(path, compile=False)
    elif path.suffix in [".joblib", ".pkl"]:
        return joblib.load(path)
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")
