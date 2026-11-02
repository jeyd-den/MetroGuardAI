import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from src.config import Config
from src.data_loader import load_sensor_data
from src.preprocessor import clean_data, scale_features

def train():
    # 1. Load Config
    model_cfg = Config.load_model_config()
    iforest_cfg = model_cfg.get("isolation_forest", {})
    n_estimators = iforest_cfg.get("n_estimators", 100)
    contamination = iforest_cfg.get("contamination", "auto")

    # 2. Load and Preprocess
    print("Loading and preprocessing data...")
    df = load_sensor_data()
    df = clean_data(df)

    # Scale (Isolation Forest is less sensitive but scaling is good for consistency)
    scaled_df, _ = scale_features(df)

    # 3. Model Training
    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42
    )
    model.fit(scaled_df)

    # 4. Save Model
    model.save_path = Config.MODEL_PATH / "isolation_forest.joblib"
    joblib.dump(model, model.save_path)

    print(f"Training complete. Model saved to {model.save_path}")

if __name__ == "__main__":
    train()
