import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Base Directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_RAW_PATH = Path(os.getenv("DATA_RAW_PATH", BASE_DIR / "data/raw"))
    DATA_PROCESSED_PATH = Path(os.getenv("DATA_PROCESSED_PATH", BASE_DIR / "data/processed"))
    MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models"))

    # File Paths
    RAW_SENSORS_CSV = DATA_RAW_PATH / "MetroPT.csv"
    FAILURE_WINDOWS_CSV = DATA_RAW_PATH / "failure_windows.csv"
    MODEL_CONFIG_JSON = MODEL_PATH / "model_config.json"
    METRICS_JSON = MODEL_PATH / "metrics.json"

    @staticmethod
    def load_model_config():
        """Loads model hyperparameters and windowing settings."""
        if not Config.MODEL_CONFIG_JSON.exists():
            raise FileNotFoundError(f"Model config not found at {Config.MODEL_CONFIG_JSON}")
        with open(Config.MODEL_CONFIG_JSON, "r") as f:
            return json.load(f)
