"""Load trusted local TensorFlow artifacts and predict power in watts."""

import json
from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "results/models"


def load_tf_model(track="B", model_dir=MODEL_DIR):
    """Load once and reuse this bundle for multiple predictions."""
    if track not in ("A", "B"):
        raise ValueError("track must be A or B")
    import tensorflow as tf
    folder = Path(model_dir)
    prefix = f"track_{track.lower()}"
    metadata = json.loads((folder / f"{prefix}_tf_metadata.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(folder / f"{prefix}_ann_final.keras", compile=False)
    x_scaler = joblib.load(folder / f"{prefix}_x_scaler.joblib")
    y_scaler = joblib.load(folder / f"{prefix}_y_scaler.joblib")
    return model, x_scaler, y_scaler, metadata


def predict_measurements(bundle, measurements):
    """Accept a DataFrame of named features; return one watt prediction per row."""
    model, x_scaler, y_scaler, metadata = bundle
    required = metadata["features"]
    missing = set(required) - set(measurements.columns)
    if missing:
        raise ValueError(f"Missing features: {', '.join(sorted(missing))}")
    X = measurements.loc[:, required].to_numpy(dtype=float)
    if len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("Provide at least one row of finite measurements")
    scaled = x_scaler.transform(X).astype("float32")
    predicted = model(scaled, training=False).numpy()
    result = y_scaler.inverse_transform(predicted).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("Model produced non-finite predictions")
    return result
