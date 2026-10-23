"""Load trusted CPU PyTorch artifacts and predict power in watts."""

import json
from pathlib import Path

import joblib
import numpy as np

from features import TRACK_A_FEATURES, TRACK_B_FEATURES
from train_ann_torch import predict_torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "results/models"


def artifact_names(track):
    if track not in ("A", "B"):
        raise ValueError("track must be A or B")
    prefix = f"track_{track.lower()}_torch"
    return {"model": f"{prefix}_ann_final.pt", "x_scaler": f"{prefix}_x_scaler.joblib",
            "y_scaler": f"{prefix}_y_scaler.joblib", "metadata": f"{prefix}_metadata.json"}


def load_torch_model(track="B", model_dir=MODEL_DIR):
    import torch
    from models_torch import TorchANN

    names, folder = artifact_names(track), Path(model_dir)
    metadata = json.loads((folder / names["metadata"]).read_text(encoding="utf-8"))
    features = TRACK_A_FEATURES if track == "A" else TRACK_B_FEATURES
    if metadata["track"] != track or metadata["features"] != features:
        raise ValueError("Metadata track or features do not match the requested model")
    model = TorchANN(len(features), metadata["hidden_layers"], metadata["activation"],
                     metadata["learning_rate"], metadata["l2"], metadata["seed"])
    state = torch.load(folder / names["model"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    x_scaler = joblib.load(folder / names["x_scaler"])
    y_scaler = joblib.load(folder / names["y_scaler"])
    if x_scaler.n_features_in_ != len(features) or y_scaler.n_features_in_ != 1:
        raise ValueError("Scaler dimensions do not match metadata")
    return model, x_scaler, y_scaler, metadata


def predict_measurements(bundle, measurements):
    required = bundle[3]["features"]
    if not measurements.columns.is_unique:
        raise ValueError("Duplicate feature columns")
    missing = set(required) - set(measurements.columns)
    if missing:
        raise ValueError(f"Missing features: {', '.join(sorted(missing))}")
    X = measurements.loc[:, required].to_numpy(dtype=float)
    if not len(X) or not np.isfinite(X).all():
        raise ValueError("Provide at least one row of finite measurements")
    return predict_torch(bundle, X)
