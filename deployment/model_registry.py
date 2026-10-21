"""Explicit artifact locations and inference for the app's saved predictors."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from features import TRACK_A_FEATURES, TRACK_B_FEATURES
from predict_tf import load_tf_model, predict_measurements
from predict_torch import artifact_names, load_torch_model, predict_measurements as predict_torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "results/models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    track: str
    kind: str
    filename: str = ""

    @property
    def description(self):
        if self.kind == "physics":
            return ("Calculates power directly as current multiplied by voltage (P = I x V). "
                    "A simple electrical baseline with no training or learned parameters.")
        explanation = {
            "tensorflow": "A TensorFlow/Keras neural network that learns nonlinear relationships, trained with the Adam optimizer.",
            "pytorch": "A PyTorch neural network trained on CPU with Adam, using the same architecture candidates and day-grouped selection protocol as TensorFlow.",
            "tuned": "A scikit-learn neural network whose original hidden-layer size and regularization were selected using day-grouped cross-validation.",
            "untuned": "A scikit-learn neural network with a fixed 10-neuron hidden layer and light regularization, used as a reference without hyperparameter tuning.",
            "linear": "Fits a weighted sum of the inputs plus an intercept, providing a simple baseline for comparison with neural networks.",
        }[self.kind]
        inputs = ("Uses irradiance, temperature, humidity and hour, without current or voltage."
                  if self.track == "B" else
                  "Uses current and voltage alongside weather measurements to estimate power at the same time.")
        return f"{explanation} {inputs}"

    @property
    def features(self):
        if self.kind == "physics":
            return ["current", "voltage"]
        return TRACK_A_FEATURES if self.track == "A" else TRACK_B_FEATURES

    @property
    def model_dir(self):
        if self.kind in ("tensorflow", "pytorch"):
            return ROOT / "results/exports" / ("tf_s2" if self.kind == "tensorflow" else "torch_s2")
        return MODEL_DIR

    def missing_files(self, directory=None):
        directory = self.model_dir if directory is None else Path(directory)
        if self.kind == "physics":
            return []
        files = [self.filename]
        if self.kind == "tensorflow":
            prefix = f"track_{self.track.lower()}"
            files += [f"{prefix}_x_scaler.joblib", f"{prefix}_y_scaler.joblib",
                      f"{prefix}_tf_metadata.json"]
        elif self.kind == "pytorch":
            files = list(artifact_names(self.track).values())
        return [name for name in files if not (Path(directory) / name).is_file()]


MODELS = {}
for track in ("B", "A"):
    prefix = f"track_{track.lower()}"
    for kind, title, suffix in [
        ("tensorflow", "TensorFlow ANN", "ann_final.keras"),
        ("pytorch", "PyTorch ANN", "torch_ann_final.pt"),
        ("tuned", "Tuned scikit-learn ANN", "ann_final.joblib"),
        ("untuned", "Untuned scikit-learn ANN", "ann_untuned.joblib"),
        ("linear", "Linear regression", "linear_final.joblib"),
    ]:
        key = f"{track.lower()}_{kind}"
        MODELS[key] = ModelSpec(key, f"Track {track} - {title}", track, kind, f"{prefix}_{suffix}")
MODELS["a_physics"] = ModelSpec("a_physics", "Track A - Physics (current x voltage)", "A", "physics")


@dataclass
class LoadedModel:
    spec: ModelSpec
    estimator: object
    metadata: dict

    def predict(self, frame):
        X = frame.loc[:, self.spec.features].to_numpy(dtype=float)
        if not len(X) or not np.isfinite(X).all():
            raise ValueError("Provide finite measurements for every required input")
        if self.spec.kind == "tensorflow":
            result = predict_measurements(self.estimator, frame)
        elif self.spec.kind == "pytorch":
            result = predict_torch(self.estimator, frame)
        elif self.spec.kind == "physics":
            result = X[:, 0] * X[:, 1]
        else:
            result = self.estimator.predict(X)
        result = np.asarray(result).reshape(-1)
        if not np.isfinite(result).all():
            raise ValueError("Model produced non-finite predictions")
        return result


def load_model(key, directory=None):
    spec = MODELS[key]
    directory = spec.model_dir if directory is None else Path(directory)
    missing = spec.missing_files(directory)
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))
    if spec.kind == "tensorflow":
        estimator = load_tf_model(spec.track, directory)
        metadata = estimator[3]
        if metadata["features"] != spec.features:
            raise ValueError("Saved feature order does not match this track")
    elif spec.kind == "pytorch":
        estimator = load_torch_model(spec.track, directory)
        metadata = estimator[3]
    elif spec.kind == "physics":
        estimator, metadata = None, {"formula": "power = current * voltage"}
    else:
        estimator = joblib.load(Path(directory) / spec.filename)
        if estimator.n_features_in_ != len(spec.features):
            raise ValueError("Saved model input count does not match this track")
        metadata = {"features": spec.features}
    return LoadedModel(spec, estimator, metadata)


def predict_selected(model, frame):
    return model.predict(frame)
