"""Explicit registry of the project's saved predictors; no training in the app."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from features import TRACK_A_FEATURES, TRACK_B_FEATURES
from predict_tf import load_tf_model, predict_measurements

MODEL_DIR = Path(__file__).resolve().parents[1] / "results/models"


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

    def missing_files(self, directory=MODEL_DIR):
        if self.kind == "physics":
            return []
        files = [self.filename]
        if self.kind == "tensorflow":
            prefix = f"track_{self.track.lower()}"
            files += [f"{prefix}_x_scaler.joblib", f"{prefix}_y_scaler.joblib",
                      f"{prefix}_tf_metadata.json"]
        return [name for name in files if not (Path(directory) / name).is_file()]


MODELS = {}
for track in ("B", "A"):
    prefix = f"track_{track.lower()}"
    for kind, title, suffix in [
        ("tensorflow", "TensorFlow ANN", "ann_final.keras"),
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
        elif self.spec.kind == "physics":
            result = X[:, 0] * X[:, 1]
        else:
            result = self.estimator.predict(X)
        result = np.asarray(result).reshape(-1)
        if not np.isfinite(result).all():
            raise ValueError("Model produced non-finite predictions")
        return result


def load_model(key, directory=MODEL_DIR):
    spec = MODELS[key]
    missing = spec.missing_files(directory)
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))
    if spec.kind == "tensorflow":
        estimator = load_tf_model(spec.track, directory)
        metadata = estimator[3]
        if metadata["features"] != spec.features:
            raise ValueError("Saved feature order does not match this track")
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
