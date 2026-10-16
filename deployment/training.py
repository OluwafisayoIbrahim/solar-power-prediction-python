"""Session-only upload training with a separate holdout evaluation."""

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from batch_prediction import validate_rows
from model_registry import LoadedModel


def train_uploaded(frame, original, group_column=None, epochs=100, progress=None):
    """Fit only on the training partition; never fit scalers on held-out rows."""
    if original.spec.kind == "physics":
        raise ValueError("The physics formula has no trainable parameters. Use saved-model mode.")
    if "power" not in frame:
        raise ValueError("Training requires measured power in a 'power' column.")
    if len(frame) > 10_000:
        raise ValueError("Limit training uploads to 10,000 rows.")
    if not 1 <= epochs <= 500:
        raise ValueError("Epochs must be between 1 and 500.")
    X, issues = validate_rows(frame, original.spec.features)
    power = pd.to_numeric(frame["power"], errors="coerce")
    bad_target = ~np.isfinite(power) | (power < 0)
    issues.loc[bad_target] = issues.loc[bad_target] + "; power: must be finite and nonnegative"
    valid = issues.eq("")
    indices = frame.index[valid].to_numpy()
    if len(indices) < 10:
        raise ValueError("At least 10 valid rows with measured power are required.")
    if group_column:
        groups = frame.loc[valid, group_column]
        if group_column == "timestamp":
            groups = pd.to_datetime(groups, errors="coerce", format="mixed", utc=True).dt.date
        if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
            raise ValueError("Every valid training row needs a valid group/date.")
        if groups.nunique() < 3:
            raise ValueError("Grouped training needs at least three distinct groups/days.")
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
            indices, groups=groups.to_numpy()))
        train_indices, validation_indices = indices[tr], indices[va]
    else:
        train_indices, validation_indices = train_test_split(indices, test_size=0.2, random_state=0)
    if min(len(train_indices), len(validation_indices)) < 2:
        raise ValueError("The split must contain at least two training and two held-out rows.")
    x_train = X.loc[train_indices].to_numpy(float)
    y_train = power.loc[train_indices].to_numpy(float)
    metadata = dict(original.metadata)
    metadata.update(features=list(original.spec.features), origin="uploaded data",
                    rows=len(train_indices), validation_rows=len(validation_indices),
                    split_group=group_column, seed=0,
                    sklearn=sklearn.__version__, numpy=np.__version__,
                    note="Trained only on training partition; evaluation rows remain held out")
    metadata.pop("source_run", None)
    convergence = 0
    if original.spec.kind == "tensorflow":
        import tensorflow as tf
        from models_tf import create_tf_ann
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        scaled_x = x_scaler.fit_transform(x_train).astype("float32")
        scaled_y = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype("float32")
        model = create_tf_ann(len(original.spec.features), tuple(metadata["hidden_layers"]),
                              metadata["activation"], metadata["learning_rate"], metadata["l2"], 0)

        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress:
                    progress((epoch + 1) / epochs)

        history = model.fit(scaled_x, scaled_y, epochs=epochs,
                            batch_size=metadata.get("batch_size", 32), verbose=0,
                            callbacks=[Progress()]).history
        metadata.update(epochs=epochs, training_loss=history["loss"])
        estimator = (model, x_scaler, y_scaler, metadata)
    else:
        estimator = clone(original.estimator)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(x_train, y_train)
        convergence = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        for item in caught:
            if not issubclass(item.category, ConvergenceWarning):
                warnings.warn(str(item.message), item.category)
        if progress:
            progress(1.0)
    loaded = LoadedModel(original.spec, estimator, metadata)
    result = frame.loc[validation_indices].copy().sort_index()
    result.insert(0, "source_row", result.index + 2)
    result["predicted_power_w"] = loaded.predict(X.loc[result.index])
    result["prediction_status"] = "predicted"
    result["prediction_issue"] = ""
    result["prediction_model"] = original.spec.label + " (uploaded training data)"
    rejected = frame.loc[~valid].copy()
    rejected["training_issue"] = issues.loc[~valid]
    return loaded, result, rejected, convergence


def export_trained(loaded):
    """Return a ZIP; temporary serialization files are removed immediately."""
    output = BytesIO()
    with TemporaryDirectory(prefix="solar-model-") as temporary:
        folder = Path(temporary)
        prefix = f"track_{loaded.spec.track.lower()}"
        if loaded.spec.kind == "tensorflow":
            model, x_scaler, y_scaler, _ = loaded.estimator
            model.save(folder / f"{prefix}_ann_final.keras")
            joblib.dump(x_scaler, folder / f"{prefix}_x_scaler.joblib")
            joblib.dump(y_scaler, folder / f"{prefix}_y_scaler.joblib")
            metadata_name = f"{prefix}_tf_metadata.json"
        else:
            joblib.dump(loaded.estimator, folder / loaded.spec.filename)
            metadata_name = "metadata.json"
        (folder / metadata_name).write_text(json.dumps(loaded.metadata, indent=2), encoding="utf-8")
        with ZipFile(output, "w") as archive:
            for path in folder.iterdir():
                archive.write(path, path.name)
    return output.getvalue()
