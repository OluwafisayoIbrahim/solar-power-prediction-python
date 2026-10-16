"""Refit a TensorFlow model using the settings from a completed comparison."""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import sklearn

from features import add_day_id, get_track_a_data, get_track_b_data, load_processed_data
from train_ann_tf import fit_tf, predict_tf
from predict_tf import load_tf_model, predict_measurements

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/tensorflow/20261014T104851616874Z"


def final_settings(config, settings, track):
    """Vote over inner-selected settings, without using outer test scores."""
    if config.get("smoke_test"):
        raise ValueError("A smoke-test run cannot supply deployment settings")
    rows = settings[(settings["track"] == track) &
                    (settings["model"] == "TensorFlow selected")].copy()
    expected = {(seed, fold) for seed in config["seeds"]
                for fold in range(1, config["folds"] + 1)}
    if len(rows) != len(expected) or set(zip(rows["seed"], rows["fold"])) != expected:
        raise ValueError(f"Track {track} does not have a complete set of fold settings")
    candidates = [tuple(units) for units in config["architectures"]]
    layers = [tuple(ast.literal_eval(value)) for value in rows["hidden"]]
    if any(layer not in candidates for layer in layers):
        raise ValueError("Saved architecture is absent from the run configuration")
    counts = Counter(layers)
    # Candidate order resolves ties deterministically, without consulting scores.
    hidden = max(candidates, key=lambda item: counts[item])
    epochs = rows.loc[[layer == hidden for layer in layers], "epochs"].to_numpy(float)
    if not np.isfinite(epochs).all() or (epochs < 1).any() or (epochs > config["epochs"]).any():
        raise ValueError("Invalid saved epoch counts")
    return hidden, int(np.median(epochs)), counts[hidden], len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--track", choices=["A", "B"], default="B")
    parser.add_argument("--output", type=Path, default=ROOT / "results/models")
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    if not (args.run / "comparison.csv").is_file():
        raise ValueError("The comparison run must be complete")
    settings = pd.read_csv(args.run / "selected_settings.csv")
    hidden, epochs, votes, total = final_settings(config, settings, args.track)
    prefix = f"track_{args.track.lower()}"
    names = {"model": f"{prefix}_ann_final.keras",
             "x_scaler": f"{prefix}_x_scaler.joblib",
             "y_scaler": f"{prefix}_y_scaler.joblib",
             "metadata": f"{prefix}_tf_metadata.json"}
    for name in names.values():
        if (args.output / name).exists():
            raise FileExistsError(f"Already exists: {args.output / name}. Use a new --output directory.")
    df = add_day_id(load_processed_data())
    if len(df) != config["rows"]:
        raise ValueError("Dataset row count differs from the comparison run")
    X_frame, y_series = (get_track_a_data if args.track == "A" else get_track_b_data)(df)
    X, y = X_frame.to_numpy(float), y_series.to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Training data must be finite")
    print(f"Track {args.track}: {hidden}, selected in {votes}/{total} folds; "
          f"refitting {len(y)} rows for {epochs} epochs, seed 0.", flush=True)
    fitted = fit_tf(X, y, hidden, 0, SimpleNamespace(**config), epochs=epochs)
    expected = predict_tf(fitted, X)
    if not np.isfinite(expected).all():
        raise ValueError("Final model produced non-finite predictions")
    args.output.mkdir(parents=True, exist_ok=True)
    fitted[0].save(args.output / names["model"])
    joblib.dump(fitted[1], args.output / names["x_scaler"])
    joblib.dump(fitted[2], args.output / names["y_scaler"])
    import tensorflow as tf
    metadata = {
        "track": args.track, "features": list(X_frame.columns), "target": "power",
        "prediction_unit": "W", "hidden_layers": hidden, "epochs": epochs, "seed": 0,
        "selection": "Most frequent inner-selected architecture across all seeds/folds; "
                     "candidate order breaks ties; median epochs for that architecture.",
        "architecture_votes": votes, "selection_folds": total,
        "source_run": str(args.run.resolve()), "rows": len(y),
        "training_data_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
        "activation": config["activation"], "learning_rate": config["learning_rate"],
        "l2": config["l2"], "batch_size": config["batch_size"],
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = args.output / names["metadata"]
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    loaded = load_tf_model(args.track, args.output)
    actual = predict_measurements(loaded, X_frame)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
    print(f"Verified save/load predictions on all {len(y)} rows.", flush=True)
    for name in names.values():
        print(args.output / name)


if __name__ == "__main__":
    main()
