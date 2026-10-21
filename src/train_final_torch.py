"""Export a new CPU PyTorch model from completed inner-selection settings."""

import argparse
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
from predict_torch import artifact_names, load_torch_model, predict_measurements
from train_ann_torch import fit_torch, predict_torch
from train_final_tf import final_settings as shared_final_settings


def final_settings(config, settings, track):
    return shared_final_settings(config, settings, track, model_label="PyTorch selected")


def save_bundle(fitted, metadata, output):
    """Refuse collisions before writing any of the four artifacts."""
    import torch
    output = Path(output)
    names = artifact_names(metadata["track"])
    for name in names.values():
        if (output / name).exists():
            raise FileExistsError(f"Already exists: {output / name}. Use a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(fitted[0].state_dict(), output / names["model"])
    joblib.dump(fitted[1], output / names["x_scaler"])
    joblib.dump(fitted[2], output / names["y_scaler"])
    (output / names["metadata"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return names


def export_final(run, track, output):
    import torch
    run, output = Path(run), Path(output)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if "torch" not in config or not (run / "comparison.csv").is_file():
        raise ValueError("A completed PyTorch comparison is required")
    settings = pd.read_csv(run / "selected_settings.csv")
    hidden, epochs, votes, total = final_settings(config, settings, track)
    # Check before the potentially long fit, as well as at save time.
    for name in artifact_names(track).values():
        if (output / name).exists():
            raise FileExistsError(f"Already exists: {output / name}. Use a new --output directory.")
    df = add_day_id(load_processed_data())
    if len(df) != config["rows"]:
        raise ValueError("Dataset row count differs from the comparison run")
    X_frame, y_series = (get_track_a_data if track == "A" else get_track_b_data)(df)
    X, y = X_frame.to_numpy(float), y_series.to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Training data must be finite")
    print(f"Track {track}: {hidden}, {votes}/{total} votes; {epochs} epochs, seed 0.", flush=True)
    fitted = fit_torch(X, y, hidden, 0, SimpleNamespace(**config), epochs=epochs)
    expected = predict_torch(fitted, X)
    metadata = {
        "track": track, "features": list(X_frame.columns), "target": "power",
        "prediction_unit": "W", "hidden_layers": hidden, "epochs": epochs,
        "epoch_cap": config["epochs"], "seed": 0, "architecture_votes": votes,
        "selection_folds": total, "source_run": str(run.resolve()), "rows": len(y),
        "selection": "Most frequent inner-selected architecture; candidate order breaks ties; "
                     "median epochs for that architecture.",
        "training_data_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
        "activation": config["activation"], "learning_rate": config["learning_rate"],
        "l2": config["l2"], "batch_size": config["batch_size"], "device": "cpu",
        "python": platform.python_version(), "torch": torch.__version__,
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    names = save_bundle(fitted, metadata, output)
    loaded = load_torch_model(track, output)
    np.testing.assert_allclose(predict_measurements(loaded, X_frame), expected, rtol=1e-6, atol=1e-5)
    print(f"Verified save/load predictions on all {len(y)} rows.", flush=True)
    for name in names.values():
        print(output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--track", choices=["A", "B"], default="B")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_final(args.run, args.track, args.output)


if __name__ == "__main__":
    main()
