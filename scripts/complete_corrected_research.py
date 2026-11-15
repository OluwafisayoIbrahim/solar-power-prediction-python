"""Finish corrected experiments after S1/S2; never promote partial evidence.

Run with the project virtualenv. --stage sklearn is independent of TF selection.
--stage frameworks waits for the two separately launched full TensorFlow jobs,
applies the documented rule, runs PyTorch and exports both frameworks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corrected_protocol import COLLECTION, data_digest, choose_tensorflow, verify_run


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def command(script, *args):
    name = Path(script).stem
    log = COLLECTION / f"{name}_{time.time_ns()}.log"
    env = os.environ.copy()
    env.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    env.setdefault("TF_NUM_INTEROP_THREADS", "2")
    print(f"Running {script}; log {log.relative_to(ROOT)}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-u", str(ROOT / script), *map(str, args)],
                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def sklearn_exports():
    import joblib
    import numpy as np
    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from features import add_day_id, load_processed_data, get_track_a_data, get_track_b_data
    from models import create_ann_model
    from train_final_model import select_final_hyperparameters

    directory = COLLECTION / "exports/sklearn"
    directory.mkdir(parents=True, exist_ok=False)
    frame = add_day_id(load_processed_data())
    for track, getter in [("a", get_track_a_data), ("b", get_track_b_data)]:
        X_frame, y_frame = getter(frame)
        X, y = X_frame.to_numpy(float), y_frame.to_numpy(float)
        hidden, alpha = select_final_hyperparameters(X, y, frame.day_id.to_numpy())
        for suffix, model in [
            ("linear_final", make_pipeline(StandardScaler(), LinearRegression())),
            ("ann_untuned", create_ann_model(10, 0.0001, 0)),
            ("ann_final", create_ann_model(hidden, alpha, 0)),
        ]:
            model.fit(X, y)
            path = directory / f"track_{track}_{suffix}.joblib"
            joblib.dump(model, path)
            expected = model.predict(X)
            if not np.isfinite(expected).all():
                raise ValueError("Nonfinite inference output")
            np.testing.assert_allclose(joblib.load(path).predict(X), expected, rtol=1e-10, atol=1e-10)
            write_json(path.with_suffix(".json"), {
                "track": track.upper(), "features": list(X_frame.columns), "rows": 180,
                "groups": 20, "training_data_sha256": data_digest(), "seed": 0,
                "sklearn": sklearn.__version__, "reload_verified": True,
                "selection": "Original full-data final-fit grid with seven grouped folds for tuned ANN; fixed settings for other baselines",
                "purpose": "Full-data inference only; held-out research scores come from separate nested evaluation",
            })
    write_json(COLLECTION / "sklearn_exports_complete.json", {"training_data_sha256": data_digest()})


def frameworks(wait_hours):
    s1, s2 = [COLLECTION / f"tensorflow_{candidate}" for candidate in ("s1", "s2")]
    deadline = time.monotonic() + wait_hours * 3600
    while not all((path / "comparison.csv").is_file() for path in (s1, s2)):
        if time.monotonic() > deadline:
            raise TimeoutError("TensorFlow S1/S2 incomplete; no corrected evidence promoted")
        write_json(COLLECTION / "progress.json", {
            "stage": "waiting_for_full_tensorflow_s1_s2", "complete": False,
            "training_data_sha256": data_digest(),
            "selected_curve_sets": {p.name: len(list((p / "learning_curves").glob("*.png"))) for p in (s1, s2)},
        })
        time.sleep(30)
    for path in (s1, s2):
        verify_run(path, "TensorFlow selected")
    selection = choose_tensorflow(s1, s2)
    write_json(COLLECTION / "selection.json", selection)
    selected = COLLECTION / f"tensorflow_{selection['selected'].lower()}"
    batch = 32 if selection["selected"] == "S1" else 16
    pytorch = COLLECTION / "pytorch_selected"
    flags = ["--tracks", "A", "B", "--seeds", "0", "1", "2", "--epochs", "1000",
             "--learning-rate", "0.01", "--batch-size", str(batch), "--patience", "30"]
    write_json(COLLECTION / "progress.json", {"stage": "pytorch", "complete": False, **selection})
    command("src/train_ann_torch.py", *flags, "--output", pytorch)
    verify_run(pytorch, "PyTorch selected")
    comparison = COLLECTION / "comparison"
    command("src/compare_frameworks.py", "--tf-run", selected, "--torch-run", pytorch, "--output", comparison)
    protocol = json.loads((comparison / "protocol.json").read_text(encoding="utf-8"))
    if protocol["warnings"]:
        raise ValueError(f"Matched comparison warnings: {protocol['warnings']}")
    for track in ["A", "B"]:
        command("src/train_final_tf.py", "--run", selected, "--track", track,
                "--output", COLLECTION / "exports/tensorflow")
        command("src/train_final_torch.py", "--run", pytorch, "--track", track,
                "--output", COLLECTION / "exports/pytorch")
    marker = COLLECTION / "sklearn_exports_complete.json"
    if not marker.is_file() or json.loads(marker.read_text())["training_data_sha256"] != data_digest():
        raise ValueError("Corrected sklearn exports are incomplete")
    manifest = {
        "status": "complete", "training_data_sha256": data_digest(), "rows": 180, "groups": 20,
        "selected": selection["selected"],
        "runs": {"TensorFlow": selected.relative_to(ROOT).as_posix(),
                 "PyTorch": pytorch.relative_to(ROOT).as_posix()},
        "comparison": (comparison / "comparison.csv").relative_to(ROOT).as_posix(),
        "exports": {kind: f"results/corrected_20_days/exports/{kind}"
                    for kind in ["tensorflow", "pytorch", "sklearn"]},
        "editorial_reference": "manuscript/editorial_reference_superseded.zip",
        "manuscript_pdf_status": "pending_update_and_visual_review",
    }
    write_json(COLLECTION / "completed.json", manifest)
    write_json(COLLECTION / "progress.json", {"stage": "experiments_and_exports_complete", "complete": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sklearn", "frameworks"], required=True)
    parser.add_argument("--wait-hours", type=float, default=24)
    args = parser.parse_args()
    try:
        sklearn_exports() if args.stage == "sklearn" else frameworks(args.wait_hours)
    except Exception as exc:
        write_json(COLLECTION / f"{args.stage}_failure.json", {"error": str(exc), "complete": False})
        raise


if __name__ == "__main__":
    main()
