"""Shared, fail-closed provenance for the 20-day research collection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_convergence import convergence_summary
from features import PROCESSED_DATA_PATH, add_day_id, load_processed_data
from load_data import MEASUREMENT_COLUMNS, validate_research_data
from evaluate import compute_metrics
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "results/research_20_days"


def data_digest():
    return hashlib.sha256(PROCESSED_DATA_PATH.read_bytes()).hexdigest()


def choose_tensorflow(s1, s2):
    """Use only inner-selected epochs/MAE; outer predictions are never read."""
    candidates = {}
    for name, path in [("S1", s1), ("S2", s2)]:
        cfg, summary = convergence_summary(path)
        if (cfg.get("smoke_test") or cfg.get("rows") != 180
                or cfg.get("groups") != 20 or cfg.get("folds") != 7
                or cfg.get("inner_folds") != 3 or cfg.get("seeds") != [0, 1, 2]
                or cfg.get("tracks") != ["A", "B"]
                or cfg.get("training_data_sha256") != data_digest()):
            raise ValueError("Selection requires full updated research settings")
        candidates[name] = {
            "cap_hits": sum(row["cap_hits"] for row in summary),
            "mean_inner_mae": float(np.mean([row["mean_inner_mae"] for row in summary])),
            "tracks": summary,
        }
    selected = min(candidates, key=lambda name: (
        candidates[name]["cap_hits"], candidates[name]["mean_inner_mae"], name))
    return {"selected": selected, "candidates": candidates,
            "rule": "Fewer total selected-refit cap hits; then lower equal-track mean inner MAE; exact tie S1. No outer scores used.",
            "training_data_sha256": data_digest()}


def verify_run(path, label):
    """Reconcile every row, split and per-seed/summary metric with updated data."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    frame = add_day_id(load_processed_data())
    validate_research_data(frame)
    if (cfg.get("smoke_test") or cfg.get("rows") != 180 or cfg.get("groups") != 20
            or cfg.get("training_data_sha256") != data_digest()
            or cfg.get("folds") != 7 or cfg.get("inner_folds") != 3
            or cfg.get("seeds") != [0, 1, 2] or cfg.get("tracks") != ["A", "B"]):
        raise ValueError("Not a full updated research run")
    predictions = pd.read_csv(path / "predictions.csv")
    metrics = pd.read_csv(path / "metrics_per_seed.csv")
    summary = pd.read_csv(path / "comparison.csv")
    expected = {(track, seed, model) for track in ["A", "B"] for seed in [0, 1, 2]
                for model in ["Linear", "Sklearn untuned", "Sklearn tuned", label]
                + (["Physics"] if track == "A" else [])}
    keys = ["track", "seed", "model"]
    if (set(predictions.groupby(keys).groups) != expected
            or set(metrics[keys].itertuples(index=False, name=None)) != expected
            or len(metrics) != len(expected) or predictions.duplicated(keys + ["row"]).any()
            or summary.duplicated(["track", "model"]).any()):
        raise ValueError("Missing or duplicate research prediction identities")
    folds = np.zeros(len(frame), dtype=int)
    identities = pd.util.hash_pandas_object(frame[MEASUREMENT_COLUMNS], index=False).to_numpy()
    for fold, (train, test) in enumerate(GroupKFold(7).split(frame, groups=frame.day_id), 1):
        if (set(frame.iloc[train].day_id) & set(frame.iloc[test].day_id)
                or set(identities[train]) & set(identities[test])):
            raise ValueError("Training/test measurement overlap")
        folds[test] = fold
    for key, rows in predictions.groupby(keys):
        rows = rows.sort_values("row")
        if (rows.row.tolist() != list(range(180)) or rows.day_id.tolist() != frame.day_id.tolist()
                or not np.array_equal(rows.fold, folds)
                or not np.allclose(rows.measured, frame.power, rtol=0, atol=1e-10)
                or not np.isfinite(rows.predicted).all()):
            raise ValueError("Saved predictions differ from updated data or fold assignments")
        recorded = metrics.set_index(keys).loc[key]
        for metric, value in compute_metrics(rows.measured.to_numpy(), rows.predicted.to_numpy()).items():
            if not np.isclose(recorded[metric], value, rtol=1e-8, atol=1e-9):
                raise ValueError(f"Prediction metric mismatch: {key} {metric}")
    for (track, model), rows in metrics.groupby(["track", "model"]):
        saved = summary.set_index(["track", "model"]).loc[(track, model)]
        for metric in ["MAE", "RMSE", "MAPE", "R2"]:
            for stat, value in [("mean", rows[metric].mean()), ("std", rows[metric].std(ddof=0))]:
                if not np.isclose(saved[f"{metric}_{stat}"], value, rtol=1e-8, atol=1e-9):
                    raise ValueError("Summary differs from pooled per-seed metrics")
    return cfg


def completed_collection():
    """Never fall back to 189-row historical or partial research evidence."""
    path = COLLECTION / "completed.json"
    if not path.is_file():
        raise ValueError("Updated experiments are incomplete; historical 189-row results are superseded")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("training_data_sha256") != data_digest():
        raise ValueError("Completed research collection does not match the active dataset")
    return manifest
