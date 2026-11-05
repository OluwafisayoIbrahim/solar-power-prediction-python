"""Read-only, checked evidence for the manuscript. Never loads or fits a model."""
import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GroupKFold

from features import TRACK_A_FEATURES, TRACK_B_FEATURES
from research_data import ROOT, UNITS, dataset
from evaluate import compute_metrics
from check_convergence import convergence_summary

RUN_PATHS = {"TensorFlow": "results/tensorflow/convergence_s2",
             "PyTorch": "results/pytorch/parity_s2"}
COMPARISON = "results/comparisons/parity_s2/comparison.csv"
METRICS = {"MAE": "W", "RMSE": "W", "MAPE": "%", "R2": "unitless"}
MEASUREMENTS = ["time", "solar_irradiance", "temperature", "humidity", "current", "voltage", "power"]


def audit_dataset(frame):
    """Report duplicates with and without file identity; do not deduplicate."""
    repeated = frame.duplicated(MEASUREMENTS, keep=False)
    times = pd.to_datetime(frame.time, errors="raise")
    intervals = times.groupby(frame.day_id).diff().dt.total_seconds().div(60).dropna()
    return {
        "rows": len(frame), "groups": frame.day_id.nunique(),
        "readings_min": int(frame.groupby("day_id").size().min()),
        "readings_max": int(frame.groupby("day_id").size().max()),
        "missing": int(frame.isna().sum().sum()),
        "duplicate_records": int(frame.drop(columns="day_id").duplicated().sum()),
        "duplicate_measurements": int(frame.duplicated(MEASUREMENTS).sum()),
        "duplicate_sources": sorted(frame.loc[repeated, "source_file"].unique()),
        "interval_min": float(intervals.min()), "interval_max": float(intervals.max()),
        "power_iv_mae": float((frame.power - frame.current * frame.voltage).abs().mean()),
    }


def validate_predictions(predictions, metrics, cfg, frame):
    """Check identities, group separation and metrics against saved OOF records."""
    keys = ["track", "seed", "model", "row"]
    if predictions.duplicated(keys).any():
        raise ValueError("Duplicate out-of-fold prediction identities")
    numeric = predictions[["row", "fold", "measured", "predicted"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite saved prediction values")
    expected_models = {"Linear", "Sklearn untuned", "Sklearn tuned",
                       "TensorFlow selected" if "tensorflow" in cfg else "PyTorch selected"}
    expected_keys = {(track, seed, model) for track in cfg["tracks"] for seed in cfg["seeds"]
                     for model in expected_models | ({"Physics"} if track == "A" else set())}
    actual_keys = set(predictions.groupby(["track", "seed", "model"]).groups)
    metric_keys = set(metrics[["track", "seed", "model"]].itertuples(index=False, name=None))
    if actual_keys != expected_keys or metric_keys != expected_keys or len(metrics) != len(expected_keys):
        raise ValueError("Incomplete or duplicate saved experiment results")
    fold_ids = np.zeros(len(frame), dtype=int)
    for fold, (_, test) in enumerate(GroupKFold(cfg["folds"]).split(frame, groups=frame.day_id), 1):
        fold_ids[test] = fold
    for key, rows in predictions.groupby(["track", "seed", "model"]):
        rows = rows.sort_values("row")
        if rows.row.tolist() != list(range(len(frame))):
            raise ValueError("Prediction row identities do not cover the research dataset")
        if rows.day_id.tolist() != frame.day_id.tolist() or not np.allclose(rows.measured, frame.power, rtol=0, atol=1e-10):
            raise ValueError("Saved prediction reference data differ from the current dataset")
        if rows.groupby("day_id").fold.nunique().max() != 1 or set(rows.fold) != set(range(1, cfg["folds"] + 1)):
            raise ValueError("Saved outer folds do not preserve source-file groups")
        if not np.array_equal(rows.fold.to_numpy(), fold_ids):
            raise ValueError("Saved outer assignments disagree with the documented GroupKFold protocol")
        recorded = metrics[(metrics.track == key[0]) & (metrics.seed == key[1]) & (metrics.model == key[2])].iloc[0]
        calculated = compute_metrics(rows.measured.to_numpy(), rows.predicted.to_numpy())
        for metric in METRICS:
            if not np.isclose(recorded[metric], calculated[metric], rtol=1e-8, atol=1e-9):
                raise ValueError(f"Saved {metric} disagrees with out-of-fold predictions: {key}")


@st.cache_data(max_entries=1)
def load_evidence():
    frame = dataset()
    table = pd.read_csv(ROOT / COMPARISON)
    protocol = json.loads((ROOT / "results/comparisons/parity_s2/protocol.json").read_text(encoding="utf-8"))
    configs, predictions, seeds = {}, {}, {}
    if table.duplicated(["track", "model"]).any():
        raise ValueError("Ambiguous model rows in matched comparison")
    for framework, path in RUN_PATHS.items():
        cfg = json.loads((ROOT / path / "config.json").read_text(encoding="utf-8"))
        if cfg.get("smoke_test") or cfg["rows"] != len(frame):
            raise ValueError("The manuscript requires completed full-data research runs")
        preds = pd.read_csv(ROOT / path / "predictions.csv")
        metrics = pd.read_csv(ROOT / path / "metrics_per_seed.csv")
        validate_predictions(preds, metrics, cfg, frame)
        configs[framework], predictions[framework], seeds[framework] = cfg, preds, metrics
        for (track, model), rows in metrics.groupby(["track", "model"]):
            matched = table[(table.track == track) & (table.model == model)]
            if len(matched) != 1:
                raise ValueError("Comparison is missing a unique research model")
            for metric in METRICS:
                for stat, value in [("mean", rows[metric].mean()), ("std", rows[metric].std(ddof=0))]:
                    if not np.isclose(matched.iloc[0][f"{metric}_{stat}"], value, rtol=1e-8, atol=1e-9):
                        raise ValueError("Comparison summary disagrees with per-seed metrics")
    controls = ["architectures", "activation", "epochs", "patience", "batch_size", "learning_rate",
                "l2", "seeds", "tracks", "folds", "inner_folds", "rows"]
    if any(configs["TensorFlow"][key] != configs["PyTorch"][key] for key in controls):
        raise ValueError("The manuscript's matched S2 configurations no longer agree")
    convergence = {candidate: convergence_summary(ROOT / f"results/tensorflow/convergence_{candidate.lower()}")[1]
                   for candidate in ("S1", "S2")}
    return {"dataset": frame, "audit": audit_dataset(frame), "comparison": table,
            "convergence": convergence,
            "configs": configs, "predictions": predictions, "seeds": seeds, "protocol": protocol}


def score(evidence, track, model, metric="MAE", stat="mean"):
    rows = evidence["comparison"]
    return float(rows[(rows.track == track) & (rows.model == model)].iloc[0][f"{metric}_{stat}"])


def result_table(evidence, track):
    rows = evidence["comparison"].query("track == @track")
    result = pd.DataFrame({"Model": rows.model})
    for metric, unit in METRICS.items():
        label = "R²" if metric == "R2" else f"{metric} ({unit})"
        result[label] = [f"{mean:.3f} ± {std:.3f}" for mean, std in zip(rows[f"{metric}_mean"], rows[f"{metric}_std"])]
    return result.reset_index(drop=True)


def dataset_table(evidence):
    frame = evidence["dataset"]
    return pd.DataFrame([
        {"Variable": name.replace("_", " ").capitalize(), "Unit": unit,
         "Minimum": frame[name].min(), "Median": frame[name].median(), "Maximum": frame[name].max()}
        for name, unit in UNITS.items()])


def track_table():
    return pd.DataFrame([
        {"Track": "A — electrical-input replication", "Inputs": ", ".join(TRACK_A_FEATURES),
         "Interpretation": "Power estimation with measured current and voltage"},
        {"Track": "B — environmental/time", "Inputs": ", ".join(TRACK_B_FEATURES),
         "Interpretation": "Power estimation from contemporaneous environmental conditions and time"}])


def model_table(evidence):
    cfg = evidence["configs"]["TensorFlow"]
    return pd.DataFrame([
        {"Model": "Physics (A only)", "Architecture / selection": "P = I × V; no fitted parameters", "Training": "None"},
        {"Model": "Linear (A, B)", "Architecture / selection": "Ordinary least squares; no tuning", "Training": "Training-partition input standardisation"},
        {"Model": "scikit-learn untuned (A, B)", "Architecture / selection": "10 tanh units; alpha=0.0001", "Training": "L-BFGS, max_iter=5000; input scaling"},
        {"Model": "scikit-learn tuned (A, B)", "Architecture / selection": "3 or 10 tanh units; alpha ∈ {0.1, 1, 10}", "Training": "3 inner grouped folds; L-BFGS; input scaling"},
        {"Model": "TensorFlow / PyTorch S2 (A, B)", "Architecture / selection": f"{cfg['architectures']}; {cfg['activation']}; linear output",
         "Training": f"Adam; lr={cfg['learning_rate']}; hidden-kernel L2={cfg['l2']}; batch={cfg['batch_size']}; cap={cfg['epochs']}; patience={cfg['patience']}; input/target scaling"},
        {"Model": "Mean prediction", "Architecture / selection": "Metric sanity check only (src/evaluate.py)", "Training": "No saved grouped-CV experiment; excluded from result rankings"}])


def diagnostic_predictions(evidence, track):
    """Fixed, disclosed selection: all seed-0 OOF predictions, never saved weights."""
    frames = []
    for framework, predictions in evidence["predictions"].items():
        allowed = ["TensorFlow selected", "Sklearn tuned", "Linear"] if framework == "TensorFlow" else ["PyTorch selected"]
        selected = predictions[(predictions.track == track) & (predictions.seed == 0) & predictions.model.isin(allowed)].copy()
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True)
    result["residual"] = result.measured - result.predicted
    return result.merge(evidence["dataset"][["hour"]].rename_axis("row").reset_index(), on="row", validate="many_to_one")
