"""Compare recorded framework runs, retaining provenance and protocol warnings."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_convergence import convergence_summary

METRICS = ["MAE", "RMSE", "MAPE", "R2"]
PROTOCOL = ["epochs", "learning_rate", "batch_size", "patience", "l2",
            "architectures", "seeds", "folds", "inner_folds", "tracks", "activation", "rows"]
BASELINES = ["Linear", "Sklearn untuned", "Sklearn tuned", "Physics"]


def read_run(path, framework):
    path = Path(path)
    config, convergence = convergence_summary(path)
    version_key = "tensorflow" if framework == "TensorFlow" else "torch"
    if version_key not in config or not (path / "comparison.csv").is_file():
        raise ValueError(f"{path}: a completed {framework} run is required")
    metrics = pd.read_csv(path / "metrics_per_seed.csv")
    expected = {(track, seed, model) for track in config["tracks"] for seed in config["seeds"]
                for model in BASELINES + [f"{framework} selected"]
                if model != "Physics" or track == "A"}
    actual = set(metrics[["track", "seed", "model"]].itertuples(index=False, name=None))
    if len(metrics) != len(expected) or actual != expected or not np.isfinite(metrics[METRICS]).all().all():
        raise ValueError(f"{path}: incomplete, duplicate or non-finite metrics")
    return config, metrics, convergence


def compare_runs(tf_run, torch_run, output):
    tf_config, tf_metrics, tf_caps = read_run(tf_run, "TensorFlow")
    pt_config, pt_metrics, pt_caps = read_run(torch_run, "PyTorch")
    notices = []
    for key in PROTOCOL:
        if key not in tf_config or key not in pt_config or tf_config[key] != pt_config[key]:
            notices.append(f"{key}: TensorFlow={tf_config.get(key)!r}, PyTorch={pt_config.get(key)!r}")
    prediction_keys = ["track", "seed", "model", "row"]
    prediction_values = ["day_id", "fold", "measured"]
    tf_predictions = pd.read_csv(Path(tf_run) / "predictions.csv")
    pt_predictions = pd.read_csv(Path(torch_run) / "predictions.csv")
    def reference(frame, config):
        selected = frame[frame["model"] == "Linear"]
        if (len(selected) != config["rows"] * len(config["tracks"]) * len(config["seeds"])
                or selected.duplicated(prediction_keys).any()):
            raise ValueError("Incomplete or duplicate reference predictions")
        return selected.set_index(prediction_keys)[prediction_values].sort_index()
    if not reference(tf_predictions, tf_config).equals(reference(pt_predictions, pt_config)):
        notices.append("Prediction data, row identities or outer split assignments differ")
    if tf_config.get("smoke_test") or pt_config.get("smoke_test"):
        notices.append("Smoke-test input: these are integration results, not research results")
    if tf_config.get("sklearn") != pt_config.get("sklearn"):
        notices.append("scikit-learn versions differ")

    combined = []
    for framework, path, config, metrics in [
            ("TensorFlow", tf_run, tf_config, tf_metrics),
            ("PyTorch", torch_run, pt_config, pt_metrics)]:
        for (track, model), rows in metrics.groupby(["track", "model"], sort=False):
            combined.append(dict(
                track=track, model=model, source_framework=framework, source_run=str(path),
                **{f"{metric}_{stat}": float(getattr(np, stat)(rows[metric].to_numpy()))
                   for metric in METRICS for stat in ("mean", "std")}))
    table = pd.DataFrame(combined)
    
    for track in sorted(table.track.unique()):
        for model in BASELINES:
            rows = table[(table.track == track) & (table.model == model)]
            if len(rows) != 2:
                continue
            columns = [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
            if np.allclose(rows.iloc[0][columns].to_numpy(float),
                           rows.iloc[1][columns].to_numpy(float), rtol=1e-9, atol=1e-10):
                table.loc[rows.index[0], "source_framework"] = "Both"
                table.loc[rows.index[0], "source_run"] = f"{tf_run}; {torch_run}"
                table = table.drop(rows.index[1])
            else:
                notices.append(f"Track {track} {model} differs: both source results retained")
    warning = ("WARNING: RESULTS ARE NOT STRICTLY COMPARABLE. " + " | ".join(notices)
               if notices else "Protocol and reference data checks match; numerical differences remain possible.")
    table["comparison_note"] = warning
    table = table.sort_values(["track", "model", "source_framework"]).reset_index(drop=True)
    lines = ["# Framework comparison", "", warning, "",
             "Mean +/- population standard deviation over seeds; not confidence intervals.", "",
             "| Track | Model | Source | MAE (W) | RMSE (W) | MAPE (%) | R2 |",
             "|---|---|---|---|---|---|---|"]
    for _, row in table.iterrows():
        values = " | ".join(f"{row[m + '_mean']:.3f} +/- {row[m + '_std']:.3f}" for m in METRICS)
        lines.append(f"| {row.track} | {row.model} | {row.source_framework} | {values} |")
    lines.extend(["", "## Selected epochs", ""])
    for row in tf_caps + pt_caps:
        lines.append(f"- {row['model']}, Track {row['track']}: "
                     f"{row['cap_hits']}/{row['selections']} at cap {row['cap']}.")
    lines.extend(["", f"TensorFlow source: `{tf_run}`", f"PyTorch source: `{torch_run}`"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "comparison.csv", index=False)
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "protocol.json").write_text(json.dumps(
        {"tensorflow": tf_config, "pytorch": pt_config, "warnings": notices,
         "convergence": tf_caps + pt_caps}, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return table, notices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf-run", required=True, type=Path)
    parser.add_argument("--torch-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    compare_runs(args.tf_run, args.torch_run, args.output)


if __name__ == "__main__":
    main()
