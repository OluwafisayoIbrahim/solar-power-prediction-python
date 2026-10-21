"""Inspect selected epochs without reading outer-test metrics."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


def convergence_summary(run):
    run = Path(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    selections = pd.read_csv(run / "selected_settings.csv")
    label = "PyTorch selected" if "torch" in config else "TensorFlow selected"
    cap = config["epochs"]
    if not isinstance(cap, int) or cap < 1:
        raise ValueError("Invalid epoch cap")
    summary = []
    for track in config["tracks"]:
        rows = selections[(selections["track"] == track) & (selections["model"] == label)]
        expected = {(seed, fold) for seed in config["seeds"]
                    for fold in range(1, config["folds"] + 1)}
        if len(rows) != len(expected) or set(zip(rows["seed"], rows["fold"])) != expected:
            raise ValueError(f"Track {track}: incomplete or duplicate selections")
        epochs = rows["epochs"].to_numpy(float)
        inner_mae = rows["inner_mae"].to_numpy(float)
        if (not np.isfinite(epochs).all() or (epochs < 1).any()
                or (epochs > cap).any() or (epochs != np.floor(epochs)).any()
                or not np.isfinite(inner_mae).all()):
            raise ValueError(f"Track {track}: invalid epochs or inner MAE")
        layers = [tuple(ast.literal_eval(item)) for item in rows["hidden"]]
        if any(layer not in [tuple(a) for a in config["architectures"]] for layer in layers):
            raise ValueError(f"Track {track}: unknown architecture")
        hits = int((epochs == cap).sum())
        summary.append(dict(track=track, model=label, selections=len(rows), cap=cap,
                            cap_hits=hits, min=int(epochs.min()), median=float(np.median(epochs)),
                            max=int(epochs.max()), votes=dict(Counter(map(str, layers))),
                            mean_inner_mae=float(inner_mae.mean()),
                            passes_threshold=hits / len(rows) < 0.2))
    return config, summary


def report_lines(run):
    config, summary = convergence_summary(run)
    lines = [f"Run: {Path(run)}",
             "Cap-hit screen only: passing this threshold does not prove convergence."]
    if config.get("smoke_test"):
        lines.append("SMOKE TEST: integration output, not research evidence.")
    for row in summary:
        lines.extend([
            f"Track {row['track']}: {row['selections']} selections, "
            f"{row['cap_hits']} at cap {row['cap']}; "
            f"epochs min/median/max = {row['min']}/{row['median']:g}/{row['max']}",
            f"Architecture votes: {row['votes']}; mean inner MAE = {row['mean_inner_mae']:.6f} W",
            ("converged: <20% of selections at cap (screening criterion)"
             if row["passes_threshold"] else
             f"NOT converged: {row['cap_hits']}/{row['selections']} at cap (possibly undertrained)")
        ])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print("\n".join(report_lines(args.run)))


if __name__ == "__main__":
    main()
