"""Compare sklearn and TensorFlow using identical held-out days.

TensorFlow architecture/epoch selection uses inner grouped CV. The selected
architecture is refitted on all outer training days for the median best epoch.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import compute_metrics
from features import (add_day_id, get_track_a_data, get_track_b_data,
                      load_processed_data, validate_features)
from models import create_ann_model
from train_ann import N_SPLITS, SEEDS, UNTUNED, select_hyperparameters


def architecture(value):
    try:
        units = tuple(int(item) for item in value.split(","))
        if not units or any(item < 1 for item in units):
            raise ValueError
        return units
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use positive layer sizes, e.g. 10 or 16,8") from exc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", type=architecture, nargs="+", default=[(10,), (16, 8)])
    parser.add_argument("--activation", choices=["tanh", "relu", "elu"], default="tanh")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--tracks", choices=["A", "B"], nargs="+", default=["A", "B"])
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-test", action="store_true", help="Short integration check, not research results")
    args = parser.parse_args()
    if args.smoke_test:
        args.epochs, args.patience = 2, 1
        args.folds, args.inner_folds = 2, 2
        args.seeds, args.tracks, args.architectures = [0], ["B"], [(3,)]
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, batch size and patience must be positive")
    if min(args.folds, args.inner_folds) < 2:
        parser.error("CV requires at least two folds")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if not np.isfinite(args.l2) or args.l2 < 0 or min(args.seeds) < 0:
        parser.error("l2 and seeds must be nonnegative (l2 must be finite)")
    return args


def fit_tf(X, y, hidden, seed, args, validation=None, epochs=None):
    import tensorflow as tf
    from models_tf import create_tf_ann

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaled = x_scaler.fit_transform(X).astype("float32")
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).astype("float32")
    model = create_tf_ann(X.shape[1], hidden, args.activation,
                          args.learning_rate, args.l2, seed)
    fit_options = {}
    if validation is not None:
        X_val, y_val = validation
        fit_options["validation_data"] = (
            x_scaler.transform(X_val).astype("float32"),
            y_scaler.transform(y_val.reshape(-1, 1)).astype("float32"),
        )
        fit_options["callbacks"] = [tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=args.patience, restore_best_weights=True,
        )]
    history = model.fit(x_scaled, y_scaled, epochs=epochs or args.epochs,
                        batch_size=args.batch_size, verbose=0, **fit_options)
    return model, x_scaler, y_scaler, history.history


def predict_tf(fitted, X):
    model, x_scaler, y_scaler, _ = fitted
    scaled = model(x_scaler.transform(X).astype("float32"), training=False).numpy()
    return y_scaler.inverse_transform(scaled).reshape(-1)


def select_tf(X, y, groups, seed, args, curve_dir, prefix):
    candidates = []
    for hidden in args.architectures:
        errors, epochs, histories = [], [], []
        for tr, va in GroupKFold(args.inner_folds).split(X, y, groups):
            fitted = fit_tf(X[tr], y[tr], hidden, seed, args, (X[va], y[va]))
            errors.append(mean_absolute_error(y[va], predict_tf(fitted, X[va])))
            history = fitted[3]
            epochs.append(int(np.argmin(history["val_mae"])) + 1)
            histories.append((history, float(fitted[2].scale_[0])))
        candidates.append((float(np.mean(errors)), hidden, epochs, histories))
    score, hidden, epochs, histories = min(candidates, key=lambda item: item[0])
    fig, ax = plt.subplots(figsize=(7, 4))
    for fold, (history, scale) in enumerate(histories, 1):
        frame = pd.DataFrame(history)
        frame.index = np.arange(1, len(frame) + 1)
        frame.index.name = "epoch"
        frame["mae_watts"] = frame["mae"] * scale
        frame["val_mae_watts"] = frame["val_mae"] * scale
        frame.to_csv(curve_dir / f"{prefix}_inner{fold}.csv")
        line, = ax.plot(frame.index, frame["mae_watts"], label=f"Train {fold}")
        ax.plot(frame.index, frame["val_mae_watts"], "--", color=line.get_color(),
                label=f"Validation {fold}")
    ax.set(xlabel="Epoch", ylabel="MAE (W)", title=f"{prefix}: selected {hidden}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(curve_dir / f"{prefix}.png", dpi=160)
    plt.close(fig)
    return hidden, max(1, int(np.median(epochs))), score


def main():
    args = parse_args()
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        if exc.name != "tensorflow":
            raise
        raise SystemExit("Install optional dependencies: python -m pip install -r requirements-tensorflow.txt") from exc
    df = add_day_id(load_processed_data())
    validate_features(df)
    groups = df["day_id"].to_numpy()
    if args.folds > len(np.unique(groups)):
        raise SystemExit("Outer folds exceed the number of measurement days")
    splits = list(GroupKFold(args.folds).split(df, groups=groups))
    if any(len(np.unique(groups[tr])) < max(args.inner_folds, 3) for tr, _ in splits):
        raise SystemExit("Not enough outer training days for inner CV")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(__file__).resolve().parents[1] / "results" / "tensorflow" / stamp
    output.mkdir(parents=True, exist_ok=False)
    curves = output / "learning_curves"
    curves.mkdir()
    metadata = vars(args).copy()
    metadata.update(output=str(output), python=platform.python_version(),
                    tensorflow=tf.__version__, sklearn=sklearn.__version__, rows=len(df))
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics, predictions, selections = [], [], []
    for track in args.tracks:
        X, y = (get_track_a_data if track == "A" else get_track_b_data)(df)
        X, y = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
        for seed in args.seeds:
            pooled = {name: np.zeros(len(y)) for name in
                      ("Linear", "Sklearn untuned", "Sklearn tuned", "TensorFlow selected")}
            fold_ids = np.zeros(len(y), dtype=int)
            for fold, (tr, te) in enumerate(splits, 1):
                print(f"Track {track}, seed {seed}, fold {fold}/{args.folds}", flush=True)
                fold_ids[te] = fold
                linear = make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], y[tr])
                pooled["Linear"][te] = linear.predict(X[te])
                for label, config in [("Sklearn untuned", UNTUNED),
                                      ("Sklearn tuned", select_hyperparameters(X[tr], y[tr], groups[tr], seed))]:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model = create_ann_model(*config, seed).fit(X[tr], y[tr])
                    for warning in caught:
                        if not issubclass(warning.category, ConvergenceWarning):
                            warnings.warn(str(warning.message), warning.category)
                    pooled[label][te] = model.predict(X[te])
                    selections.append(dict(track=track, seed=seed, fold=fold, model=label,
                                           hidden=str(config[0]), alpha=config[1],
                                           convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in caught)))
                hidden, epochs, inner_mae = select_tf(
                    X[tr], y[tr], groups[tr], seed, args, curves, f"track{track}_seed{seed}_fold{fold}")
                fitted = fit_tf(X[tr], y[tr], hidden, seed, args, epochs=epochs)
                pooled["TensorFlow selected"][te] = predict_tf(fitted, X[te])
                selections.append(dict(track=track, seed=seed, fold=fold, model="TensorFlow selected",
                                       hidden=str(hidden), epochs=epochs, inner_mae=inner_mae))
            if track == "A":
                pooled["Physics"] = (df["current"] * df["voltage"]).to_numpy()
            for label, preds in pooled.items():
                metrics.append(dict(track=track, seed=seed, model=label, **compute_metrics(y, preds)))
                predictions.append(pd.DataFrame(dict(track=track, seed=seed, model=label,
                                                     row=np.arange(len(y)), day_id=groups,
                                                     fold=fold_ids, measured=y, predicted=preds)))
            pd.DataFrame(metrics).to_csv(output / "metrics_per_seed.csv", index=False)
            pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
            pd.DataFrame(selections).to_csv(output / "selected_settings.csv", index=False)
    summary = pd.DataFrame(metrics).groupby(["track", "model"])[["MAE", "RMSE", "MAPE", "R2"]].agg(
        ["mean", lambda values: np.std(values, ddof=0)])
    summary.columns = [f"{metric}_{'std' if stat == '<lambda_0>' else stat}" for metric, stat in summary.columns]
    summary.to_csv(output / "comparison.csv")
    print(summary.to_string())
    print(f"\nSaved to {output}")
    if args.smoke_test:
        print("SMOKE TEST ONLY: these scores are not research results.")


if __name__ == "__main__":
    main()
