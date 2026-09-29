from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from features import (
    add_day_id,
    get_track_a_data,
    get_track_b_data,
    load_processed_data,
)
from train_ann import (
    SEEDS,
    compute_metrics,
    out_of_fold_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

OUTPUT_PATH = RESULTS_DIR / "model_comparison.csv"

N_SPLITS = 7


def linear_out_of_fold(
    X,
    y,
    groups,
) -> np.ndarray:
    """
    Generate pooled out-of-fold predictions using
    grouped cross-validation.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    predictions = np.zeros(len(y))

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    for train_idx, test_idx in splitter.split(
        X,
        y,
        groups,
    ):
        model = LinearRegression()

        model.fit(
            X[train_idx],
            y[train_idx],
        )

        predictions[test_idx] = model.predict(
            X[test_idx]
        )

    return predictions


def metrics_row(
    track: str,
    model_name: str,
    metrics: dict,
    seed: str | int = "n/a",
) -> dict:
    """
    Convert model metrics into one results-table row.
    """

    return {
        "track": track,
        "model": model_name,
        "seed": seed,
        "mae_w": metrics["MAE"],
        "rmse_w": metrics["RMSE"],
        "mape_percent": metrics["MAPE"],
        "r2": metrics["R2"],
    }


def main() -> None:
    df = add_day_id(
        load_processed_data()
    )

    groups = df["day_id"].to_numpy()

    X_a, y_a = get_track_a_data(df)
    X_b, y_b = get_track_b_data(df)

    X_a = np.asarray(X_a)
    y_a = np.asarray(y_a)

    X_b = np.asarray(X_b)
    y_b = np.asarray(y_b)

    rows = []

    # --------------------------------------------------
    # Physics baseline
    # --------------------------------------------------

    physics_predictions = (
        df["current"].to_numpy()
        * df["voltage"].to_numpy()
    )

    physics_metrics = compute_metrics(
        y_a,
        physics_predictions,
    )

    rows.append(
        metrics_row(
            track="A",
            model_name="Physics baseline",
            metrics=physics_metrics,
        )
    )

    # --------------------------------------------------
    # Linear regression
    # --------------------------------------------------

    track_a_linear_predictions = linear_out_of_fold(
        X_a,
        y_a,
        groups,
    )

    track_a_linear_metrics = compute_metrics(
        y_a,
        track_a_linear_predictions,
    )

    rows.append(
        metrics_row(
            track="A",
            model_name="Linear Regression",
            metrics=track_a_linear_metrics,
        )
    )

    track_b_linear_predictions = linear_out_of_fold(
        X_b,
        y_b,
        groups,
    )

    track_b_linear_metrics = compute_metrics(
        y_b,
        track_b_linear_predictions,
    )

    rows.append(
        metrics_row(
            track="B",
            model_name="Linear Regression",
            metrics=track_b_linear_metrics,
        )
    )

    # --------------------------------------------------
    # ANN models
    # --------------------------------------------------

    for track_name, X, y in (
        ("A", X_a, y_a),
        ("B", X_b, y_b),
    ):
        for tuned in (
            False,
            True,
        ):
            model_name = (
                "ANN Tuned"
                if tuned
                else "ANN Untuned"
            )

            seed_metrics = []

            for seed in SEEDS:
                predictions, _ = out_of_fold_predictions(
                    X,
                    y,
                    groups,
                    seed,
                    tuned,
                )

                metrics = compute_metrics(
                    y,
                    predictions,
                )

                seed_metrics.append(metrics)

            averaged_metrics = {}

            for metric_name in (
                "MAE",
                "RMSE",
                "MAPE",
                "R2",
            ):
                averaged_metrics[metric_name] = np.mean(
                    [
                        metrics[metric_name]
                        for metrics in seed_metrics
                    ]
                )

            rows.append(
                metrics_row(
                    track=track_name,
                    model_name=model_name,
                    metrics=averaged_metrics,
                    seed="mean_of_seeds",
                )
            )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results_df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nFinal Model Comparison")
    print("----------------------")
    print(results_df.to_string(index=False))

    print(
        f"\nResults saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()