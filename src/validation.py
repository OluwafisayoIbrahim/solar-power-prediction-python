import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from evaluate import evaluate_predictions


def grouped_cross_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 7,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Evaluate a regression model using GroupKFold.

    Each source day is treated as a group, ensuring that all
    measurements from the same day stay entirely in either
    training or testing for a given fold.
    """

    splitter = GroupKFold(n_splits=n_splits)

    fold_results = []

    all_predictions = np.full(
        shape=len(y),
        fill_value=np.nan,
        dtype=float,
    )

    for fold_number, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        all_predictions[test_index] = predictions

        metrics = evaluate_predictions(
            y_test,
            predictions,
        )

        metrics["fold"] = fold_number

        fold_results.append(metrics)

        print(
            f"\nFold {fold_number}: "
            f"{groups.iloc[test_index].nunique()} test days"
        )

        print(
            f"MAE={metrics['mae']:.4f} W | "
            f"RMSE={metrics['rmse']:.4f} W | "
            f"MAPE={metrics['mape']:.4f}% | "
            f"R2={metrics['r2']:.4f}"
        )

    results_df = pd.DataFrame(fold_results)

    return results_df, all_predictions


def summarize_cv_results(
    results: pd.DataFrame,
) -> None:
    """Print mean and standard deviation across CV folds."""

    metric_columns = [
        "mae",
        "rmse",
        "mape",
        "r2",
    ]

    print("\nCross-Validation Summary")
    print("------------------------")

    for metric in metric_columns:
        mean_value = results[metric].mean()
        std_value = results[metric].std()

        print(
            f"{metric.upper()}: "
            f"{mean_value:.4f} +/- {std_value:.4f}"
        )
