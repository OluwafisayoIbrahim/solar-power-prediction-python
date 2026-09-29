from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_mape(
    y_true: Any,
    y_pred: Any,
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Rows where the true value is zero are excluded to avoid division by zero.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    non_zero_mask = y_true != 0

    if not np.any(non_zero_mask):
        raise ValueError(
            "MAPE cannot be calculated because all true values are zero."
        )

    percentage_errors = np.abs(
        (
            y_true[non_zero_mask]
            - y_pred[non_zero_mask]
        )
        / y_true[non_zero_mask]
    )

    return float(np.mean(percentage_errors) * 100)


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
) -> dict[str, float]:
    """
    Calculate regression performance metrics.

    Returns:
        MAE  - Mean Absolute Error
        RMSE - Root Mean Squared Error
        MAPE - Mean Absolute Percentage Error
        R2   - Coefficient of determination
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mape = calculate_mape(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


def compute_metrics(
    y_true: Any,
    y_pred: Any,
) -> dict[str, float]:
    """
    Return regression metrics using the legacy uppercase key names.

    Some experiment scripts print compact tables with MAE/RMSE/MAPE/R2 keys.
    This wrapper keeps that output format while reusing evaluate_predictions().
    """

    metrics = evaluate_predictions(
        y_true,
        y_pred,
    )

    return {
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "MAPE": metrics["mape"],
        "R2": metrics["r2"],
    }


def print_metrics(
    metrics: dict[str, float],
    title: str = "Model Performance",
) -> None:
    """Print regression metrics in a readable format."""

    print(f"\n{title}")
    print("-" * len(title))

    print(f"MAE:  {metrics['mae']:.4f} W")
    print(f"RMSE: {metrics['rmse']:.4f} W")
    print(f"MAPE: {metrics['mape']:.4f}%")
    print(f"R2:   {metrics['r2']:.4f}")


def main() -> None:
    """
    Small sanity test.

    Predicting the mean for every sample should produce
    an R2 score close to 0.
    """

    y_true = np.array(
        [200, 220, 240, 260, 280],
        dtype=float,
    )

    mean_prediction = np.full(
        shape=y_true.shape,
        fill_value=y_true.mean(),
    )

    metrics = evaluate_predictions(
        y_true,
        mean_prediction,
    )

    print_metrics(
        metrics,
        title="Mean Prediction Sanity Check",
    )


if __name__ == "__main__":
    main()
