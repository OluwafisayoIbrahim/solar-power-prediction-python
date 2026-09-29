import pandas as pd
from sklearn.linear_model import LinearRegression

from evaluate import evaluate_predictions, print_metrics
from features import (
    add_day_id,
    get_track_a_data,
    get_track_b_data,
    load_processed_data,
)
from validation import grouped_cross_validate, summarize_cv_results


def evaluate_physics_baseline(df: pd.DataFrame) -> dict[str, float]:
    """
    Physics baseline:
    predicted power = current x voltage
    """

    y_true = df["power"]
    y_pred = df["current"] * df["voltage"]

    metrics = evaluate_predictions(
        y_true,
        y_pred,
    )

    print_metrics(
        metrics,
        title="Physics Baseline: Current x Voltage",
    )

    return metrics


def evaluate_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    title: str,
) -> dict[str, float]:
    """
    Fit a simple linear regression model and evaluate it
    on the same dataset.

    This is only an initial baseline check.
    Proper grouped cross-validation comes next.
    """

    model = LinearRegression()

    model.fit(X, y)

    predictions = model.predict(X)

    metrics = evaluate_predictions(
        y,
        predictions,
    )

    print_metrics(
        metrics,
        title=title,
    )

    return metrics


def main() -> None:
    df = load_processed_data()
    df = add_day_id(df)

    X_a, y_a = get_track_a_data(df)
    X_b, y_b = get_track_b_data(df)

    evaluate_physics_baseline(df)

    evaluate_linear_regression(
        X_a,
        y_a,
        title="Linear Regression - Track A",
    )

    evaluate_linear_regression(
        X_b,
        y_b,
        title="Linear Regression - Track B",
    )

    groups = df["day_id"]

    print("\n")
    print("=" * 60)
    print("GROUPED CROSS-VALIDATION")
    print("=" * 60)

    print("\nTrack A - Linear Regression")

    track_a_model = LinearRegression()

    track_a_results, _ = grouped_cross_validate(
        model=track_a_model,
        X=X_a,
        y=y_a,
        groups=groups,
        n_splits=7,
    )

    summarize_cv_results(track_a_results)

    print("\nTrack B - Linear Regression")

    track_b_model = LinearRegression()

    track_b_results, _ = grouped_cross_validate(
        model=track_b_model,
        X=X_b,
        y=y_b,
        groups=groups,
        n_splits=7,
    )

    summarize_cv_results(track_b_results)
    


if __name__ == "__main__":
    main()
