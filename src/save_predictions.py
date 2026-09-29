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
from train_ann import SEEDS, out_of_fold_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
N_SPLITS = 7


def linear_out_of_fold(X, y, groups):
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    predictions = np.zeros(len(y))
    for train_idx, test_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])
    return predictions


def save(name, df, day_id, hour, measured, predicted):
    out = pd.DataFrame({
        "day_id": day_id, "hour": hour,
        "measured": measured, "predicted": predicted,
    })
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PRED_DIR / f"{name}.csv", index=False)
    print(f"  saved {name}.csv  ({len(out)} rows)")


def main():
    df = add_day_id(load_processed_data())
    groups = df["day_id"].to_numpy()
    hour = df["hour"].to_numpy()

    X_a, y_a = get_track_a_data(df)
    X_b, y_b = get_track_b_data(df)
    X_a, y_a, X_b, y_b = map(np.asarray, (X_a, y_a, X_b, y_b))

    print("Linear:")
    save("track_a_linear", df, groups, hour, y_a, linear_out_of_fold(X_a, y_a, groups))
    save("track_b_linear", df, groups, hour, y_b, linear_out_of_fold(X_b, y_b, groups))

    print("ANN (seed 0 only, for plotting — matches the 'Largest errors' printout):")
    for track_name, X, y in (("a", X_a, y_a), ("b", X_b, y_b)):
        for tuned, label in ((False, "untuned"), (True, "tuned")):
            preds, _ = out_of_fold_predictions(X, y, groups, SEEDS[0], tuned)
            save(f"track_{track_name}_ann_{label}", df, groups, hour, y, preds)

    print(f"\nAll predictions saved to {PRED_DIR}")


if __name__ == "__main__":
    main()