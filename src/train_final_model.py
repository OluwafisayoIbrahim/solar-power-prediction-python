from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from features import (
    add_day_id,
    get_track_a_data,
    get_track_b_data,
    load_processed_data,
)
from models import create_ann_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "results" / "models"

HIDDEN_GRID = [3, 10]
ALPHA_GRID = [0.1, 1.0, 10.0]
SEED = 0
N_SPLITS = 7


def select_final_hyperparameters(X, y, groups):
    cv = GroupKFold(n_splits=N_SPLITS)

    best_mae = np.inf
    best_config = None

    for hidden_units in HIDDEN_GRID:
        for alpha in ALPHA_GRID:
            fold_errors = []

            for train_idx, val_idx in cv.split(X, y, groups):
                model = create_ann_model(
                    hidden_layer_size=hidden_units,
                    alpha=alpha,
                    random_state=SEED,
                )

                model.fit(
                    X[train_idx],
                    y[train_idx],
                )

                predictions = model.predict(
                    X[val_idx]
                )

                mae = mean_absolute_error(
                    y[val_idx],
                    predictions,
                )

                fold_errors.append(mae)

            mean_mae = np.mean(fold_errors)

            print(
                f"Hidden={hidden_units}, "
                f"alpha={alpha}: "
                f"MAE={mean_mae:.4f} W"
            )

            if mean_mae < best_mae:
                best_mae = mean_mae
                best_config = (
                    hidden_units,
                    alpha,
                )

    return best_config


def train_and_save_final_model(
    track_name,
    X,
    y,
    groups,
):
    print("\n" + "=" * 60)
    print(f"Training final {track_name} ANN")
    print("=" * 60)

    hidden_units, alpha = select_final_hyperparameters(
        X,
        y,
        groups,
    )

    print("\nSelected configuration:")
    print(f"Hidden units: {hidden_units}")
    print(f"Alpha: {alpha}")

    final_model = create_ann_model(
        hidden_layer_size=hidden_units,
        alpha=alpha,
        random_state=SEED,
    )

    print(
        f"\nTraining {track_name} final model "
        f"on all {len(y)} observations..."
    )

    final_model.fit(
        X,
        y,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / f"{track_name.lower().replace(' ', '_')}_ann_final.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
    )

    print("\nSaved model:")
    print(model_path)


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

    train_and_save_final_model(
        track_name="Track A",
        X=X_a,
        y=y_a,
        groups=groups,
    )

    train_and_save_final_model(
        track_name="Track B",
        X=X_b,
        y=y_b,
        groups=groups,
    )


if __name__ == "__main__":
    main()