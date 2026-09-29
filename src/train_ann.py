import warnings
 
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
 
from evaluate import compute_metrics
from features import (
    add_day_id,
    get_track_a_data,
    get_track_b_data,
    load_processed_data,
)
from models import create_ann_model
 
N_SPLITS = 7
SEEDS = [0, 1, 2]
HIDDEN_GRID = [3, 10]
ALPHA_GRID = [0.1, 1.0, 10.0]
UNTUNED = (10, 0.0001)
 
 
def select_hyperparameters(X_train, y_train, groups_train, seed):
    """Pick (hidden, alpha) by inner grouped CV on the training days only."""
    inner = GroupKFold(n_splits=3)
    best_error, best_config = np.inf, None
 
    for hidden in HIDDEN_GRID:
        for alpha in ALPHA_GRID:
            errors = []
            for tr, va in inner.split(X_train, y_train, groups_train):
                model = create_ann_model(hidden, alpha, seed)
                model.fit(X_train[tr], y_train[tr])
                errors.append(mean_absolute_error(y_train[va], model.predict(X_train[va])))
            if np.mean(errors) < best_error:
                best_error, best_config = np.mean(errors), (hidden, alpha)
 
    return best_config
 
 
def out_of_fold_predictions(X, y, groups, seed, tuned):
    predictions = np.zeros(len(y))
    chosen_configs = []
 
    for train_idx, test_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        if tuned:
            config = select_hyperparameters(X[train_idx], y[train_idx], groups[train_idx], seed)
        else:
            config = UNTUNED
        model = create_ann_model(config[0], config[1], seed)
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])
        chosen_configs.append(config)
 
    return predictions, chosen_configs
 
 
def run_ann_experiment(X, y, groups, track_name: str, tuned: bool):
    label = "tuned" if tuned else "untuned (10 units, alpha=1e-4)"
    print("\n" + "=" * 60)
    print(f"{track_name} - ANN, {label}")
    print("=" * 60)
 
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
 
    per_seed, first_seed_predictions, first_seed_configs = [], None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for seed in SEEDS:
            preds, configs = out_of_fold_predictions(X, y, groups, seed, tuned)
            per_seed.append(compute_metrics(y, preds))
            if first_seed_predictions is None:
                first_seed_predictions, first_seed_configs = preds, configs
 
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
 
    print("Pooled out-of-fold metrics (mean +/- std over seeds):")
    for key in ("MAE", "RMSE", "MAPE", "R2"):
        vals = [m[key] for m in per_seed]
        unit = {"MAE": " W", "RMSE": " W", "MAPE": " %", "R2": ""}[key]
        print(f"  {key:<5} {np.mean(vals):8.4f} +/- {np.std(vals):.4f}{unit}")
 
    if tuned:
        print(f"Settings chosen per fold (seed {SEEDS[0]}): {first_seed_configs}")
    print(f"Convergence warnings across all fits: {n_conv}")
 
   
    residuals = np.abs(first_seed_predictions - y)
    print(f"Largest errors (seed {SEEDS[0]}):")
    for i in np.argsort(residuals)[::-1][:5]:
        print(f"  {groups[i]:<24} measured {y[i]:7.2f} W  predicted {first_seed_predictions[i]:7.2f} W")
 
    return per_seed
 
 
def main() -> None:
    df = add_day_id(load_processed_data())
    groups = df["day_id"]
 
    X_a, y_a = get_track_a_data(df)
    X_b, y_b = get_track_b_data(df)
 
    for track_name, X, y in (("Track A", X_a, y_a), ("Track B", X_b, y_b)):
        run_ann_experiment(X, y, groups, track_name, tuned=False)
        run_ann_experiment(X, y, groups, track_name, tuned=True)
 
 
if __name__ == "__main__":
    main()
