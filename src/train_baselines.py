import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
from evaluate import compute_metrics
from features import (
    add_day_id,
    get_track_a_data,
    get_track_b_data,
    load_processed_data,
)
from train_ann import N_SPLITS
 
 
def linear_out_of_fold(X, y, groups):
    predictions = np.zeros(len(y))
    for train_idx, test_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])
    return predictions
 
 
def print_row(name, m):
    print(f"{name:<28} MAE {m['MAE']:7.3f} W | RMSE {m['RMSE']:7.3f} W | "
          f"MAPE {m['MAPE']:6.3f} % | R2 {m['R2']:.4f}")
 
 
def main() -> None:
    df = add_day_id(load_processed_data())
    groups = df["day_id"].to_numpy()
 
    X_a, y_a = get_track_a_data(df)
    X_b, y_b = get_track_b_data(df)
    X_a, y_a, X_b, y_b = map(np.asarray, (X_a, y_a, X_b, y_b))
 
    physics = df["current"].to_numpy() * df["voltage"].to_numpy()
 
    print("Baselines (pooled out-of-fold, held-out days)")
    print("-" * 78)
    print_row("Physics: current x voltage", compute_metrics(y_a, physics))
    print_row("Track A - linear", compute_metrics(y_a, linear_out_of_fold(X_a, y_a, groups)))
    print_row("Track B - linear", compute_metrics(y_b, linear_out_of_fold(X_b, y_b, groups)))
 
 
if __name__ == "__main__":
    main()
 
