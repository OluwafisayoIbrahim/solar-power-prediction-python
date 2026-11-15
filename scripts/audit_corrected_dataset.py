"""Write reproducible data/fold counts and the no-fitting electrical baseline."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from sklearn.model_selection import GroupKFold
from corrected_protocol import COLLECTION, data_digest
from features import add_day_id, load_processed_data
from load_data import MEASUREMENT_COLUMNS, validate_research_data
from evaluate import compute_metrics


def main():
    frame = add_day_id(load_processed_data())
    validate_research_data(frame)
    folds, fold_rows = [], [0] * len(frame)
    for fold, (train, test) in enumerate(GroupKFold(7).split(frame, groups=frame.day_id), 1):
        folds.append({"fold": fold, "training_groups": frame.iloc[train].day_id.nunique(),
                      "training_rows": len(train), "test_groups": frame.iloc[test].day_id.nunique(),
                      "test_rows": len(test)})
        for row in test:
            fold_rows[row] = fold
    predicted = (frame.current * frame.voltage).to_numpy()
    metrics = compute_metrics(frame.power.to_numpy(), predicted)
    COLLECTION.mkdir(parents=True, exist_ok=True)
    (COLLECTION / "dataset_audit.json").write_text(json.dumps({
        "training_data_sha256": data_digest(), "rows": len(frame), "groups": frame.day_id.nunique(),
        "rows_per_group": sorted(frame.groupby("day_id").size().unique().tolist()),
        "repeated_measurements_excluding_filename": int(frame.duplicated(MEASUREMENT_COLUMNS).sum()),
        "excluded_file": "solar_data_tue_28st.csv", "folds": folds,
        "physics_metrics": metrics, "physics_fitting": "none",
        "status": "Data and deterministic baseline complete; full neural comparison is separate",
    }, indent=2), encoding="utf-8")
    pd.DataFrame([dict(track="A", seed=seed, model="Physics", **metrics) for seed in [0, 1, 2]]).to_csv(
        COLLECTION / "physics_metrics_per_seed.csv", index=False)
    pd.concat([pd.DataFrame(dict(track="A", seed=seed, model="Physics", row=range(len(frame)),
                                day_id=frame.day_id, fold=fold_rows, measured=frame.power,
                                predicted=predicted)) for seed in [0, 1, 2]], ignore_index=True).to_csv(
        COLLECTION / "physics_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
