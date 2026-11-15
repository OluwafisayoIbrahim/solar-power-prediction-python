"""Regression checks for the researcher-confirmed accidental duplicate."""
from pathlib import Path
import sys
import tempfile
import unittest
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import add_day_id, load_processed_data
from load_data import (MEASUREMENT_COLUMNS, load_all_solar_data, preprocess_data,
                       validate_research_data)
from corrected_protocol import choose_tensorflow, data_digest, verify_run
from evaluate import compute_metrics


class CorrectedDataTests(unittest.TestCase):
    def test_selection_uses_cap_hits_then_inner_evidence_without_outer_files(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ["s1", "s2"]]
            for path in paths:
                path.mkdir()
                cfg = dict(epochs=1000, architectures=[[10]], seeds=[0, 1, 2],
                           tracks=["A", "B"], folds=7, inner_folds=3, rows=180,
                           groups=20, training_data_sha256=data_digest(), tensorflow="test")
                (path / "config.json").write_text(json.dumps(cfg))
                rows = [dict(track=track, seed=seed, fold=fold, model="TensorFlow selected",
                             hidden="(10,)", epochs=50, inner_mae=1 if path == paths[0] else 2)
                        for track in ["A", "B"] for seed in [0, 1, 2] for fold in range(1, 8)]
                pd.DataFrame(rows).to_csv(path / "selected_settings.csv", index=False)
            self.assertEqual(choose_tensorflow(*paths)["selected"], "S1")
            rows = pd.read_csv(paths[0] / "selected_settings.csv")
            rows.loc[0, "epochs"] = 1000
            rows.to_csv(paths[0] / "selected_settings.csv", index=False)
            self.assertEqual(choose_tensorflow(*paths)["selected"], "S2")
            rows.loc[0, "epochs"] = 50
            rows["inner_mae"] = 2
            rows.to_csv(paths[0] / "selected_settings.csv", index=False)
            self.assertEqual(choose_tensorflow(*paths)["selected"], "S1")

    def test_historical_runs_cannot_be_promoted_as_corrected(self):
        with self.assertRaisesRegex(ValueError, "Not a full corrected"):
            verify_run(ROOT / "results/tensorflow/convergence_s2", "TensorFlow selected")

    def test_corrected_prediction_reconciliation_rejects_corruption(self):
        # Synthetic predictions exist only in a temporary test directory.
        frame = add_day_id(load_processed_data())
        folds = np.zeros(180, dtype=int)
        for fold, (_, test) in enumerate(GroupKFold(7).split(frame, groups=frame.day_id), 1):
            folds[test] = fold
        predictions, metrics = [], []
        for track in ["A", "B"]:
            for seed in [0, 1, 2]:
                for model in ["Linear", "Sklearn tuned", "Sklearn untuned", "TensorFlow selected"] + (["Physics"] if track == "A" else []):
                    values = (frame.current * frame.voltage).to_numpy()
                    predictions.append(pd.DataFrame(dict(track=track, seed=seed, model=model,
                                                        row=range(180), day_id=frame.day_id,
                                                        fold=folds, measured=frame.power, predicted=values)))
                    metrics.append(dict(track=track, seed=seed, model=model,
                                        **compute_metrics(frame.power.to_numpy(), values)))
        predictions = pd.concat(predictions, ignore_index=True)
        metrics = pd.DataFrame(metrics)
        summary = metrics.groupby(["track", "model"])[["MAE", "RMSE", "MAPE", "R2"]].agg(
            ["mean", lambda values: np.std(values, ddof=0)])
        summary.columns = [f"{metric}_{'std' if stat == '<lambda_0>' else stat}" for metric, stat in summary.columns]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(dict(rows=180, groups=20,
                folds=7, inner_folds=3, seeds=[0, 1, 2], tracks=["A", "B"],
                training_data_sha256=data_digest(), tensorflow="test")))
            metrics.to_csv(path / "metrics_per_seed.csv", index=False)
            summary.to_csv(path / "comparison.csv")
            predictions.to_csv(path / "predictions.csv", index=False)
            verify_run(path, "TensorFlow selected")
            for column, value in [("row", 1), ("day_id", "incorrect_day.csv"),
                                  ("fold", 0), ("measured", 999), ("predicted", 999)]:
                with self.subTest(column=column):
                    bad = predictions.copy()
                    bad.loc[0, column] = value
                    bad.to_csv(path / "predictions.csv", index=False)
                    with self.assertRaises(ValueError):
                        verify_run(path, "TensorFlow selected")

    def test_processed_data_reproduce_raw_and_have_no_measurement_copies(self):
        repeated = pd.read_csv(ROOT / "data/excluded/solar_data_tue_28st.csv")
        pd.testing.assert_frame_equal(repeated, pd.read_csv(ROOT / "data/raw/solar_data_tue_21st.csv"))
        self.assertFalse(repeated.equals(pd.read_csv(ROOT / "data/raw/solar_data_tue_28th.csv")))
        expected = preprocess_data(load_all_solar_data())
        actual = load_processed_data()
        expected["time"] = expected.time.astype(str)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        validate_research_data(actual)
        self.assertEqual((len(actual), actual.source_file.nunique()), (180, 20))
        self.assertFalse(actual.duplicated(MEASUREMENT_COLUMNS).any())
        self.assertIn("solar_data_tue_21st.csv", set(actual.source_file))
        self.assertIn("solar_data_tue_28th.csv", set(actual.source_file))

    def test_duplicate_reentry_fails_even_with_a_new_filename(self):
        frame = load_processed_data()
        bad = frame.copy()
        bad.loc[0, "source_file"] = "solar_data_tue_28st.csv"
        with self.assertRaisesRegex(ValueError, "Accidentally repeated"):
            validate_research_data(bad)
        bad = frame.copy()
        bad.loc[9, MEASUREMENT_COLUMNS] = bad.loc[0, MEASUREMENT_COLUMNS].values
        with self.assertRaisesRegex(ValueError, "Identical measurement"):
            validate_research_data(bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solar_data_tue_28st.csv"
            path.write_bytes((ROOT / "data/excluded/solar_data_tue_28st.csv").read_bytes())
            with self.assertRaisesRegex(ValueError, "Excluded accidental"):
                load_all_solar_data(Path(directory))

    def test_outer_and_inner_partitions_have_no_shared_groups_or_measurements(self):
        frame = add_day_id(load_processed_data())
        identities = pd.util.hash_pandas_object(frame[MEASUREMENT_COLUMNS], index=False).to_numpy()
        def check(part, train, test):
            self.assertFalse(set(frame.iloc[part[train]].day_id) & set(frame.iloc[part[test]].day_id))
            self.assertFalse(set(identities[part[train]]) & set(identities[part[test]]))
        sizes = []
        for train, test in GroupKFold(7).split(frame, groups=frame.day_id):
            check(np.arange(len(frame)), train, test)
            sizes.append((len(train), len(test)))
            for inner_train, inner_test in GroupKFold(3).split(train, groups=frame.iloc[train].day_id):
                check(train, inner_train, inner_test)
        self.assertEqual(sorted(sizes), [(153, 27)] * 6 + [(162, 18)])


if __name__ == "__main__":
    unittest.main()
