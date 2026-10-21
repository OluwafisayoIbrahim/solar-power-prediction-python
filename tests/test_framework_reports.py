"""Report tests use temporary fixtures, never published research results."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from check_convergence import convergence_summary
from compare_frameworks import compare_runs
from train_ann_tf import parse_args


def fixture(path, framework, **changes):
    path.mkdir()
    config = dict(epochs=1000, learning_rate=0.01, batch_size=32, patience=30,
                  l2=0.0001, architectures=[[10]], seeds=[0, 1], folds=2,
                  inner_folds=2, tracks=["B"], activation="tanh", rows=4,
                  smoke_test=False, sklearn="test")
    config["torch" if framework == "PyTorch" else "tensorflow"] = "test"
    config.update(changes)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    selections, metrics, predictions = [], [], []
    for seed in config["seeds"]:
        for fold in range(1, 3):
            selections.append(dict(track="B", seed=seed, fold=fold, model=f"{framework} selected",
                                   hidden="(10,)", epochs=1000 if seed == 0 else 50, inner_mae=2))
        for model in ["Linear", "Sklearn untuned", "Sklearn tuned", f"{framework} selected"]:
            metrics.append(dict(track="B", seed=seed, model=model, MAE=1 + seed,
                                RMSE=2 + seed, MAPE=3 + seed, R2=0.5))
            for row in range(4):
                predictions.append(dict(track="B", seed=seed, model=model, row=row,
                                        day_id=f"d{row}", fold=1 + row % 2,
                                        measured=10 + row, predicted=9 + row))
    pd.DataFrame(selections).to_csv(path / "selected_settings.csv", index=False)
    pd.DataFrame(metrics).to_csv(path / "metrics_per_seed.csv", index=False)
    pd.DataFrame(predictions).to_csv(path / "predictions.csv", index=False)
    (path / "comparison.csv").write_text("completed fixture\n", encoding="utf-8")


class ReportTests(unittest.TestCase):
    def test_historical_convergence_and_tf_defaults(self):
        for stamp, counts in [("20261014T104851616874Z", [21, 2]),
                              ("20261019T100248937256Z", [18])]:
            _, rows = convergence_summary(ROOT / "results/tensorflow" / stamp)
            self.assertEqual([r["cap_hits"] for r in rows], counts)
        self.assertEqual(parse_args([]).epochs, 300)

    def test_comparison_population_std_and_no_overwrite(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            fixture(root / "tf", "TensorFlow")
            fixture(root / "pt", "PyTorch")
            table, notices = compare_runs(root / "tf", root / "pt", root / "out")
            self.assertFalse(notices)
            self.assertEqual(len(table), 5)
            self.assertEqual(table.iloc[0]["MAE_std"], 0.5)
            self.assertIn("2/4 at cap 1000", (root / "out/comparison.md").read_text())
            with self.assertRaises(FileExistsError):
                compare_runs(root / "tf", root / "pt", root / "out")

    def test_mismatches_and_baseline_differences_are_written(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            fixture(root / "tf", "TensorFlow")
            fixture(root / "pt", "PyTorch", batch_size=16, smoke_test=True)
            path = root / "pt/metrics_per_seed.csv"
            metrics = pd.read_csv(path)
            metrics["MAE"] = metrics["MAE"].astype(float)
            metrics.loc[metrics.model == "Sklearn untuned", "MAE"] += 0.1
            metrics.to_csv(path, index=False)
            table, notices = compare_runs(root / "tf", root / "pt", root / "out")
            self.assertEqual(len(table), 6)
            self.assertTrue(any("batch_size" in x for x in notices))
            self.assertTrue(any("Smoke-test" in x for x in notices))
            self.assertTrue(any("Sklearn untuned differs" in x for x in notices))
            for filename in ["comparison.csv", "comparison.md"]:
                self.assertIn("NOT STRICTLY COMPARABLE", (root / "out" / filename).read_text())

    def test_missing_selections_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tf"
            fixture(path, "TensorFlow")
            rows = pd.read_csv(path / "selected_settings.csv")
            rows.iloc[:2].to_csv(path / "selected_settings.csv", index=False)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                convergence_summary(path)


if __name__ == "__main__":
    unittest.main()
