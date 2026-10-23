"""Fast CPU checks for numerical behavior, selection and checkpoint integrity."""

from importlib.util import find_spec
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from models_torch import TorchANN, l2_penalty
    from train_ann_torch import fit_torch, predict_torch, select_torch, parse_args
    from train_final_torch import save_bundle, final_settings, export_final
    from predict_torch import load_torch_model, predict_measurements
    from features import TRACK_B_FEATURES


@unittest.skipUnless(HAS_TORCH, "Optional PyTorch is not installed")
class TorchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = rng.normal(size=(32, 4))
        self.y = 200 + 8 * self.X[:, 0] - 2 * self.X[:, 1]
        self.args = SimpleNamespace(activation="tanh", learning_rate=0.01, l2=0.0001,
                                    batch_size=7, epochs=60, patience=4,
                                    architectures=[(10,), (16, 8)], inner_folds=2)

    def test_shapes_initialization_and_seed(self):
        for hidden in [(10,), (16, 8)]:
            a = TorchANN(4, hidden, seed=3)
            b = TorchANN(4, hidden, seed=3)
            x = torch.ones((6, 4))
            self.assertEqual(tuple(a(x).shape), (6, 1))
            torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
            for layer in [*a.hidden, a.output]:
                bound = np.sqrt(6 / (layer.in_features + layer.out_features))
                self.assertLessEqual(layer.weight.detach().abs().max().item(), bound)
                self.assertGreater(layer.weight.detach().std().item(), 0)
                self.assertEqual(torch.count_nonzero(layer.bias).item(), 0)

    def test_l2_excludes_output_and_biases(self):
        model = TorchANN(4, (10, 8))
        penalty = l2_penalty(model, 0.02)
        expected = 0.02 * sum(layer.weight.square().sum() for layer in model.hidden)
        torch.testing.assert_close(penalty, expected)
        with torch.no_grad():
            for layer in [*model.hidden, model.output]:
                layer.bias.fill_(100)
            model.output.weight.fill_(100)
        torch.testing.assert_close(l2_penalty(model, 0.02), penalty)
        l2_penalty(model, 0.02).backward()
        self.assertIsNone(model.output.weight.grad)
        self.assertTrue(all(layer.bias.grad is None for layer in model.hidden))

    def test_fit_reduces_loss_and_repeats_exactly(self):
        first = fit_torch(self.X, self.y, (10,), 0, self.args)
        second = fit_torch(self.X, self.y, (10,), 0, self.args)
        self.assertLess(first[3]["loss"][-1], first[3]["loss"][0] * 0.5)
        np.testing.assert_array_equal(predict_torch(first, self.X), predict_torch(second, self.X))

    def test_stopping_restores_best_and_scalers_use_training_only(self):
        self.args.learning_rate = 1e-30 
        self.args.patience = 2
        x_val, y_val = self.X[:5] + 20, self.y[:5] + 100
        fitted = fit_torch(self.X[5:], self.y[5:], (10,), 0, self.args, (x_val, y_val))
        history = fitted[3]
        self.assertEqual(len(history["val_mae"]), 3)
        best_epoch = np.argmin(history["val_mae"]) + 1
        self.assertTrue(1 <= best_epoch <= self.args.epochs)
        np.testing.assert_allclose(fitted[1].mean_, self.X[5:].mean(axis=0))
        np.testing.assert_allclose(fitted[2].mean_, self.y[5:].mean())
        restored_mae = np.abs(predict_torch(fitted, x_val) - y_val).mean() / fitted[2].scale_[0]
        self.assertAlmostEqual(restored_mae, min(history["val_mae"]), places=5)
        fixed = fit_torch(self.X, self.y, (10,), 0, self.args, epochs=5)
        self.assertEqual(len(fixed[3]["mae"]), 5)
        self.assertNotIn("val_mae", fixed[3])

    def test_selection_day_isolation_and_curves(self):
        self.args.epochs = 2
        groups = np.repeat(np.arange(8), 4)
        calls = []
        def tracked(X, y, hidden, seed, args, validation=None, epochs=None):
            train_rows = [np.flatnonzero((self.X == row).all(axis=1))[0] for row in X]
            val_rows = [np.flatnonzero((self.X == row).all(axis=1))[0] for row in validation[0]]
            self.assertFalse(set(groups[train_rows]) & set(groups[val_rows]))
            calls.append(hidden)
            return fit_torch(X, y, hidden, seed, args, validation, epochs)
        with TemporaryDirectory() as directory, patch("train_ann_torch.fit_torch", side_effect=tracked):
            hidden, epochs, score = select_torch(
                self.X, self.y, groups, 0, self.args, Path(directory), "test")
            self.assertIn(hidden, self.args.architectures)
            self.assertTrue(1 <= epochs <= 2)
            self.assertTrue(np.isfinite(score))
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(list(Path(directory).glob("*.csv"))), 2)
            self.assertTrue((Path(directory) / "test.png").exists())

    def test_checkpoint_and_feature_order_and_no_overwrite(self):
        fitted = fit_torch(self.X, self.y, (10,), 0, self.args, epochs=3)
        metadata = dict(track="B", features=TRACK_B_FEATURES, hidden_layers=[10],
                        activation="tanh", learning_rate=0.01, l2=0.0001, seed=0)
        frame = pd.DataFrame(self.X, columns=TRACK_B_FEATURES)
        with TemporaryDirectory() as directory:
            save_bundle(fitted, metadata, directory)
            loaded = load_torch_model("B", directory)
            np.testing.assert_array_equal(predict_measurements(loaded, frame[frame.columns[::-1]]),
                                          predict_torch(fitted, self.X))
            with self.assertRaises(FileExistsError):
                save_bundle(fitted, metadata, directory)
            for invalid in [frame.drop(columns="hour"), frame.iloc[:0], frame.assign(hour=np.nan)]:
                with self.assertRaises(ValueError):
                    predict_measurements(loaded, invalid)

    def test_final_vote_median_smoke_rejection_and_cli(self):
        config = dict(smoke_test=False, seeds=[0], folds=3, architectures=[[10], [16, 8]], epochs=1000)
        settings = pd.DataFrame(dict(track=["A"] * 3, model=["PyTorch selected"] * 3,
                                     seed=[0] * 3, fold=[1, 2, 3],
                                     hidden=["(10,)", "(16, 8)", "(10,)"], epochs=[20, 800, 40]))
        self.assertEqual(final_settings(config, settings, "A"), ((10,), 30, 2, 3))
        with self.assertRaises(ValueError):
            final_settings({**config, "smoke_test": True}, settings, "A")
        with self.assertRaises(ValueError):
            final_settings(config, settings.iloc[:2], "A")
        self.assertEqual(parse_args([]).epochs, 1000)
        smoke = parse_args(["--smoke-test"])
        self.assertEqual((smoke.epochs, smoke.tracks, smoke.seeds), (2, ["B"], [0]))

    def test_final_export_records_provenance_and_rejects_existing_artifacts(self):
        frame = pd.DataFrame(self.X, columns=TRACK_B_FEATURES)
        frame["power"] = self.y
        frame["source_file"] = np.repeat([f"day{i}" for i in range(8)], 4)
        config = dict(vars(self.args), rows=len(frame), seeds=[0], tracks=["B"],
                      folds=2, torch=torch.__version__, smoke_test=False)
        config["architectures"] = [[10]]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "comparison.csv").write_text("test completion marker\n", encoding="utf-8")
            pd.DataFrame(dict(track=["B"] * 2, model=["PyTorch selected"] * 2,
                              seed=[0, 0], fold=[1, 2], hidden=["(10,)"] * 2,
                              epochs=[2, 4])).to_csv(run / "selected_settings.csv", index=False)
            with patch("train_final_torch.load_processed_data", return_value=frame):
                export_final(run, "B", root / "export")
            loaded = load_torch_model("B", root / "export")
            metadata = loaded[3]
            self.assertEqual(metadata["epochs"], 3)
            self.assertEqual(metadata["epoch_cap"], self.args.epochs)
            self.assertEqual(metadata["training_data_sha256"],
                             hashlib.sha256(self.X.tobytes() + self.y.tobytes()).hexdigest())
            with patch("train_final_torch.fit_torch") as fit:
                with self.assertRaises(FileExistsError):
                    export_final(run, "B", root / "export")
                fit.assert_not_called()
            config["smoke_test"] = True
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke-test"):
                export_final(run, "B", root / "forbidden")
            self.assertFalse((root / "forbidden").exists())


if __name__ == "__main__":
    unittest.main()
