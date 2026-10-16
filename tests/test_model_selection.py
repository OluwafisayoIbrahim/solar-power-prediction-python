"""Integration checks for all model families and isolated upload training."""

from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import joblib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "deployment")]
from features import load_processed_data
from model_registry import MODELS, load_model, predict_selected
from model_image import predictor_image
from batch_prediction import read_sheet, run_batch
from training import train_uploaded, export_trained
from streamlit.testing.v1 import AppTest


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_processed_data()

    def test_all_saved_models_and_images(self):
        for key, spec in MODELS.items():
            with self.subTest(model=key):
                self.assertFalse(spec.missing_files())
                model = load_model(key)
                frame = self.frame.head(5)
                expected = model.predict(frame)
                self.assertTrue(np.isfinite(expected).all())
                np.testing.assert_allclose(model.predict(frame[frame.columns[::-1]]), expected)
                result = run_batch(frame, model, features=spec.features, predictor=predict_selected)
                np.testing.assert_allclose(result["predicted_power_w"], expected)
                image = Image.open(BytesIO(predictor_image(model)))
                self.assertEqual(image.format, "PNG")
                self.assertGreater(np.asarray(image).std(), 5)
                if spec.kind == "physics":
                    np.testing.assert_allclose(expected, frame["current"] * frame["voltage"])

    def test_training_isolation_holdout_and_exports(self):
        for key in ["b_tensorflow", "b_tuned", "b_untuned", "a_linear"]:
            with self.subTest(model=key):
                saved = load_model(key)
                before = saved.predict(self.frame.head())
                trained, evaluation, rejected, _ = train_uploaded(
                    self.frame, saved, "source_file", epochs=2)
                self.assertTrue(rejected.empty)
                heldout = evaluation.index
                train_frame = self.frame.drop(index=heldout)
                self.assertFalse(set(train_frame.source_file) & set(evaluation.source_file))
                scaler = trained.estimator[1] if saved.spec.kind == "tensorflow" else trained.estimator.steps[0][1]
                np.testing.assert_allclose(scaler.mean_, train_frame[saved.spec.features].mean())
                np.testing.assert_allclose(saved.predict(self.frame.head()), before)
                np.testing.assert_allclose(trained.predict(evaluation), evaluation["predicted_power_w"])
                with ZipFile(BytesIO(export_trained(trained))) as archive:
                    self.assertTrue(any(name.endswith(".json") for name in archive.namelist()))
                    with TemporaryDirectory() as temp:
                        archive.extractall(temp)
                        if saved.spec.kind == "tensorflow":
                            exported = load_model(key, temp)
                            np.testing.assert_allclose(exported.predict(evaluation), trained.predict(evaluation), rtol=1e-6)
                        else:
                            exported = joblib.load(Path(temp) / saved.spec.filename)
                            np.testing.assert_allclose(exported.predict(evaluation[saved.spec.features].to_numpy()), trained.predict(evaluation))

    def test_training_rejections(self):
        model = load_model("b_linear")
        for frame in [self.frame.drop(columns="power"), self.frame.head(3), self.frame.assign(power=np.nan)]:
            with self.assertRaises(ValueError):
                train_uploaded(frame, model)
        with self.assertRaises(ValueError):
            train_uploaded(self.frame, load_model("a_physics"))

    def test_cached_model_without_description(self):
        cached = load_model("a_physics")
        spec = cached.spec
        cached.spec = SimpleNamespace(key=spec.key, label=spec.label, track=spec.track,
                                      kind=spec.kind, features=spec.features)
        app = AppTest.from_file(str(ROOT / "deployment/app.py"), default_timeout=30)
        app.session_state["prediction_model"] = "a_physics"
        app.session_state["diagram_model"] = "a_physics"
        with patch("model_registry.load_model", return_value=cached):
            app.run()
        self.assertFalse(len(app.exception), str(app.exception))
        self.assertFalse(len(app.error), str(app.error))
        self.assertTrue(any("no training or learned parameters" in item.value for item in app.markdown))

    def test_ui_model_switching_and_training(self):
        app = AppTest.from_file(str(ROOT / "deployment/app.py"), default_timeout=90).run()
        for key in MODELS:
            app.selectbox(key="prediction_model").select(key).run()
            app.selectbox(key="diagram_model").select(key).run()
            next(button for button in app.button if button.label == "Predict power").click().run()
            self.assertFalse(len(app.exception), str(app.exception))
            self.assertFalse(len(app.error), str(app.error))
            self.assertTrue(any(metric.label == "Predicted power" for metric in app.metric))
        app.selectbox(key="prediction_model").select("b_linear").run()
        app.radio(key="model_source").set_value("Train from uploaded data").run()
        upload = BytesIO(self.frame.to_csv(index=False).encode())
        upload.name = "training.csv"
        with patch("streamlit.file_uploader", return_value=upload):
            app.run()
            next(button for button in app.button if button.label == "Train selected model").click().run()
            self.assertFalse(len(app.exception), str(app.exception))
            self.assertFalse(len(app.error), str(app.error))
            self.assertIn("rows", app.session_state["trained_model"].metadata)
            app.selectbox(key="prediction_model").select("a_linear").run()
            with self.assertRaises(KeyError):
                app.session_state["trained_model"]


if __name__ == "__main__":
    unittest.main()
