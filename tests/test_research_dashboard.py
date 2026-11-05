"""Read-only data provenance, transformations and dashboard navigation checks."""
from pathlib import Path
import sys
import unittest
from io import BytesIO
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "deployment"), str(ROOT / "src")]
from research_data import dataset, summary, errors, metric_values, range_warnings, UNITS, config, repository_source, source_links

class DashboardTests(unittest.TestCase):
    def test_portable_repository_sources(self):
        expected = "results/tensorflow/20261019T100248937256Z"
        source = r"C:\Users\user\Documents\fisayo\solar-power-prediction-python\results\tensorflow\20261019T100248937256Z"
        self.assertEqual(repository_source(source), expected)
        self.assertEqual(repository_source("/home/app/project/" + expected), expected)
        self.assertEqual(repository_source(expected), expected)
        self.assertIn("/tree/dev/" + expected, source_links(source))
        self.assertNotIn("C:", source_links(source))
        self.assertEqual(repository_source("C:/private/file"), "Repository source unavailable")

    def test_prediction_comparisons(self):
        app = AppTest.from_file(str(ROOT / "deployment/prediction_workspace.py"), default_timeout=90).run()
        next(item for item in app.multiselect if item.label == "Compare saved models on these inputs").set_value(["b_linear", "b_pytorch"]).run()
        next(item for item in app.button if item.label == "Compare predictions").click().run()
        self.assertFalse(len(app.exception), str(app.exception))
        self.assertFalse(len(app.error), str(app.error))
        upload = BytesIO(dataset().head(5).to_csv(index=False).encode())
        upload.name = "comparison.csv"
        with patch("streamlit.file_uploader", return_value=upload):
            app.run()
            next(item for item in app.multiselect if item.label == "Compare saved models on this spreadsheet").set_value(["a_physics", "b_pytorch"]).run()
            next(item for item in app.button if item.label == "Compare spreadsheet predictions").click().run()
            self.assertFalse(len(app.exception), str(app.exception))
            self.assertFalse(len(app.error), str(app.error))

    def test_dataset_and_transformations(self):
        frame = dataset()
        self.assertEqual(summary(frame)["Measurements"], 189)
        self.assertEqual(summary(frame)["Collection days"], 21)
        self.assertEqual(frame.groupby("day_id").size().unique().tolist(), [9])
        self.assertEqual(UNITS["power"], "W")
        self.assertEqual(config("TensorFlow S2")["batch_size"], 16)
        sample = pd.DataFrame({"measured": [0., 4.], "predicted": [1., 2.]})
        self.assertEqual(errors(sample).residual.tolist(), [-1., 2.])
        self.assertEqual(metric_values(sample.measured, sample.predicted)["MAE (W)"], 1.5)
        self.assertIsNone(metric_values([0., 0.], [1., 2.])["MAPE (%)"])
        self.assertTrue(range_warnings(pd.DataFrame({"temperature": [1000.]})))

    def test_all_research_pages_and_views(self):
        app = AppTest.from_file(str(ROOT / "deployment/app.py"), default_timeout=90).run()
        self.assertEqual(app.session_state["dashboard_page"], "Overview")
        options = list(app.radio(key="dashboard_page").options)
        self.assertEqual(options[:5], ["Overview", "Research Manuscript", "Results Explorer", "Predict", "Train / Experiment"])
        self.assertNotIn("Research Methodology & Limitations", options)
        for page in options:
            app.radio(key="dashboard_page").set_value(page).run()
            self.assertFalse(len(app.exception), str(app.exception))
            self.assertFalse(len(app.error), str(app.error))
        app.radio(key="dashboard_page").set_value("Results Explorer").run()
        for collection in ["Historical TensorFlow", "Original scikit-learn / baselines", "Matched TensorFlow/PyTorch S2"]:
            next(item for item in app.selectbox if item.label == "Experiment collection").select(collection).run()
            self.assertFalse(len(app.exception), str(app.exception))
        app.radio(key="dashboard_page").set_value("Overview").run()
        next(item for item in app.button if item.label == "Explore the Dataset").click().run()
        self.assertEqual(app.session_state["dashboard_page"], "Dataset Explorer")
        self.assertFalse(len(app.exception), str(app.exception))

    def test_overview_consistency_links_and_workspace_separation(self):
        from manuscript_content import TITLE
        from manuscript_data import load_evidence, score
        evidence = load_evidence()
        app = AppTest.from_file(str(ROOT / "deployment/app.py"), default_timeout=90).run()
        self.assertEqual(app.title[0].value, TITLE)
        prose = " ".join(item.value for item in app.markdown)
        self.assertIn(f"{score(evidence, 'B', 'Sklearn tuned'):.3f} W", prose)
        self.assertIn("?view=manuscript#discussion", prose)
        self.assertIn("?view=manuscript#limitations", prose)
        app.query_params["view"] = "manuscript"
        app.run()
        self.assertEqual(app.session_state["dashboard_page"], "Research Manuscript")
        self.assertEqual(app.title[0].value, TITLE)
        app.radio(key="dashboard_page").set_value("Predict").run()
        self.assertNotIn("Train From Uploaded Data", [item.label for item in app.tabs])
        self.assertFalse(any(item.label == "Model source" for item in app.radio))
        self.assertFalse(any(item.label == "Training measurements" for item in app.get("file_uploader")))
        app.session_state["batch_result"] = pd.DataFrame({"prediction_model": ["stale saved prediction"]})
        app.radio(key="dashboard_page").set_value("Train / Experiment").run()
        self.assertIn("Train From Uploaded Data", [item.label for item in app.tabs])
        self.assertTrue(any(item.label == "Training measurements" for item in app.get("file_uploader")))
        self.assertNotIn("batch_result", app.session_state)
        self.assertFalse(len(app.exception), str(app.exception))
        self.assertFalse(len(app.error), str(app.error))

    def test_training_route_fits_and_predicts_with_session_model(self):
        app = AppTest.from_file(str(ROOT / "deployment/app.py"), default_timeout=90).run()
        app.radio(key="dashboard_page").set_value("Train / Experiment").run()
        app.selectbox(key="prediction_model").select("b_linear").run()
        upload = BytesIO(dataset().head(30).to_csv(index=False).encode())
        upload.name = "training.csv"
        def uploaded(label, **kwargs):
            return upload if label == "Training measurements" else None
        with patch("streamlit.file_uploader", side_effect=uploaded):
            app.run()
            next(item for item in app.button if item.label == "Train selected model").click().run()
            self.assertEqual(app.session_state["trained_model"].spec.key, "b_linear")
            next(item for item in app.button if item.label == "Predict power").click().run()
            self.assertTrue(any(item.label == "Predicted power" for item in app.metric))
            self.assertFalse(len(app.exception), str(app.exception))
            self.assertFalse(len(app.error), str(app.error))

    def test_observed_prediction_defaults_and_both_tracks(self):
        app = AppTest.from_file(str(ROOT / "deployment/prediction_workspace.py"), default_timeout=90).run()
        for track, model in [("A", "a_linear"), ("B", "b_linear")]:
            next(item for item in app.selectbox if item.label == "Prediction track").select(track).run()
            app.selectbox(key="prediction_model").set_value(model).run()
            self.assertEqual(app.number_input(key="input_solar_irradiance").value,
                             float(dataset().solar_irradiance.median()))
            next(item for item in app.button if item.label == "Predict power").click().run()
            self.assertFalse(len(app.exception), str(app.exception))
            self.assertFalse(len(app.error), str(app.error))
            self.assertTrue(any(item.label == "Predicted power" for item in app.metric))

if __name__ == "__main__":
    unittest.main()
