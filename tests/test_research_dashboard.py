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
        for page in ["Research Overview", "Dataset Explorer", "Experiments & Results", "Model Explorer", "Research Methodology & Limitations", "Prediction Workspace"]:
            app.radio(key="dashboard_page").set_value(page).run()
            self.assertFalse(len(app.exception), str(app.exception))
            self.assertFalse(len(app.error), str(app.error))
        app.radio(key="dashboard_page").set_value("Experiments & Results").run()
        for collection in ["Historical TensorFlow", "Original scikit-learn / baselines", "Matched TensorFlow/PyTorch S2"]:
            next(item for item in app.selectbox if item.label == "Experiment collection").select(collection).run()
            self.assertFalse(len(app.exception), str(app.exception))
        app.radio(key="dashboard_page").set_value("Research Overview").run()
        next(item for item in app.button if item.label == "Explore the Dataset").click().run()
        self.assertEqual(app.session_state["dashboard_page"], "Dataset Explorer")
        self.assertFalse(len(app.exception), str(app.exception))

if __name__ == "__main__":
    unittest.main()
