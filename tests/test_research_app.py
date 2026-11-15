"""Exercise the updated-data transition without substituting historical scores."""
from pathlib import Path
import sys
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "deployment"), str(ROOT / "src")]
from research_protocol import COLLECTION, verify_run
from model_registry import MODELS
from model_statistics import research_scores


class ResearchAppTests(unittest.TestCase):
    def test_registry_uses_fresh_exports_and_labels_history(self):
        for spec in MODELS.values():
            if spec.research_run:
                self.assertIn("superseded", spec.label)
            elif spec.kind != "physics":
                self.assertIn("research_20_days", spec.model_dir.as_posix())

    def test_manuscript_and_overview_do_not_present_partial_results(self):
        app = AppTest.from_file(str(ROOT / "deployment/app.py"), default_timeout=90).run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertFalse(app.error, str(app.error))
        app.radio(key="dashboard_page").set_value("Research Manuscript").run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertFalse(app.error, str(app.error))
        if not (ROOT / "manuscript/publication_ready.json").is_file():
            self.assertTrue(any("updated manuscript" in item.value.lower() for item in app.warning))
            self.assertEqual(len(app.dataframe), 0)
        if not (COLLECTION / "completed.json").is_file():
            with self.assertRaisesRegex(ValueError, "incomplete"):
                research_scores(MODELS["a_physics"])

    def test_physics_prediction_remains_available(self):
        app = AppTest.from_file(str(ROOT / "deployment/app.py"), default_timeout=90).run()
        app.radio(key="dashboard_page").set_value("Predict").run()
        app.selectbox(key="prediction_model").set_value("a_physics").run()
        app.number_input(key="input_current").set_value(7.)
        app.number_input(key="input_voltage").set_value(35.)
        next(button for button in app.button if button.label == "Predict power").click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertFalse(app.error, str(app.error))
        result = next(metric for metric in app.metric if metric.label == "Predicted power")
        self.assertEqual(result.value, "245.00 W")

    def test_completed_research_is_reconciled_if_available(self):
        if not (COLLECTION / "completed.json").is_file():
            self.skipTest("Full updated research runs are still in progress; no replacement scores accepted")
        from research_protocol import completed_collection
        for framework, path in completed_collection()["runs"].items():
            verify_run(ROOT / path, f"{framework} selected")


if __name__ == "__main__":
    unittest.main()
