"""Scientific evidence integrity and the continuous manuscript presentation."""
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "deployment"), str(ROOT / "src")]
import manuscript_content as content
from manuscript_data import (COMPARISON, load_evidence, score, result_table,
                             validate_predictions, diagnostic_predictions)
from manuscript_figures import daily_errors


class ManuscriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = load_evidence()

    def test_sources_reconcile_and_duplicate_audit_is_explicit(self):
        # Loading verifies every saved seed score against raw held-out predictions.
        audit = self.evidence["audit"]
        self.assertEqual((audit["rows"], audit["groups"]), (189, 21))
        self.assertEqual(audit["duplicate_records"], 0)
        self.assertEqual(audit["duplicate_measurements"], 9)
        self.assertEqual(audit["duplicate_sources"], ["solar_data_tue_21st.csv", "solar_data_tue_28st.csv"])
        self.assertEqual((audit["interval_min"], audit["interval_max"]), (35, 92))
        for track in ("A", "B"):
            table = result_table(self.evidence, track)
            for _, row in table.iterrows():
                expected = f"{score(self.evidence, track, row.Model):.3f} ± {score(self.evidence, track, row.Model, stat='std'):.3f}"
                self.assertEqual(row["MAE (W)"], expected)
        text = content.abstract(self.evidence)
        self.assertTrue(180 <= len(text.split()) <= 250)
        for track, model in [("A", "Linear"), ("A", "TensorFlow selected"),
                             ("B", "Sklearn tuned"), ("B", "Linear")]:
            self.assertIn(f"{score(self.evidence, track, model):.3f} W", text)

    def test_corrupt_prediction_records_are_rejected(self):
        cfg = self.evidence["configs"]["TensorFlow"]
        rows = self.evidence["predictions"]["TensorFlow"]
        metrics = self.evidence["seeds"]["TensorFlow"]
        frame = self.evidence["dataset"]
        for corruption in ("prediction", "identity", "group", "partition"):
            damaged = rows.copy()
            if corruption == "prediction":
                damaged.loc[0, "predicted"] += 100
            elif corruption == "identity":
                damaged.loc[0, "row"] = damaged.loc[1, "row"]
            elif corruption == "group":
                damaged.loc[0, "fold"] = 1
            else:
                # Preserves each group but changes the documented outer partition IDs.
                damaged["fold"] = cfg["folds"] + 1 - damaged.fold
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                validate_predictions(damaged, metrics, cfg, frame)

    def test_diagnostics_use_fixed_oof_rows_and_units(self):
        for track in ("A", "B"):
            predictions = diagnostic_predictions(self.evidence, track)
            self.assertEqual(set(predictions.seed), {0})
            self.assertEqual(len(predictions), 4 * 189)
            self.assertEqual(predictions.groupby("model").day_id.nunique().tolist(), [21] * 4)
            np.testing.assert_allclose(predictions.residual, predictions.measured - predictions.predicted)
            daily = daily_errors(predictions)
            self.assertEqual(len(daily), 4 * 21)
            self.assertTrue((daily["MAE (W)"] >= 0).all())

    def test_citations_and_evidence_inventory(self):
        prose = " ".join(str(getattr(content, name)) for name in
                         ("INTRODUCTION", "RELATED_WORK", "HISTORICAL", "REASSESSMENT", "VALIDATION"))
        citations = {int(n) for n in re.findall(r"\[\[(\d+)\]\]", prose)}
        self.assertEqual(citations, set(range(1, len(content.REFERENCES) + 1)))
        inventory = json.loads((ROOT / "deployment/manuscript_evidence.json").read_text(encoding="utf-8"))
        for claim in inventory["claims"]:
            for source in claim["sources"]:
                if not source.startswith("https://"):
                    self.assertTrue((ROOT / source).exists(), source)
        self.assertEqual(len(inventory["references"]), len(content.REFERENCES))
        self.assertTrue((ROOT / COMPARISON).is_file())

    def test_manuscript_all_sections_figures_and_track_controls_without_models(self):
        with patch("model_registry.load_model", side_effect=AssertionError("The manuscript must not load fitted models")):
            app = AppTest.from_file(str(ROOT / "deployment/app.py"), default_timeout=90)
            # Seed a retired session before the updated widget is constructed.
            app.session_state["dashboard_page"] = "Research Methodology & Limitations"
            app.run()
            self.assertEqual(app.session_state["dashboard_page"], "Research Manuscript")
            self.assertFalse(len(app.exception), str(app.exception))
            self.assertFalse(len(app.error), str(app.error))
            headers = {item.value for item in app.header}
            self.assertTrue({label for _, label in content.SECTIONS} <= headers)
            captions = [item.value for item in app.caption]
            diagrams = app.get("graphviz_chart")
            self.assertEqual(len(diagrams), 2)
            for diagram in diagrams:
                self.assertIn('bgcolor="transparent"', diagram.proto.spec)
                self.assertNotIn('filled', diagram.proto.spec)
                self.assertNotIn('#ffffff', diagram.proto.spec)
            for number in range(1, 8):
                self.assertTrue(any(value.startswith(f"Figure {number}.") for value in captions))
            for number in range(1, 7):
                self.assertTrue(any(value.startswith(f"Table {number}.") for value in captions))
            self.assertTrue(any("9 repeated measurement" in item.value for item in app.warning))
            self.assertTrue(any("S1 had 1 selected outer refit(s) at the epoch cap and S2 had 0" in item.value
                                for item in app.markdown))
            app.selectbox(key="manuscript_track").select("A").run()
            app.selectbox(key="manuscript_metric").select("RMSE").run()
            self.assertFalse(len(app.exception), str(app.exception))
            self.assertFalse(len(app.error), str(app.error))
            self.assertTrue(any("Track A, seed 0" in item.value for item in app.caption))


if __name__ == "__main__":
    unittest.main()
