"""Manuscript presentation and evidence integrity, including pending research."""
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "deployment"), str(ROOT / "src")]
import manuscript_content as content
from manuscript_data import (load_evidence, audit_dataset, validate_predictions,
                             diagnostic_predictions, result_table)
from manuscript_figures import daily_errors, VALIDATION
from research_data import dataset
from research_protocol import COLLECTION
from evaluate import compute_metrics


class ManuscriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = dataset()
        cls.audit = audit_dataset(cls.frame)
        # In-memory fixtures exercise reconciliation and presentation only.
        cls.cfg = dict(tracks=["A", "B"], seeds=[0], folds=7, tensorflow="fixture")
        folds = np.zeros(len(cls.frame), dtype=int)
        for fold, (_, test) in enumerate(GroupKFold(7).split(cls.frame, groups=cls.frame.day_id), 1):
            folds[test] = fold
        cls.predictions, cls.metrics = {}, {}
        for framework in ["TensorFlow", "PyTorch"]:
            frames, scores = [], []
            for track in ["A", "B"]:
                models = ["Linear", "Sklearn tuned", "Sklearn untuned", f"{framework} selected"]
                if track == "A":
                    models.append("Physics")
                for offset, model in enumerate(models):
                    predicted = cls.frame.power.to_numpy() + (offset + 1) * .1
                    frames.append(pd.DataFrame(dict(track=track, seed=0, model=model,
                        row=range(len(cls.frame)), day_id=cls.frame.day_id, fold=folds,
                        measured=cls.frame.power, predicted=predicted)))
                    scores.append(dict(track=track, seed=0, model=model,
                        **compute_metrics(cls.frame.power.to_numpy(), predicted)))
            cls.predictions[framework] = pd.concat(frames, ignore_index=True)
            cls.metrics[framework] = pd.DataFrame(scores)

    def test_dataset_and_methods_describe_daily_partitions(self):
        self.assertEqual((self.audit["rows"], self.audit["groups"]), (180, 20))
        self.assertEqual(self.audit["duplicate_measurements"], 0)
        self.assertEqual(self.audit["duplicate_sources"], [])
        self.assertEqual((self.audit["interval_min"], self.audit["interval_max"]), (35, 92))
        self.assertIn("153 rows", content.VALIDATION)
        self.assertIn("18 rows", content.VALIDATION)
        self.assertIn("20 daily groups / 180 rows", VALIDATION)
        self.assertIn("One repeated daily file was excluded before evaluation", content.LIMITATIONS)
        self.assertNotIn("folds 1 and 7", content.LIMITATIONS)

    def test_corrupt_prediction_records_are_rejected(self):
        predictions = self.predictions["TensorFlow"]
        metrics = self.metrics["TensorFlow"]
        validate_predictions(predictions, metrics, self.cfg, self.frame)
        for corruption in ["predicted", "row", "day_id", "fold", "partition"]:
            damaged = predictions.copy()
            if corruption == "partition":
                damaged["fold"] = 8 - damaged.fold
            else:
                damaged.loc[0, corruption] = "unknown.csv" if corruption == "day_id" else 999
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                validate_predictions(damaged, metrics, self.cfg, self.frame)

    def test_diagnostics_use_all_seed_zero_heldout_rows(self):
        evidence = dict(dataset=self.frame, predictions=self.predictions)
        for track in ["A", "B"]:
            predictions = diagnostic_predictions(evidence, track)
            self.assertEqual(len(predictions), 4 * 180)
            self.assertEqual(set(predictions.seed), {0})
            self.assertEqual(predictions.groupby("model").day_id.nunique().tolist(), [20] * 4)
            np.testing.assert_allclose(predictions.residual, predictions.measured - predictions.predicted)
            self.assertEqual(len(daily_errors(predictions)), 4 * 20)

    def test_citations_and_inventory_keep_history_separate(self):
        prose = " ".join(getattr(content, name) for name in
            ["INTRODUCTION", "RELATED_WORK", "HISTORICAL", "REASSESSMENT", "VALIDATION"])
        self.assertEqual({int(n) for n in re.findall(r"\[\[(\d+)\]\]", prose)}, set(range(1, 7)))
        inventory = json.loads((ROOT / "deployment/manuscript_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(len(inventory["references"]), 6)
        self.assertTrue((ROOT / inventory["historical_inventory"]).is_file())
        for claim in inventory["claims"]:
            for source in claim["sources"]:
                self.assertTrue((ROOT / source).exists(), source)
        self.assertIn("README at this commit does not state an input count", content.HISTORICAL)
        self.assertIn("not a dataset-specific finding", content.RELATED_WORK)
        self.assertIn("does not independently establish when the rule was decided", content.VALIDATION)

    def test_research_loader_requires_completed_collection(self):
        load_evidence.clear()
        if not (COLLECTION / "completed.json").is_file():
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_evidence()
            return
        evidence = load_evidence()
        self.assertEqual(evidence["audit"]["rows"], 180)
        self.assertEqual(evidence["audit"]["groups"], 20)
        for track in ["A", "B"]:
            self.assertFalse(result_table(evidence, track).empty)
        self.assertIn(f"{evidence['audit']['power_iv_mae']:.3f} W", content.abstract(evidence))

    def test_manuscript_route_never_loads_inference_models(self):
        with patch("model_registry.load_model", side_effect=AssertionError("Inference is separate from manuscript evidence")):
            app = AppTest.from_file(str(ROOT / "deployment/app.py"), default_timeout=90)
            app.session_state["dashboard_page"] = "Research Methodology & Limitations"
            app.run()
            self.assertEqual(app.session_state["dashboard_page"], "Research Manuscript")
            self.assertFalse(app.exception, str(app.exception))
            self.assertFalse(app.error, str(app.error))
            if not (ROOT / "manuscript/publication_ready.json").is_file():
                self.assertTrue(any("not available yet" in item.value for item in app.warning))
                self.assertEqual(len(app.dataframe), 0)
            else:
                self.assertTrue({label for _, label in content.SECTIONS} <= {item.value for item in app.header})
                for track in ["A", "B"]:
                    app.selectbox(key="manuscript_track").select(track).run()
                    self.assertFalse(app.exception, str(app.exception))


if __name__ == "__main__":
    unittest.main()
