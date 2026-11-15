"""Explicit historical inference fixtures for UI and session-training tests.

These models exercise supported formats; they never supply active research
evidence or change the application's production registry.
"""
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "deployment")]
import streamlit as st
from model_registry import MODELS

PRODUCTION_SPECS = MODELS.copy()


class SavedModelFixtures:
    def setUp(self):
        super().setUp()
        fixtures = {}
        for key, spec in PRODUCTION_SPECS.items():
            if spec.kind == "physics" or spec.research_run:
                fixtures[key] = spec
                continue
            directory = {"tensorflow": "results/exports/tf_s2",
                         "pytorch": "results/exports/torch_s2"}.get(spec.kind, "results/models")
            fixtures[key] = replace(spec, artifact_directory=directory,
                                    research_run="PyTorch S2" if spec.kind == "pytorch" else "TensorFlow S2",
                                    label=f"Track {spec.track} - {spec.kind} (historical test fixture)")
        st.cache_resource.clear()
        st.cache_data.clear()
        registry = patch.dict(MODELS, fixtures)
        registry.start()

        def cleanup():
            registry.stop()
            st.cache_resource.clear()
            st.cache_data.clear()
        self.addCleanup(cleanup)
