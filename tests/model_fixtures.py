"""Cache isolation for tests using the current verified model exports."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "deployment")]
import streamlit as st
from model_registry import MODELS
PRODUCTION_SPECS = MODELS.copy()

class SavedModelFixtures:
    def setUp(self):
        super().setUp()
        st.cache_resource.clear()
        st.cache_data.clear()
        self.addCleanup(st.cache_resource.clear)
        self.addCleanup(st.cache_data.clear)
