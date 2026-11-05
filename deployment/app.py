"""Research dashboard entrypoint."""
from pathlib import Path
import runpy
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "deployment")]
from research_pages import overview, dataset_explorer, experiments, models
from manuscript import render_manuscript

st.set_page_config(page_title="Solar Power Prediction", layout="wide")
st.html("<style>" + (ROOT / "deployment/button_styles.css").read_text() + "</style>")
PAGES = ["Overview", "Research Manuscript", "Results Explorer", "Predict",
         "Train / Experiment", "Dataset Explorer", "Model Explorer"]
ALIASES = {"Research Overview": "Overview", "Experiments & Results": "Results Explorer",
           "Prediction Workspace": "Predict", "Research Methodology & Limitations": "Research Manuscript"}
previous = st.session_state.get("dashboard_page")
if previous in ALIASES:
    st.session_state["dashboard_page"] = ALIASES[previous]
DESTINATIONS = {"manuscript": "Research Manuscript", "results": "Results Explorer",
                "predict": "Predict", "train": "Train / Experiment",
                "dataset": "Dataset Explorer", "models": "Model Explorer"}
requested = st.query_params.get("view")
if requested in DESTINATIONS:
    st.session_state["dashboard_page"] = DESTINATIONS[requested]
    del st.query_params["view"]
with st.sidebar:
    st.title("Solar Power Prediction")
    page = st.radio("Research dashboard", PAGES, key="dashboard_page")
    st.caption("Read the research manuscript, explore saved evidence, or use the interactive demonstration.")
    st.link_button("GitHub repository", "https://github.com/OluwafisayoIbrahim/solar-power-prediction-python", icon=":material/code:", width="stretch")
    st.link_button("Original study preprint (SSRN)", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4771898", icon=":material/article:", width="stretch")
try:
    if page in ("Predict", "Train / Experiment"):
        runpy.run_path(str(ROOT / "deployment/prediction_workspace.py"),
                       init_globals={"WORKSPACE_VIEW": "predict" if page == "Predict" else "train"})
    else:
        {"Research Manuscript": render_manuscript, "Overview": overview,
         "Dataset Explorer": dataset_explorer, "Results Explorer": experiments,
         "Model Explorer": models}[page]()
except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
    st.warning(f"This research view is unavailable: {exc}")
