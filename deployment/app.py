"""Research dashboard entrypoint."""
from pathlib import Path
import runpy
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "deployment")]
from research_pages import overview, dataset_explorer, experiments, models, methodology

st.set_page_config(page_title="Solar Power Prediction", layout="wide")
st.html("<style>" + (ROOT / "deployment/button_styles.css").read_text() + "</style>")
PAGES = ["Research Overview", "Dataset Explorer", "Experiments & Results",
         "Prediction Workspace", "Model Explorer", "Research Methodology & Limitations"]
with st.sidebar:
    st.title("Solar Power Prediction")
    page = st.radio("Research dashboard", PAGES, key="dashboard_page")
    st.caption("Solar measurements, independent evaluation and interactive predictions.")
    st.link_button("GitHub repository", "https://github.com/OluwafisayoIbrahim/solar-power-prediction-python", icon=":material/code:", width="stretch")
    st.link_button("Research preprint", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4771898", icon=":material/article:", width="stretch")
try:
    if page == "Prediction Workspace":
        runpy.run_path(str(ROOT / "deployment/prediction_workspace.py"))
    else:
        {PAGES[0]: overview, PAGES[1]: dataset_explorer, PAGES[2]: experiments,
         PAGES[4]: models, PAGES[5]: methodology}[page]()
except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
    st.warning(f"This research view is unavailable: {exc}")
