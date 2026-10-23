"""Read-only research artifacts and chart transformations."""
import json
from pathlib import PurePosixPath
from urllib.parse import quote
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from features import add_day_id, load_processed_data
from evaluate import calculate_mape

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_TREE = "https://github.com/OluwafisayoIbrahim/solar-power-prediction-python/tree/dev/"

def repository_source(value):
    """Display portable artifact paths without the export machine's directory."""
    sources = []
    for source in str(value).split(";"):
        parts = PurePosixPath(source.strip().replace("\\", "/")).parts
        if "results" not in parts:
            sources.append("Repository source unavailable")
            continue
        relative = parts[parts.index("results"):]
        sources.append("/".join(relative) if ".." not in relative else "Repository source unavailable")
    return "; ".join(sources)

def source_links(value):
    return " | ".join(f"[{path}]({REPOSITORY_TREE}{quote(path, safe='/')})"
                      for path in repository_source(value).split("; ") if path.startswith("results/"))
UNITS = {"solar_irradiance": "W/m2", "temperature": "deg C", "humidity": "%",
         "current": "A", "voltage": "V", "power": "W", "hour": "hour"}
DESCRIPTIONS = {"solar_irradiance": "Incident solar radiation per unit area",
                "temperature": "Ambient temperature", "humidity": "Relative humidity",
                "current": "Measured electrical current", "voltage": "Measured electrical voltage",
                "power": "Measured power output", "hour": "Decimal measurement time of day"}
RUNS = {"Original TensorFlow (300 epochs)": "tensorflow/20261014T104851616874Z",
        "Historical TensorFlow Track A (1000 epochs)": "tensorflow/20261019T100248937256Z",
        "TensorFlow S1": "tensorflow/convergence_s1",
        "TensorFlow S2": "tensorflow/convergence_s2", "PyTorch S2": "pytorch/parity_s2"}

@st.cache_data
def dataset():
    return add_day_id(load_processed_data())

@st.cache_data
def csv(relative):
    frame = pd.read_csv(ROOT / relative)
    if "source_run" in frame:
        frame["source_run"] = frame["source_run"].map(repository_source)
    return frame

@st.cache_data
def config(run):
    return json.loads((ROOT / "results" / RUNS[run] / "config.json").read_text())

def summary(frame):
    return {"Measurements": len(frame), "Collection days": frame.day_id.nunique(),
            "Duplicate records": int(frame.drop(columns="day_id").duplicated().sum()),
            "Missing values": int(frame.isna().sum().sum())}

def errors(frame):
    result = frame.copy()
    result["residual"] = result.measured - result.predicted
    result["absolute_error"] = result.residual.abs()
    return result

def metric_values(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    residual = actual - predicted
    denominator = ((actual - actual.mean()) ** 2).sum()
    return {"MAE (W)": float(np.mean(abs(residual))),
            "RMSE (W)": float(np.sqrt(np.mean(residual ** 2))),
            "MAPE (%)": calculate_mape(actual, predicted) if np.any(actual != 0) else None,
            "R2": float(1 - (residual ** 2).sum() / denominator) if denominator > 0 else None}

def range_warnings(frame):
    reference = dataset()
    return [f"{name}: outside observed {reference[name].min():g} to {reference[name].max():g} {unit}"
            for name, unit in UNITS.items() if name in frame and name in reference
            and ((pd.to_numeric(frame[name], errors="coerce") < reference[name].min()) |
                 (pd.to_numeric(frame[name], errors="coerce") > reference[name].max())).any()]
