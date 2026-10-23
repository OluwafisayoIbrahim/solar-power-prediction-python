"""Saved model settings and the corresponding grouped-CV research scores."""
import pandas as pd
import streamlit as st
from model_registry import MODELS
from research_data import csv, config, RUNS, source_links
from research_pages import model_details

RESEARCH_LABELS = {
    "tensorflow": "TensorFlow selected", "pytorch": "PyTorch selected",
    "tuned": "Sklearn tuned", "untuned": "Sklearn untuned",
    "linear": "Linear", "physics": "Physics",
}


def research_scores(spec):
    if spec.research_run:
        run = RUNS[spec.research_run]
        table = csv(f"results/{run}/comparison.csv").assign(
            source_run=f"results/{run}", source_framework="TensorFlow")
    else:
        table = csv("results/comparisons/parity_s2/comparison.csv")
    rows = table[(table.track == spec.track) & (table.model == RESEARCH_LABELS[spec.kind])]
    if len(rows) != 1:
        raise ValueError("A unique matching research result is unavailable.")
    return rows.iloc[0]


def render_model_statistics(loaded):
    spec = MODELS[loaded.spec.key]
    st.subheader("Model settings")
    details = model_details(loaded)
    details = {name.replace("_", " "): value for name, value in details.items()}
    for field, label in [("batch_size", "Batch size"), ("seed", "Final fit seed"),
                         ("architecture_votes", "Architecture votes"),
                         ("selection_folds", "Architecture selections")]:
        if field in loaded.metadata:
            details[label] = str(loaded.metadata[field])
    st.dataframe(pd.Series(details, name="Saved value"), width="stretch")
    if "source_run" in loaded.metadata:
        st.markdown("Repository source: " + source_links(loaded.metadata["source_run"]))
    st.subheader("Held-out research performance")
    try:
        scores = research_scores(spec)
        for column, metric, unit in zip(st.columns(4), ["MAE", "RMSE", "MAPE", "R2"], [" W", " W", "%", ""]):
            column.metric(f"CV {metric}", f"{scores[metric + '_mean']:.3f}{unit}")
        variation = pd.DataFrame([
            {"Metric": metric, "Mean": scores[metric + "_mean"],
             "Seed standard deviation": scores[metric + "_std"], "Unit": unit}
            for metric, unit in [("MAE", "W"), ("RMSE", "W"), ("MAPE", "%"), ("R2", "unitless")]])
        st.dataframe(variation, hide_index=True, width="stretch", column_config={
            name: st.column_config.NumberColumn(format="%.6f")
            for name in ("Mean", "Seed standard deviation")})
        collection = spec.research_run or "Matched S2"
        st.caption(f"{collection} grouped cross-validation: metrics pool held-out predictions per seed. "
                   "Seed standard deviations are not confidence intervals. These scores evaluate "
                   "the training procedure, not an independent test of the final saved weights.")
        st.markdown("Metric source runs: " + source_links(scores.source_run))
        cfg = config(spec.research_run or ("PyTorch S2" if spec.kind == "pytorch" else "TensorFlow S2"))
        st.caption(f"Evaluation: {cfg['rows']} measurements | {cfg['folds']} outer folds | "
                   f"{cfg['inner_folds']} inner folds for selected ANNs | seeds {cfg['seeds']}")
        st.download_button("Download model research metrics", variation.to_csv(index=False),
                           f"{spec.key}_research_metrics.csv", "text/csv", key=f"metrics_{spec.key}")
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        st.info(f"Research metrics unavailable: {exc}")
