"""Continuous, read-only research article; operational workspaces stay separate."""
import json

import pandas as pd
import streamlit as st

import manuscript_content as content
import manuscript_figures as figures
from manuscript_data import (COMPARISON, RUN_PATHS, load_evidence, score, result_table,
                             dataset_table, track_table, model_table, diagnostic_predictions)
from research_data import ROOT, REPOSITORY_TREE


def heading(anchor, title):
    st.header(title, anchor=anchor)


def table(number, title, frame, source):
    st.caption(f"Table {number}. {title}")
    st.dataframe(frame, hide_index=True, width="stretch", alt=f"Table {number}. {title}",
                 column_config={name: st.column_config.NumberColumn(format="%.3f")
                                for name in frame.select_dtypes("number").columns})
    st.caption("Source: " + source)


def figure(number, title, chart, interpretation, source, diagram=False):
    if diagram:
        st.graphviz_chart(chart, width="stretch", alt=f"Figure {number}. {title}")
    else:
        st.altair_chart(chart, width="stretch", alt=f"Figure {number}. {title}")
    st.caption(f"Figure {number}. {title}")
    st.markdown(interpretation)
    st.caption("Source: " + source)


def render_manuscript():
    # Graphviz's SVG defaults are black, independent of the app theme. Preserve
    # its original geometry and transparent canvas while inheriting text colour.
    st.html("""<style>
        .st-key-research_manuscript [id^="st-graphviz-chart-"] svg text {
            fill: currentColor;
        }
        .st-key-research_manuscript [id^="st-graphviz-chart-"] svg .node path,
        .st-key-research_manuscript [id^="st-graphviz-chart-"] svg .node polygon,
        .st-key-research_manuscript [id^="st-graphviz-chart-"] svg .edge path,
        .st-key-research_manuscript [id^="st-graphviz-chart-"] svg .edge polygon {
            stroke: currentColor;
        }
        .st-key-research_manuscript [id^="st-graphviz-chart-"] svg .edge polygon {
            fill: currentColor;
        }
    </style>""")
    # Fixed maximum reading width shrinks to the parent on narrow screens.
    with st.container(horizontal_alignment="center"):
        with st.container(width=940, key="research_manuscript"):
            _article()


def _article():
    st.title(content.TITLE)
    st.markdown(f"**{content.AUTHOR}** · Engineering / applied machine learning")
    st.markdown(f"[Python code]({content.REPOSITORY}) · [Original MATLAB study]({content.ORIGINAL}) · "
                "[Original preprint](https://doi.org/10.2139/ssrn.4771898) · "
                f"[Dataset and preparation]({REPOSITORY_TREE}data)")
    st.caption("Research manuscript: fixed saved experiments. Predict uses saved models; Train / Experiment fits session models. Uploads never replace the evidence below.")
    with st.expander("Article contents", expanded=False):
        st.markdown("\n\n".join(f"[{label}](#{anchor})" for anchor, label in content.SECTIONS))
    try:
        evidence = load_evidence()
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        st.error(f"Verified manuscript evidence is unavailable: {exc}. No substitute metrics are displayed.")
        return
    a = evidence["audit"]
    heading("abstract", "Abstract")
    st.markdown(content.abstract(evidence))
    st.caption("Keywords: " + content.KEYWORDS)
    heading("introduction", "1. Introduction")
    st.markdown(content.INTRODUCTION)
    figure(1, "Experimental workflow of the Python re-evaluation. Arrows indicate processing stages, not a time-horizon forecast.",
           figures.WORKFLOW,
           "The two feature configurations enter the same evaluation structure. Model selection is confined to the outer training groups; the final comparison uses saved held-out predictions.",
           "src/load_data.py; src/features.py; src/train_ann_tf.py; src/train_ann_torch.py.", diagram=True)

    heading("related-work", "2. Related work")
    st.markdown(content.RELATED_WORK)
    heading("methods", "3. Materials and methods")
    st.subheader("3.1 Dataset, provenance and quality")
    st.markdown(f"The processed dataset contains **{a['rows']} observations in {a['groups']} source-file groups**, "
                f"with {a['readings_min']}–{a['readings_max']} readings per group. The project README attributes collection "
                "to Olabisi Onabanjo University, Ogun State, Nigeria, using solar panels and measurement instruments. "
                "The raw daily CSVs accompany the processed dataset. The original project was completed in 2023, "
                "but the exact measurement dates cannot be reconstructed from the placeholder 1900 timestamps.")
    st.markdown(f"Consecutive within-file intervals range from **{a['interval_min']:g} to {a['interval_max']:g} minutes**. "
                f"There are {a['missing']} missing cells and {a['duplicate_records']} exact full-record duplicates when source identity is included. "
                "These counts do not establish independence or instrument accuracy.")
    st.warning(f"Provenance finding: {a['duplicate_measurements']} repeated measurement records appear under different filenames "
               f"({', '.join(a['duplicate_sources'])}). The experiments group by filename, so 21 identifiers do not establish 21 unique days. "
               "Saved data and results have been retained unchanged; resolving provenance and re-evaluating performance is future work.")
    table(1, "Observed variables, units and empirical ranges; all stored rows retained.", dataset_table(evidence),
          "data/processed/solar_data_combined.csv. Hour is decimal time; irradiance units follow repository documentation.")
    figure(2, "Distribution of measured electrical power across all stored observations, including repeated records.",
           figures.power_distribution(evidence["dataset"]),
           f"Measured power spans {evidence['dataset'].power.min():.2f}–{evidence['dataset'].power.max():.2f} W. "
           "This is the observed operating range, not evidence of behaviour at night or across other installations.",
           "data/processed/solar_data_combined.csv; bins summarise counts, not independent days.")
    st.subheader("3.2 Original MATLAB implementation")
    st.markdown(content.HISTORICAL)
    st.subheader("3.3 Methodological reassessment")
    st.latex(r"P=V\times I")
    st.caption("P: electrical power (W); V: voltage (V); I: current (A).")
    st.markdown(content.REASSESSMENT)
    st.subheader("3.4 Experimental tracks")
    table(2, "Feature sets and legitimate task interpretations.", track_table(), "src/features.py; electrical physics uses only current and voltage in Track A.")
    st.subheader("3.5 Data preparation and preprocessing")
    st.markdown(content.PREPROCESSING)
    st.subheader("3.6 Models and baselines")
    table(3, "Training configurations for the matched S2 comparison; the mean sanity check is not a ranked experiment.",
          model_table(evidence), "src/models.py; src/models_tf.py; src/models_torch.py; both S2 config.json files; src/evaluate.py.")
    st.markdown("Both neural frameworks use MSE on standardised targets plus hidden-kernel L2, excluding output kernels and biases. "
                "The PyTorch implementation explicitly uses Glorot-uniform kernels, zero biases, seeded shuffled batches, "
                "Adam epsilon 1e-7, no optimiser weight decay, and deterministic single-thread CPU execution. "
                "Keras uses its Dense initialisation and Adam implementation. Shared settings do not imply identical numerical optimisation. "
                "The scikit-learn MLP is a practical approximation of the historical network, not an implementation of MATLAB trainlm.")
    st.subheader("3.7 Grouped validation and model selection")
    st.markdown(content.VALIDATION)
    caps = {candidate: sum(row["cap_hits"] for row in rows)
            for candidate, rows in evidence["convergence"].items()}
    st.markdown(f"**S1/S2 convergence diagnostics.** Across both tracks, S1 had {caps['S1']} selected outer refit(s) "
                f"at the epoch cap and S2 had {caps['S2']}. The predeclared rule preferred fewer total cap hits, "
                "so S2 was selected using inner evidence. The screening threshold is fewer than 20% of "
                "selected refits at the cap per track. Reaching the cap can indicate undertraining; "
                "avoiding it or passing this screen does not prove convergence. These counts come from "
                "the saved selected settings, not outer-test accuracy.")
    figure(3, "One outer iteration of the nested grouped design; repeat for seven outer folds and each ANN seed.",
           figures.VALIDATION,
           "The outer test branch does not enter inner selection. Counts refer to filename groups, not verified independent dates; "
           "duplicate measurements across identifiers are a separate overlap risk. Inner branches rotate over three partitions.",
           "src/train_ann.py; src/train_ann_tf.py; saved S2 configurations and fold assignments.", diagram=True)
    st.subheader("3.8 Metrics and aggregation")
    st.markdown("Let $y_i$ and $\\hat y_i$ be measured and predicted power in watts, $n$ the number of pooled held-out rows, "
                "$\\bar y$ their measured mean and $S=\\{i:y_i\\ne0\\}$. Each seed pools all outer rows before scoring.")
    for equation, description in [
        (r"\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|", "MAE (W): average absolute error."),
        (r"\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}", "RMSE (W): gives larger errors greater influence."),
        (r"\mathrm{MAPE}=\frac{100}{|S|}\sum_{i\in S}\left|\frac{y_i-\hat y_i}{y_i}\right|", "MAPE (%): zero measured values are excluded; small nonzero denominators can dominate. Undefined if S is empty."),
        (r"R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}", "R² (unitless): may be negative; the manuscript treats constant-target R² as undefined.")]:
        st.latex(equation)
        st.caption(description)
    st.markdown("MAE, RMSE and MAPE favour lower values; R² favours higher values. Tables report mean ± population seed "
                "standard deviation (`ddof=0`), not the average or spread of fold metrics. The mean-prediction code in "
                "`src/evaluate.py` is a synthetic metric check, not a fitted outer-training-mean baseline experiment.")

    heading("results", "4. Experimental results")
    st.markdown("The following tables use the completed matched S2 collection. A read-only loader checks the summary against "
                "per-seed metrics and recomputes those metrics from saved out-of-fold records. It also checks row identities, "
                "targets and filename-group fold assignments. It does not train models. The audit qualification about duplicate "
                "records applies to every interpretation of these scores.")
    if evidence["protocol"]["warnings"]:
        st.warning("Saved comparison warnings: " + "; ".join(evidence["protocol"]["warnings"]))
    st.subheader("4.1 Historical findings and comparability")
    st.markdown("The original preprint's numerical findings are not reproduced as a verified table here because its full text "
                "and historical prediction records were unavailable to this audit. The existing Results explorer retains original "
                "Python and historical TensorFlow outputs. Those runs have different training settings and are not silently merged into S2.")
    for number, track in [(4, "A"), (5, "B")]:
        st.subheader(f"4.{number - 2} Track {track}")
        table(number, f"Track {track}: pooled out-of-fold performance, mean ± population standard deviation across seeds.",
              result_table(evidence, track), COMPARISON)
        st.markdown((f"Linear regression has the lowest Track A mean MAE ({score(evidence, track, 'Linear'):.3f} W). "
                     "The measured-current × measured-voltage baseline already gives low error without learning. This is electrical-input estimation."
                     if track == "A" else
                     f"The tuned scikit-learn ANN has the lowest Track B mean MAE ({score(evidence, track, 'Sklearn tuned'):.3f} W), "
                     f"versus {score(evidence, track, 'Linear'):.3f} W for linear regression. The small difference is descriptive; "
                     "no significance or external forecasting claim follows."))
    # Numbering follows the text's logical figure order: diagnostics introduce Fig. 4 before Fig. 5.
    st.subheader("4.4 Error diagnostics and framework variability")
    track = st.selectbox("Diagnostic track (Figures 4, 6 and 7)", ["B", "A"], key="manuscript_track")
    predictions = diagnostic_predictions(evidence, track)
    figure(4, f"Measured versus predicted power, Track {track}, seed 0; each model includes every saved outer row. Dashed line: identity.",
           figures.parity_chart(predictions),
           "Scatter around the identity line shows individual errors that pooled averages conceal. This disclosed seed is a diagnostic illustration, "
           "not a best-seed selection or an independent test of the full-data deployment weights.",
           "; ".join(f"{path}/predictions.csv" for path in RUN_PATHS.values()))
    metric = st.selectbox("Figure 5 comparison metric", ["MAE", "RMSE", "MAPE"], key="manuscript_metric")
    figure(5, f"Matched S2 mean {metric} with ± one population seed standard deviation. Track panels use independent scales; bars start at zero.",
           figures.comparison_chart(evidence["comparison"], metric),
           "Within-track comparisons hold feature availability fixed. Across-track bar lengths must not be read as a fair contest between equivalent tasks. "
           "Whiskers describe three-seed variation, not confidence intervals; near-zero baseline variation reflects repeated deterministic predictions.", COMPARISON)
    variability = pd.concat([rows.assign(Framework=name) for name, rows in evidence["seeds"].items()], ignore_index=True)
    variability = variability[variability.model.isin(["TensorFlow selected", "PyTorch selected"])]
    table(6, "Framework-specific per-seed pooled scores; historical runs excluded.",
          variability.rename(columns={"track": "Track", "model": "Model", "seed": "Seed", "MAE": "MAE (W)",
                                      "RMSE": "RMSE (W)", "MAPE": "MAPE (%)", "R2": "R²"}),
          "; ".join(f"{path}/metrics_per_seed.csv" for path in RUN_PATHS.values()))
    figure(6, f"Signed residual histograms, Track {track}, seed 0. Residual = measured − predicted power; vertical scales vary by model.",
           figures.residual_chart(predictions),
           "Negative residuals indicate overestimation; positive residuals indicate underestimation. The distributions expose the balance and spread "
           "of errors, but do not provide calibrated uncertainty intervals.", "Saved S2 predictions.csv files; same rows and models as Figure 4.")
    figure(7, f"Within-source-group MAE, Track {track}, seed 0; filename order is categorical rather than chronological.",
           figures.daily_chart(predictions),
           "Group-level errors vary across the measurement conditions, which a single pooled score can hide. Each point averages the nine stored readings "
           "for one filename and model. These group metrics are descriptive and are not substituted for the pooled per-seed results.",
           "Saved S2 predictions.csv files; mean absolute residual grouped by day_id and model.")
    st.download_button("Download matched research metrics", evidence["comparison"].to_csv(index=False),
                       "manuscript_s2_metrics.csv", "text/csv")

    heading("discussion", "5. Discussion")
    st.markdown(content.discussion(evidence))
    heading("limitations", "6. Limitations")
    st.markdown(content.LIMITATIONS)
    heading("conclusion", "7. Conclusion and future work")
    st.markdown(content.CONCLUSION)
    heading("references", "8. References")
    for index, (citation, url) in enumerate(content.REFERENCES, 1):
        st.subheader(f"[{index}]", anchor=f"ref-{index}")
        st.markdown(f"{citation} [Source]({url})")
    st.caption("Reference metadata checked against publisher/author records. SSRN full-text access was blocked; its indexed bibliographic record was available. The original preprint is not presented as peer reviewed.")
    heading("reproducibility", "9. Research transparency and reproducibility")
    st.markdown(f"**Data availability.** [Raw daily CSVs]({REPOSITORY_TREE}data/raw) and "
                f"[processed measurements]({REPOSITORY_TREE}data/processed/solar_data_combined.csv) are included in the repository. "
                "Full calendar dates, instrument calibration records and the provenance of the repeated source file remain unresolved. "
                "No missing metadata are inferred from filenames.")
    st.markdown(f"**Code and artifacts.** [Python source]({REPOSITORY_TREE}src), "
                f"[saved experiments]({REPOSITORY_TREE}results) and "
                f"[deployment documentation]({REPOSITORY_TREE}deployment/README.md) are available. "
                "The source mapping below identifies the files behind manuscript claims. Saved model bundles are for inference; "
                "their full-data fitted weights do not supply the outer-CV scores in this report.")
    versions = [{"Recorded environment": name, "Python": cfg["python"], "scikit-learn": cfg["sklearn"],
                 "Framework": cfg.get("tensorflow", cfg.get("torch"))} for name, cfg in evidence["configs"].items()]
    st.caption("Recorded research-run versions (rather than inferred current runtime versions)")
    st.dataframe(pd.DataFrame(versions), hide_index=True, width="stretch", alt="Recorded software versions for each S2 research run")
    st.markdown("Install the pinned deployment environment and run the app from the repository root:")
    st.code("python -m pip install -r deployment/requirements.txt\npython -m streamlit run deployment/app.py\npython -m unittest discover -s tests -v", language="bash")
    st.markdown("To reproduce S2 separately, use new, empty output paths. These commands fit models and are documentation only; "
                "opening this page does not execute them. Both frameworks must use the recorded settings:")
    cfg = evidence["configs"]["TensorFlow"]
    architectures = " ".join(",".join(map(str, units)) for units in cfg["architectures"])
    flags = (f"--tracks {' '.join(cfg['tracks'])} --seeds {' '.join(map(str, cfg['seeds']))} "
             f"--architectures {architectures} --activation {cfg['activation']} --epochs {cfg['epochs']} "
             f"--patience {cfg['patience']} --batch-size {cfg['batch_size']} --learning-rate {cfg['learning_rate']} "
             f"--l2 {cfg['l2']} --folds {cfg['folds']} --inner-folds {cfg['inner_folds']}")
    st.code(f"python src/train_ann_tf.py {flags} --output results/reproduction/tf_s2\n"
            f"python src/train_ann_torch.py {flags} --output results/reproduction/torch_s2\n"
            "python src/compare_frameworks.py --tf-run results/reproduction/tf_s2 --torch-run results/reproduction/torch_s2 --output results/reproduction/comparison_s2", language="bash")
    st.markdown("The original Python reports can be regenerated with `src/save_results.py`, `src/save_predictions.py` and "
                "`src/plots.py`; those scripts write their original report locations. The README documents S1/S2 selection, "
                "convergence diagnostics and final exports. Use the committed outputs to read this manuscript without rerunning research.")
    st.markdown("**Declarations and availability gaps.** No verified funding declaration, ethics approval, conflict-of-interest statement "
                "or additional acknowledgement is supplied here. Those details are undocumented in the inspected research files. "
                "The original repository attributes the undergraduate project to Olabisi Onabanjo University; no current affiliation is inferred.")
    inventory = json.loads((ROOT / "deployment/manuscript_evidence.json").read_text(encoding="utf-8"))
    with st.expander("Evidence inventory and audit trail"):
        st.json(inventory)
    st.download_button("Download evidence inventory", json.dumps(inventory, indent=2),
                       "manuscript_evidence.json", "application/json")
    st.caption("PDF export is deferred: a reliable paginated export with equations, figures and references needs a separate render-and-verify workflow. This page is the current manuscript presentation.")
    st.markdown("[Back to article contents](#abstract)")
