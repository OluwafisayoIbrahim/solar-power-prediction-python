"""Research views built from saved data, predictions and model artifacts."""
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from research_data import (ROOT, UNITS, DESCRIPTIONS, RUNS, dataset, csv, config,
                           summary, errors, metric_values, repository_source, source_links)
from model_registry import MODELS, load_model
from model_image import predictor_image

@st.cache_resource
def saved_model(key):
    return load_model(key)

def track_note():
    st.info("Track A includes current and voltage: power is nearly computable as I x V. "
            "Track B uses irradiance, temperature, humidity and hour to estimate power "
            "from measured conditions. It does not forecast future weather.")

def histogram(values, label):
    frame = pd.DataFrame({"value": values}).dropna()
    st.altair_chart(alt.Chart(frame).mark_bar(color="#168b80").encode(
        x=alt.X("value:Q", title=label, bin=alt.Bin(maxbins=25)), y="count()", tooltip=["count()"]), width="stretch")

def overview():
    from manuscript_content import TITLE, ORIGINAL, REPOSITORY
    from manuscript_data import load_evidence, score

    st.title(TITLE)
    st.markdown("A Python re-evaluation of an undergraduate MATLAB ANN study, examining how input availability, "
                "simple baselines and validation design affect the interpretation of solar power estimates.")
    evidence = load_evidence()
    audit = evidence["audit"]
    cfg = evidence["configs"]["TensorFlow"]
    for column, (label, value) in zip(st.columns(3), [
            ("Stored observations", audit["rows"]), ("Source-file groups", audit["groups"]),
            ("Outer validation folds", cfg["folds"]) ]):
        column.metric(label, value)
    st.subheader("Why revisit the original study?")
    st.markdown("The original ANN produced low errors using measured electrical and environmental inputs. "
                "This re-evaluation asks what those errors demonstrate when current and voltage already closely "
                "constrain power, and whether neural networks improve on simpler models under held-out-group evaluation.")
    st.subheader("Research approach")
    st.markdown("**Track A** includes measured current and voltage alongside environmental inputs. "
                "**Track B** uses irradiance, temperature, humidity and hour without electrical inputs. "
                "Both estimate power from contemporaneous conditions. Physics and linear baselines are compared "
                "with scikit-learn, TensorFlow/Keras and PyTorch networks using grouped validation, nested selection "
                "where applicable, and repeated ANN seeds. [Read the methods](?view=manuscript#methods).")
    st.subheader("Key findings")
    st.markdown(
        f"- **Electrical inputs change the task.** The measured-power versus current × voltage difference averages "
        f"{audit['power_iv_mae']:.3f} W. Low Track A error does not establish weather-only forecasting. "
        "[Interpretation](?view=manuscript#discussion).\n"
        f"- **Complexity does not consistently improve performance.** Linear regression leads Track A by mean MAE "
        f"({score(evidence, 'A', 'Linear'):.3f} W). Track B tuned scikit-learn MAE is "
        f"{score(evidence, 'B', 'Sklearn tuned'):.3f} W versus {score(evidence, 'B', 'Linear'):.3f} W for linear regression; "
        "the small difference is descriptive, not statistically established. [Results](?view=manuscript#results).\n"
        f"- **Generalisation remains qualified.** {audit['duplicate_measurements']} measurement records repeat under "
        "different filenames assigned to outer folds 1 and 7. Identical copies cross training/test boundaries "
        "when either fold is held out; the effect on reported error remains unmeasured. No external or future-horizon "
        "forecasting validation is established. [Limitations](?view=manuscript#limitations).")
    st.caption("Findings use the same checked S2 evidence as the manuscript. Full methods, references and scientific qualifications are in the report.")
    st.subheader("Explore the application")
    with st.container(horizontal=True):
        for label, destination in [
                ("Read the Manuscript", "Research Manuscript"), ("View Experimental Results", "Results Explorer"),
                ("Try a Prediction", "Predict"), ("Train / Experiment", "Train / Experiment"),
                ("Explore the Dataset", "Dataset Explorer"), ("Explore Models", "Model Explorer")]:
            st.button(label, icon=":material/arrow_forward:",
                      on_click=lambda target=destination: st.session_state.update(dashboard_page=target))
    st.markdown(f"**Research resources:** [GitHub]({REPOSITORY}) · "
                f"[Dataset documentation]({REPOSITORY}/tree/dev/data) · "
                f"[Original MATLAB research]({ORIGINAL}) · "
                "[Original SSRN preprint](https://doi.org/10.2139/ssrn.4771898). "
                "The preprint concerns the original study; the Python re-evaluation is documented in the manuscript.")


def dataset_explorer():
    st.title("Dataset Explorer")
    with st.spinner("Loading measurements"):
        df = dataset()
    for col, (name, value) in zip(st.columns(4), summary(df).items()):
        col.metric(name, value)
    st.caption("Source: data/processed/solar_data_combined.csv. Source filenames identify measurement days. "
               "The 1900 timestamp year is a placeholder; calendar dates and seasonal coverage cannot be reconstructed.")
    st.caption("The displayed collection-day count is a count of source files. Full-record duplicate checks include filenames; the manuscript separately identifies nine repeated measurements across two different filenames.")
    st.dataframe(df[list(UNITS)].agg(["min", "max"]).T.assign(unit=pd.Series(UNITS)), width="stretch")
    st.dataframe(df.isna().sum().rename("Missing values"), width="stretch")
    days = st.multiselect("Measurement days", sorted(df.day_id.unique()), default=sorted(df.day_id.unique())[:2])
    selected = df[df.day_id.isin(days)]
    feature = st.selectbox("Range filter", list(UNITS))
    low, high = float(df[feature].min()), float(df[feature].max())
    if low < high:
        bounds = st.slider(f"{feature} ({UNITS[feature]})", low, high, (low, high))
        selected = selected[selected[feature].between(*bounds)]
    columns = st.multiselect("Visible columns", list(df), default=list(df))
    sort = st.selectbox("Sort by", list(df), index=list(df).index("hour"))
    selected = selected.sort_values(sort, ascending=not st.toggle("Descending"))
    st.dataframe(selected[columns], hide_index=True, width="stretch")
    st.download_button("Download filtered dataset", selected[columns].to_csv(index=False), "filtered_measurements.csv", "text/csv")
    if selected.empty:
        st.info("No measurements match these filters.")
        return
    st.subheader("Measurements within each day")
    for name in ("solar_irradiance", "temperature", "humidity", "power"):
        chart = alt.Chart(selected).mark_line(point=True).encode(
            x=alt.X("hour:Q", title="Recorded hour of day"), y=alt.Y(f"{name}:Q", title=f"{name} ({UNITS[name]})"),
            color="day_id:N", tooltip=["day_id", "hour", name])
        st.altair_chart(chart.interactive(), width="stretch")
    variable = st.selectbox("Feature distribution", list(UNITS)[:-1])
    histogram(selected[variable], f"{variable} ({UNITS[variable]})")
    corr = selected[list(UNITS)].corr().rename_axis("feature").reset_index().melt("feature", var_name="other", value_name="correlation")
    st.altair_chart(alt.Chart(corr).mark_rect().encode(x="feature:N", y="other:N",
        color=alt.Color("correlation:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")), tooltip=list(corr)), width="stretch")
    st.caption("Correlations describe association, not causation. Filters apply to these charts.")
    st.subheader("Data dictionary")
    dictionary = pd.DataFrame({"Description": DESCRIPTIONS, "Unit": UNITS})
    st.dataframe(dictionary, width="stretch")
    st.caption("time: recorded measurement time, with a placeholder date. source_file/day_id: source measurement-day identifier.")
    track_note()
    with st.expander("Dataset limitations"):
        st.write("Small sample; 21 days at one installation; seasonal coverage is not established. Consecutive readings may be correlated. "
                 "Different installations, weather regimes and seasons require independent validation.")

def learning_curves():
    st.subheader("Inner-fold learning curves")
    run = st.selectbox("Learning-curve run", list(RUNS))
    folder = ROOT / "results" / RUNS[run] / "learning_curves"
    files = sorted(folder.glob("*_inner*.csv"))
    if not files:
        st.info("No saved epoch curves are available for this run.")
        return
    path = st.selectbox("Track / seed / outer fold / inner fold", files, format_func=lambda p: p.stem)
    curve = csv(str(path.relative_to(ROOT)))
    st.line_chart(curve, x="epoch", y=["mae_watts", "val_mae_watts"])
    st.caption("Training and validation MAE in W for the selected candidate's inner fit. Outer test days were excluded. These are not final-model test curves.")

def experiments():
    st.title("Results Explorer")
    track_note()
    with st.expander("Original research summary figures"):
        st.graphviz_chart('digraph { rankdir=LR; node [shape=box]; "Data collection" -> "Preparation" -> "Feature selection" -> "Training" -> "Grouped cross-validation" -> "Evaluation" -> "Comparison"; }')
        st.image(str(ROOT / "results/figures/model_comparison_bars.png"), caption="Original baseline/scikit-learn comparison; matched S2 scores are shown separately below.")
    view = st.selectbox("Experiment collection", ["Matched TensorFlow/PyTorch S2", "Historical TensorFlow", "Original scikit-learn / baselines"])
    run = None
    if view.startswith("Matched"):
        table = csv("results/comparisons/parity_s2/comparison.csv")
        st.caption("Source: results/comparisons/parity_s2/comparison.csv; means and population standard deviations across seeds, not confidence intervals.")
        with st.expander("Matched experiment settings"):
            st.json({name: config(name) for name in ("TensorFlow S2", "PyTorch S2")})
    elif view.startswith("Historical"):
        run = st.selectbox("Historical run", list(RUNS)[:3])
        table = csv(f"results/{RUNS[run]}/comparison.csv").assign(source_framework="TensorFlow / sklearn baselines", source_run=RUNS[run])
        st.json(config(run))
    else:
        raw = csv("results/metrics/model_comparison.csv")
        table = raw.groupby(["track", "model"])[["mae_w", "rmse_w", "mape_percent", "r2"]].agg(["mean", lambda s: s.std(ddof=0)]).reset_index()
        table.columns = ["track", "model"] + [f"{m}_{s}" for m in ["MAE", "RMSE", "MAPE", "R2"] for s in ["mean", "std"]]
        table = table.assign(source_framework="scikit-learn / physics", source_run="results/metrics/model_comparison.csv")
        st.caption("Original reports use grouped outer CV, nested sklearn tuning and seeds 0, 1, 2 for ANNs; deterministic baselines have one recorded row. No separate run configuration was saved with this CSV.")
    track = st.selectbox("Experimental track", ["All tracks", "A", "B"])
    frameworks = st.multiselect("Frameworks", table.source_framework.unique(), default=list(table.source_framework.unique()))
    shown = table[table.source_framework.isin(frameworks)]
    if track != "All tracks":
        shown = shown[shown.track == track]
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        name: st.column_config.NumberColumn(format="%.3f")
        for name in shown.columns if name.endswith(("_mean", "_std"))})
    st.download_button("Download displayed metrics", shown.to_csv(index=False), "research_metrics.csv", "text/csv")
    st.markdown("Explore the saved evidence below. [Full interpretation](?view=manuscript#discussion) "
                "and [validation limitations](?view=manuscript#limitations) are in the manuscript.")
    metric = st.selectbox("Comparison metric", ["MAE", "RMSE", "MAPE", "R2"])
    st.caption("Higher is better." if metric == "R2" else "Lower is better. MAE/RMSE: W; MAPE: percent.")
    if not shown.empty:
        bars = shown.assign(label=shown.track + " / " + shown.model)
        st.altair_chart(alt.Chart(bars).mark_bar().encode(x=alt.X(f"{metric}_mean:Q", scale=alt.Scale(zero=True)),
            y="label:N", color="track:N", tooltip=["label", f"{metric}_mean", f"{metric}_std", "source_run"]), width="stretch")
    st.subheader("Saved out-of-fold predictions")
    if view.startswith("Original"):
        files = sorted((ROOT / "results/predictions").glob("*.csv"))
        if not files:
            st.info("No saved original out-of-fold prediction files are available.")
            learning_curves()
            return
        path = st.selectbox("Prediction file", files, format_func=lambda p: p.name)
        predictions = csv(str(path.relative_to(ROOT)))
        st.caption(str(path.relative_to(ROOT)))
    else:
        prediction_run = run or st.selectbox("Prediction framework", ["TensorFlow S2", "PyTorch S2"])
        predictions = csv(f"results/{RUNS[prediction_run]}/predictions.csv")
        prediction_track = st.selectbox("Prediction track", sorted(predictions.track.unique()))
        predictions = predictions[predictions.track == prediction_track]
        model = st.selectbox("Held-out model", sorted(predictions.model.unique()))
        seed = st.selectbox("Prediction seed", sorted(predictions.seed.unique()))
        predictions = predictions[(predictions.model == model) & (predictions.seed == seed)]
        if "hour" not in predictions:
            predictions = predictions.merge(dataset()[["day_id", "hour"]].assign(row=lambda d: d.index), on=["row", "day_id"], validate="many_to_one")
        st.caption(f"Source: results/{RUNS[prediction_run]}/predictions.csv; seed {seed}; held-out days only.")
    if predictions.empty:
        st.info("No saved predictions for this selection.")
        return
    residuals = errors(predictions)
    st.subheader("Measured versus predicted power")
    scatter = alt.Chart(residuals).mark_circle(size=45).encode(
        x=alt.X("measured:Q", title="Measured power (W)"),
        y=alt.Y("predicted:Q", title="Predicted power (W)"), tooltip=["day_id", "measured", "predicted"])
    limits = [float(residuals[["measured", "predicted"]].min().min()), float(residuals[["measured", "predicted"]].max().max())]
    diagonal = alt.Chart(pd.DataFrame({"measured": limits, "predicted": limits})).mark_line(color="#c45858").encode(x="measured:Q", y="predicted:Q")
    st.altair_chart((scatter + diagonal).interactive(), width="stretch")
    st.caption("Diagonal: perfect predictions. Residual = measured power - predicted power (W).")
    st.subheader("Error distributions")
    histogram(residuals.residual, "Residual (W)")
    histogram(residuals.absolute_error, "Absolute error (W)")
    st.subheader("Residuals versus predicted power")
    st.scatter_chart(residuals, x="predicted", y="residual", x_label="Predicted power (W)", y_label="Residual (W)")
    st.caption("Structure or changing spread can reveal systematic errors; the distributions above summarise signed and absolute errors.")
    st.subheader("Held-out daily power profile")
    day = st.selectbox("Daily performance", sorted(residuals.day_id.unique()))
    daily = residuals[residuals.day_id == day].sort_values("hour")
    st.line_chart(daily, x="hour", y=["measured", "predicted"])
    st.dataframe(pd.DataFrame([metric_values(daily.measured, daily.predicted)]), hide_index=True)
    st.caption("Daily metrics are descriptive; published scores pool held-out predictions per seed. Small differences on 21 days do not establish significance. Baselines offer a check on whether network complexity helps.")
    learning_curves()

def model_details(loaded):
    spec, meta = MODELS[loaded.spec.key], loaded.metadata
    details = dict(Model=spec.label, Track=spec.track, Framework=spec.kind, Features=", ".join(spec.features),
                   Artifacts=spec.model_dir.relative_to(ROOT).as_posix(),
                   Source=repository_source(meta["source_run"]) if "source_run" in meta else "No separate source-run metadata saved")
    if spec.kind in ("tensorflow", "pytorch"):
        network = loaded.estimator[0]
        details.update(Hidden_layers=len(meta["hidden_layers"]), Neurons=str(meta["hidden_layers"]),
                       Training_readings=meta["rows"],
                       Activation=meta["activation"], Output_activation="linear", Optimiser="Adam", L2=meta["l2"],
                       Learning_rate=meta["learning_rate"], Epochs=meta["epochs"],
                       Parameters=network.count_params() if spec.kind == "tensorflow" else sum(p.numel() for p in network.parameters()))
    elif spec.kind == "physics":
        details.update(Structure="P = I x V", Parameters=0, Optimiser="None")
    else:
        estimator = loaded.estimator.steps[-1][1]
        details.update(Structure=type(estimator).__name__, Training_readings=loaded.estimator.steps[0][1].n_samples_seen_)
        if spec.kind == "linear":
            details.update(Parameters=estimator.coef_.size + np.size(estimator.intercept_), Optimiser="Least squares")
        else:
            details.update(Hidden_layers=len(estimator.hidden_layer_sizes) if isinstance(estimator.hidden_layer_sizes, tuple) else 1,
                           Neurons=str(estimator.hidden_layer_sizes), Activation=estimator.activation,
                           Optimiser=estimator.solver, L2_alpha=estimator.alpha, Iterations=estimator.n_iter_,
                           Parameters=sum(w.size for w in estimator.coefs_) + sum(b.size for b in estimator.intercepts_))
    return {key: str(value) for key, value in details.items()}

def models():
    st.title("Model Explorer")
    track = st.selectbox("Model track", ["All", "A", "B"])
    family = st.selectbox("Model family", ["All", "Neural network", "Linear regression", "Physics"])
    kinds = st.multiselect("Model frameworks", sorted({s.kind for s in MODELS.values()}), default=sorted({s.kind for s in MODELS.values()}))
    keys = [key for key, spec in MODELS.items() if (track == "All" or spec.track == track) and spec.kind in kinds
            and (family == "All" or (family == "Physics" and spec.kind == "physics") or
                 (family == "Linear regression" and spec.kind == "linear") or
                 (family == "Neural network" and spec.kind not in ("linear", "physics")))]
    if not keys:
        st.info("No models match these filters.")
        return
    chosen = st.multiselect("Compare up to two models", keys, default=keys[:1], max_selections=2, format_func=lambda k: MODELS[k].label)
    details = {}
    for column, key in zip(st.columns(max(len(chosen), 1)), chosen):
        with column:
            try:
                loaded = saved_model(key)
                st.subheader(loaded.spec.label)
                st.write(loaded.spec.description)
                details[loaded.spec.label] = model_details(loaded)
                st.dataframe(pd.Series(details[loaded.spec.label], name="Value"), width="stretch")
                if "source_run" in loaded.metadata:
                    st.markdown("Repository source: " + source_links(loaded.metadata["source_run"]))
                picture = predictor_image(loaded)
                st.image(picture, width="stretch")
                st.download_button("Download architecture", picture, f"{key}.png", "image/png", key=f"diagram_{key}")
                st.caption("Raw measurements pass through the saved input scaler, fitted network or estimator, and target inverse scaler where applicable.")
            except (FileNotFoundError, ValueError, ImportError) as exc:
                st.warning(f"Model unavailable: {exc}")
    if details:
        st.dataframe(pd.DataFrame(details), width="stretch")
    st.caption("Applicable metrics describe the grouped-CV training procedure, not independent testing of the final fitted weights.")
    from model_statistics import research_scores
    scores = []
    for key in chosen:
        try:
            row = research_scores(MODELS[key]).copy()
            row["Saved model"] = MODELS[key].label
            scores.append(row)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            st.info(f"Research metrics unavailable: {exc}")
    if scores:
        st.dataframe(pd.DataFrame(scores), hide_index=True, width="stretch")
        st.caption("Historical and S2 runs have different training settings; inspect each source before comparing their scores.")
    st.write("An epoch is one pass over training data. Adam updates network weights using gradients; sklearn ANNs use L-BFGS. "
             "L2 discourages large weights. Inner-validation early stopping selects useful training durations. Reaching an epoch cap can indicate undertraining; avoiding it does not prove convergence.")
    learning_curves()
