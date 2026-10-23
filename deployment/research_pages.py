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
    st.title("Solar Power Prediction")
    st.write("An interactive research dashboard for evaluating machine-learning models using solar and environmental measurements.")
    st.write("This work revisits an undergraduate MATLAB ANN study in Python. Day-grouped evaluation, "
             "nested model selection and baseline comparisons examine whether low prediction errors "
             "reflect generalisation or electrical inputs that nearly determine the target.")
    df, cfg = dataset(), config("TensorFlow S2")
    counts = df.groupby("day_id").size()
    values = {"Measurements": len(df), "Collection days": len(counts),
              "Readings per day": str(counts.min()) if counts.nunique() == 1 else f"{counts.min()}-{counts.max()}",
              "Outer CV folds": cfg["folds"], "Random seeds": len(cfg["seeds"]), "Predictors": len(MODELS)}
    for column, (label, value) in zip(st.columns(6), values.items()):
        column.metric(label, value)
    track_note()
    st.subheader("Research workflow")
    st.graphviz_chart('digraph { rankdir=LR; node [shape=box]; "Data collection" -> "Preparation" -> "Feature selection" -> "Training" -> "Grouped cross-validation" -> "Evaluation" -> "Comparison"; }')
    st.subheader("What the saved experiments show")
    table = csv("results/comparisons/parity_s2/comparison.csv")
    for track in ("A", "B"):
        rows = table[table.track == track]
        best = rows.loc[rows.MAE_mean.idxmin()]
        st.write(f"Track {track}: {best['model']} has the lowest mean MAE ({best.MAE_mean:.3f} W) in the matched S2 comparison.")
    st.caption("Three-seed differences do not establish statistical significance. Only 21 measurement days are available; external validation remains necessary.")
    st.image(str(ROOT / "results/figures/model_comparison_bars.png"), caption="Original baseline/scikit-learn comparison; S2 results are shown separately in Experiments & Results.")
    actions = st.columns(2)
    for index, (label, destination) in enumerate([
            ("Explore the Dataset", "Dataset Explorer"), ("View Experimental Results", "Experiments & Results"),
            ("Try a Prediction", "Prediction Workspace"), ("Explore Models", "Model Explorer")]):
        with actions[index % 2]:
            st.button(label, icon=":material/arrow_forward:", width="stretch",
                      on_click=lambda target=destination: st.session_state.update(dashboard_page=target))
    st.link_button("Original MATLAB research", "https://github.com/OluwafisayoIbrahim/ANN-Forecasting-for-Solar-Power")

def dataset_explorer():
    st.title("Dataset Explorer")
    with st.spinner("Loading measurements"):
        df = dataset()
    for col, (name, value) in zip(st.columns(4), summary(df).items()):
        col.metric(name, value)
    st.caption("Source: data/processed/solar_data_combined.csv. Source filenames identify measurement days. "
               "The 1900 timestamp year is a placeholder; calendar dates and seasonal coverage cannot be reconstructed.")
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
    st.title("Experiments & Results")
    track_note()
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
    scatter = alt.Chart(residuals).mark_circle(size=45).encode(x="measured:Q", y="predicted:Q", tooltip=["day_id", "measured", "predicted"])
    limits = [float(residuals[["measured", "predicted"]].min().min()), float(residuals[["measured", "predicted"]].max().max())]
    diagonal = alt.Chart(pd.DataFrame({"measured": limits, "predicted": limits})).mark_line(color="#c45858").encode(x="measured:Q", y="predicted:Q")
    st.altair_chart((scatter + diagonal).interactive(), width="stretch")
    st.caption("Diagonal: perfect predictions. Residual = measured power - predicted power (W).")
    histogram(residuals.residual, "Residual (W)")
    histogram(residuals.absolute_error, "Absolute error (W)")
    st.scatter_chart(residuals, x="predicted", y="residual")
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

def methodology():
    st.title("Research Methodology & Limitations")
    st.write("The MATLAB study was revisited to audit validation and task definition, then extended with baseline comparisons and TensorFlow/PyTorch experiments.")
    track_note()
    sections = {
        "Data preparation": "Raw files are combined into the processed CSV; feature sets and day identifiers are explicit. Input scalers are fitted only on each training partition. TensorFlow/PyTorch also standardise targets and convert predictions back to W.",
        "Day-grouped evaluation": "Seven outer folds hold out three complete measurement days each. Every row receives an out-of-fold prediction while its day is excluded from fitting. This evaluates unseen days, not chronological future forecasting.",
        "Nested selection": "Three inner grouped folds select sklearn hyperparameters or TensorFlow/PyTorch architecture. Neural-network validation MAE determines early stopping; median selected epochs determine outer refit duration. Outer test days do not choose architecture or epochs. Reusing inner validation for stopping and selection can make inner scores optimistic.",
        "Repeated experiments": "Seeds 0, 1 and 2 assess initialisation variation. Reported means and population standard deviations describe three seeds; they are not confidence intervals. Metrics pool all out-of-fold rows per seed.",
        "Validation improvements": "Day grouping addresses random observation splits and correlated readings. Held-out predictions avoid evaluating fits on reused training observations. Shared metrics, explicit baseline comparisons, nested regularisation selection and saved per-seed artifacts make evaluation more consistent and reproducible. Electrical inputs still define a substantially easier task.",
        "S2 choice and convergence": "S1 and S2 were compared by the predeclared total cap-hit rule; S2 had zero selected refits at the cap versus one for S1. S2 was selected using inner evidence. Historical outer results had already been inspected, so independent confirmation is still needed. Matching protocols does not make framework initialisation or floating-point operations identical.",
        "Remaining limitations": "189 measurements, 21 days and one collection location limit generalisation. Seasonal coverage is unverified, adjacent readings can correlate, and no external-site or genuine future-time experiment is provided. Repeated development on this dataset can bias conclusions; improved CV does not remove small-sample limits.",
        "Proposed future work": "Evaluate larger independent photovoltaic datasets and multiple sites. Define a prediction horizon for chronological forecasting; investigate lagged environmental variables and additional baselines. Measure performance as training volume increases. These experiments have not been completed."}
    for title, text in sections.items():
        st.subheader(title)
        st.write(text)
    st.subheader("Metric definitions")
    for formula, explanation in [
        (r"MAE=\frac{1}{n}\sum_i |y_i-\hat y_i|", "Typical absolute error, in W."),
        (r"RMSE=\sqrt{\frac{1}{n}\sum_i(y_i-\hat y_i)^2}", "Error in W, with greater weight on large misses."),
        (r"MAPE=\frac{100}{|S|}\sum_{i\in S}|(y_i-\hat y_i)/y_i|,\quad S=\{i:y_i\ne0\}", "Percentage error; zero measured values are excluded."),
        (r"R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}", "Relative to predicting the mean; it may be negative and is undefined for constant targets.")]:
        st.latex(formula)
        st.caption(explanation)
