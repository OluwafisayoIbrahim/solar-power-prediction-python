from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_registry import MODELS, load_model, predict_selected
from evaluate import calculate_mape
from batch_prediction import (FEATURES, chart_data, download_csv, measured_pairs,
                              read_sheet, run_batch, validate_rows)
from model_image import predictor_image
from training import train_uploaded, export_trained
from research_data import range_warnings, metric_values, DESCRIPTIONS, UNITS
from research_pages import histogram


st.title("Solar Power Prediction using Python")
st.caption("Research demonstration | Power from measured conditions")


def clear_results():
    for key in ("batch_result", "batch_key", "plot_axis", "trained_model", "training_result", "trained_zip", "training_rejected", "training_warnings"):
        st.session_state.pop(key, None)


prediction_track = st.selectbox("Prediction track", ["All tracks", "A", "B"], on_change=clear_results)
compatible_keys = [key for key, item in MODELS.items() if prediction_track == "All tracks" or item.track == prediction_track]
model_key = st.selectbox("Prediction model", compatible_keys,
                         format_func=lambda key: MODELS[key].label,
                         key="prediction_model", on_change=clear_results)
spec = MODELS[model_key]
st.write(spec.description)
mode = st.radio("Model source", ["Saved models", "Train from uploaded data"],
                horizontal=True, key="model_source", on_change=clear_results)
st.caption("Required inputs: " + ", ".join(spec.features))
if spec.track == "A":
    st.info("Track A uses current and voltage to estimate contemporaneous power; it is not a weather-only forecast.")
missing = spec.missing_files()
if missing:
    st.warning("This model is not available: " + ", ".join(missing))


@st.cache_resource
def get_model(key):
    return load_model(key)


@st.cache_data
def get_model_image(key):
    return predictor_image(get_model(key))


def active_model():
    if mode == "Saved models":
        return get_model(model_key)
    if "trained_model" not in st.session_state:
        raise ValueError("Train the selected model in the Training tab first.")
    return st.session_state["trained_model"]


def render_results(result, prefix=""):
    st.caption("Results: " + result["prediction_model"].iloc[0])
    valid = result["prediction_status"].eq("predicted")
    columns = st.columns(3)
    columns[0].metric("Readings", f"{len(result):,}")
    columns[1].metric("Predicted", f"{valid.sum():,}")
    columns[2].metric("Invalid", f"{(~valid).sum():,}")
    if (~valid).any():
        st.warning("Invalid readings have no prediction. Details are included in the results table and download.")
    if (result["predicted_power_w"] < 0).any():
        st.warning("Some model outputs are negative. These are retained as predicted; check the measurements and model applicability.")
    st.download_button("Download predictions", download_csv(result),
                       "solar_power_predictions.csv", "text/csv", icon=":material/download:", key=prefix + "download_results")
    st.dataframe(result, hide_index=True, width="stretch", height=300)
    histogram(result.loc[valid, "predicted_power_w"], "Predicted power (W)")
    if not valid.any():
        return
    st.subheader("Power profile")
    excluded = set(FEATURES + ["current", "voltage", "power", "source_row", "predicted_power_w", "prediction_status", "prediction_issue", "prediction_model"])
    candidates = [column for column in result.columns if column not in excluded]
    timestamp = st.selectbox("Horizontal axis", [None] + candidates,
                             format_func=lambda value: "Source row" if value is None else value,
                             key=prefix + "plot_axis")
    try:
        chart, x = chart_data(result, timestamp)
    except ValueError as exc:
        st.warning(str(exc))
        chart, x = chart_data(result)
    if x == "Timestamp (UTC)":
        st.caption("Timestamps are displayed in UTC; dates without a timezone are interpreted as UTC.")
    series = [column for column in ("Predicted (W)", "Measured (W)") if column in chart]
    if chart[x].duplicated().any():
        st.caption("Repeated timestamps are shown as separate points.")
        st.scatter_chart(chart, x=x, y=series, color=["#087f8c", "#b44952"][:len(series)])
    else:
        st.line_chart(chart, x=x, y=series, color=["#087f8c", "#b44952"][:len(series)])
    pairs = measured_pairs(result)
    if "power" not in result:
        return
    if pairs.empty:
        st.info("No valid measured power values are available for error metrics.")
        return
    st.subheader("Measured vs predicted")
    st.caption(f"Metrics use {len(pairs):,} paired readings from this upload; they are not independent model validation.")
    actual, predicted = pairs["Measured (W)"].to_numpy(), pairs["Predicted (W)"].to_numpy()
    columns = st.columns(4)
    columns[0].metric("MAE", f"{mean_absolute_error(actual, predicted):.3f} W")
    columns[1].metric("RMSE", f"{np.sqrt(mean_squared_error(actual, predicted)):.3f} W")
    columns[2].metric("MAPE", f"{calculate_mape(actual, predicted):.3f}%" if np.any(actual != 0) else "N/A")
    r2 = r2_score(actual, predicted) if len(actual) > 1 and np.ptp(actual) > 0 else None
    columns[3].metric("R2", f"{r2:.3f}" if r2 is not None else "N/A")
    if (actual == 0).any() or r2 is None:
        st.caption("MAPE excludes zero measured power. R2 is unavailable for a single reading or constant measured power.")
    st.scatter_chart(pairs, x="Measured (W)", y="Predicted (W)", color="#087f8c")
    histogram(actual - predicted, "Residual: measured - predicted (W)")
    report = pd.DataFrame([metric_values(actual, predicted)])
    st.download_button("Download evaluation summary", report.to_csv(index=False), "upload_evaluation.csv", "text/csv", key=prefix + "evaluation_download")


def batch_tab():
    example = {"solar_irradiance": [850, 720], "temperature": [31, 29],
               "humidity": [65, 70], "hour": [12, 13], "current": [8, 7], "voltage": [35, 34]}
    template_frame = pd.DataFrame({name: example[name] for name in spec.features})
    template_frame.insert(0, "timestamp", ["2026-10-16 12:00:00", "2026-10-16 13:00:00"])
    template = template_frame.to_csv(index=False)
    st.download_button("Download CSV template", template, "solar_measurements_template.csv",
                       "text/csv", icon=":material/download:")
    uploaded = st.file_uploader("Measurements", type=["csv", "xlsx"], max_upload_size=20,
                                help="Required: " + ", ".join(spec.features) + ". Optional: timestamp and measured power. Up to 50,000 rows.")
    if uploaded is None:
        st.session_state.pop("batch_result", None)
        st.session_state.pop("batch_key", None)
        return
    data = uploaded.getvalue()
    fingerprint = sha256(data).hexdigest()
    if st.session_state.get("upload_key") != fingerprint:
        st.session_state.pop("batch_result", None)
        st.session_state.pop("plot_axis", None)
        st.session_state["upload_key"] = fingerprint
    try:
        sheet = 0
        if uploaded.name.lower().endswith(".xlsx"):
            with pd.ExcelFile(BytesIO(data), engine="openpyxl") as book:
                sheet = st.selectbox("Worksheet", book.sheet_names)
        key = (fingerprint, sheet, model_key)
        if st.session_state.get("batch_key") != key:
            st.session_state.pop("batch_result", None)
            st.session_state.pop("plot_axis", None)
            st.session_state["batch_key"] = key
        frame = read_sheet(data, uploaded.name, sheet, features=spec.features)
    except Exception as exc:
        st.session_state.pop("batch_result", None)
        st.error(f"Unable to read measurements: {exc}")
        return
    st.caption(f"{len(frame):,} readings | Irradiance: W/m2 | Temperature: deg C | Humidity: % | Hour: 0 to <24 | Current: A | Voltage: V")
    with st.expander("Input preview"):
        st.dataframe(frame.head(20), hide_index=True, width="stretch")
        _, issues = validate_rows(frame, spec.features)
        st.write(f"Valid rows: {issues.eq('').sum()} | Invalid rows: {issues.ne('').sum()}")
        st.dataframe(frame.isna().sum().rename("Missing values"))
    try:
        for warning in range_warnings(frame):
            st.warning(warning)
    except FileNotFoundError:
        st.caption("Historical measurement ranges are unavailable.")
    if st.button("Predict uploaded readings", type="primary", icon=":material/play_arrow:", disabled=bool(missing)):
        st.session_state.pop("batch_result", None)
        progress = st.progress(0, text="Predicting power")
        try:
            result = run_batch(frame, active_model(), progress.progress,
                               features=spec.features, predictor=predict_selected)
            result["prediction_model"] = spec.label + (" (uploaded training data)" if mode != "Saved models" else "")
            st.session_state["batch_result"] = result
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        finally:
            progress.empty()
    if "batch_result" in st.session_state:
        render_results(st.session_state["batch_result"])
    compare_keys = st.multiselect("Compare saved models on this spreadsheet",
        [key for key, item in MODELS.items() if set(item.features) <= set(frame.columns)],
        format_func=lambda key: MODELS[key].label)
    if compare_keys and st.button("Compare spreadsheet predictions", icon=":material/compare_arrows:"):
        comparisons = pd.DataFrame({"source_row": np.arange(len(frame)) + 2})
        for key in compare_keys:
            try:
                candidate = MODELS[key]
                output = run_batch(frame, get_model(key), features=candidate.features, predictor=predict_selected)
                comparisons[candidate.label] = output["predicted_power_w"].to_numpy()
            except (ValueError, FileNotFoundError, ImportError) as exc:
                st.warning(f"{MODELS[key].label}: {exc}")
        series = list(comparisons.columns[1:])
        if series:
            st.line_chart(comparisons, x="source_row", y=series)
            st.dataframe(comparisons, hide_index=True)
            st.download_button("Download model comparison", comparisons.to_csv(index=False), "model_comparison.csv", "text/csv")
            st.caption("Saved-model predictions; invalid measurements retain missing outputs. Model agreement does not establish correctness.")


def training_tab():
    if spec.kind == "physics":
        st.info("P = current x voltage is a fixed formula with no parameters to train. Select saved-model mode to use it.")
        return
    st.caption("Fresh fit using the selected model's architecture/settings. Measured power is required. Up to 10,000 rows; 20% held out. Saved project models are never overwritten.")
    with st.expander("Settings reused for session training"):
        try:
            from research_pages import model_details
            st.json(model_details(get_model(model_key)))
        except (FileNotFoundError, ValueError, ImportError) as exc:
            st.warning(f"Settings unavailable: {exc}")
        st.caption("This session uses one holdout split, not the research experiment's nested grouped cross-validation. Held-out rows are excluded from fitting and stopping.")
    uploaded = st.file_uploader("Training measurements", type=["csv", "xlsx"],
                                max_upload_size=20, key="training_upload", on_change=clear_results)
    if uploaded is None:
        return
    try:
        sheet = 0
        if uploaded.name.lower().endswith(".xlsx"):
            with pd.ExcelFile(BytesIO(uploaded.getvalue()), engine="openpyxl") as book:
                sheet = st.selectbox("Training worksheet", book.sheet_names,
                                      key="training_sheet", on_change=clear_results)
        frame = read_sheet(uploaded.getvalue(), uploaded.name, sheet, features=spec.features)
        if "power" not in frame:
            raise ValueError("Include a 'power' column containing measured power in watts.")
    except Exception as exc:
        clear_results()
        st.error(f"Training data unavailable: {exc}")
        return
    st.dataframe(frame.head(10), hide_index=True, width="stretch")
    candidates = [name for name in frame.columns if name not in spec.features + ["power"]]
    preferred = next((name for name in ("day_id", "source_file", "timestamp") if name in candidates), None)
    options = [None] + candidates
    group = st.selectbox("Holdout groups", options, index=options.index(preferred),
                          format_func=lambda value: "Random rows (no grouping)" if value is None else value,
                          key="training_group", on_change=clear_results)
    if group is None:
        st.warning("Random rows may put readings from the same day in both partitions. Use day/source groups for grouped evaluation.")
    elif group == "timestamp":
        st.caption("Timestamp groups are calendar days in UTC.")
    epochs = 100
    if spec.kind in ("tensorflow", "pytorch"):
        epochs = st.number_input("Training epochs", min_value=1, max_value=500, value=100,
                                  key="training_epochs", on_change=clear_results)
    if st.button("Train selected model", type="primary", disabled=bool(missing), icon=":material/play_arrow:"):
        clear_results()
        progress = st.progress(0, text="Training selected model")
        try:
            trained, evaluation, rejected, warnings_count = train_uploaded(
                frame, get_model(model_key), group, int(epochs), progress.progress)
            st.session_state["trained_model"] = trained
            st.session_state["training_result"] = evaluation
            st.session_state["training_rejected"] = rejected
            st.session_state["training_warnings"] = warnings_count
            st.session_state["trained_zip"] = export_trained(trained)
        except Exception as exc:
            clear_results()
            st.error(f"Training failed: {exc}")
        finally:
            progress.empty()
    if "trained_model" in st.session_state:
        trained = st.session_state["trained_model"]
        st.success(f"Session model ready: {trained.metadata['rows']} training rows, {trained.metadata['validation_rows']} held-out rows.")
        st.caption("Single and spreadsheet predictions now use this session model. The evaluation below uses only held-out rows, which were not fitted.")
        if st.session_state["training_warnings"]:
            st.warning("The optimizer reached a convergence limit. Review held-out performance before using this model.")
        rejected = st.session_state["training_rejected"]
        if not rejected.empty:
            st.warning(f"{len(rejected)} invalid rows were excluded from training/evaluation.")
            with st.expander("Excluded training rows"):
                st.dataframe(rejected, hide_index=True)
        st.download_button("Download trained model", st.session_state["trained_zip"],
                           f"{model_key}_trained.zip", "application/zip", icon=":material/download:")
        picture = predictor_image(trained)
        with st.expander("Trained model image"):
            st.image(picture, width="stretch")
            st.download_button("Download trained model image", picture, f"{model_key}_trained.png", "image/png")
        render_results(st.session_state["training_result"], prefix="training_")


single, batch, training, model_tab = st.tabs(["Single Prediction", "Spreadsheet Prediction", "Train From Uploaded Data", "Model diagram"])
with training:
    if mode == "Train from uploaded data":
        training_tab()
    else:
        st.info("Select Train from uploaded data in Model source to fit and evaluate a session model.")
with single:
    with st.form("prediction"):
        columns = st.columns(2)
        controls = {
            "solar_irradiance": ("Solar irradiance (W/m2)", 850.0, 0.0, None),
            "temperature": ("Temperature (deg C)", 31.0, None, None),
            "humidity": ("Humidity (%)", 65.0, 0.0, 100.0),
            "hour": ("Hour of day", 12.5, 0.0, 23.99),
            "current": ("Current (A)", 8.0, 0.0, None),
            "voltage": ("Voltage (V)", 35.0, 0.0, None),
        }
        values = {}
        for index, feature in enumerate(spec.features):
            label, default, lower, upper = controls[feature]
            values[feature] = columns[index % 2].number_input(
                label, value=default, min_value=lower, max_value=upper, key=f"input_{feature}",
                help=DESCRIPTIONS.get(feature, feature) + ". " + (f"Minimum {lower}. " if lower is not None else "Finite values. ") + (f"Maximum {upper}." if upper is not None else ""))
        submitted = st.form_submit_button("Predict power", type="primary", disabled=bool(missing), icon=":material/bolt:", width="stretch")
    if submitted:
        measurements = pd.DataFrame([values])
        try:
            _, issues = validate_rows(measurements, spec.features)
            if issues.ne("").any():
                raise ValueError(issues.iloc[0])
            prediction = predict_selected(active_model(), measurements)[0]
            st.metric("Predicted power", f"{prediction:.2f} W")
            st.caption(f"{spec.label} | {spec.kind} | {mode}. Estimated power for the supplied measured conditions.")
            try:
                for warning in range_warnings(measurements):
                    st.warning(warning)
            except FileNotFoundError:
                st.caption("Historical measurement ranges are unavailable.")
            if prediction < 0:
                st.warning("Negative model output; check the measurements and model applicability.")
        except Exception as exc:
            st.error(f"Prediction unavailable: {exc}")
with batch:
    batch_tab()
with single:
    comparison = st.multiselect("Compare saved models on these inputs", [key for key, item in MODELS.items() if set(item.features) <= set(spec.features)], format_func=lambda key: MODELS[key].label)
    if comparison and st.button("Compare predictions", icon=":material/compare_arrows:"):
        rows = []
        for key in comparison:
            try:
                rows.append({"Model": MODELS[key].label, "Power (W)": float(get_model(key).predict(pd.DataFrame([values]))[0])})
            except (ValueError, FileNotFoundError, ImportError) as exc:
                st.warning(f"{MODELS[key].label}: {exc}")
        if rows:
            table = pd.DataFrame(rows)
            table["Difference from first model (W)"] = table["Power (W)"] - table["Power (W)"].iloc[0]
            st.dataframe(table, hide_index=True)
            st.bar_chart(table, x="Model", y="Power (W)")
            st.caption("Agreement between models is not proof of accuracy. These are saved-model predictions.")
with model_tab:
    diagram_key = st.selectbox("Saved model diagram", list(MODELS),
                               format_func=lambda key: MODELS[key].label, key="diagram_model")
    try:
        loaded = get_model(diagram_key)
        metadata = loaded.metadata
        picture = get_model_image(diagram_key)
        st.subheader(loaded.spec.label)
        st.write(MODELS[diagram_key].description)
        st.image(picture, width="stretch")
        st.download_button("Download model image", picture, f"{diagram_key}_model.png", "image/png",
                           icon=":material/download:")
        st.caption("Inputs: " + ", ".join(loaded.spec.features))
        columns = st.columns(3)
        if loaded.spec.kind in ("tensorflow", "pytorch"):
            columns[0].metric("Training readings", metadata["rows"])
            columns[1].metric("Epochs", metadata["epochs"])
            network = loaded.estimator[0]
            parameters = (network.count_params() if loaded.spec.kind == "tensorflow"
                          else sum(parameter.numel() for parameter in network.parameters()))
            columns[2].metric("Parameters", f"{parameters:,}")
        elif loaded.spec.kind == "physics":
            columns[0].metric("Formula", "P = I x V")
            columns[1].metric("Learned parameters", "0")
        else:
            estimator = loaded.estimator.steps[-1][1]
            columns[0].metric("Training readings", int(loaded.estimator.steps[0][1].n_samples_seen_))
            if loaded.spec.kind == "linear":
                columns[1].metric("Parameters", len(estimator.coef_) + 1)
            else:
                columns[1].metric("Iterations", estimator.n_iter_)
                columns[2].metric("L2 alpha", estimator.alpha)
        st.caption("189 measurements from 21 days. Predictions outside the observed conditions may be unreliable; this is not a future-weather forecast.")
        from model_statistics import render_model_statistics
        render_model_statistics(loaded)
        curve = None
        if "source_run" in metadata:
            framework = "pytorch" if loaded.spec.kind == "pytorch" else "tensorflow"
            run_name = Path(metadata["source_run"].replace("\\", "/")).name
            st.caption(f"Source run: {framework}/{run_name}")
            curve = ROOT / "results" / framework / run_name / f"learning_curves/track{loaded.spec.track}_seed0_fold1.png"
        if curve is not None and curve.is_file():
            with st.expander("Training and validation curves"):
                st.image(str(curve), width="stretch")
                st.caption("Comparison run: seed 0, outer fold 1. Inner-fold curves, not the final refit's test performance.")
    except Exception as exc:
        st.error(f"Model unavailable: {exc}")
