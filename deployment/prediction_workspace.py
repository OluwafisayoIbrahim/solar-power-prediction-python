import json
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

from dataclasses import replace
from evaluate import calculate_mape
from model_image import predictor_image
from session_runtime import load_session_runtime

session_runtime = load_session_runtime()
MODELS = session_runtime.model_registry.MODELS
ModelSpec = session_runtime.model_registry.ModelSpec
predict_selected = session_runtime.model_registry.predict_selected
map_frame = session_runtime.feature_mapping.map_frame
target_labels = session_runtime.feature_mapping.target_labels
dump_mapping = session_runtime.feature_mapping.dump_mapping
training_mapping = session_runtime.mapping_ui.training_mapping
prediction_mapping = session_runtime.mapping_ui.prediction_mapping
batch_backend = session_runtime.batch_prediction
FEATURES = batch_backend.FEATURES
chart_data = batch_backend.chart_data
download_csv = batch_backend.download_csv
measured_pairs = batch_backend.measured_pairs
read_sheet = batch_backend.read_sheet
run_batch = batch_backend.run_batch
validate_rows = batch_backend.validate_rows
training_backend = session_runtime.training
train_uploaded = training_backend.train_uploaded
export_trained = training_backend.export_trained
from research_data import range_warnings, metric_values, DESCRIPTIONS, UNITS, dataset
from research_pages import histogram, saved_model


st.title("Predict")
st.caption("Use saved models or train a session model from uploaded measurements.")
if not (ROOT / "results/completed.json").is_file():
    st.warning("Research models are not available yet. "
               ""
               "The electrical-product baseline remains available.")
st.info("Track A combines electrical and environmental inputs, including measured current and voltage (P = I x V). "
        "Track B uses irradiance, temperature, humidity and measurement hour without electrical inputs. "
        "Both estimate power from contemporaneous conditions; neither is validated for future forecasting.")


def clear_results():
    for key in ("batch_result", "batch_key", "plot_axis", "trained_model", "training_result", "trained_zip", "training_rejected", "training_warnings"):
        st.session_state.pop(key, None)


def clear_prediction_results():
    for key in ('batch_result', 'batch_key', 'plot_axis'):
        st.session_state.pop(key, None)


prediction_track = st.selectbox("Prediction track", ["All tracks", "A", "B"], on_change=clear_prediction_results)
compatible_keys = [key for key, item in MODELS.items()
                   if (prediction_track == "All tracks" or item.track == prediction_track)]
model_key = st.selectbox("Prediction model", compatible_keys,
                         format_func=lambda key: MODELS[key].label,
                         key="prediction_model", on_change=clear_prediction_results)
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


def get_model(key):
    return saved_model(key)


@st.cache_data
def get_model_image(key):
    return predictor_image(get_model(key))


def active_model():
    if mode == "Saved models":
        return get_model(model_key)
    if "trained_model" not in st.session_state:
        raise ValueError("Train the selected model in the Train From Uploaded Data tab first.")
    return st.session_state["trained_model"]


def render_results(result, prefix="", feature_config=None):
    feature_config = feature_config or result.attrs.get('feature_config')
    target_name, target_unit = target_labels(feature_config)
    measured_label, predicted_label = f'Measured {target_name} ({target_unit})', f'Predicted {target_name} ({target_unit})'
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
    display = result.rename(columns={'predicted_power_w': predicted_label, 'power': measured_label, 'linear_prediction_w': f'Linear baseline ({target_unit})'}) if feature_config else result
    st.dataframe(display, hide_index=True, width="stretch", height=300)
    histogram(result.loc[valid, "predicted_power_w"], predicted_label)
    if not valid.any():
        return
    st.subheader(target_name + " profile")
    excluded = set(FEATURES + ["current", "voltage", "power", "source_row", "predicted_power_w", "prediction_status", "prediction_issue", "prediction_model"])
    if feature_config:
        excluded.update(e['feature_id'] for e in feature_config['features'])
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
    chart = chart.rename(columns={'Predicted (W)': predicted_label, 'Measured (W)': measured_label})
    series = [column for column in (predicted_label, measured_label) if column in chart]
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
    if prefix == 'training_':
        st.caption(f"Metrics use {len(pairs):,} held-out readings excluded from fitting.")
    else:
        st.caption(f"Metrics use {len(pairs):,} paired upload readings. These are descriptive unless independently held out.")
    st.caption("MAPE excludes zero measured power and is unstable near zero. WAPE is undefined when all measured power is zero.")
    evaluation_metrics = training_backend.evaluation_metrics
    st.dataframe(pd.DataFrame([evaluation_metrics(pairs['Measured (W)'], pairs['Predicted (W)'])]))
    if (pairs['Measured (W)'].abs() < max(np.finfo(float).eps, pairs['Measured (W)'].abs().max()*.01)).any():
        st.warning("Small measured-power values can make percentage errors unstable.")
    actual, predicted = pairs["Measured (W)"].to_numpy(), pairs["Predicted (W)"].to_numpy()
    columns = st.columns(4)
    columns[0].metric("MAE", f"{mean_absolute_error(actual, predicted):.3f} {target_unit}")
    columns[1].metric("RMSE", f"{np.sqrt(mean_squared_error(actual, predicted)):.3f} {target_unit}")
    columns[2].metric("MAPE", f"{calculate_mape(actual, predicted):.3f}%" if np.any(actual != 0) else "N/A")
    r2 = r2_score(actual, predicted) if len(actual) > 1 and np.ptp(actual) > 0 else None
    columns[3].metric("R2", f"{r2:.3f}" if r2 is not None else "N/A")
    if (actual == 0).any() or r2 is None:
        st.caption("MAPE excludes zero measured power. R2 is unavailable for a single reading or constant measured power.")
    scatter = pairs.rename(columns={'Measured (W)': measured_label, 'Predicted (W)': predicted_label})
    st.scatter_chart(scatter, x=measured_label, y=predicted_label, color="#087f8c")
    histogram(actual - predicted, f"Residual: measured - predicted ({target_unit})")
    report = pd.DataFrame([dict(evaluation_metrics(actual, predicted), target_name=target_name, target_unit=target_unit)])
    st.download_button("Download evaluation summary", report.to_csv(index=False), "upload_evaluation.csv", "text/csv", key=prefix + "evaluation_download")


def batch_tab():
    session = st.session_state.get('trained_model') if mode != 'Saved models' else None
    prediction_spec = session.spec if session else spec
    config = session.metadata.get('feature_config') if session else None
    example = {"solar_irradiance": [850, 720], "temperature": [31, 29],
               "humidity": [65, 70], "hour": [12, 13], "current": [8, 7], "voltage": [35, 34]}
    template_frame = pd.DataFrame({name: example.get(name, [0., 0.]) for name in prediction_spec.features})
    if config:
        template_frame = template_frame.rename(columns={e['feature_id']: e['source_column'] for e in config['features']})
    template_frame.insert(0, "timestamp", ["2026-10-16 12:00:00", "2026-10-16 13:00:00"])
    template = template_frame.to_csv(index=False)
    st.download_button("Download CSV template", template, "solar_measurements_template.csv",
                       "text/csv", icon=":material/download:")
    uploaded = st.file_uploader("Measurements", type=["csv", "xlsx"], max_upload_size=20,
                                help="Required: " + ", ".join(prediction_spec.features) + ". Optional: timestamp and measured power. Up to 50,000 rows.")
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
        frame = read_sheet(data, uploaded.name, sheet, features=[] if config else prediction_spec.features, preserve_columns=bool(config))
        if config:
            frame = prediction_mapping(frame, config, clear_prediction_results)
    except Exception as exc:
        st.session_state.pop("batch_result", None)
        st.error(f"Unable to read measurements: {exc}")
        return
    st.caption(f"{len(frame):,} readings | Target: {target_labels(config)[0]} ({target_labels(config)[1]})" if config else f"{len(frame):,} readings | Irradiance: W/m2 | Temperature: deg C | Humidity: % | Hour: 0 to <24 | Current: A | Voltage: V")
    with st.expander("Input preview"):
        st.dataframe(frame.head(20), hide_index=True, width="stretch")
        _, issues = validate_rows(frame, prediction_spec.features)
        st.write(f"Valid rows: {issues.eq('').sum()} | Invalid rows: {issues.ne('').sum()}")
        st.dataframe(frame.isna().sum().rename("Missing values"))
    try:
        historical = frame if not config or config['target'].get('unit') == 'W' else frame.drop(columns=['power'], errors='ignore')
        for warning in range_warnings(historical):
            st.warning(warning)
    except FileNotFoundError:
        st.caption("Historical measurement ranges are unavailable.")
    if st.button("Predict uploaded readings", type="primary", icon=":material/play_arrow:", disabled=bool(missing) and mode == "Saved models"):
        st.session_state.pop("batch_result", None)
        progress = st.progress(0, text="Predicting power")
        try:
            result = run_batch(frame, active_model(), progress.progress,
                               features=prediction_spec.features, predictor=predict_selected)
            result["prediction_model"] = session.metadata['model_name'] + ' (session model)' if session else spec.label
            if config:
                result.attrs['feature_config'] = config
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
    validate_training = training_backend.validate_training
    split_uploaded = training_backend.split_uploaded
    evaluation_metrics = training_backend.evaluation_metrics
    st.subheader("Step 1 \u2014 Select training mode")
    training_mode = st.radio("Training mode", ["Train a New Model", "Continue Training an Existing Model"], key="training_mode", on_change=clear_results)
    continuing = training_mode.startswith("Continue")
    st.subheader("Step 2 \u2014 Configure the model")
    original = None
    if continuing:
        keys = [key for key, item in MODELS.items() if item.kind in ("tensorflow", "pytorch")]
        source = st.selectbox("Existing trained model", keys, format_func=lambda key: MODELS[key].label, on_change=clear_results)
        training_spec = MODELS[source]
        st.caption("Copies pretrained weights and retains original input and target scalers. Architecture and regularisation are preserved. A new optimizer is used. The original model is never changed.")
        st.caption("Continued training is unavailable for saved scikit-learn pipelines: their complete preprocessing and estimator pipelines do not support reliable incremental fitting.")
    else:
        family = st.selectbox("Model family", ["tensorflow", "pytorch", "tuned", "linear"], format_func=lambda value: {"tensorflow":"TensorFlow/Keras ANN", "pytorch":"PyTorch ANN", "tuned":"scikit-learn MLP", "linear":"Linear Regression"}[value], on_change=clear_results)
        track = st.selectbox("Training track", ["A", "B", "Custom"],
                             format_func=lambda value: {'A': 'Track A - Electrical and Environmental', 'B': 'Track B - Weather-Only', 'Custom': 'Custom Features'}[value],
                             key='training_track', on_change=clear_results)
        training_spec = ModelSpec(f"{track.lower()}_{family}", "Session model", track, family, f"track_{track.lower()}_session.joblib")
        st.caption("Creates fresh weights and fits new preprocessing using only the training partition. No saved model is required.")
    name = st.text_input("Model name", value="Session model", on_change=clear_results)
    hidden = st.text_input("Hidden layers (comma-separated neurons)", value="10", disabled=continuing or training_spec.kind=="linear", on_change=clear_results)
    neural = training_spec.kind in ('tensorflow', 'pytorch')
    epochs = st.number_input("Maximum additional epochs" if continuing else "Maximum epochs" if neural else "Iterations", min_value=1, max_value=1000 if neural else 500, value=200 if neural else 100, key="training_epochs", on_change=clear_results)
    early_stopping, patience, min_delta = False, 30, .0001
    if neural:
        early_stopping = st.checkbox("Enable early stopping", value=True, key="training_early_stopping", on_change=clear_results)
        patience = st.number_input("Early-stopping patience", min_value=1, value=30, key="training_patience", disabled=not early_stopping, on_change=clear_results)
        min_delta = st.number_input("Minimum validation-loss improvement", min_value=0., value=.0001, format="%.6f", key="training_min_delta", disabled=not early_stopping, on_change=clear_results)
        st.caption("20% of the training partition is reserved for validation, respecting the selected split strategy. Scalers use fitting rows only for new models. The best validation-loss weights are restored, including when the full epoch budget is used. The held-out test partition is never used for stopping.")
    learning_rate = st.number_input("Learning rate", min_value=0.000001, max_value=1., value=.001, format="%.6f", on_change=clear_results)
    l2 = st.number_input("Regularisation strength", min_value=0., value=.0001, format="%.6f", disabled=continuing or training_spec.kind=="linear", on_change=clear_results)
    batch_size = st.number_input("Batch size", min_value=1, max_value=4096, value=32, on_change=clear_results)
    seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=0, on_change=clear_results)
    baseline = st.checkbox("Compare with linear regression on the same split", disabled=training_spec.kind=="linear", on_change=clear_results)
    st.subheader("Step 3 \u2014 Upload training data")
    st.caption("Upload original column names, then map the inputs, target, groups and timestamps below.")
    st.caption("Set and confirm the units for each input and the target. Standard definitions retain research-compatible units; custom identifiers support other units. No unit conversions are performed. Up to 50,000 rows / 20 MB.")
    uploaded = st.file_uploader("Training measurements", type=["csv", "xlsx"], max_upload_size=20, key="training_upload", on_change=clear_results)
    if uploaded is None:
        clear_results()
        return
    try:
        sheet = 0
        if uploaded.name.lower().endswith('.xlsx'):
            with pd.ExcelFile(BytesIO(uploaded.getvalue()), engine='openpyxl') as book:
                sheet = st.selectbox("Training worksheet", book.sheet_names, on_change=clear_results)
        raw_frame = read_sheet(uploaded.getvalue(), uploaded.name, sheet, features=[], preserve_columns=True)
        st.dataframe(raw_frame.head(20), hide_index=True, alt='Uploaded data with original column headers')
        feature_config = training_mapping(raw_frame, training_spec.track, clear_results,
                                           training_spec.features if continuing else None)
        frame = map_frame(raw_frame, feature_config)
        training_spec = replace(training_spec, input_features=tuple(e['feature_id'] for e in feature_config['features']))
        _, _, issues = validate_training(frame, training_spec.features)
        st.dataframe(frame.head(20), hide_index=True, width="stretch")
        st.write(f"Total: {len(frame):,} | Valid: {issues.eq('').sum():,} | Rejected: {issues.ne('').sum():,}")
        if issues.ne('').any():
            st.dataframe(issues[issues.ne('')].rename('Rejection reason'))
        st.dataframe(frame.describe(include='all').astype(str))
        duplicates = int(frame.duplicated().sum())
        if duplicates:
            st.warning(f"{duplicates:,} duplicate records are retained. Repeated records can bias evaluation, especially random splits; resolve accidental duplicates before upload.")
        if 'timestamp' in frame:
            parsed = pd.to_datetime(frame.timestamp, errors='coerce', format='mixed', utc=True)
            repeated = int(parsed[parsed.notna()].duplicated().sum())
            if repeated:
                st.warning(f"{repeated:,} repeated timestamps are retained as separate observations.")
        try:
            historical = frame if feature_config['target'].get('unit') == 'W' else frame.drop(columns=['power'], errors='ignore')
            for warning in range_warnings(historical):
                st.warning(warning)
        except FileNotFoundError:
            st.caption("Original research ranges unavailable.")
        units = st.checkbox("I confirm the uploaded measurements use the stated units", on_change=clear_results)
        st.subheader("Step 4 \u2014 Choose evaluation strategy")
        candidates = [c for c in ('day_id', 'source_file', 'timestamp') if c in frame]
        suitable = []
        for candidate in candidates:
            try:
                split_uploaded(frame.loc[issues.eq('')], 'grouped', candidate)
                suitable.append(candidate)
            except ValueError:
                pass
        strategy = st.selectbox("Split strategy", ['grouped', 'random', 'chronological'], index=0 if suitable else 1, on_change=clear_results)
        group = st.selectbox("Grouping column", suitable or candidates or [None], on_change=clear_results) if strategy=='grouped' else None
        holdout = st.slider("Held-out proportion", .05, .5, .2, .05, on_change=clear_results)
        if strategy=='random':
            st.warning("Random splitting may place readings from the same measurement day in both partitions.")
        if strategy=='grouped':
            st.caption("Day/source identifiers must identify complete measurement days. Random grouped evaluation does not evaluate future forecasting.")
        tr, te, details = split_uploaded(frame.loc[issues.eq('')], strategy, group, holdout, int(seed))
        st.write(f"Training: {len(tr):,} | Held out: {len(te):,} | Training groups: {len(details['training_groups'])} | Held-out groups: {len(details['evaluation_groups'])}")
        st.caption("Timestamp calendar days and ranges use UTC; timestamps without a timezone are interpreted as UTC.")
        st.json(details)
        signature = sha256(uploaded.getvalue() + json.dumps([sheet, training_mode, training_spec.key, name, hidden, epochs, learning_rate, l2, batch_size, seed, baseline, strategy, group, holdout, early_stopping, patience, min_delta, feature_config], sort_keys=True).encode()).hexdigest()
        if st.session_state.get('training_signature') != signature:
            clear_results()
            st.session_state['training_signature'] = signature
    except Exception as exc:
        clear_results()
        st.error(f"Training data/configuration unavailable: {exc}")
        return
    st.subheader("Step 5 \u2014 Train")
    st.caption("Held-out observations are excluded from weights, preprocessing and stopping. Repeated hyperparameter adjustment against this holdout can bias its evaluation. This session experiment is separate from the published nested grouped cross-validation study.")
    if st.button("Train selected model", type='primary', disabled=not units or 'trained_model' in st.session_state):
        clear_results()
        progress = st.progress(0, text='Training session model')
        try:
            if continuing:
                original = get_model(source)
            trained, evaluation, rejected, count = train_uploaded(raw_frame, original,
                family=training_spec.kind, track=training_spec.track,
                training_mode='continue' if continuing else 'new',
                hidden_layers=(10,) if continuing else tuple(int(n.strip()) for n in hidden.split(',')),
                epochs=int(epochs), learning_rate=learning_rate, l2=l2,
                early_stopping=early_stopping, patience=int(patience), min_delta=min_delta,
                batch_size=int(batch_size), seed=int(seed), strategy=strategy,
                group_column=group, holdout=holdout, baseline=baseline,
                dataset_filename=uploaded.name, dataset_bytes=uploaded.getvalue(), dataset_sheet=sheet,
                model_name=name, progress=progress.progress, feature_config=feature_config)
            st.session_state.update(trained_model=trained, training_result=evaluation,
                training_rejected=rejected, training_warnings=count,
                trained_zip=export_trained(trained, evaluation, rejected))
        except Exception as exc:
            clear_results()
            st.error(f"Training failed: {exc}")
        finally:
            progress.empty()
    if 'trained_model' in st.session_state:
        trained = st.session_state.trained_model
        result = st.session_state.training_result
        st.subheader("Step 6 \u2014 Review results")
        st.subheader("Held-Out Evaluation \u2014 Uploaded Dataset")
        st.caption("The model was trained on one partition and evaluated on another. Single and spreadsheet predictions automatically use this session model and its configured inputs.")
        st.write({key: trained.metadata[key] for key in ('model_name', 'model_family', 'track', 'training_mode', 'valid_rows', 'rejected_rows', 'rows', 'validation_rows', 'epochs', 'split_strategy')})
        if 'completed_epochs' in trained.metadata:
            columns = st.columns(4)
            for column, label, key in zip(columns,
                    ['Requested maximum epochs', 'Completed epochs', 'Best epoch', 'Early stopping occurred'],
                    ['requested_max_epochs', 'completed_epochs', 'best_epoch', 'early_stopped']):
                value = trained.metadata[key]
                column.metric(label, ('Yes' if value else 'No') if key == 'early_stopped' else value)
            st.caption(f"Fitting rows: {trained.metadata['rows']:,}; internal validation rows: {trained.metadata['early_stopping_validation_rows']:,}; held-out test rows: {trained.metadata['validation_rows']:,}.")
            curves = pd.DataFrame({'Training loss': trained.metadata['training_loss'],
                                   'Validation loss': trained.metadata['validation_loss']})
            curves.index = pd.RangeIndex(1, len(curves) + 1, name='Epoch')
            st.line_chart(curves, x_label='Epoch', y_label='Scaled-target MSE + regularisation', alt='Training and internal validation loss by epoch')
        with st.expander('Training metadata'):
            st.json(trained.metadata)
        if st.session_state.training_warnings:
            st.warning("Optimizer reached its iteration limit; review held-out performance.")
        table = {'Selected model': trained.metadata['evaluation_metrics']}
        if 'baseline_metrics' in trained.metadata:
            table['Linear Regression'] = trained.metadata['baseline_metrics']
        st.caption('Target: ' + ' ('.join(target_labels(trained.metadata.get('feature_config'))) + '). MAE/RMSE use this target unit; MAPE/WAPE are percentages.')
        st.dataframe(pd.DataFrame.from_dict(table, orient='index'))
        render_results(result, prefix='training_', feature_config=trained.metadata.get('feature_config'))
        actual = pd.to_numeric(result.power)
        errors = (actual-result.predicted_power_w).abs()
        _, target_unit = target_labels(trained.metadata.get('feature_config'))
        histogram(errors, f'Absolute error ({target_unit})')
        bins = pd.cut(actual, bins=min(5, max(1, actual.nunique())), duplicates='drop')
        st.dataframe(pd.DataFrame({f'Target range ({target_unit})':bins, f'Absolute error ({target_unit})':errors}).groupby(f'Target range ({target_unit})', observed=True)[f'Absolute error ({target_unit})'].agg(['count','mean','max']))
        st.subheader("Step 7 \u2014 Download")
        st.download_button('Download trained model', st.session_state.trained_zip, 'session_model.zip', 'application/zip')
        st.download_button('Download feature configuration', dump_mapping(trained.metadata['feature_config']), 'feature_config.json', 'application/json')
        st.download_button('Download training metadata', json.dumps(trained.metadata, indent=2), 'training_metadata.json', 'application/json')
        if not st.session_state.training_rejected.empty:
            st.download_button('Download rejected rows', download_csv(st.session_state.training_rejected), 'rejected_rows.csv', 'text/csv')


single, batch, training, model_tab = st.tabs(["Single Prediction", "Spreadsheet Prediction", "Train From Uploaded Data", "Model diagram"])
with training:
    if mode == "Train from uploaded data":
        training_tab()
    else:
        st.info("Select Train from uploaded data in Model source to fit and evaluate a session model.")
with single:
    session = st.session_state.get('trained_model') if mode != 'Saved models' else None
    prediction_spec = session.spec if session else spec
    prediction_config = session.metadata.get('feature_config') if session else None
    feature_entries = {entry['feature_id']: entry for entry in prediction_config['features']} if prediction_config else {}
    try:
        reference = dataset()
        defaults = reference[list(UNITS)].median().to_dict()
        st.caption("Defaults are medians of the research measurements. Individual medians need not represent one observed operating condition.")
    except (FileNotFoundError, ValueError, KeyError, OSError):
        reference, defaults = None, {}
        st.caption("Research measurements unavailable; reference defaults are shown.")
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
        for index, feature in enumerate(prediction_spec.features):
            label, default, lower, upper = controls.get(feature, (feature, 0., None, None))
            if feature in feature_entries:
                entry = feature_entries[feature]
                label = entry['display_name'] + (f" ({entry['unit']})" if entry.get('unit') else ' (unit unspecified)')
            default = float(defaults.get(feature, default))
            observed = (f" Observed range: {reference[feature].min():g} to {reference[feature].max():g} {UNITS[feature]}."
                        if reference is not None and feature in reference else "")
            values[feature] = columns[index % 2].number_input(
                label, value=default, min_value=lower, max_value=upper, key=f"input_{feature}",
                help=DESCRIPTIONS.get(feature, feature) + ". " + (f"Minimum {lower}. " if lower is not None else "Finite values. ") + (f"Maximum {upper}." if upper is not None else "") + observed)
        submitted = st.form_submit_button("Predict power", type="primary", disabled=bool(missing) and mode == "Saved models", icon=":material/bolt:", width="stretch")
    if submitted:
        measurements = pd.DataFrame([values])
        try:
            _, issues = validate_rows(measurements, prediction_spec.features)
            if issues.ne("").any():
                raise ValueError(issues.iloc[0])
            prediction = predict_selected(active_model(), measurements)[0]
            target_name, target_unit = target_labels(prediction_config)
            st.metric('Predicted ' + target_name if prediction_config else 'Predicted power', f"{prediction:.2f} {target_unit}")
            st.caption(f"{session.metadata['model_name'] if session else spec.label} | {prediction_spec.kind} | {mode}. Estimate for the supplied conditions.")
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
    comparison = st.multiselect("Compare saved models on these inputs", [key for key, item in MODELS.items() if set(item.features) <= set(prediction_spec.features)], format_func=lambda key: MODELS[key].label)
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
        st.caption(("Historical experiment: 189 observations across 21 daily files. " if MODELS[diagram_key].research_run else
                    "Updated research dataset: 180 measurements from 20 daily files. ") +
                   "Predictions outside the observed conditions may be unreliable; this is not a future-weather forecast.")
        from model_statistics import render_model_statistics
        render_model_statistics(loaded)
        curve = None
        if "source_run" in metadata:
            framework = "pytorch" if loaded.spec.kind == "pytorch" else "tensorflow"
            from research_data import repository_source
            run_name = repository_source(metadata["source_run"])
            st.caption(f"Source run: {run_name}")
            curve = ROOT / run_name / f"learning_curves/track{loaded.spec.track}_seed0_fold1.png"
        if curve is not None and curve.is_file():
            with st.expander("Training and validation curves"):
                st.image(str(curve), width="stretch")
                st.caption("Comparison run: seed 0, outer fold 1. Inner-fold curves, not the final refit's test performance.")
    except (FileNotFoundError, ValueError) as exc:
        st.info(f"Model unavailable: {exc}")
    except Exception as exc:
        st.error(f"Model unavailable: {exc}")
