"""Spreadsheet parsing and row-preserving batch inference for the app."""

import csv
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from predict_tf import predict_measurements

FEATURES = ["solar_irradiance", "temperature", "humidity", "hour"]
MAX_ROWS = 50_000
MAX_BYTES = 20 * 1024 * 1024
RESULT_COLUMNS = {"source_row", "predicted_power_w", "prediction_status", "prediction_issue", "prediction_model"}


def normalize_columns(columns):
    names = [str(column).strip().lower().replace(" ", "_") for column in columns]
    names = ["solar_irradiance" if name == "solarirradiance" else name for name in names]
    if len(names) != len(set(names)):
        raise ValueError("Column names must be unique after normalizing spaces and case.")
    return names


def read_sheet(data, filename, sheet=0, features=FEATURES):
    if len(data) > MAX_BYTES:
        raise ValueError("Upload a file of 20 MB or less.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        text = data.decode("utf-8-sig")
        names = normalize_columns(next(csv.reader(StringIO(text)), []))
        frame = pd.read_csv(StringIO(text), nrows=MAX_ROWS + 1)
    elif suffix == ".xlsx":
        with pd.ExcelFile(BytesIO(data), engine="openpyxl") as book:
            header = book.parse(sheet_name=sheet, header=None, nrows=1)
            if header.empty:
                raise ValueError("The worksheet is empty.")
            names = normalize_columns(header.iloc[0])
            frame = book.parse(sheet_name=sheet, nrows=MAX_ROWS + 1)
    else:
        raise ValueError("Use a UTF-8 CSV or an Excel .xlsx workbook.")
    if frame.empty:
        raise ValueError("The selected file or worksheet has no data rows.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Limit each upload to {MAX_ROWS:,} data rows.")
    if len(names) != len(frame.columns):
        raise ValueError("The header does not match the data columns.")
    frame.columns = names
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError("Some rows have more fields than the CSV header.")
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")
    reserved = RESULT_COLUMNS & set(frame.columns)
    if reserved:
        raise ValueError(f"Rename these reserved output columns: {', '.join(sorted(reserved))}.")
    return frame.reset_index(drop=True)


def validate_rows(frame, features=FEATURES):
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    issues = pd.Series("", index=frame.index, dtype="str")

    def flag(mask, message):
        issues.loc[mask] = issues.loc[mask] + message + "; "

    for feature in features:
        flag(~np.isfinite(numeric[feature]), f"{feature}: missing or nonnumeric/nonfinite")
    for feature in ("solar_irradiance", "current", "voltage"):
        if feature in numeric:
            flag(numeric[feature] < 0, f"{feature}: must be >= 0")
    if "humidity" in numeric:
        flag((numeric["humidity"] < 0) | (numeric["humidity"] > 100), "humidity: must be 0-100")
    if "hour" in numeric:
        flag((numeric["hour"] < 0) | (numeric["hour"] >= 24), "hour: must be >= 0 and < 24")
    return numeric, issues.str.rstrip("; ")


def run_batch(frame, bundle, progress=None, features=FEATURES, predictor=None):
    numeric, issues = validate_rows(frame, features)
    predictor = predictor or predict_measurements
    valid = issues.eq("")
    result = frame.copy()
    result.insert(0, "source_row", np.arange(2, len(frame) + 2))
    result["predicted_power_w"] = np.nan
    indices = frame.index[valid]
    for start in range(0, len(indices), 1024):
        batch = indices[start:start + 1024]
        result.loc[batch, "predicted_power_w"] = predictor(bundle, numeric.loc[batch])
        if progress:
            progress(min(start + len(batch), len(indices)) / len(indices))
    result["prediction_status"] = np.where(valid, "predicted", "invalid")
    result["prediction_issue"] = issues
    return result


def measured_pairs(result):
    if "power" not in result:
        return pd.DataFrame(columns=["Measured (W)", "Predicted (W)"])
    measured = pd.to_numeric(result["power"], errors="coerce")
    valid = np.isfinite(measured) & np.isfinite(result["predicted_power_w"])
    return pd.DataFrame({"Measured (W)": measured[valid],
                         "Predicted (W)": result.loc[valid, "predicted_power_w"]})


def chart_data(result, timestamp=None):
    chart = pd.DataFrame({"Source row": result["source_row"],
                          "Predicted (W)": result["predicted_power_w"]})
    if "power" in result:
        measured = pd.to_numeric(result["power"], errors="coerce")
        chart["Measured (W)"] = measured.where(np.isfinite(measured))
    if timestamp is not None:
        values = result[timestamp]
        if pd.api.types.is_numeric_dtype(values):
            raise ValueError("Choose a full date/time column, not an hour or numeric column.")
        if not pd.api.types.is_datetime64_any_dtype(values) and not values.astype("str").str.match(
                r"^\d{4}-\d{2}-\d{2}(?:[T\s]|$)").all():
            raise ValueError("Use full ISO dates such as 2026-10-16 12:00:00, not times alone.")
        parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
        if parsed.isna().any():
            raise ValueError("Some timestamps are missing or invalid. Use source-row order or correct them.")
        chart["Timestamp (UTC)"] = parsed
        return chart.sort_values("Timestamp (UTC)", kind="stable"), "Timestamp (UTC)"
    return chart, "Source row"


def download_csv(result):
    safe = result.copy()
    for name in safe.select_dtypes(include=["object", "string"]).columns:
        safe[name] = safe[name].map(
            lambda value: "'" + value if isinstance(value, str) and
            value.lstrip().startswith(("=", "+", "-", "@")) else value)
    return safe.to_csv(index=False).encode("utf-8-sig")
