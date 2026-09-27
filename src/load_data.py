from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_COLUMNS = {
    "time",
    "solar_irradiance",
    "temperature",
    "humidity",
    "current",
    "voltage",
    "power",
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized snake_case column names."""
    df = df.copy()

    df.columns = (
        df.columns.str.strip()
        .str.replace(r"(?<=[a-z0-9])(?=[A-Z])", "_", regex=True)
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]+", "", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    return df


def load_all_solar_data(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    csv_files = sorted(raw_data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in {raw_data_dir}")

    dataframes: list[pd.DataFrame] = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df = clean_column_names(df)
        df["source_file"] = file_path.name
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        format="%H:%M",
        errors="coerce",
    )

    df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60
    df["calculated_power"] = df["current"] * df["voltage"]
    df["power_difference"] = df["power"] - df["calculated_power"]

    return df


def validate_data(df: pd.DataFrame) -> None:
    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\nBasic numeric statistics:")
    print(df.describe(include="number"))

    invalid_times = df["time"].isna().sum()

    print("\nInvalid time values:")
    print(invalid_times)

    correlation = df[["power", "calculated_power"]].corr()

    print("\nPower vs Current x Voltage correlation:")
    print(correlation)

    mean_abs_power_difference = df["power_difference"].abs().mean()

    print("\nMean absolute difference between Power and Current x Voltage:")
    print(f"{mean_abs_power_difference:.4f} W")

    max_abs_power_difference = df["power_difference"].abs().max()

    print("\nMaximum absolute difference between Power and Current x Voltage:")
    print(f"{max_abs_power_difference:.4f} W")

    duplicate_rows = df.duplicated().sum()

    print("\nDuplicate rows:")
    print(duplicate_rows)

    print("\nRows per source file:")
    print(df["source_file"].value_counts().sort_index())
    

def save_processed_data(df: pd.DataFrame) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "solar_data_combined.csv"
    df.to_csv(output_path, index=False)

    print(f"\nProcessed dataset saved to: {output_path}")

    return output_path

def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = EXPECTED_COLUMNS - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing expected columns: {missing}")

def main() -> None:
    df = load_all_solar_data()

    validate_required_columns(df)

    df = preprocess_data(df)

    validate_data(df)
    save_processed_data(df)

    print("\nFirst 10 rows:")
    print(df.head(10))


if __name__ == "__main__":
    main()
