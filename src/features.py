from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "solar_data_combined.csv"
)


TRACK_A_FEATURES = [
    "solar_irradiance",
    "current",
    "voltage",
    "temperature",
    "humidity",
]

TRACK_B_FEATURES = [
    "solar_irradiance",
    "temperature",
    "humidity",
    "hour",
]

TARGET_COLUMN = "power"


def load_processed_data(
    data_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Load the cleaned and combined solar dataset."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {data_path}"
        )

    df = pd.read_csv(data_path)

    return df


def add_day_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a day identifier derived from the original source filename.

    Each CSV represents one measurement day.
    """

    df = df.copy()

    if "source_file" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'source_file' column."
        )

    df["day_id"] = df["source_file"]

    return df


def validate_features(df: pd.DataFrame) -> None:
    """Check that all required feature and target columns exist."""

    required_columns = set(
        TRACK_A_FEATURES
        + TRACK_B_FEATURES
        + [TARGET_COLUMN, "day_id"]
    )

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"Missing required columns: {missing}"
        )


def get_track_a_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Original MATLAB-style feature set.

    Includes current and voltage.
    """

    X = df[TRACK_A_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def get_track_b_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Environmental/time-only feature set.

    Excludes current and voltage.
    """

    X = df[TRACK_B_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def main() -> None:
    df = load_processed_data()
    df = add_day_id(df)

    validate_features(df)

    X_a, y_a = get_track_a_data(df)
    X_b, y_b = get_track_b_data(df)

    print("\nTrack A features:")
    print(TRACK_A_FEATURES)

    print("\nTrack A shape:")
    print(X_a.shape)

    print("\nTrack B features:")
    print(TRACK_B_FEATURES)

    print("\nTrack B shape:")
    print(X_b.shape)

    print("\nTarget shape:")
    print(y_a.shape)

    print("\nNumber of unique days:")
    print(df["day_id"].nunique())

    print("\nRows per day:")
    print(
        df["day_id"]
        .value_counts()
        .sort_index()
    )

    print("\nFirst 5 Track A rows:")
    print(X_a.head())

    print("\nFirst 5 Track B rows:")
    print(X_b.head())


if __name__ == "__main__":
    main()