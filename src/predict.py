from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "results" / "models"


def predict_track_a(
    solar_irradiance: float,
    current: float,
    voltage: float,
    temperature: float,
    humidity: float,
) -> float:
    model_path = MODEL_DIR / "track_a_ann_final.joblib"
    model = joblib.load(model_path)

    
    X = np.array(
        [[
            solar_irradiance,
            current,
            voltage,
            temperature,
            humidity,
        ]],
        dtype=float,
    )

    prediction = model.predict(X)

    return float(prediction[0])


def predict_track_b(
    solar_irradiance: float,
    temperature: float,
    humidity: float,
    hour: float,
) -> float:
    model_path = MODEL_DIR / "track_b_ann_final.joblib"
    model = joblib.load(model_path)

    X = np.array(
        [[
            solar_irradiance,
            temperature,
            humidity,
            hour,
        ]],
        dtype=float,
    )

    prediction = model.predict(X)

    return float(prediction[0])


def main() -> None:
    track_a_prediction = predict_track_a(
        solar_irradiance=850,
        current=8.0,
        voltage=35.0,
        temperature=31,
        humidity=65,
    )

    track_b_prediction = predict_track_b(
        solar_irradiance=850,
        temperature=31,
        humidity=65,
        hour=12.5,
    )

    print("\nTrack A predicted power:")
    print(f"{track_a_prediction:.2f} W")

    print("\nTrack B predicted power:")
    print(f"{track_b_prediction:.2f} W")


if __name__ == "__main__":
    main()