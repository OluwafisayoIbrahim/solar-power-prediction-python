"""Save missing linear and untuned ANN predictors for the deployment selector."""

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import load_processed_data, get_track_a_data, get_track_b_data
from models import create_ann_model


def main():
    directory = Path(__file__).resolve().parents[1] / "results/models"
    directory.mkdir(parents=True, exist_ok=True)
    df = load_processed_data()
    for track, getter in [("a", get_track_a_data), ("b", get_track_b_data)]:
        X, y = getter(df)
        for suffix, model in [
            ("linear_final", make_pipeline(StandardScaler(), LinearRegression())),
            ("ann_untuned", create_ann_model(10, 0.0001, 0)),
        ]:
            path = directory / f"track_{track}_{suffix}.joblib"
            if path.exists():
                print(f"Keeping existing {path.name}", flush=True)
                continue
            print(f"Fitting {path.name} on {len(y)} rows", flush=True)
            model.fit(X.to_numpy(), y.to_numpy())
            expected = model.predict(X.to_numpy())
            if not np.isfinite(expected).all():
                raise ValueError("Non-finite model output")
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict(X.to_numpy()), expected)
            path.with_suffix(".json").write_text(json.dumps({
                "track": track.upper(), "features": list(X.columns), "rows": len(y),
                "seed": 0, "sklearn": sklearn.__version__,
                "purpose": "Final fit on all observations; not an independent evaluation",
            }, indent=2), encoding="utf-8")
            print(f"Saved and verified {path.name}", flush=True)


if __name__ == "__main__":
    main()
