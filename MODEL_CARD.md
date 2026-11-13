---
language:
  - en
tags:
  - solar-power
  - regression
  - tabular-regression
  - renewable-energy
  - scikit-learn
  - tensorflow
  - pytorch
  - research
---

# Solar Power Prediction: Scikit-learn, TensorFlow and PyTorch

## Model description

This repository contains regression models that estimate solar panel power output in watts from environmental and electrical measurements. It accompanies a Python reimplementation and re-evaluation of an earlier MATLAB artificial neural network project.

The release includes linear regression, tuned and untuned scikit-learn neural networks, and TensorFlow/Keras and PyTorch neural networks. Two input tracks show how access to current and voltage changes the prediction task.

**These models estimate power from measurements taken at the same time. They have not been validated for future power forecasting.**

## Inputs and outputs

| Track | Required features, in order | Interpretation |
|---|---|---|
| A | `solar_irradiance`, `current`, `voltage`, `temperature`, `humidity` | Replicates the electrical and environmental input setting. Current and voltage already nearly determine the target through `power = current * voltage`. |
| B | `solar_irradiance`, `temperature`, `humidity`, `hour` | Estimates power from measured environmental conditions and hour of day without current or voltage. |

Output: one predicted power value in **watts (W)** per input row.

Use the source dataset's measurement units and conventions. `current` and `voltage` represent amperes and volts; humidity is represented on a 0–100 scale. `hour` is numeric hour of day, including fractions, such as `12.5` for 12:30. Inputs must be finite numeric values. Preserve the feature order when passing arrays to scikit-learn models; the TensorFlow and PyTorch helpers select columns using saved metadata.

## Intended use

- Research and teaching about solar power regression on small tabular datasets.
- Comparing linear regression and neural networks under a shared grouped evaluation protocol.
- Examining how electrical inputs can create a shortcut when predicting power.
- Experimental estimation under conditions comparable to the collection setting.

Validate on independent data from the intended installation before operational use. These models have not been established as reliable for grid dispatch, safety controls, financial planning, other sites or panel configurations, nighttime conditions, or long-term forecasting.

## Training data and provenance

The processed dataset contains **189 observations from 21 source-file groups**, with nine measurements per group nominally covering 08:00–16:00. Measurements were collected using pyranometers, solar panels and multimeters at Olabisi Onabanjo University. The project records environmental measurements, current, voltage, time and power.

The source project stores raw files under `data/raw/` and the combined dataset at `data/processed/solar_data_combined.csv`. The model upload does not itself imply that these data files are included or licensed for redistribution.

**Data audit limitation:** the project's 2026-11-05 audit identified nine repeated measurement records across `solar_data_tue_21st.csv` and `solar_data_tue_28st.csv`. The 21 source groups are not independently verified unique collection dates. Identical observations may therefore occur across nominally held-out partitions. Within-file measurement intervals range from 35 to 92 minutes, so sampling is approximately hourly rather than uniformly hourly. The reported results preserve the original data; no duplicate-corrected performance is claimed.

## Models and training

| Model family | Training approach |
|---|---|
| Linear regression | Standardized inputs followed by linear regression. |
| Untuned scikit-learn ANN | One hidden layer with 10 units, `tanh`, `lbfgs`, and L2 penalty `0.0001`. |
| Tuned scikit-learn ANN | Hidden size from `{3, 10}` and L2 penalty from `{0.1, 1, 10}`, selected within grouped cross-validation for evaluation. |
| TensorFlow ANN, S2 | Hidden layers `(10,)` or `(16, 8)`, `tanh`, linear output, Adam, learning rate `0.01`, L2 `0.0001`, batch size `16`, and an epoch cap of `1000`. |
| PyTorch ANN, S2 | Matching architecture candidates and selection controls, with an explicit CPU training loop. |

For TensorFlow and PyTorch evaluation, three inner grouped folds select architecture. Early stopping uses validation MAE with patience 30; scaling is fitted within training partitions. S2 was selected using the project's declared epoch-cap screening rule rather than the lowest outer-test error. The broader research process had already inspected historical results, so these comparisons still require independent confirmation.

The published deployment models are refitted on all 189 observations. The tuned scikit-learn final-fit script selects settings using seven grouped folds and refits with seed 0. The S2 export scripts choose the most frequently selected architecture across the research folds and seeds, then use its median selected epoch count for a seed-0 final fit. Consult each bundle's metadata for the actual architecture and epoch count.

## Evaluation

Evaluation uses seven outer `GroupKFold` folds over filename-derived day groups, with three groups held out per fold. Neural-network experiments use seeds 0, 1 and 2. For each seed, metrics are calculated on pooled out-of-fold predictions over all 189 rows, then averaged across seeds. These are grouped holdouts, not chronological forecasts.

The following results are rounded values from the project's committed baseline and matched S2 comparison reports. They describe the evaluation procedures, **not an independent test of the final models refitted on all observations**. The repeated-record issue above limits their interpretation.

| Track | Model | MAE (W) | RMSE (W) | MAPE (%) | R² |
|---|---|---:|---:|---:|---:|
| A | Physics: current × voltage | 1.168 | 1.321 | 0.445 | 0.998 |
| A | Linear regression | 0.992 | 1.245 | 0.382 | 0.998 |
| A | Scikit-learn ANN, untuned | 1.319 | 3.814 | 0.518 | 0.978 |
| A | Scikit-learn ANN, tuned | 1.141 | 1.574 | 0.442 | 0.997 |
| A | TensorFlow ANN, S2 | 1.545 | 2.108 | 0.594 | 0.994 |
| A | PyTorch ANN, S2 | 1.575 | 2.135 | 0.606 | 0.994 |
| B | Linear regression | 12.165 | 15.149 | 4.854 | 0.701 |
| B | Scikit-learn ANN, untuned | 14.302 | 23.980 | 5.698 | 0.191 |
| B | Scikit-learn ANN, tuned | 11.965 | 15.089 | 4.786 | 0.703 |
| B | TensorFlow ANN, S2 | 12.131 | 15.283 | 4.854 | 0.695 |
| B | PyTorch ANN, S2 | 11.981 | 15.310 | 4.809 | 0.694 |

The physics baseline has no learned parameters and no model artifact. Track A's low errors mainly reflect the relationship between power, current and voltage. Track B provides a more informative test of estimation without those electrical inputs; its tuned ANN performs similarly to linear regression on this dataset. Seed variation measures sensitivity to initialization and is not a confidence interval for generalization.

Historical TensorFlow models are also included when `results/models/` is uploaded. Their results and training budgets differ from S2; the S2 TensorFlow rows above apply only to `results/exports/tf_s2/`.

## Repository files

Paths below assume the project folder structure is preserved on the Hub. In filenames, `{a,b}` means that separate Track A and Track B files exist.

| Path | Contents |
|---|---|
| `results/models/track_{a,b}_ann_final.joblib` | Tuned scikit-learn predictors, including preprocessing. |
| `results/models/track_{a,b}_ann_untuned.joblib` | Untuned scikit-learn predictors, including preprocessing. |
| `results/models/track_{a,b}_linear_final.joblib` | Linear-regression pipelines, including preprocessing. |
| `results/exports/tf_s2/` | S2 `.keras` models, input and target scalers, and JSON metadata. |
| `results/exports/torch_s2/` | S2 PyTorch `.pt` state dictionaries, input and target scalers, and JSON metadata. |
| `results/models/*.keras` | Historical TensorFlow models with their associated scalers and metadata in the same directory. |
| `src/` | Model definitions, prediction helpers, training and evaluation code. |
| `requirements.txt` | Project dependency versions and CPU PyTorch package index. |

Keep each TensorFlow or PyTorch model together with its input scaler, target scaler and metadata. The neural-network output must be transformed back to watts with the target scaler. Scikit-learn predictors already contain their preprocessing and should receive unscaled inputs.

## Usage

Replace `YOUR_USERNAME/solar-power-prediction` with the actual Hub repository ID. Install `huggingface_hub` and the project dependencies in a compatible environment. For private repositories, authenticate locally with `hf auth login` first.

### Scikit-learn: Track B

```python
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="OluwafisayoIbrahim/solar-power-prediction",
    filename="results/models/track_b_ann_final.joblib",
)
model = joblib.load(model_path)

measurements = np.array([[850.0, 31.0, 65.0, 12.5]])
power_watts = model.predict(measurements)
print(f"Predicted power: {power_watts[0]:.2f} W")
```

### TensorFlow or PyTorch: Track B

Download the artifacts and the project's helper code:

```python
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

root = Path(snapshot_download(
    repo_id="OluwafisayoIbrahim/solar-power-prediction",
    allow_patterns=["src/*.py", "results/exports/*/*"],
))
sys.path.insert(0, str(root / "src"))

measurements = pd.DataFrame([{
    "solar_irradiance": 850.0,
    "temperature": 31.0,
    "humidity": 65.0,
    "hour": 12.5,
}])

from predict_tf import load_tf_model, predict_measurements

tf_bundle = load_tf_model("B", root / "results/exports/tf_s2")
print("TensorFlow power (W):", predict_measurements(tf_bundle, measurements))

from predict_torch import load_torch_model
from predict_torch import predict_measurements as predict_with_torch

torch_bundle = load_torch_model("B", root / "results/exports/torch_s2")
print("PyTorch power (W):", predict_with_torch(torch_bundle, measurements))
```

For Track A, select track `"A"` and supply its five required feature columns. The PyTorch loader reconstructs the project-specific `TorchANN` architecture and loads its state dictionary on CPU with `weights_only=True`.

## Reproducibility and limitations

- The collection is small and limited to one measurement setting. Performance across seasons, sites, hardware and extreme conditions has not been established.
- The repeated records and uncertain date provenance may make grouped validation optimistic. A duplicate-aware re-evaluation and an independent dataset are needed.
- Track A benefits from target-related electrical inputs and should not be presented as evidence of weather-based forecasting accuracy.
- No lagged inputs or future-weather forecasts are used. Grouped validation does not establish future forecasting performance.
- The untuned Track B ANN has substantial seed sensitivity and recorded convergence warnings.
- Linear outputs are not physically constrained; predictions can be implausible outside the observed data range.
- Framework parity does not imply identical random draws or numerical behavior.
- Final exports are trained on all observations and have no separate held-out deployment test set.

Recorded environments include Python `3.13.1`, scikit-learn `1.9.1`, NumPy `2.5.3`, TensorFlow `2.21.0`, and CPU PyTorch `2.9.1+cpu`. Use the included dependency file and per-model metadata to reproduce the relevant environment. Serialized estimators may be incompatible with different dependency versions. Only load trusted `.joblib` artifacts or execute trusted downloaded Python code.

## Evidence and attribution

The model card is based on the source project's `README.md`, `deployment/model_registry.py`, `deployment/manuscript_evidence.json`, saved export metadata, `results/metrics/model_comparison.csv`, and `results/comparisons/parity_s2/comparison.md`. These evidence files remain in the source codebase unless separately uploaded to the Hub.

The earlier MATLAB project is [ANN-Forecasting-for-Solar-Power](https://github.com/OluwafisayoIbrahim/ANN-Forecasting-for-Solar-Power). Its reported results are distinct from this Python re-evaluation.
