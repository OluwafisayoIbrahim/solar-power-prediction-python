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

The active models use **180 observations across 20 daily files**. One repeated daily file was excluded before evaluation. Historical 189-row models and results remain stored separately.

## Task and intended use

These regression models estimate solar power in watts from contemporaneous measurements. They support research and teaching; future-horizon forecasts, other sites, grid operations and safety controls have not been validated.

Track A uses `solar_irradiance`, `current`, `voltage`, `temperature`, `humidity`. Track B uses `solar_irradiance`, `temperature`, `humidity`, `hour`. Current and voltage use amperes and volts; humidity uses a 0-100 scale and hour is decimal time. Preserve the saved feature order. Inputs must be finite.

## Training and evaluation

Linear regression and tuned/untuned scikit-learn MLPs are compared with TensorFlow and PyTorch networks. Neural framework candidates are `(10,)` and `(16, 8)`, tanh, Adam learning rate 0.01, hidden-kernel L2 0.0001, maximum 1,000 epochs, patience 30 and selected batch size 16. Seven outer grouped folds and three inner grouped folds separate evaluation from selection. Seeds are 0, 1 and 2. S2 was selected using cap hits, then inner MAE; outer scores were excluded from selection.

| Track | Model | MAE (W) | RMSE (W) | MAPE (%) | R2 |
|---|---|---:|---:|---:|---:|
| A | Linear | 0.978 +/- 0.000 | 1.235 +/- 0.000 | 0.377 +/- 0.000 | 0.998 +/- 0.000 |
| A | Physics | 1.115 +/- 0.000 | 1.254 +/- 0.000 | 0.424 +/- 0.000 | 0.998 +/- 0.000 |
| A | PyTorch selected | 1.624 +/- 0.148 | 2.405 +/- 0.145 | 0.622 +/- 0.062 | 0.992 +/- 0.001 |
| A | Sklearn tuned | 1.069 +/- 0.060 | 1.393 +/- 0.089 | 0.413 +/- 0.026 | 0.997 +/- 0.000 |
| A | Sklearn untuned | 1.304 +/- 0.538 | 3.580 +/- 3.118 | 0.505 +/- 0.214 | 0.970 +/- 0.039 |
| A | TensorFlow selected | 1.334 +/- 0.159 | 1.863 +/- 0.297 | 0.515 +/- 0.059 | 0.995 +/- 0.002 |
| B | Linear | 11.835 +/- 0.000 | 14.928 +/- 0.000 | 4.724 +/- 0.000 | 0.702 +/- 0.000 |
| B | PyTorch selected | 12.032 +/- 0.262 | 15.027 +/- 0.246 | 4.806 +/- 0.097 | 0.698 +/- 0.010 |
| B | Sklearn tuned | 12.179 +/- 0.037 | 15.633 +/- 0.278 | 4.865 +/- 0.012 | 0.673 +/- 0.012 |
| B | Sklearn untuned | 14.044 +/- 0.267 | 18.890 +/- 1.219 | 5.584 +/- 0.136 | 0.521 +/- 0.063 |
| B | TensorFlow selected | 11.951 +/- 0.317 | 14.989 +/- 0.372 | 4.761 +/- 0.107 | 0.700 +/- 0.015 |

Values are mean +/- population standard deviation across seeds, calculated from pooled held-out predictions per seed. Six outer test folds contain 27 rows and one contains 18. These are daily grouped holdouts, not chronological forecasts. Final inference weights are fitted on all 180 observations and do not have a separate deployment test set.

## Artifacts and inference

Active exports are under `results/research_20_days/exports/{sklearn,tensorflow,pytorch}`. Each final export passed save/reload prediction equality checks. Metadata records features, settings and dataset provenance. Earlier artifacts under `results/models` and `results/exports` describe historical 189-row models. The Hugging Face repository may retain that earlier release; this change does not publish new Hub artifacts.

```python
import sys
import pandas as pd
sys.path[:0] = ['src', 'deployment']
from model_registry import load_model

measurements = pd.DataFrame([dict(solar_irradiance=850., temperature=31., humidity=65., hour=12.5)])
# The Streamlit Predict view also accepts single readings and batch uploads.
```

Use the Streamlit app or the helpers in `src/predict_tf.py` and `src/predict_torch.py` with the corresponding active export directory. Only load trusted model bundles. Serialized estimators may depend on the recorded package versions.

## Limitations and provenance

The data are attributed to one university installation. Full calendar dates and calibration records are unavailable. No exact measurement records repeat in the active dataset, but small sample size and potentially correlated readings limit independence. No external validation, significance tests or calibrated uncertainty intervals are supplied. Linear outputs can be implausible outside the observed range. Access to electrical inputs changes the task; lower Track A errors do not establish weather-only forecasting accuracy.

Evidence: `results/research_20_days/comparison/`, `selection.json`, `completed.json`, and `deployment/manuscript_evidence.json`. The original MATLAB study and preprint are distinct from this Python evaluation.
