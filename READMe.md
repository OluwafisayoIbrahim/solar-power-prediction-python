# Solar Power Forecasting: Python Reimplementation and Audit

A Python reimplementation and re-evaluation of an earlier MATLAB feedforward-ANN
project for solar power forecasting. The original MATLAB project is here:
[https://github.com/OluwafisayoIbrahim/ANN-Forecasting-for-Solar-Power].

## Background

The original project trained a feedforward Artificial Neural Network (ANN) in
MATLAB to forecast solar power output from environmental and electrical
measurements, reporting a Mean Absolute Percentage Error (MAPE) of 0.32%–0.60%.

Revisiting the project in Python surfaced a methodological issue with that
original evaluation (see **Key finding** below). This repository reimplements
the analysis with corrected, held-out validation, and reports what the ANN can
and cannot actually predict.

The project now also compares a TensorFlow/Keras ANN with the scikit-learn
models using the same held-out days. The completed comparisons include custom
architecture selection, early stopping, and saved training/validation curves.

## Dataset

- 21 days of measurements, 9 readings per day (8am–4pm), 189 rows total
- Collected with pyranometers, solar panels and multimeters at Olabisi Onabanjo
  University
- Columns: `time, solar_irradiance, temperature, humidity, current, voltage, power`

Raw data: `data/raw/`. Combined, cleaned data: `data/processed/solar_data_combined.csv`,
produced by `src/load_data.py`.

## Key finding: Power ≈ Current × Voltage

`power` is almost exactly the product of `current` and `voltage` (that's simple
physics — P = IV). The original MATLAB model used current and voltage as ANN
*inputs* to predict power as the *output*. A model given both of those inputs
isn't really forecasting; it has a shortcut available that makes the problem
close to trivial. Consistent with this, a plain calculation of
`current × voltage` (no learning at all) reaches 0.45% MAPE on this data —
almost the same accuracy the original ANN reported.

This repository reports two tracks so the effect is visible directly.

## Track A vs. Track B

| Track | Inputs | Purpose |
|---|---|---|
| **A — replication** | irradiance, current, voltage, temperature, humidity | Matches the original MATLAB inputs. Included to show what the leakage does. |
| **B — weather only** | irradiance, temperature, humidity, hour of day | Predicts power from measured conditions without current and voltage; not a future-weather forecast. |

## Validation

With only 189 rows from 21 days, how the data is split matters a lot:

- **Grouped, by day:** all readings from a given day are kept together, in
  either training or testing, never split across both (`GroupKFold` on a
  `day_id`, 7 folds of 3 days each). This prevents same-day readings from
  leaking between train and test.
- **Nested tuning:** the scikit-learn ANN's hidden layer size and L2 regularization strength
  are chosen using only the training days in each fold (inner 3-fold grouped
  CV), never the held-out days. This keeps model selection honest.
- **Multiple seeds:** ANN results are averaged over 3 random initializations,
  since a network this small on this little data is sensitive to the seed.
- Metrics are computed on **pooled out-of-fold predictions** (every one of the
  189 rows predicted exactly once, while its day was held out), not averaged
  per fold, since per-fold R² on 3-day folds is unstable.

## Models compared

- **Physics baseline** (Track A only): `current × voltage`, no learning
- **Linear regression**
- **ANN, untuned**: 10 hidden units, `tanh` activation, `lbfgs` solver, L2
  `alpha=0.0001` — a practical approximation of the original MATLAB
  `fitnet(10)` (scikit-learn has no Levenberg-Marquardt solver, so `lbfgs`
  is used as the nearest available second-order alternative)
- **ANN, tuned**: hidden size (3 or 10) and L2 penalty (0.1, 1, 10) chosen
  by nested cross-validation
- **TensorFlow ANN, selected**: Keras network with a linear output, Adam
  optimizer and `tanh` hidden layers. Inner grouped CV selects either one
  10-unit hidden layer or two hidden layers of 16 and 8 units.

## Results

![Model comparison](results/figures/model_comparison_bars.png)

The figure above shows the original scikit-learn/baseline comparison. The table
below includes the completed TensorFlow run. Values are means over seeds
0, 1 and 2, with metrics calculated on pooled out-of-fold predictions per seed.

| Track | Model | MAE (W) | RMSE (W) | MAPE (%) | R² |
|---|---|---|---|---|---|
| A | Physics baseline | 1.168 | 1.321 | 0.445 | 0.998 |
| A | Linear Regression | 0.992 | 1.245 | 0.382 | 0.998 |
| A | ANN, untuned | 1.319 | 3.814 | 0.518 | 0.978 |
| A | ANN, tuned | 1.141 | 1.574 | 0.442 | 0.997 |
| A | TensorFlow ANN, selected | 1.237 | 1.811 | 0.471 | 0.996 |
| B | Linear Regression | 12.165 | 15.149 | 4.854 | 0.701 |
| B | ANN, untuned | 14.302 | 23.980 | 5.698 | 0.191 |
| B | ANN, tuned | 11.965 | 15.089 | 4.786 | 0.703 |
| B | TensorFlow ANN, selected | 12.015 | 14.979 | 4.797 | 0.707 |

Full comparisons, including standard deviations:
[Track A, 1,000-epoch ceiling](results/tensorflow/20261019T100248937256Z/comparison.csv)
and [Track B](results/tensorflow/20261014T104851616874Z/comparison.csv).
The original baseline/scikit-learn table remains in
`results/metrics/model_comparison.csv`.

Both completed runs used Python 3.13.1, TensorFlow 2.21.0 and scikit-learn
1.9.1. Their configurations record the settings and versions. Standard
deviations describe variation across initialization seeds, not confidence
intervals for generalization.

![Predicted vs measured, Track A](results/figures/pred_vs_measured_track_a_ann_tuned.png)
![Predicted vs measured, Track B](results/figures/pred_vs_measured_track_b_ann_tuned.png)
![Daily profiles, Track B](results/figures/daily_profiles_track_b_ann_tuned.png)

## Discussion

Track A achieves near-perfect accuracy across every model, including the
no-learning physics baseline, because current and voltage make power almost
directly computable rather than forecastable. This confirms the leakage
described above and reproduces the original MATLAB project's low error
figures for the reasons given there.

Once current and voltage are excluded (Track B), errors rise substantially
and reflect the harder task of prediction from measured conditions. Here, model choice
matters: the untuned ANN — architecturally the closest match to the original
MATLAB network — is the worst performer on Track B (5.70% MAPE, R² 0.19),
consistent with overfitting the 162 training rows available in each outer fold. Once tuned
against that overfitting, the ANN (4.79% MAPE, R² 0.70) performs about the
same as plain linear regression (4.85% MAPE, R² 0.70). On a dataset this
size, there is no evidence that a neural network offers a meaningful
advantage over a simpler model, and the lightly regularized untuned ANN performs
measurably worse.

On Track B, TensorFlow has slightly lower RMSE (14.979 W) and higher R²
(0.707) than the tuned scikit-learn ANN, but slightly higher MAE (12.015 W)
and MAPE (4.797%). These small differences do not establish a meaningful
advantage on 21 measurement days. Raising Track A's training ceiling from 300
to 1,000 epochs reduced TensorFlow MAPE from 0.715% to 0.471%, but linear
regression remained better at 0.382%. Adding TensorFlow provides more training
controls and learning curves, but does not automatically improve accuracy.

This compares training approaches, not just frameworks: TensorFlow uses Adam
and standardized targets, while scikit-learn uses L-BFGS and unscaled targets.
Keras L2 and scikit-learn alpha are not numerically equivalent penalties.

## TensorFlow training and learning curves

Each outer fold holds out three complete days. Within its remaining training
days, three inner grouped folds select the architecture by mean validation MAE.
Early stopping monitors inner validation MAE with patience 30 and restores the
best weights. The selected architecture is then rebuilt on all outer training
days for the median best epoch count from the inner folds. Outer test days
never select the architecture or training duration.

Defaults are 300 maximum epochs, batch size 32, learning rate 0.001 and hidden
kernel L2 of 0.0001. Track A was rerun with a 1,000-epoch ceiling after every
one of its original 21 selected refits chose epoch 300. Input and target
scalers are fitted only on the training portion of each fit. Predictions are
converted back to watts before evaluation.

Example learning curves for Track B, seed 0, outer fold 1:

![TensorFlow inner-fold learning curves](results/tensorflow/20261014T104851616874Z/learning_curves/trackB_seed0_fold1.png)

Solid lines show training MAE; dashed lines show inner validation MAE, both in
watts. These curves describe the selected candidate's inner fits, not outer-test
performance or the final refit. Epoch CSVs also contain raw loss in standardized
target units, including regularization; that loss is not MSE in watts squared.

Every run creates a new `results/tensorflow/<UTC timestamp>/` directory:

| Output | Contents |
|---|---|
| `comparison.csv` | Mean and population standard deviation of pooled metrics over seeds |
| `metrics_per_seed.csv` | Metrics for each track, model and seed |
| `predictions.csv` | Held-out predictions with row, source day and fold identifiers |
| `selected_settings.csv` | Chosen settings, refit epoch counts and final sklearn fit convergence-warning counts |
| `config.json` | Run controls and Python/model-library versions |
| `learning_curves/` | PNG curves and per-epoch CSVs for selected candidates' inner fits |

The completed Track A run emitted scikit-learn L-BFGS convergence warnings
during inner tuning. Among the 21 final outer fits, 13 untuned and 14 tuned
fits also reached the 5,000-iteration limit. Completion does not mean every
fit converged. Inner tuning warnings appear in the terminal; the counts in
`selected_settings.csv` cover final sklearn fits only.

The daily profile plots show the tuned Track B model tracking the general
shape of a day's power curve reasonably well, but with real, visible
deviation in magnitude on some days (see `fri_10th`, `fri_17th` in particular).

## Limitations

- 189 samples is a small dataset for any neural network; results should be
  read as indicative, not as a robust generalization claim.
- Weather features here are simple instantaneous readings (irradiance,
  temperature, humidity, hour); no lagged or forecasted weather variables
  were available.
- Grouped CV evaluates held-out days, not a chronological forecast into the
  future. The TensorFlow addition does not change that distinction.
- Architecture is selected inside CV, but other TensorFlow controls are fixed
  in advance. Repeatedly adjusting them based on outer results would bias evaluation.
- Each inner validation fold supplies both the early-stopping epoch and the
  score used to select the architecture. This can make inner selection mildly
  optimistic, but the untouched outer test days keep the reported scores fair.
- Track A remained training-limited after the rerun: 18 of 21 selected refits
  chose epoch 1,000, and 50 of 63 selected inner histories had their best
  validation MAE at the final epoch. Its 0.471% MAPE should therefore be read
  as the result under a 1,000-epoch budget, not a fully converged optimum.
- The untuned Track B ANN is unstable across seeds (R2 standard deviation
  0.463) and produced convergence warnings. Its exact score may be sensitive
  to numerical environment even though the conclusion that it performs worst
  is unchanged.
- Grouped-by-day validation and a random row-based split give fairly similar
  results on this dataset, suggesting same-day leakage is not the dominant
  source of the original inflated numbers — the current/voltage leakage is.

## Repository structure

```
solar-power-prediction-python/
|-- .github/workflows/              # automated Python tests
|-- data/                            # raw and processed measurements
|-- deployment/                      # Streamlit app and deployment helpers
|-- notebooks/                       # exploratory analysis
|-- results/
|   |-- figures/                     # comparison plots
|   |-- metrics/                     # saved metrics
|   |-- models/                      # deployable sklearn and TensorFlow models
|   |-- predictions/                 # pooled out-of-fold predictions
|   `-- tensorflow/                  # timestamped comparisons and curves
|-- src/
|   |-- baselines.py                 # baseline CV, using validation.py
|   |-- validation.py                # shared grouped-validation helpers
|   |-- predict.py                   # sklearn inference
|   |-- predict_tf.py                # TensorFlow inference and scaling
|   |-- train_final_model.py         # final sklearn model export
|   |-- train_final_tf.py            # final TensorFlow model export
|   |-- export_baseline_models.py    # linear and untuned model exports
|   `-- ...                          # loading, features, training and plots
|-- tests/                            # unit and Streamlit integration tests
|-- requirements.txt                 # pinned research environment
`-- READMe.md
```

## How to run

From the project root on Windows, install the pinned research dependencies
(including TensorFlow 2.21.0) into the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/load_data.py
```

Run the original baseline/scikit-learn reports and plots:

```powershell
.\.venv\Scripts\python.exe src/save_results.py
.\.venv\Scripts\python.exe src/save_predictions.py
.\.venv\Scripts\python.exe src/plots.py
```

Run the full TensorFlow comparison:

```powershell
.\.venv\Scripts\python.exe src/train_ann_tf.py
```

Defaults require 294 TensorFlow fits across both tracks, three seeds and seven
outer folds, plus the scikit-learn fits. Training can take a long time. Progress
prints once per outer fold; per-epoch terminal output is disabled. Existing
reports are preserved in their original locations.

For an integration check or a smaller exploratory run:

```powershell
.\.venv\Scripts\python.exe src/train_ann_tf.py --smoke-test
.\.venv\Scripts\python.exe src/train_ann_tf.py --tracks B --seeds 0 --architectures 10 --epochs 100
```

The smoke test uses two epochs, two outer folds and one seed on Track B. Its
scores are not research results; sklearn inner tuning still runs.

Customize candidate architectures and training controls:

```powershell
.\.venv\Scripts\python.exe src/train_ann_tf.py --architectures 10 16,8 32,16 --activation relu --epochs 500 --patience 40 --learning-rate 0.001 --batch-size 16 --l2 0.001
.\.venv\Scripts\python.exe src/train_ann_tf.py --help
```

Spaces separate candidate architectures; commas separate hidden layers within
one candidate. Use `--output PATH` for a named run directory that does not yet
exist. The original `src/plots.py` generates the original comparison figures;
TensorFlow learning curves are generated by `src/train_ann_tf.py` itself.

## Saving a final TensorFlow model

The comparison saves evaluation outputs, not deployable network weights. To
refit and save a final Track B model using the completed comparison's settings:

```powershell
.\.venv\Scripts\python.exe src/train_final_tf.py
```

The default source run is `results/tensorflow/20261014T104851616874Z`.
It supplies the current Track B model. Recreate the current Track A model from
its 1,000-epoch comparison with:

```powershell
.\.venv\Scripts\python.exe src/train_final_tf.py --track A --run results/tensorflow/20261019T100248937256Z --output PATH_TO_EMPTY_DIRECTORY
```

Use `--run PATH` for another completed run and `--track A` for Track A.
The final architecture is the most frequent inner-selected architecture across
all seeds and folds (ties use candidate order). Its median selected epoch count
sets the final training duration. This is a deployment heuristic using existing
tuning decisions, not a new search or selection by outer test accuracy. A fresh
network with seed 0 is fitted on all observations; the published nested-CV
metrics remain the evaluation of the original training procedure, not an
independent test of this particular final model.

The script writes `track_b_ann_final.keras`, `track_b_x_scaler.joblib`,
`track_b_y_scaler.joblib`, and `track_b_tf_metadata.json` into `results/models`.
The metadata records feature order, settings, source run and library versions.
Both scalers must accompany the network. Existing files are never overwritten;
use `--output PATH` for a new export. The existing scikit-learn models are unchanged.
After saving, the script reloads the files and checks that predictions match
the in-memory model on every training row. This verifies serialization, not accuracy.

For inference, import `load_tf_model` and `predict_measurements` from
`src/predict_tf.py` (or `predict_tf` when `src` is on the Python import path).
Load once, then pass a DataFrame with named feature columns; the helper orders
inputs, applies input scaling, and converts outputs back to watts. Only load
trusted model/scaler files. The `.gitignore` exceptions allow both tracks'
scalers to be included with the networks for deployment.

## Interactive prediction app

`deployment/app.py` includes a selector for all nine comparison predictors,
single-reading predictions, CSV/Excel (`.xlsx`) batch uploads, prediction graphs,
measured-power comparisons, CSV downloads, and downloadable model diagrams.
Choose saved-model inference or train a fresh session model from uploaded
measurements with a `power` target. Session training uses a separate holdout
and can export its model without overwriting the project artifacts.

```powershell
.\.venv\Scripts\python.exe -m pip install -r deployment/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run deployment/app.py
```

Track B uploads require `solar_irradiance`, `temperature`, `humidity`, and `hour`.
Track A replaces hour with `current` and `voltage`; physics needs only those
two electrical inputs. Forms and templates adapt to the selected model.
An optional full `timestamp` supports multi-day graphs; optional measured
`power` enables error metrics and measured-versus-predicted plots. Invalid rows
are retained with an explanation and no prediction. Prediction uploads do not
retrain the model; training is a separate explicit action. See
[deployment instructions](deployment/README.md) for limits, units,
output columns, timestamp handling and Streamlit hosting files.

## Relationship to the original project

This project corrects and extends an earlier undergraduate MATLAB project. The original repository has been updated with a
note pointing to this correction. The original SSRN preprint has also been
revised accordingly: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4771898].
