# Streamlit prediction app

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r deployment/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run deployment/app.py
```

The model selector covers nine predictors: TensorFlow, tuned/untuned
scikit-learn ANNs and linear regression for both tracks, plus the Track A
physics baseline. The Models tab offers a separate diagram selector and PNG
download for every saved model. Neural-network images use actual saved layer
sizes; regression and physics images show their computation structure.

Choose **Saved models** for inference only, or **Train from uploaded data**
to fit a fresh session model. Changing the model or mode clears previous results.
Existing comparison learning curves are shown when available.

## Spreadsheet predictions

Upload a UTF-8 CSV or `.xlsx` workbook, then select a worksheet for Excel.
Maximum: 20 MB and 50,000 data rows, with inference in batches of 1,024.
The first row must contain column names. A CSV template is downloadable in the app.

| Column | Values |
|---|---|
| `solar_irradiance` | Nonnegative W/m2 (ANN/regression) |
| `temperature` | Finite degrees Celsius |
| `humidity` | Percentage from 0 to 100 |
| `hour` | Decimal hour, 0 inclusive to 24 exclusive (Track B) |
| `current` | Nonnegative amperes (Track A and physics) |
| `voltage` | Nonnegative volts (Track A and physics) |
| `timestamp` (optional) | Full ISO date/time for multi-day plots |
| `power` (optional) | Measured power in watts for comparison |

Track A ANN/regression requires irradiance, current, voltage, temperature and
humidity. Track B requires irradiance, temperature, humidity and hour. The
physics baseline requires only current and voltage. The template and form
adapt to the selection. Headers are normalized to lowercase with spaces replaced by underscores.
Invalid feature rows stay in the output, with a reason and no prediction.
Output columns are `source_row` (header is row 1), `predicted_power_w`,
`prediction_status`, `prediction_issue`, and `prediction_model`. These names are reserved.
Blank CSV lines are skipped by the parser, so source rows refer to parsed data
positions, not necessarily physical text-file line numbers. Rows remain in
their original order in the download; only the chart is sorted by timestamp.

The horizontal-axis selector defaults to source-row order. Full timestamps
are parsed in UTC; timezone-naive dates are treated as UTC. Invalid timestamps
fall back to source-row order. Duplicate timestamps are shown as separate points.
Numeric Excel serial dates should be formatted as actual dates before upload.

If `power` is present, charts compare measured and predicted values. Metrics
use rows with both a valid prediction and finite measured power. MAPE excludes
zero measured values; R2 is unavailable for constant targets or fewer than two
paired readings. These metrics describe the upload, not independent validation.
Negative model outputs are retained and flagged, not silently clipped.

Prediction uploads/results live in the current app session; they are not written
to the repository or used for training. Download results before closing the session.
Uploaded text beginning with formula characters is escaped in CSV exports.

## Training from uploads

Choose **Train from uploaded data**, select a model, and use the Training tab.
CSV/Excel training requires the model's input columns plus measured `power`.
Training is limited to 10,000 rows, at least 10 valid readings. Invalid inputs
and missing/nonfinite/negative targets are excluded and listed.

The app holds out 20% of rows, or 20% of groups when a group column is selected.
It defaults to `day_id`, `source_file`, or calendar days derived from `timestamp`
when available. Grouped evaluation requires at least three groups. Without
grouping, correlated readings can occur in both partitions. The random seed
is fixed at 0. All scalers and model weights are fitted on the training partition
only; held-out readings are never used to fit or stop training.

The selected saved model supplies the architecture and hyperparameters. A tuned
ANN reuses its existing chosen settings; upload training does not rerun nested
hyperparameter search. TensorFlow uses a fixed epoch count (1-500, default 100)
chosen before training. These upload scores are not the published nested-CV
scores, and repeated tuning against the same holdout can bias them.

After training, single and batch predictions in this mode use the session model.
Download its ZIP before closing the session: TensorFlow exports the network,
both scalers and metadata; sklearn exports its complete preprocessing pipeline
and metadata. The export remains fitted on the training partition, not all rows.
Serialization uses temporary files that are removed after creating the download.
Saved project artifacts are never overwritten. The physics baseline cannot be
trained because `P = I * V` has no learned parameters; use saved-model mode for it.

## Deployment files

Choose `deployment/app.py` as the Streamlit entrypoint and Python 3.13.
The adjacent requirements file includes Excel/image dependencies and pins
TensorFlow, NumPy and scikit-learn to the saved model environment.
Include the deployment folder, `src/predict_tf.py`, `src/evaluate.py`,
`src/features.py`, `src/models_tf.py`, and the artifacts in `results/models`.
Both tracks require their `.keras` network, x/y scalers and TF metadata, plus
their tuned ANN, untuned ANN and linear `.joblib` pipelines. The root `.gitignore`
explicitly allows these artifacts. Missing predictors are clearly marked and
cannot make predictions. No dataset is required for inference/upload training.

To regenerate missing baseline exports locally, run `python src/export_baseline_models.py`.
To recreate the current Track A TensorFlow model, use the completed 1,000-epoch
comparison and an empty output directory:
`python src/train_final_tf.py --track A --run results/tensorflow/20261019T100248937256Z --output PATH_TO_EMPTY_DIRECTORY`.
Existing artifacts are preserved. The initial Track A untuned ANN export reached
its 5,000-iteration limit and emitted a convergence warning.

Run the spreadsheet regression checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## GitHub Actions checks

The `Python Tests` workflow in `.github/workflows/tests.yml` runs on pushes and
pull requests using Ubuntu, Python 3.13, and `deployment/requirements.txt`.
It checks installed dependencies and runs all tests, including saved-model
predictions, diagrams, spreadsheet uploads, and short upload-training checks.
It does not rerun the research experiments or overwrite saved models.

Commit and push the workflow to see runs in the repository's Actions tab.
Manual runs are also available once the workflow is on the default branch.
Streamlit continues deploying independently from its configured branch; a failed
test does not automatically block Streamlit deployment. Require the
`Python 3.13 tests` status check in a branch rule if merges should require passing
tests, and deploy from that protected branch.
