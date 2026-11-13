# Streamlit prediction app

## Research dashboard navigation

The sidebar opens **Overview → Research Manuscript → Results Explorer → Predict →
Train / Experiment**, followed by Dataset Explorer and Model Explorer.

- **Overview:** a short introduction with the same title and checked S2 evidence as
  the manuscript, research motivation, approach, three findings and section links.
- **Research Manuscript:** the full continuous scientific report with in-page contents,
  methods, results, discussion, limitations, references and reproducibility information.
- **Results Explorer:** separate original, historical TensorFlow and matched
  S2 collections; metric filters, saved out-of-fold scatter/residual plots,
  daily profiles and inner-fold learning curves.
- **Predict:** the existing thirteen saved predictors, single and spreadsheet
  predictions, optional compatible-model comparison, range warnings, evaluation
  downloads. This page always uses saved project models.
- **Train / Experiment:** explicit uploaded-data training, held-out evaluation, model
  downloads and single/spreadsheet predictions using the trained session model.
- **Dataset Explorer:** source-day and numeric filters, column selection,
  sorting/downloads, measurement profiles, distributions, correlations and units.
- **Model Explorer:** track/framework/family filters, up to two saved models,
  architecture downloads, actual estimator settings, parameters and relevant metrics.

The former Research Methodology & Limitations page has been removed. Its methods,
metric definitions and S1/S2 cap-hit diagnostics are consolidated in the manuscript.

The overview summarises the manuscript rather than maintaining a separate research account.
Its section links route to manuscript anchors using `?view=manuscript#section`.
The original overview workflow and comparison image are retained in Results Explorer
under Original research summary figures. Detailed interpretation lives in the manuscript.
The original study preprint is distinguished from the subsequent Python results.
Prediction form defaults use cached dataset medians when available, with observed
ranges in input help and warnings for values beyond those ranges. Model inspection
and prediction share one cached loader. No saved weights or experimental results
are changed by this refactor.

`app.py` owns navigation; `prediction_workspace.py` retains the prediction UI;
`research_pages.py` renders research views; `research_data.py` caches read-only
artifact loading and supplies statistics/error transformations. Existing registry,
batch validation, training and diagram modules remain shared.

The two operational routes reuse `prediction_workspace.py` with an explicit view:
Predict hides training controls and uses saved models; Train / Experiment opens on
training and provides session-inference tabs. Switching routes clears stale batch
outputs, while a compatible trained session model can remain available. Standalone
execution keeps the combined workspace for existing integrations and regression tests.

### Manuscript evidence and maintenance

`manuscript.py` renders the article in a native container with a maximum reading
width of 940 px that shrinks to its parent on narrow screens. `manuscript_content.py`
holds prose and the bibliography; `manuscript_data.py` loads and checks the saved S2
outputs; `manuscript_figures.py` generates charts without fitting models. The JSON
evidence inventory (`manuscript_evidence.json`) maps claims, numbers and figures to
source files and records citation verification and unresolved provenance issues.

Tables 4–6, the abstract, discussion and comparison chart share the same checked
comparison/per-seed data. The loader recomputes metrics from saved out-of-fold
predictions, checks row/target identities and filename-group assignments, and rejects
missing, duplicate or inconsistent results. Figures 4, 6 and 7 disclose seed 0 and
use saved outer predictions rather than final deployment weights. Figure controls
change only the view of fixed evidence. Uploads/session training never replace it.

The November 5, 2026 audit found nine repeated measurement records across `solar_data_tue_21st.csv`
and `solar_data_tue_28st.csv`. There are zero full-record duplicates when the source
filename is included. These different definitions must not be conflated: 21 filenames
do not prove 21 unique collection days. The November 13 follow-up checked saved
TensorFlow and PyTorch S2 predictions: the files occupy outer folds 1 and 7,
respectively. Identical copies cross training/test boundaries when either fold
is held out. The effect on reported error remains unmeasured. Intervals range from 35 to 92 minutes;
measurements are approximately hourly. Data and historical metrics are preserved.
Resolving provenance and publishing a duplicate-aware re-evaluation are separate
research tasks. No corrected performance is claimed in this UI change.

The original MATLAB README and `NeuralNet.m` were inspected at commit
`578401bb0d106c53f0e152380441a8338854ed57`. The README does not state an input count;
the script selects numeric time alongside five measured predictors. A daily function discussed in the README
is absent from that tree. The manuscript explicitly distinguishes these sources. This corrects the initial
November 5 input-count attribution. Location and 2023 completion are attributed
to project documentation, without independent institutional certification.
The original SSRN work is a preprint; full-text access was blocked during the audit,
so its numerical results are not imported as verified benchmarks. A mean-prediction
sanity check exists, but no saved grouped-CV mean-baseline experiment was found.

The November 13, 2026 reference review checked all six identities and bibliographic
records: references 1, 2 and 4 at record/abstract level, and references 3, 5 and 6
additionally against relevant full-text passages. SSRN full text and historical
MATLAB numerical results remain unverified. Antonanzas et al. classify studies
by forecast horizon and input origin and propose comparability guidelines; the
classification of Tracks A/B as contemporaneous estimation is this project's
interpretation of its features and protocol. The README-documented selection
rule is not independently established as predeclared.

No packages, weights or experiment outputs are added or changed for this manuscript.
Reliable PDF export is deferred pending a separate pagination/render-verification
workflow; the Streamlit article is the current presentation. All legacy charts,
downloads and operational views remain available.

Run the focused checks with `python -m unittest discover -s tests -p test_manuscript.py -v`.
When adding a new verified experiment, update the source mapping and run these
checks before changing claims or the chosen research collection. The current
manuscript has six numbered tables and seven numbered figures; equations are
rendered with native LaTeX and references use matching numbered links.

Research pages require `data/processed/solar_data_combined.csv`, original reports
under `results/metrics` and `results/predictions`, TensorFlow run directories,
`results/pytorch/parity_s2`, and `results/comparisons/parity_s2`. Keep each run's
configuration, predictions and learning-curve CSVs with its metrics. Historical
settings are shown separately; no fitted-model predictions replace held-out records.
The dataset's 1900 date is a placeholder: profiles use recorded hour and source day.
Calendar coverage cannot be established from those timestamps. Original prediction
CSVs do not identify a seed; their filenames are shown without inventing one.

No new packages are required; interactive charts use Streamlit's existing Altair
dependency. Missing artifacts display an unavailable-view message. Upload metrics
remain descriptive and separate from published CV results. All source datasets,
research results and saved weights are read-only throughout dashboard operation.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run deployment/app.py
```

The model selector covers thirteen saved predictors: eleven current predictors
plus historical TensorFlow Track A (1000 epochs) and Track B (82 epochs).
Historical models are explicitly labelled in prediction and diagram selectors
and Model Explorer, and their metrics come from their own saved experiments.
Current predictors include TensorFlow S2, PyTorch S2, tuned/untuned
scikit-learn ANNs and linear regression for both tracks, plus the Track A
physics baseline. The Models tab offers a separate diagram selector and PNG
download for every saved model. Neural-network images use actual saved layer
sizes; regression and physics images show their computation structure.

Open **Predict** for saved-model inference, or **Train / Experiment**
to fit a fresh session model. Changing the selected model clears previous results;
switching workspaces clears batch outputs so saved and session results are not mixed.
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

Open **Train / Experiment**, select a model, and use **Train From Uploaded Data**.
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
hyperparameter search. TensorFlow and PyTorch use a fixed epoch count (1-500, default 100)
chosen before training. These upload scores are not the published nested-CV
scores, and repeated tuning against the same holdout can bias them.

After training, single and batch predictions in this mode use the session model.
Download its ZIP before closing the session: TensorFlow exports the network,
both scalers and metadata; PyTorch exports a CPU `.pt` state dictionary,
both scalers and metadata; sklearn exports its complete preprocessing pipeline
and metadata. The export remains fitted on the training partition, not all rows.
Serialization uses temporary files that are removed after creating the download.
Saved project artifacts are never overwritten. The physics baseline cannot be
trained because `P = I * V` has no learned parameters; use saved-model mode for it.

## Deployment files

Choose `deployment/app.py` as the Streamlit entrypoint and Python 3.13.
The adjacent `requirements.txt` delegates to `../requirements.txt`; all package
versions live in the root requirements file, including CPU PyTorch. Keep both
files in the repository. CI uses the same dependency list.
Include the deployment folder, the full `src/` folder and these artifacts:

| Model family | Active artifact directory | Required files per track |
|---|---|---|
| TensorFlow S2 | `results/exports/tf_s2` | `.keras`, x/y scalers, TF metadata |
| PyTorch S2 | `results/exports/torch_s2` | `.pt` state dictionary, x/y scalers, torch metadata |
| scikit-learn | `results/models` | tuned ANN, untuned ANN and linear `.joblib` pipelines |

The registry chooses these directories explicitly. Older TensorFlow files in
`results/models` are preserved but are no longer the app's default models.
Both S2 frameworks use hidden layers `(16, 8)`. TensorFlow A/B use 167/5 epochs;
PyTorch A/B use 124/16. The app shows each network's parameter count, source run
and diagram. Include `results/tensorflow/convergence_s2/learning_curves` and
`results/pytorch/parity_s2/learning_curves` to display their comparison curves.
Missing predictors are marked and cannot make predictions. No dataset is
required for inference/upload training. Only load trusted model/scaler files.

To regenerate missing baseline exports locally, run `python src/export_baseline_models.py`.
To recreate an S2 TensorFlow model, use the completed comparison and an empty
output directory:
`python src/train_final_tf.py --track A --run results/tensorflow/convergence_s2 --output PATH_TO_EMPTY_DIRECTORY`.
For PyTorch use `src/train_final_torch.py` with `--run results/pytorch/parity_s2`.
Repeat with `--track B` for Track B. Do not rerun exports into existing folders.
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

Before pushing, include both S2 export directories with their scaler files,
the PyTorch source modules and the updated app/tests. Push to Streamlit's
configured deployment branch only after local tests pass. These local edits
do not update the hosted app until that push/redeployment happens.

Commit and push the workflow to see runs in the repository's Actions tab.
Manual runs are also available once the workflow is on the default branch.
Streamlit continues deploying independently from its configured branch; a failed
test does not automatically block Streamlit deployment. Require the
`Python 3.13 tests` status check in a branch rule if merges should require passing
tests, and deploy from that protected branch.
