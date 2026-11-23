# Streamlit prediction app

The active dataset contains 180 observations across 20 daily files. The dashboard uses the completed 20-day collection exclusively. Model exports are stored in
`results/exports`. The completed research collection and all
final model exports passed prediction, metric and save/reload checks.

## Research dashboard navigation

All six views use neutral research names. Overview and Research Manuscript
present the completed 20-day experiments and reviewed manuscript.
Results Explorer reads only `results/comparison/comparison.csv` and the corresponding saved predictions. Dataset Explorer reads the 180-row dataset. Predict and Model Explorer load eleven predictors from `results/exports/`. The active evidence inventory identifies the same completed runs.

The sidebar opens **Overview → Research Manuscript → Results Explorer → Predict**, followed by Dataset Explorer and Model Explorer.

- **Overview:** a short introduction with the same title and checked S2 evidence as
  the manuscript, research motivation, approach, three findings and section links.
- **Research Manuscript:** the full continuous scientific report with in-page contents,
  methods, results, discussion, limitations, references and reproducibility information.
- **Results Explorer:** the 20-day matched comparison, metric downloads, held-out predictions, daily profiles, residual plots and inner-fold learning curves.
- **Predict:** the eleven saved predictors, single and spreadsheet
  predictions, optional compatible-model comparison, range warnings, evaluation
  downloads, and explicit uploaded-data training with held-out evaluation. Choose Saved models or Train from uploaded data using Model source on the same page.
- **Dataset Explorer:** source-day and numeric filters, column selection,
  sorting/downloads, measurement profiles, distributions, correlations and units.
- **Model Explorer:** track/framework/family filters, up to two saved models,
  architecture downloads, actual estimator settings, parameters and relevant metrics.

The former Research Methodology & Limitations page has been removed. Its methods,
metric definitions and S1/S2 cap-hit diagnostics are consolidated in the manuscript.

The overview summarises the manuscript rather than maintaining a separate research account.
Its section links route to manuscript anchors using `?view=manuscript#section`.
Results Explorer charts the completed 20-day comparison and its saved predictions.
Detailed interpretation lives in the manuscript.
The original study preprint is distinguished from the subsequent Python results.
Prediction form defaults use cached dataset medians when available, with observed
ranges in input help and warnings for values beyond those ranges. Model inspection
and prediction share one cached loader. No saved weights or experimental results
are changed by this refactor.

`app.py` owns navigation; `prediction_workspace.py` retains the prediction UI;
`research_pages.py` renders research views; `research_data.py` caches read-only
artifact loading and supplies statistics/error transformations. Existing registry,
batch validation, training and diagram modules remain shared.

Predict combines saved-model inference and session training in `prediction_workspace.py`. The Model source selector switches modes and clears previous results. Use the Train From Uploaded Data tab to fit and evaluate a session model, then use the prediction tabs with that model.

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
measurements are approximately hourly. The dashboard presents only the completed 20-day metrics.
Resolving provenance and publishing a duplicate-aware re-evaluation are separate
research tasks. No updated performance is claimed in this UI change.

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

The manuscript page presents the article without PDF or LaTeX download buttons. The source files remain under `manuscript/`.
All pages were rendered and visually reviewed after the tables and figures were
regenerated from the completed 20-day experiments. Historical charts, downloads
and operational views remain available separately.

Run the focused checks with `python -m unittest discover -s tests -p test_manuscript.py -v`.
When adding a new verified experiment, update the source mapping and run these
checks before changing claims or the chosen research collection. The current
manuscript has six numbered tables and seven numbered figures; equations are
rendered with native LaTeX and references use matching numbered links.

Research pages require `data/processed/solar_data_combined.csv` and `results/`, including the completed manifest, matched comparison, both selected runs, S1/S2 convergence evidence, and final exports. Keep each run's configuration, predictions and learning-curve CSVs with its metrics. The placeholder date does not establish calendar coverage. Charts use held-out predictions, not fitted-model predictions.

No new packages are required; interactive charts use Streamlit's existing Altair
dependency. Missing artifacts display an unavailable-view message. Upload metrics
remain descriptive and separate from published CV results. All source datasets,
research results and saved weights are read-only throughout dashboard operation.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run deployment/app.py
```

The model selector covers eleven predictors from the 20-day collection.
Current predictors include TensorFlow S2, PyTorch S2, tuned/untuned
scikit-learn ANNs and linear regression for both tracks, plus the Track A
physics baseline. The Models tab offers a separate diagram selector and PNG
download for every saved model. Neural-network images use actual saved layer
sizes; regression and physics images show their computation structure.

Open **Predict** and select Saved models or Train from uploaded data. Changing the selected model clears previous results;
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

Open **Predict**, set Model source to Train from uploaded data, select a model, and use **Train From Uploaded Data**.
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

Run Streamlit from the repository root so `.streamlit/config.toml` is loaded.
The deployment disables source-file watching to avoid shared-module invalidation
during live updates, which can surface as import-time `KeyError: 'manuscript'`
or `KeyError: 'model_registry'`. Widget interactions and normal script reruns
remain enabled. Restart the app process after source updates; on Community Cloud,
use **Manage app → Reboot app** after this configuration reaches the deployed
branch. A browser refresh alone does not restart the Python process.

For local development only, opt back into source watching explicitly:

```bash
python -m streamlit run deployment/app.py --server.fileWatcherType=auto
```

The adjacent `requirements.txt` delegates to `../requirements.txt`; all package
versions live in the root requirements file, including CPU PyTorch. Keep both
files in the repository. CI uses the same dependency list.
Include the deployment folder, the full `src/` folder and these artifacts:

| Model family | Active artifact directory | Required files per track |
|---|---|---|
| TensorFlow S2 | `results/exports/tensorflow` | `.keras`, x/y scalers, TF metadata |
| PyTorch S2 | `results/exports/pytorch` | `.pt` state dictionary, x/y scalers, torch metadata |
| scikit-learn | `results/exports/sklearn` | tuned ANN, untuned ANN and linear `.joblib` pipelines |

The registry chooses these directories explicitly. Architecture, epoch counts, parameter counts and source runs are read from each current export's metadata. The app's learning-curve selector uses `tensorflow_s1`, `tensorflow_s2` and `pytorch_selected` within `results`.
Missing predictors are marked and cannot make predictions. No dataset is
required for inference/upload training. Only load trusted model/scaler files.

To regenerate missing baseline exports locally, run `python src/export_baseline_models.py`.
To recreate an S2 TensorFlow model, use the completed comparison and an empty
output directory:
`python src/train_final_tf.py --track A --run results/tensorflow_s2 --output PATH_TO_EMPTY_DIRECTORY`.
For PyTorch use `src/train_final_torch.py` with `--run results/pytorch_selected`.
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

Dataset fingerprints use CSV bytes normalised to LF line endings. This keeps model and result verification consistent on Windows and Linux.

### Uploaded training experiments

The Predict page's training tab separates **Train a New Model** (fresh weights,
training-only scalers; no saved model required) from **Continue Training an
Existing Model** (independent copy of TensorFlow/Keras or PyTorch weights, original
input/target scalers and architecture preserved). Continuation starts a new Adam
optimizer; it does not resume optimizer momentum. Saved scikit-learn pipelines
are excluded from continuation because end-to-end incremental training is not
supported reliably. All artifacts stay in session memory or temporary export
folders. Nothing is written into research export or publication directories.

CSV/XLSX uploads accept up to 50,000 observations and 20 MB. Required features
follow Track A/B plus measured `power` in watts. The interface previews data,
reports rejected observations, retains and warns about duplicates, and requires
confirmation of the displayed units. It cannot infer units reliably from numeric
values. Use complete ISO timestamps (naive timestamps are interpreted as UTC).

Choose random, grouped by day_id/source_file/calendar day, or chronological
holdout (20% default). Group identifiers must represent complete measurement
days. Chronological splitting requires timestamps and keeps calendar days intact.
Preprocessing and model fitting use training rows only; the held-out partition is
not used for early stopping. Optional linear regression uses exactly the same
features and partitions. MAE/RMSE/MAPE/WAPE/R2 use raw outputs, including negative
predictions. MAPE omits zero targets and becomes unstable near zero; WAPE is
undefined if all targets are zero. Repeated tuning against the same holdout biases
evaluation. This workflow estimates contemporaneous power, not future horizons.

Downloads contain weights/pipeline, neural preprocessing, run metadata, held-out
predictions, evaluation summary and rejected rows where applicable. Extract a
trusted package into a separate directory and use
`model_registry.load_model('<track>_<family>', directory)` with the corresponding
registry key (for example `b_tuned`, `a_pytorch`, `b_tensorflow`, or `a_linear`).
Never extract a session package over the research model directory.

Keep the Romanian dataset outside version control and use it as an interface
acceptance case: check units/feature mapping first, then run grouped and
chronological evaluations and reload the downloaded model. No Romanian file is
bundled with this change. `tests/test_uploaded_training.py` includes a synthetic
12,000-observation installation to exercise uploads larger than the research data.
The original Nigerian study remains its own reproducible research record.


#### Uploaded ANN epochs and early stopping

TensorFlow/Keras and PyTorch session ANNs default to a maximum of 200 epochs,
configurable from 1 to 1,000, for both fresh and continued training. Early stopping
is enabled by default with patience 30 and min_delta 0.0001. These controls do
not change the published 20-day cross-validation experiments or saved models.

After the outer evaluation split, 20% of the training partition is reserved for
internal validation. Random splitting uses random rows; grouped splitting keeps
the selected day/source groups disjoint; chronological splitting reserves later
complete UTC days, after the fitting days and before the outer evaluation days.
Insufficient groups/days produce an explicit error rather than a random fallback.
Fresh-model input and target scalers fit only on the fitting subset. Continued
training retains copied source-model scalers. The optional linear baseline uses
the same fitting rows and outer evaluation rows as the ANN.

Validation loss is scaled-target MSE plus hidden-kernel L2 regularisation.
Patience counts consecutive epochs without an improvement greater than min_delta,
relative to the last significant improvement. The absolute lowest-validation-loss
checkpoint is restored even when the maximum epoch budget is reached. Disabling
early stopping runs all requested epochs; internal validation and best-checkpoint
restoration remain active. Chronological runs do not shuffle training batches.
No outer held-out measurements enter fitting, scaling or checkpoint selection.

Results show requested maximum epochs, actual completed epochs, best epoch,
whether training stopped early, and training/validation loss curves. Existing ZIP
formats and model filenames remain compatible; run metadata now also records
these settings, curves, fitting/validation/evaluation source-row numbers and the
internal split details. `rows` counts fitting rows; `validation_rows` retains its
existing meaning of outer held-out evaluation rows; the separate
`early_stopping_validation_rows` field counts internal validation observations.


The Predict page loads its upload-training backend by local source version. This
prevents a Streamlit hot rerun from retaining an older `train_uploaded` signature
while displaying newer training controls. In-progress runs keep their original
module instance. For deployments, publish the page and training modules together;
restart/reboot the app if a running server still uses an older deployed release.

#### Column mappings, custom inputs and reusable JSON

In **Train a New Model**, choose Track A, Track B, or Custom Features. Upload a
CSV/XLSX with its original headers. Select and name the target, specify its unit,
and assign optional grouping and timestamp columns. Track presets suggest known
aliases (for example GHI, AmbientTemp, RH and AC_Power); review these suggestions.
For Custom Features, add/remove numerical source columns and edit their display
names, stable internal identifiers and units. Standard definitions keep their
established units. For other units, use a separate custom identifier. No values
are converted, and the original uploaded frame is never renamed in place.

**Download mapping configuration** produces a portable `column_mapping.json`.
In a later session, upload a dataset with compatible headers, upload that JSON,
and click **Apply mapping configuration**. It restores input order, source names,
display labels, identifiers, units, target and grouping/timestamp roles. Model
hyperparameters are configured separately. Invalid/missing/duplicate mappings
are rejected; a malformed imported JSON does not replace an applied valid mapping.
**Restore feature preset** resets mapping controls. Actual mapping edits clear
stale trained-model and prediction results while preserving other UI settings.

Target and grouping/timestamp columns cannot be inputs. Raw datetime/categorical
inputs require explicit external preprocessing into suitable numerical columns.
Invalid measurements retain rejection reasons. Inputs that duplicate the target
and current/voltage combinations produce leakage/methodology warnings. Adding
inputs is not a guarantee of better accuracy. Confirm all units before training.

The existing training-only scalers, internal validation, early stopping and
held-out evaluation rules apply unchanged to custom feature matrices. The linear
baseline receives the same selected features, target and fitting/evaluation rows.
Charts and metric summaries use the configured target name/unit. For non-watt
outputs, downloaded predictions use generic `predicted_value`/`target_value`
columns and explicitly include the target unit; no watt conversion is implied.

Single Prediction automatically follows the active session model's input schema.
Spreadsheet Prediction suggests its saved source columns and lets you explicitly
map alternatives, once per required input. No restart or selection of a matching
saved research model is needed. Continuing a saved ANN also allows source-column
mapping, but preserves its input identifiers/order, units, architecture and scalers.
Choose a new model to change the input structure.

Mapped session ZIPs include `feature_config.json`, `metadata.json`, a
`session_manifest.json`, weights/estimator, neural scalers, evaluation summary and
held-out predictions. The complete uploaded training dataset is not embedded.
Extract a trusted package to a separate directory and call
`model_registry.load_session_model(directory)`. Its `predict(frame)` accepts the
saved source headers or an internally mapped frame; use
`feature_mapping.map_frame(..., require_target=False, overrides={...})` to map
alternative prediction headers explicitly. Legacy model downloads/loaders remain
supported, and no research artifacts are overwritten.

The upload spreadsheet reader is also loaded by source version, together with the
session trainer. This avoids stale `read_sheet(..., preserve_columns=True)`
signature errors during hot reruns; export helpers use the matching reader module.
After deploying changes to several application modules, reboot the running app
once to load the complete updated release.


`session_runtime.py` loads the registry, feature mappings, spreadsheet reader,
trainer and mapping UI as one immutable source snapshot. Imports within that
snapshot resolve to the matching versions, even when Streamlit retains older
modules in `sys.modules`. New source versions do not reload or mutate an active
training run. Legacy cached source-model specifications are copied into the
current specification before applying input mappings. Deploy all workflow files
together, including `session_runtime.py`; `training_runtime.py` remains a
compatibility entrypoint for existing callers.
