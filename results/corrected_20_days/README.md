# Duplicate-corrected research collection — in progress

The active data contain 180 observations across 20 daily CSVs. The researcher
confirmed that `solar_data_tue_28st.csv` was an accidental second save of the
21st. It is preserved in `data/excluded`, while both `solar_data_tue_21st.csv`
and the distinct `solar_data_tue_28th.csv` remain active. The original processed
189-row data are preserved in `data/historical/solar_data_combined_189.csv`.
All older research outputs are superseded because they included the duplicate.

## Completed independent work

- `dataset_audit.json`: dataset checksum, row/group counts and outer fold sizes.
- `physics_metrics_per_seed.csv` and `physics_predictions.csv`: the no-fitting
  electrical-product baseline on all corrected records (MAE 1.114574 W).
- `exports/sklearn`: six new full-data inference predictors, each reloaded and
  checked against the pre-save predictions. The original seven-fold full-data
  final-selection grid is retained for tuned model exports. Inference fits do
  not supply held-out evaluation scores.

## Full runs

`tensorflow_s1` and `tensorflow_s2` are fresh full research runs, not smoke tests.
Both tracks, seeds 0/1/2, seven outer folds and three inner folds are retained.
The original feature sets, implementations and grids are unchanged. TensorFlow
uses the original S1/S2 settings: `(10,)` and `(16,8)` candidates, tanh, Adam
learning rate 0.01, hidden-kernel L2 0.0001, cap 1000, patience 30, batch sizes
32 and 16 respectively. Training partitions obtain fresh input/target scalers.

There are 42 outer refits and 252 inner neural fits per framework candidate,
plus the original scikit-learn comparisons. On this CPU, the initial outer
fold took roughly ten minutes. Several hours are expected; the estimate is
provisional because stopping times vary. `*.log`, learning curves and per-seed
CSV checkpoints record progress. `progress.json` is an operational checkpoint,
not evidence of completed research.

Six outer test folds have 3 groups/27 rows and train on 17 groups/153 rows;
one has 2 groups/18 rows and trains on 18 groups/162 rows. For 17 training
groups, inner fitting uses 11 or 12 groups and validation uses 6 or 5; for
18 groups, inner fitting uses 12 and validation uses 6.

`scripts/complete_corrected_research.py --stage frameworks` waits for the two
full TensorFlow jobs, validates their corrected identities and metrics, then
chooses by fewer selected-refit cap hits, lower equal-track mean inner MAE,
and finally S1 for an exact tie. It never selects using outer-test scores.
It then runs `pytorch_selected`, writes `comparison`, and exports/reloads
TensorFlow and PyTorch models for both tracks. `completed.json` is written
only after the collection and exports finish. It does not assert that the
manuscript/PDF has been updated or visually reviewed.

Do not rerun an already-running command against the same output directory.
The pipeline refuses collisions. If a process fails, retain its directory and
logs, diagnose the failure, and use a fresh clearly named collection for a new
attempt. The completion worker times out after 24 hours without promoting
incomplete evidence. Keep the computer awake while these CPU jobs run.

```powershell
.\.venv\Scripts\python.exe scripts/audit_corrected_dataset.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_corrected_data.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_corrected_app.py -v
```

## Publication still required

The supplied latest editorial source is preserved in
`manuscript/editorial_reference_superseded.zip`. It includes the old results
and is **not** a corrected deliverable. After completed runs, update every
affected result, rank, statement, table, diagram and figure in the manuscript
and app. Render the editable LaTeX, inspect every PDF page, reconcile the
evidence inventory and update the existing completed-research regression
tests. Run the full automated suite and fitted-predictor app checks. Only
after those steps create `manuscript/publication_ready.json`. Until then,
the manuscript/overview withhold model rankings. No merge or deployment is
authorized by this collection's completion worker.
