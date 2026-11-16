# Solar power estimation: Python research and Streamlit app

This study re-evaluates an earlier MATLAB neural-network project using electrical and environmental measurements. Both tracks estimate power from measurements at the target time; no advance forecasting horizon is evaluated.

## Dataset and evaluation

The active dataset contains **180 observations across 20 daily files**, with nine readings per file. One repeated daily file, `solar_data_tue_28st.csv`, was excluded before evaluation and is retained under `data/excluded/`. The distinct `solar_data_tue_21st.csv` and `solar_data_tue_28th.csv` remain included. No exact measurement records repeat when filename identity is excluded.

Raw files are in `data/raw/`; `src/load_data.py` produces `data/processed/solar_data_combined.csv`. The previous 189-row dataset is preserved under `data/historical/`. Full calendar dates and calibration records are unavailable. Collection is attributed to Olabisi Onabanjo University in the original project documentation.

| Track | Inputs | Interpretation |
|---|---|---|
| A | Irradiance, current, voltage, temperature, humidity | Contemporaneous electrical-input estimation |
| B | Irradiance, temperature, humidity, observation hour | Contemporaneous environmental/time estimation |

Seven outer grouped folds hold out complete daily files: six test partitions have 27 rows and one has 18. Three inner grouped folds select tuned models within the outer training days. Scaling is fitted on training partitions only. The neural-network seeds are 0, 1 and 2. Metrics pool all 180 held-out predictions per seed, then report mean +/- population seed standard deviation. This variation is descriptive, not a confidence interval.

## Completed results

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

Linear regression has the lowest mean MAE in both tracks. The no-fitting current x voltage baseline has MAE **1.115 W**, RMSE **1.254 W**, MAPE **0.424%**, and R2 **0.997898**. Comparisons remain descriptive; no significance test or external validation is supplied.

The main results are in `results/research_20_days/comparison/`. The dashboard uses this collection exclusively. The former 21-day result directories and model exports have been removed from the working tree; earlier commits retain their history.

## Neural training settings

Both TensorFlow candidates use architectures `(10,)` and `(16, 8)`, tanh hidden layers, a linear output, Adam learning rate 0.01, hidden-kernel L2 0.0001, maximum 1,000 epochs and early-stopping patience 30. S1 uses batch size 32; S2 uses 16. Architecture selection uses mean inner MAE in watts. Each outer refit uses the selected architecture's median best inner epoch.

S2 was selected by fewer selected-refit epoch-cap hits: **0 for S2 versus 2 for S1**. The next tie-breakers are lower equal-track mean inner MAE, then S1 for an exact tie. Outer scores do not enter this selection. PyTorch completed the same selected settings, folds and seeds. Framework implementations can still differ numerically.

```powershell
.\.venv\Scripts\python.exe -u src/train_ann_tf.py --tracks A B --seeds 0 1 2 --epochs 1000 --learning-rate 0.01 --batch-size 32 --patience 30 --output results/reproduction/tensorflow_s1
.\.venv\Scripts\python.exe -u src/train_ann_tf.py --tracks A B --seeds 0 1 2 --epochs 1000 --learning-rate 0.01 --batch-size 16 --patience 30 --output results/reproduction/tensorflow_s2
.\.venv\Scripts\python.exe -u src/train_ann_torch.py --tracks A B --seeds 0 1 2 --epochs 1000 --learning-rate 0.01 --batch-size 16 --patience 30 --output results/reproduction/pytorch_selected
```

Use new output directories when reproducing runs. Final inference models are fitted on all 180 observations, independently of the held-out scores above. TensorFlow/PyTorch final exports use the most frequent inner-selected architecture and its median selected epoch count, with seed 0. All exports passed save/reload prediction equality checks. Their metadata records the active dataset digest.

## Streamlit and manuscript

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run deployment/app.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The app has seven views: Overview, Research Manuscript, Results Explorer, Predict, Train / Experiment, Dataset Explorer and Model Explorer. Eleven predictors from the 20-day collection are available. Predictions use saved models. Upload training creates an isolated session model and does not alter research artifacts.

The manuscript uses the supplied edited title and six references. Editable LaTeX, regenerated tables, main figures and supplementary views are under `manuscript/source/`. The PDF and LaTeX bundle are downloadable from the manuscript page. See [deployment documentation](deployment/README.md), [model card](MODEL_CARD.md), and [manuscript files](manuscript/README.md).

Small sample size, uncertain temporal coverage and absent external validation limit generalisation. Track A needs electrical measurements; Track B still needs contemporaneous environmental measurements. Neither demonstrates future weather forecasting or reliability at other installations.

## Automated Streamlit Health Monitoring

The [Streamlit Health Check workflow](.github/workflows/streamlit-health-check.yml)
opens the public deployment in headless Chromium every six hours (`0 */6 * * *`,
UTC). It checks the real Overview UI, including the embedded app frame, rather
than treating an HTTP 200 landing page as proof of health. If it sees the normal
Community Cloud sleep screen, it clicks a recognized wake button once and waits
for the Overview to load. It never opens prediction/training tools or uploads data.
The monitor uses only Playwright; production/ML requirements are unchanged.

After merging these files into the repository's **default branch**, enable GitHub
Actions if necessary, open **Actions → Streamlit Health Check → Run workflow**,
and inspect the run's **Verify public Streamlit interface** logs. Successful runs
report `already_awake` or `awakened`. The workflow fails if readiness cannot be
verified. Failed checks provide a `streamlit-health-check-…` artifact with a
viewport screenshot and bounded JSON diagnostics, retained for seven days.
Authentication/access-gate screenshots are deliberately omitted. No cookies,
credentials, page HTML, raw console messages or URL queries are recorded. Browser
installation failures appear in the job logs and may have no screenshot. Use
GitHub's built-in Actions notification settings for failure notifications.

Change the workflow's `schedule` cron to adjust frequency. Use its Actions menu
**Disable workflow**, or remove the `schedule` entry (retaining manual runs), to
disable scheduled monitoring. Schedules run from the default branch and may be
delayed or dropped under GitHub load. GitHub can disable scheduled workflows in
inactive public repositories after 60 days; check that the workflow remains enabled.
See [GitHub's scheduling rules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

To run locally in a separate environment (PowerShell):

```powershell
python -m venv .venv-health
.\.venv-health\Scripts\python.exe -m pip install -r scripts/requirements-health-check.txt
.\.venv-health\Scripts\python.exe -m playwright install --only-shell chromium
.\.venv-health\Scripts\python.exe scripts/check_streamlit.py --timeout 180
$env:RUN_HEALTH_BROWSER_TESTS = '1'
.\.venv-health\Scripts\python.exe -m unittest discover -s tests -p test_streamlit_health_check.py -v
```

On Linux, install the browser with `python -m playwright install --with-deps
--only-shell chromium`. The script accepts `--url`, `--timeout` (total budget,
default 180 seconds), `--attempts` (default two, maximum three), and
`--artifacts-dir`. It retries only transient connectivity/hosting failures with
short exponential backoff, within that budget. It does not refresh while the app
is initializing or retry a wake click. An unknown sleep-screen button fails with
diagnostics instead of clicking an arbitrary control. Exit code 0 means the
Overview was verified; 1 means the check failed; 2 indicates invalid arguments.
The regular ML test suite skips browser fixtures unless explicitly enabled.

**This is best-effort monitoring and cannot guarantee prevention of Streamlit
Community Cloud hibernation.** A network, browser or readiness timeout means
health is unverified, not a confirmed application outage. Deployment errors,
access restrictions, sleep/wake failures and connectivity failures are reported
separately. The check does not sign in, solve CAPTCHAs or bypass hosting controls.
Cloud UI changes may require updating the wake labels or Overview readiness checks.
Local fixtures test wake behavior; only a real sleeping deployment can establish
that the live wake-up flow works.

## Original project

[Original MATLAB repository](https://github.com/OluwafisayoIbrahim/ANN-Forecasting-for-Solar-Power) | [Original preprint](https://doi.org/10.2139/ssrn.4771898). Historical findings are distinct from this Python re-evaluation.
