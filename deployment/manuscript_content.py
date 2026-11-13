"""Manuscript prose and verified bibliography, separate from Streamlit rendering."""
from manuscript_data import score

TITLE = "Re-evaluating neural networks for solar power estimation: features, baselines and held-out-day validation"
AUTHOR = "Oluwafisayo Ibrahim"
ORIGINAL_COMMIT = "578401bb0d106c53f0e152380441a8338854ed57"
ORIGINAL = "https://github.com/OluwafisayoIbrahim/ANN-Forecasting-for-Solar-Power"
REPOSITORY = "https://github.com/OluwafisayoIbrahim/solar-power-prediction-python"
KEYWORDS = "Photovoltaic power; neural networks; feature availability; grouped cross-validation; baselines; reproducibility"
SECTIONS = [
    ("abstract", "Abstract"), ("introduction", "1. Introduction"),
    ("related-work", "2. Related work"), ("methods", "3. Materials and methods"),
    ("results", "4. Experimental results"), ("discussion", "5. Discussion"),
    ("limitations", "6. Limitations"), ("conclusion", "7. Conclusion and future work"),
    ("references", "8. References"), ("reproducibility", "9. Research transparency and reproducibility")]

# Initial metadata audit: 2026-11-05; scoped reference follow-up: 2026-11-13.
REFERENCES = [
    ("O. Ibrahim, “Forecasting Solar Power Generation using Artificial Neural Networks,” SSRN preprint, "
     "manuscript dated May 25, 2023; posted Apr. 18, 2024. doi: 10.2139/ssrn.4771898.",
     "https://doi.org/10.2139/ssrn.4771898"),
    ("J. Antonanzas, N. Osorio, R. Escobar, R. Urraca, F. J. Martinez-de-Pison, and F. Antonanzas-Torres, "
     "“Review of photovoltaic power forecasting,” Solar Energy, vol. 136, pp. 78–111, 2016. doi: 10.1016/j.solener.2016.06.069.",
     "https://doi.org/10.1016/j.solener.2016.06.069"),
    ("K. Hornik, M. Stinchcombe, and H. White, “Multilayer feedforward networks are universal approximators,” "
     "Neural Networks, vol. 2, no. 5, pp. 359–366, 1989. doi: 10.1016/0893-6080(89)90020-8.",
     "https://doi.org/10.1016/0893-6080(89)90020-8"),
    ("S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” "
     "ACM Transactions on Knowledge Discovery from Data, vol. 6, no. 4, Art. 15, 2012. doi: 10.1145/2382577.2382579.",
     "https://doi.org/10.1145/2382577.2382579"),
    ("D. R. Roberts et al., “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” "
     "Ecography, vol. 40, no. 8, pp. 913–929, 2017. doi: 10.1111/ecog.02881.",
     "https://doi.org/10.1111/ecog.02881"),
    ("G. C. Cawley and N. L. C. Talbot, “On over-fitting in model selection and subsequent selection bias in performance evaluation,” "
     "Journal of Machine Learning Research, vol. 11, pp. 2079–2107, 2010.",
     "https://www.jmlr.org/papers/v11/cawley10a.html"),
]

REFERENCE_REVIEW = (
    "Reference review, November 13, 2026: all six reference identities and bibliographic records "
    "were checked. References [1], [2] and [4] were checked at record/abstract level; "
    "[3], [5] and [6] were additionally checked against relevant full-text passages. "
    "This does not mean all six papers were fully read or independently verified in their entirety. "
    "SSRN full text and historical MATLAB numerical results remain unverified. "
    "The original preprint is not presented as peer reviewed."
)

INTRODUCTION = """
Photovoltaic power forecasting supports the scheduling and integration of variable solar generation.
Forecasting studies must specify both the prediction horizon and the information available before
that horizon; weather measurements, weather forecasts and electrical measurements imply different
prediction problems [[2]](#ref-2).

Project documentation describes the original undergraduate investigation as completed at
Olabisi Onabanjo University in 2023; this is not independent institutional certification. It
implemented a MATLAB feedforward ANN to estimate power from measured conditions [[1]](#ref-1).
The subsequent Python study revisited its task definition, evaluation and baseline comparisons.
It asks whether low errors reflect useful generalisation or access to electrical measurements
that already constrain the target. It also examines whether network complexity provides value
relative to linear regression under the same source-day splits.

**Research questions.** RQ1: What changes when measured electrical inputs are included?
RQ2: How do ANNs compare with linear and electrical physics baselines within each track?
RQ3: What performance is obtained on held-out source-day groups?
RQ4: How consistent are matched TensorFlow/Keras and PyTorch results across the recorded seeds?
These are questions about this dataset and protocol, rather than claims of a new forecasting algorithm.
"""

RELATED_WORK = """
**Solar prediction and feature availability.** Antonanzas et al. classify photovoltaic forecasting
studies by forecast horizon and input origin and propose comparability guidelines [[2]](#ref-2).
This project interprets Tracks A/B as contemporaneous estimation based on their features and
protocol, which use measurements at the target time without an advance horizon. This is our
interpretation of these experiments, not a dataset-specific finding of that review.
The electrical baseline implemented here is the direct product of measured voltage and current;
it is not a weather-driven photovoltaic performance model.

**Feedforward regression.** Hornik et al. establish approximation properties for sufficiently
large feedforward networks under stated assumptions [[3]](#ref-3). Representational capacity
does not establish that a particular fitted ANN will generalise from a small solar dataset.
This study therefore evaluates small networks against a simpler linear model instead of treating
nonlinearity as evidence of superior performance.

**Leakage and structured validation.** Kaufman et al. relate leakage to the modelling goal and
information that is legitimately available [[4]](#ref-4). Current and voltage are legitimate
inputs for electrical estimation, but unavailable shortcuts for a weather-only deployment.
This feature-availability issue differs from fitting preprocessing on test observations.
Roberts et al. explain why validation must respect dependence structures [[5]](#ref-5).
Here, source-file grouping is used to separate nominal measurement days; the duplicate-file
finding below limits the interpretation of that separation.

**Selection bias and baseline comparisons.** Cawley and Talbot show that finite-sample model
selection can itself overfit and bias evaluation [[6]](#ref-6). The Python experiments separate
inner selection from outer scoring and retain simple baselines, shared metrics, configurations
and per-seed predictions. These design choices make the comparisons inspectable; they do not
remove all uncertainty from repeated development on one dataset.
"""

HISTORICAL = f"""
The original repository describes `fitnet(10)`, ten hidden neurons, `tansig` hidden and
`purelin` output activations, Levenberg–Marquardt (`trainlm`) training and `mapminmax`
normalisation in a daily training function. It reports observation-level random splitting,
per-day evaluation and reuse of daily observations for training and testing.

**Historical source distinction.** The inspected
[NeuralNet.m]({ORIGINAL}/blob/{ORIGINAL_COMMIT}/NeuralNet.m) combines five named daily files,
uses `dividerand(..., 0.8, 0.1, 0.1)`, sets a 9,000-epoch cap, and does not disable
`fitnet`'s internal division. It converts time to numeric form and selects `1:end-1` as
inputs. Thus, this script includes numeric time alongside five measured predictors. The
README at this commit does not state an input count; no input-count conflict is established.
The daily function mentioned in the README
is absent from this inspected tree. Hidden/output activations and daily normalisation are
README-reported details, not explicit settings in this script.

The preprint [[1]](#ref-1) belongs to the original study. Its full text was not accessible
during this audit; no independently verified MATLAB result table or held-out predictions
are available locally. Historical error ranges in the Python README are therefore not
treated as a comparable benchmark. The Python protocols below were introduced later.
"""

REASSESSMENT = """
The baseline relation $P=VI$ makes the task definition consequential. Contemporaneous electrical
inputs strongly constrain measured power. That is **feature–target dependence**, not by itself
proof that test observations contaminated training. It becomes potential target leakage if
those inputs would be unavailable for the stated deployment objective [[4]](#ref-4).
**Preprocessing leakage** instead occurs when held-out data influence fitted transformations;
the Python fold-level scalers address that separate issue. **Observation overlap** is another
problem: identical records under different source identifiers can cross nominal group boundaries.

Track A audits electrical-input estimation. Track B evaluates estimation without current and
voltage, but still uses contemporaneous environmental measurements. The error levels of the two
tracks cannot be interpreted as a fair contest between models with the same information budget.
Neither experiment defines an advance forecasting horizon.
"""

PREPROCESSING = """
`src/load_data.py` normalises headers, concatenates daily CSVs and adds source filenames.
It parses time using `%H:%M`, constructs decimal hour as hour + minute/60, and adds
calculated power and measured-minus-calculated power diagnostics. The diagnostic columns,
source identifier and placeholder date are excluded from model inputs (`src/features.py`).
No outlier-removal or imputation procedure is implemented in this preparation path; extreme
values are retained. Missing-value reporting is diagnostic rather than a missing-data model.

The scikit-learn MLP pipeline fits `StandardScaler` on each training partition; targets remain
in watts. The matched-run linear pipeline also standardises inputs. TensorFlow/PyTorch fit
separate input and target standardisers on training observations only, transform validation
inputs using those fitted values, and inverse-transform outputs to watts. Inner models and
outer refits obtain fresh scalers. The historical `src/baselines.py` also contains an in-sample
linear check; manuscript scores instead use the saved grouped-CV outputs.
"""

VALIDATION = """
Seven outer `GroupKFold` partitions hold out three source-file groups (27 rows) and fit on
18 groups (162 rows). Each row receives one outer prediction per model and seed. Three
inner grouped folds divide those 18 groups into 12 fitting and six validation groups.
The tuned scikit-learn MLP selects hidden size and alpha by mean inner MAE; its untuned
reference has fixed settings. Linear regression has no nested search; the physics product
has no fitted parameters.

TensorFlow/PyTorch select architecture by mean inner MAE in watts. Within each inner fit,
validation MAE controls early stopping; best weights are restored. The median best epoch
of the selected architecture's inner folds determines the outer training-only refit duration.
This reuses inner validation for stopping and selection, which can favour optimistic inner
scores. Outer test rows do not choose the architecture or epochs.

Seeds 0, 1 and 2 vary ANN initialisation. Scores pool all outer predictions per seed before
computing a mean and population standard deviation across seeds. Deterministic baselines
repeat the same values across these records. These are not confidence intervals or
independent dataset replications. The nominal day grouping reduces same-file overlap,
but the duplicate-source issue prevents a claim of complete observational independence.

S1 and S2 were compared using the README-documented selection rule and inner evidence.
The inspected evidence does not independently establish when the rule was decided. S2 was selected.
Historical outer scores had already been inspected,
so this remains a development study requiring independent confirmation. Matching framework
settings does not make random initialisations or floating-point operations identical.
"""

LIMITATIONS = """
**Sample size and effective independence.** Only 189 rows and 21 source-file identifiers
are available. Nine measurement records repeat under two different identifiers. File-based
grouping keeps each filename intact, but the saved predictions assign
`solar_data_tue_21st.csv` to outer fold 1 and `solar_data_tue_28st.csv` to outer fold 7.
Identical measurement copies therefore cross training/test boundaries when either fold is
held out. This confirms overlap; its effect on reported error remains unmeasured.
This report preserves the published outputs rather
than silently deduplicating data or claiming corrected performance.

**Temporal and geographic coverage.** Collection is attributed to one university installation
in the project documentation. The processed timestamps contain a placeholder date, and source
names do not establish a full calendar period. Observations are approximately hourly, with
irregular intervals. Cross-season, multi-site and changed-installation performance are untested.

**Measurement uncertainty.** The original project names measurement instruments, but calibration
records and quantified measurement uncertainty are not available in this repository. Differences
between measured power and $VI$ are observed, but cannot be assigned to a particular instrument
or physical mechanism from these files alone.

**Deployment information.** Track A requires contemporaneous electrical measurements. Track B
does not, but still requires measured environmental conditions at the target time. Neither has
been validated using advance weather forecasts, independently collected external data, or a
production operational workflow. The interactive app demonstrates inference, not deployment validity.

**Selection and variability.** Three seeds provide a limited view of initialisation sensitivity.
They do not quantify uncertainty over sites or seasons, and the same small dataset has supported
multiple rounds of development. Historical cap hits and untuned-network instability caution against
equating additional complexity or a completed optimisation run with reliable generalisation.
No confidence intervals, significance tests or calibrated prediction intervals are supplied.
"""

CONCLUSION = """
The principal contribution is a methodological re-evaluation: it separates electrical-input
estimation from environmental/time-based estimation, supplies physics and linear baselines,
and records grouped, nested evaluation across three neural-network implementations. The
saved results do not justify a general claim that ANNs are superior to linear regression,
nor that this system forecasts future solar generation.

**Proposed work, not completed experiments.** First resolve the duplicate-file provenance
and predefine an evaluation that keeps identical observations together. Re-evaluation would
be a separate scientific change with newly versioned outputs. Collect longer records across
seasons and installations, preserve full timestamps and instrument metadata, and evaluate
external datasets. Define one-hour-ahead or day-ahead tasks using only inputs available before
the target time. Compare justified temporal and tree-based baselines; assess uncertainty
calibration and seasonal robustness. Physics-informed learning is a possible extension only
where the available measurements and physical model warrant it.
"""


def abstract(evidence):
    a = evidence["audit"]
    return f"""Solar power models require evaluation that reflects the information available at prediction time.
This study re-examines an undergraduate MATLAB artificial-neural-network investigation through
a Python comparison of feature configurations, baselines and validation procedures. The available
dataset contains {a['rows']} observations assigned to {a['groups']} source-file groups, nominally measurement days.
Track A includes irradiance, current, voltage, temperature and humidity; Track B excludes electrical
measurements and includes observation hour. Linear regression, a Track A electrical-product baseline,
and scikit-learn, TensorFlow/Keras and PyTorch networks are assessed using seven outer grouped folds,
nested selection where applicable, and three ANN seeds. Metrics pool held-out predictions per seed.
In matched S2 experiments, Track A linear regression attains mean MAE {score(evidence, 'A', 'Linear'):.3f} W,
compared with {score(evidence, 'A', 'TensorFlow selected'):.3f} W for TensorFlow and {score(evidence, 'A', 'PyTorch selected'):.3f} W for PyTorch.
Track B tuned scikit-learn MAE is {score(evidence, 'B', 'Sklearn tuned'):.3f} W versus {score(evidence, 'B', 'Linear'):.3f} W for linear regression;
advantages are small and metric dependent. Electrical-input accuracy consequently does not establish
weather-only forecasting, and network complexity does not consistently improve estimation. An artifact
audit identifies {a['duplicate_measurements']} repeated measurement records under different filenames, limiting the
independence implied by file-based grouping. The study contributes transparent task definition and
reproducible comparisons, while small sample size, uncertain temporal coverage, duplicated records
and absent external validation constrain generalisation. Neither track establishes future-horizon
forecasting capability."""


def discussion(evidence):
    return f"""**RQ1 — information changes the task.** The mean absolute difference between measured power
and the electrical product is {evidence['audit']['power_iv_mae']:.3f} W. Track A can exploit this near-identity,
whereas Track B cannot. Lower Track A errors reflect richer contemporaneous information; they are
not evidence of superior weather-only forecasting.

**RQ2 — complexity does not guarantee improvement.** Linear regression has the lowest matched Track A
mean MAE ({score(evidence, 'A', 'Linear'):.3f} W). Track B's tuned scikit-learn ANN improves mean MAE by
{score(evidence, 'B', 'Linear') - score(evidence, 'B', 'Sklearn tuned'):.3f} W over linear regression, but this small
descriptive difference has no supplied significance test. PyTorch's Track B mean RMSE
({score(evidence, 'B', 'PyTorch selected', 'RMSE'):.3f} W) exceeds linear regression's
({score(evidence, 'B', 'Linear', 'RMSE'):.3f} W), illustrating metric-dependent rankings.

**RQ3 — held-out groups, with a qualification.** The Python protocol excludes each test filename from
fitting and nests model selection, improving the experimental design relative to the documented
historical reuse and observation-level splits. There is no controlled experiment here isolating the
effect of changing split strategy. The repeated files occupy outer folds 1 and 7, so identical
measurement copies cross training/test boundaries when either fold is held out; the error effect
remains unmeasured. These scores
cannot be described as wholly independent-day validation. External-site or chronological forecasting
performance cannot be inferred from them.

**RQ4 — implementation comparison, not a universal framework ranking.** TensorFlow and PyTorch use
matching recorded settings, reference data and outer assignments. Their error and seed variation
describe this protocol; non-identical initialisations, updates and early stopping can still yield
different fits. The evidence does not establish a statistically supported framework winner or make
software framework choice a substitute for task definition and baseline evaluation."""
