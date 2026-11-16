# Main research results

`research_20_days/` is the results collection used throughout the dashboard.
It evaluates 180 observations across 20 daily files.

| Artifact | Path |
|---|---|
| Main comparison | `research_20_days/comparison/comparison.csv` |
| TensorFlow S1 | `research_20_days/tensorflow_s1/` |
| Selected TensorFlow S2 | `research_20_days/tensorflow_s2/` |
| Selected PyTorch | `research_20_days/pytorch_selected/` |
| Final inference models | `research_20_days/exports/{tensorflow,pytorch,sklearn}/` |
| Selection evidence | `research_20_days/selection.json` |
| Completed collection | `research_20_days/completed.json` |

Each run contains its configuration, held-out predictions, per-seed metrics,
selected settings and inner-fold learning curves. The main comparison reports
pooled per-seed metrics followed by means and population standard deviations.

The former 21-day result directories are removed. Earlier Git commits retain
their history. Local smoke checks and partial attempts are not dashboard evidence.
