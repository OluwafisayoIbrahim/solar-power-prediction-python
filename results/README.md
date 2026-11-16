# Main research results

This folder contains the results collection used throughout the dashboard.
It evaluates 180 observations across 20 daily files.

| Artifact | Path |
|---|---|
| Main comparison | `comparison/comparison.csv` |
| TensorFlow S1 | `tensorflow_s1/` |
| Selected TensorFlow S2 | `tensorflow_s2/` |
| Selected PyTorch | `pytorch_selected/` |
| Final inference models | `exports/{tensorflow,pytorch,sklearn}/` |
| Selection evidence | `selection.json` |
| Completed collection | `completed.json` |
| Portfolio plots | `plots/` (PNG and SVG) |

Each run contains its configuration, held-out predictions, per-seed metrics,
selected settings and inner-fold learning curves. The main comparison reports
pooled per-seed metrics followed by means and population standard deviations.

The former 21-day result directories are removed. Earlier Git commits retain
their history. Local smoke checks and partial attempts are not dashboard evidence.
