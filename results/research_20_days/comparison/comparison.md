# Framework comparison

Protocol and reference data checks match; numerical differences remain possible.

Mean +/- population standard deviation over seeds; not confidence intervals.

| Track | Model | Source | MAE (W) | RMSE (W) | MAPE (%) | R2 |
|---|---|---|---|---|---|---|
| A | Linear | Both | 0.978 +/- 0.000 | 1.235 +/- 0.000 | 0.377 +/- 0.000 | 0.998 +/- 0.000 |
| A | Physics | Both | 1.115 +/- 0.000 | 1.254 +/- 0.000 | 0.424 +/- 0.000 | 0.998 +/- 0.000 |
| A | PyTorch selected | PyTorch | 1.624 +/- 0.148 | 2.405 +/- 0.145 | 0.622 +/- 0.062 | 0.992 +/- 0.001 |
| A | Sklearn tuned | Both | 1.069 +/- 0.060 | 1.393 +/- 0.089 | 0.413 +/- 0.026 | 0.997 +/- 0.000 |
| A | Sklearn untuned | Both | 1.304 +/- 0.538 | 3.580 +/- 3.118 | 0.505 +/- 0.214 | 0.970 +/- 0.039 |
| A | TensorFlow selected | TensorFlow | 1.334 +/- 0.159 | 1.863 +/- 0.297 | 0.515 +/- 0.059 | 0.995 +/- 0.002 |
| B | Linear | Both | 11.835 +/- 0.000 | 14.928 +/- 0.000 | 4.724 +/- 0.000 | 0.702 +/- 0.000 |
| B | PyTorch selected | PyTorch | 12.032 +/- 0.262 | 15.027 +/- 0.246 | 4.806 +/- 0.097 | 0.698 +/- 0.010 |
| B | Sklearn tuned | Both | 12.179 +/- 0.037 | 15.633 +/- 0.278 | 4.865 +/- 0.012 | 0.673 +/- 0.012 |
| B | Sklearn untuned | Both | 14.044 +/- 0.267 | 18.890 +/- 1.219 | 5.584 +/- 0.136 | 0.521 +/- 0.063 |
| B | TensorFlow selected | TensorFlow | 11.951 +/- 0.317 | 14.989 +/- 0.372 | 4.761 +/- 0.107 | 0.700 +/- 0.015 |

## Selected epochs

- TensorFlow selected, Track A: 0/21 at cap 1000.
- TensorFlow selected, Track B: 0/21 at cap 1000.
- PyTorch selected, Track A: 0/21 at cap 1000.
- PyTorch selected, Track B: 0/21 at cap 1000.

TensorFlow source: `C:\Users\user\Documents\fisayo\solar-power-prediction-python\results\research_20_days\tensorflow_s2`
PyTorch source: `C:\Users\user\Documents\fisayo\solar-power-prediction-python\results\research_20_days\pytorch_selected`
