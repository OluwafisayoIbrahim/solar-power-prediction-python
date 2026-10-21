# Framework comparison

Protocol and reference data checks match; numerical differences remain possible.

Mean +/- population standard deviation over seeds; not confidence intervals.

| Track | Model | Source | MAE (W) | RMSE (W) | MAPE (%) | R2 |
|---|---|---|---|---|---|---|
| A | Linear | Both | 0.992 +/- 0.000 | 1.245 +/- 0.000 | 0.382 +/- 0.000 | 0.998 +/- 0.000 |
| A | Physics | Both | 1.168 +/- 0.000 | 1.321 +/- 0.000 | 0.445 +/- 0.000 | 0.998 +/- 0.000 |
| A | PyTorch selected | PyTorch | 1.575 +/- 0.071 | 2.135 +/- 0.029 | 0.606 +/- 0.025 | 0.994 +/- 0.000 |
| A | Sklearn tuned | Both | 1.141 +/- 0.112 | 1.574 +/- 0.276 | 0.442 +/- 0.044 | 0.997 +/- 0.001 |
| A | Sklearn untuned | Both | 1.319 +/- 0.143 | 3.814 +/- 1.590 | 0.518 +/- 0.063 | 0.978 +/- 0.018 |
| A | TensorFlow selected | TensorFlow | 1.545 +/- 0.162 | 2.108 +/- 0.263 | 0.594 +/- 0.062 | 0.994 +/- 0.001 |
| B | Linear | Both | 12.165 +/- 0.000 | 15.149 +/- 0.000 | 4.854 +/- 0.000 | 0.701 +/- 0.000 |
| B | PyTorch selected | PyTorch | 11.981 +/- 0.132 | 15.310 +/- 0.163 | 4.809 +/- 0.040 | 0.694 +/- 0.007 |
| B | Sklearn tuned | Both | 11.965 +/- 0.172 | 15.089 +/- 0.280 | 4.786 +/- 0.072 | 0.703 +/- 0.011 |
| B | Sklearn untuned | Both | 14.302 +/- 0.773 | 23.980 +/- 6.756 | 5.698 +/- 0.239 | 0.191 +/- 0.463 |
| B | TensorFlow selected | TensorFlow | 12.131 +/- 0.379 | 15.283 +/- 0.198 | 4.854 +/- 0.122 | 0.695 +/- 0.008 |

## Selected epochs

- TensorFlow selected, Track A: 0/21 at cap 1000.
- TensorFlow selected, Track B: 0/21 at cap 1000.
- PyTorch selected, Track A: 0/21 at cap 1000.
- PyTorch selected, Track B: 0/21 at cap 1000.

TensorFlow source: `results\tensorflow\convergence_s2`
PyTorch source: `results\pytorch\parity_s2`
