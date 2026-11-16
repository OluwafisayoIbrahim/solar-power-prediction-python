# Portfolio plots

PNG images use twice the chart resolution; SVG files scale without losing clarity. Copy either format into your website's assets folder.

Start with `model_comparison_mae.png`, `track_b_measured_vs_predicted.png`, and `track_b_daily_mae.png`. Every plot uses the same completed 180-row, 20-day evidence as the dashboard. Comparison bars show three-seed means and population standard deviations; diagnostic plots use seed 0. Track A includes electrical measurements; Track B uses environmental measurements and hour. Both estimate contemporaneous power without a future forecast horizon.

`manifest.json` describes all ten plots; `comparison_metrics.csv` supplies numerical summaries. Regenerate with `python src/plots.py` after installing `vl-convert-python`.
