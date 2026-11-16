"""Export PNG and SVG portfolio charts from the dashboard's verified evidence."""
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "deployment"), str(ROOT / "src")]
from manuscript_data import load_evidence, diagnostic_predictions
import manuscript_figures as figures


def main():
    import vl_convert as vlc
    evidence = load_evidence()
    output = ROOT / "results/plots"
    output.mkdir(parents=True, exist_ok=True)
    charts = {"power_distribution": (figures.power_distribution(evidence["dataset"]),
                                     "Measured power across all 180 observations.")}
    for metric in ["MAE", "RMSE", "MAPE"]:
        charts[f"model_comparison_{metric.lower()}"] = (
            figures.comparison_chart(evidence["comparison"], metric),
            f"Mean {metric} with one population seed standard deviation. Track panels use independent scales; variation is not a confidence interval.")
    for track in ["A", "B"]:
        predictions = diagnostic_predictions(evidence, track)
        for kind, factory, description in [
            ("measured_vs_predicted", figures.parity_chart, "Dashed line denotes equality."),
            ("residuals", figures.residual_chart, "Residual is measured minus predicted power."),
            ("daily_mae", figures.daily_chart, "Filename groups are categorical, not chronological.")]:
            charts[f"track_{track.lower()}_{kind}"] = (
                factory(predictions), f"Track {track}, seed 0, all 180 held-out rows per model. {description}")
    inventory = []
    for name, (chart, description) in charts.items():
        spec = chart.configure(background="white", font="Arial").configure_axis(
            labelFontSize=13, titleFontSize=14).configure_legend(
            labelFontSize=13, titleFontSize=14).to_dict()
        if "vconcat" in spec:
            for panel in spec["vconcat"]:
                panel["width"] = 800
        elif "facet" in spec:
            spec["spec"]["width"] = 800
        else:
            spec["width"] = 800
        (output / f"{name}.png").write_bytes(vlc.vegalite_to_png(spec, scale=2))
        (output / f"{name}.svg").write_text(vlc.vegalite_to_svg(spec), encoding="utf-8")
        inventory.append({"name": name, "description": description,
                          "png": name + ".png", "svg": name + ".svg"})
        print(f"Exported {name}", flush=True)
    columns = ["track", "model"] + [f"{metric}_{stat}" for metric in ["MAE", "RMSE", "MAPE", "R2"]
                                     for stat in ["mean", "std"]]
    evidence["comparison"][columns].to_csv(output / "comparison_metrics.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "rows": 180, "daily_files": 20, "source_comparison": "results/comparison/comparison.csv",
        "source_runs": json.loads((ROOT / "results/completed.json").read_text())["runs"],
        "plots": inventory}, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Portfolio plots\n\nPNG images use twice the chart resolution; SVG files scale without losing clarity. "
        "Copy either format into your website's assets folder.\n\n"
        "Start with `model_comparison_mae.png`, `track_b_measured_vs_predicted.png`, and `track_b_daily_mae.png`. "
        "Every plot uses the same completed 180-row, 20-day evidence as the dashboard. "
        "Comparison bars show three-seed means and population standard deviations; diagnostic plots use seed 0. "
        "Track A includes electrical measurements; Track B uses environmental measurements and hour. "
        "Both estimate contemporaneous power without a future forecast horizon.\n\n"
        "`manifest.json` describes all ten plots; `comparison_metrics.csv` supplies numerical summaries. "
        "Regenerate with `python src/plots.py` after installing `vl-convert-python`.\n", encoding="utf-8")
    with zipfile.ZipFile(ROOT / "results/portfolio_plots.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, "plots/" + path.name)
    print("Ready: results/plots and results/portfolio_plots.zip", flush=True)


if __name__ == "__main__":
    main()
