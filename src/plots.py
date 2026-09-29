from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "model_comparison.csv"
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
FIG_DIR = PROJECT_ROOT / "results" / "figures"


def fig1_model_comparison_bars(metrics: pd.DataFrame):
    """MAPE by model, split by track — shows Track A vs B at a glance."""
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Physics baseline", "Linear Regression", "ANN Untuned", "ANN Tuned"]
    width = 0.35
    x = np.arange(len(order))

    for offset, track, color in ((-width / 2, "A", "#4c72b0"), (width / 2, "B", "#dd8452")):
        vals = []
        for model in order:
            row = metrics[(metrics.track == track) & (metrics.model == model)]
            vals.append(row.mape_percent.iloc[0] if len(row) else np.nan)
        ax.bar(x + offset, vals, width, label=f"Track {track}", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Model comparison: MAPE by track\n(Track A includes current & voltage; Track B is weather-only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison_bars.png", dpi=150)
    plt.close(fig)


def fig2_predicted_vs_measured(track: str, model_file: str, title: str):
    df = pd.read_csv(PRED_DIR / f"{model_file}.csv")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df.measured, df.predicted, alpha=0.6, s=20)
    lims = [df.measured.min() - 5, df.measured.max() + 5]
    ax.plot(lims, lims, "--", color="grey", label="Perfect prediction")
    ax.set_xlabel("Measured power (W)")
    ax.set_ylabel("Predicted power (W)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"pred_vs_measured_{model_file}.png", dpi=150)
    plt.close(fig)


def fig3_daily_profiles(model_file: str, title: str, n_days: int = 4):
    df = pd.read_csv(PRED_DIR / f"{model_file}.csv")
    days = sorted(df.day_id.unique())
    chosen = [days[i] for i in np.linspace(0, len(days) - 1, n_days).astype(int)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, day in zip(axes.ravel(), chosen):
        d = df[df.day_id == day].sort_values("hour")
        ax.plot(d.hour, d.measured, "o-", label="Measured")
        ax.plot(d.hour, d.predicted, "s--", label="Predicted")
        ax.set_title(day.replace("solar_data_", "").replace(".csv", ""))
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    for ax in axes[1]:
        ax.set_xlabel("Hour of day")
    for ax in axes[:, 0]:
        ax.set_ylabel("Power (W)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"daily_profiles_{model_file}.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_PATH)

    fig1_model_comparison_bars(metrics)
    print("saved model_comparison_bars.png")

    fig2_predicted_vs_measured("A", "track_a_ann_tuned", "Track A (current & voltage as inputs): tuned ANN")
    fig2_predicted_vs_measured("B", "track_b_ann_tuned", "Track B (weather only): tuned ANN")
    print("saved pred_vs_measured plots")

    fig3_daily_profiles("track_b_ann_tuned", "Track B (weather only), tuned ANN: measured vs predicted")
    print("saved daily_profiles plot")

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()