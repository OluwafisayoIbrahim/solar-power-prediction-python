"""Reproducible manuscript charts from checked measurements and saved OOF outputs."""
import altair as alt
import pandas as pd


WORKFLOW = '''digraph {
  graph [rankdir=TB, bgcolor="transparent"];
  node [shape=box, style=rounded, fontname="Arial"];
  "Recorded daily CSVs" -> "Header normalisation and decimal hour";
  "Header normalisation and decimal hour" -> "Track A: electrical + environment";
  "Header normalisation and decimal hour" -> "Track B: environment + hour";
  "Track A: electrical + environment" -> "Grouped outer evaluation";
  "Track B: environment + hour" -> "Grouped outer evaluation";
  "Grouped outer evaluation" -> "Inner selection on training groups only";
  "Inner selection on training groups only" -> "Outer refit and held-out predictions";
  "Outer refit and held-out predictions" -> "Pooled per-seed metrics and comparison";
}'''

VALIDATION = '''digraph {
  graph [rankdir=TB, bgcolor="transparent"];
  node [shape=box, style=rounded, fontname="Arial"];
  "21 source groups / 189 rows" -> "Outer training: 18 groups / 162 rows";
  "21 source groups / 189 rows" -> "Outer test: 3 groups / 27 rows";
  "Outer training: 18 groups / 162 rows" -> "Inner fit: 12 groups / 108 rows";
  "Outer training: 18 groups / 162 rows" -> "Inner validation: 6 groups / 54 rows";
  "Inner fit: 12 groups / 108 rows" -> "Select settings across 3 inner folds";
  "Inner validation: 6 groups / 54 rows" -> "Select settings across 3 inner folds";
  "Select settings across 3 inner folds" -> "Refit on all 18 outer training groups";
  "Refit on all 18 outer training groups" -> "Score saved outer predictions";
  "Outer test: 3 groups / 27 rows" -> "Score saved outer predictions";
}'''


def power_distribution(frame):
    return alt.Chart(frame).mark_bar().encode(
        x=alt.X("power:Q", bin=alt.Bin(maxbins=20), title="Measured power (W)"),
        y=alt.Y("count():Q", title="Observations (count)"),
        tooltip=[alt.Tooltip("count():Q", title="Observations")]).properties(height=220)


def comparison_chart(table, metric="MAE"):
    unit = "%" if metric == "MAPE" else "W"
    charts = []
    for track in ("A", "B"):
        rows = table[table.track == track].copy()
        rows["low"] = rows[f"{metric}_mean"] - rows[f"{metric}_std"]
        rows["high"] = rows[f"{metric}_mean"] + rows[f"{metric}_std"]
        base = alt.Chart(rows).encode(y=alt.Y("model:N", title="Model", sort="ascending"))
        bars = base.mark_bar(opacity=0.7).encode(
            x=alt.X(f"{metric}_mean:Q", title=f"Mean {metric} ({unit})", scale=alt.Scale(zero=True)),
            tooltip=["model", alt.Tooltip(f"{metric}_mean:Q", format=".3f"), alt.Tooltip(f"{metric}_std:Q", format=".3f")])
        whiskers = base.mark_rule().encode(x="low:Q", x2="high:Q")
        charts.append((bars + whiskers).properties(title=f"Track {track} — separate information setting", height=180))
    return alt.vconcat(*charts).resolve_scale(x="independent")


def parity_chart(predictions):
    limits = [float(predictions[["measured", "predicted"]].min().min()),
              float(predictions[["measured", "predicted"]].max().max())]
    points = alt.Chart(predictions).mark_circle(size=25, opacity=0.5).encode(
        x=alt.X("measured:Q", title="Measured power (W)", scale=alt.Scale(domain=limits, zero=False)),
        y=alt.Y("predicted:Q", title="Predicted power (W)", scale=alt.Scale(domain=limits, zero=False)),
        color=alt.Color("model:N", title="Model"), tooltip=["model", "day_id", "measured", "predicted"])
    diagonal = alt.Chart(pd.DataFrame({"measured": limits, "predicted": limits})).mark_line(
        color="#666666", strokeDash=[5, 4]).encode(x="measured:Q", y="predicted:Q")
    return (points + diagonal).properties(height=330)


def residual_chart(predictions):
    return alt.Chart(predictions).mark_bar(opacity=0.65).encode(
        x=alt.X("residual:Q", bin=alt.Bin(maxbins=25), title="Measured − predicted power (W)"),
        y=alt.Y("count():Q", title="Observations (count)"),
        color=alt.Color("model:N", title="Model", legend=None), tooltip=["model", "count()"]
    ).properties(height=100).facet(row=alt.Row("model:N", title="Model")).resolve_scale(y="independent")


def daily_errors(predictions):
    return predictions.assign(absolute_error=predictions.residual.abs()).groupby(
        ["day_id", "model"], as_index=False).absolute_error.mean().rename(columns={"absolute_error": "MAE (W)"})


def daily_chart(predictions):
    daily = daily_errors(predictions)
    return alt.Chart(daily).mark_point(filled=True, size=40).encode(
        x=alt.X("MAE (W):Q", title="Within-group MAE (W)", scale=alt.Scale(zero=True)),
        y=alt.Y("day_id:N", title="Source-file day group (not calendar order)"),
        color=alt.Color("model:N", title="Model"), tooltip=["day_id", "model", alt.Tooltip("MAE (W):Q", format=".3f")]
    ).properties(height=430)
