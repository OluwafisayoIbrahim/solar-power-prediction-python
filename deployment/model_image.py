"""Render the loaded Dense network as a downloadable PNG."""

from io import BytesIO

import numpy as np
from matplotlib.figure import Figure


def architecture_image(model):
    layers = [(int(model.input_shape[-1]), "Inputs", "Standardized measurements")]
    dense = [layer for layer in model.layers if hasattr(layer, "units")]
    for index, layer in enumerate(dense):
        layers.append((layer.units, "Output" if index == len(dense) - 1 else f"Hidden {index + 1}",
                       layer.activation.__name__))
    return render_network(layers, "Raw inputs -> input scaler -> network -> target inverse scaler -> power (W)")


def predictor_image(loaded):
    if loaded.spec.kind == "tensorflow":
        return architecture_image(loaded.estimator[0])
    if loaded.spec.kind == "pytorch":
        model = loaded.estimator[0]
        layers = [(model.hidden[0].in_features, "Inputs", "Standardized measurements")]
        for index, layer in enumerate(model.hidden, 1):
            layers.append((layer.out_features, f"Hidden {index}", loaded.metadata["activation"]))
        layers.append((model.output.out_features, "Output", "linear"))
        return render_network(layers, "Raw inputs -> input scaler -> network -> target inverse scaler -> power (W)")
    if loaded.spec.kind == "physics":
        return render_network([(2, "Inputs", "Current (A), voltage (V)"),
                               (1, "Multiplication", "P = I x V")],
                              "Physics baseline: no training, scaling or learned weights. Output: watts")
    estimator = loaded.estimator.steps[-1][1]
    if loaded.spec.kind == "linear":
        layers = [(len(loaded.spec.features), "Inputs", "Standardized measurements"),
                  (1, "Linear regression", "Weighted sum + intercept")]
    else:
        sizes = [estimator.coefs_[0].shape[0]] + [weights.shape[1] for weights in estimator.coefs_]
        layers = [(sizes[0], "Inputs", "Standardized measurements")]
        for index, size in enumerate(sizes[1:]):
            last = index == len(sizes) - 2
            layers.append((size, "Output" if last else f"Hidden {index + 1}",
                           estimator.out_activation_ if last else estimator.activation))
    return render_network(layers, "Raw inputs -> saved StandardScaler -> fitted estimator -> power (W)")


def render_network(layers, footer):
    figure = Figure(figsize=(10, 5), facecolor="white")
    ax = figure.subplots()
    ax.set(xlim=(-0.5, len(layers) - 0.5), ylim=(-0.22, 1.24))
    ax.axis("off")
    colors = ["#087f8c", "#426b46", "#426b46", "#b44952"]
    previous = None
    for column, (units, name, activation) in enumerate(layers):
        positions = np.linspace(0.05, 0.95, units) if units > 1 else [0.5]
        if previous is not None:
            for source in previous:
                for target in positions:
                    ax.plot([column - 1, column], [source, target], color="#a8b4b9",
                            alpha=0.25, linewidth=0.6, zorder=1)
        ax.scatter([column] * units, positions, s=65, color=colors[column % len(colors)], zorder=2)
        ax.text(column, 1.16, f"{name}\n{units} {'unit' if units == 1 else 'units'}",
                ha="center", va="center", fontsize=11, color="#202b30")
        ax.text(column, -0.08, activation, ha="center", fontsize=9, color="#46535a")
        previous = positions
    ax.text((len(layers) - 1) / 2, -0.2,
            footer,
            ha="center", fontsize=9, color="#46535a")
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=180)
    return output.getvalue()
