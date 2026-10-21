"""Small CPU MLP with the same layers and regularization as models_tf.py."""

import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    """Repeat a fit on this environment, including DataLoader shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


class TorchANN(nn.Module):
    """nn.Module registers layers so their parameters can receive gradients."""

    def __init__(self, n_features, hidden_layers=(10,), activation="tanh",
                 learning_rate=0.001, l2=0.0001, seed=42):
        super().__init__()
        if n_features < 1 or not hidden_layers or any(n < 1 for n in hidden_layers):
            raise ValueError("Feature and hidden-layer sizes must be positive")
        activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        seed_everything(seed)
        self.learning_rate, self.l2 = learning_rate, l2
        sizes = (n_features, *hidden_layers)
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.activation = activations[activation]()
        self.output = nn.Linear(sizes[-1], 1)
        for layer in [*self.hidden, self.output]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.hidden:
            x = self.activation(layer(x))
        return self.output(x)


def l2_penalty(model, l2):
    """Keras regularizes hidden kernels only; its output Dense has no L2."""
    return l2 * sum(layer.weight.square().sum() for layer in model.hidden)
