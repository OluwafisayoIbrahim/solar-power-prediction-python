"""PyTorch ANN comparison with the same grouped protocol as TensorFlow."""

from copy import deepcopy

import numpy as np
from sklearn.preprocessing import StandardScaler

from train_ann_tf import parse_args as parse_shared_args, run_comparison, select_ann


def parse_args(argv=None):
    return parse_shared_args(argv, default_epochs=1000, description=__doc__)


def fit_torch(X, y, hidden, seed, args, validation=None, epochs=None, progress=None):
    """Fit scalers on training data and restore the best validation checkpoint."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from models_torch import TorchANN, l2_penalty

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x = torch.from_numpy(x_scaler.fit_transform(X).astype("float32"))
    target = torch.from_numpy(y_scaler.fit_transform(y.reshape(-1, 1)).astype("float32"))
    model = TorchANN(X.shape[1], hidden, args.activation, args.learning_rate, args.l2, seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x, target), batch_size=args.batch_size,
                        shuffle=True, drop_last=False, generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 eps=1e-7, weight_decay=0)
    mse = nn.MSELoss()
    history = {"loss": [], "mae": []}
    if validation is not None:
        x_val = torch.from_numpy(x_scaler.transform(validation[0]).astype("float32"))
        y_val = torch.from_numpy(y_scaler.transform(validation[1].reshape(-1, 1)).astype("float32"))
        history.update(val_loss=[], val_mae=[])
    best_mae, best_state, wait = float("inf"), None, 0
    epoch_limit = epochs if epochs is not None else args.epochs
    for epoch in range(epoch_limit):
        model.train()
        total_loss, total_mae = 0.0, 0.0
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = mse(predictions, targets) + l2_penalty(model, args.l2)
            if not torch.isfinite(loss):
                raise ValueError("Training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(inputs)
            total_mae += (predictions.detach() - targets).abs().sum().item()
        history["loss"].append(total_loss / len(x))
        history["mae"].append(total_mae / len(x))
        if progress is not None:
            progress((epoch + 1) / epoch_limit)
        if validation is not None:
            model.eval()
            with torch.no_grad():
                predictions = model(x_val)
                val_mae = (predictions - y_val).abs().mean().item()
                val_loss = (mse(predictions, y_val) + l2_penalty(model, args.l2)).item()
            if not np.isfinite([val_mae, val_loss]).all():
                raise ValueError("Validation produced non-finite metrics")
            history["val_mae"].append(val_mae)
            history["val_loss"].append(val_loss)
            if val_mae < best_mae:
                best_mae, wait = val_mae, 0
                best_state = deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= args.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, x_scaler, y_scaler, history


def predict_torch(fitted, X):
    import torch
    model, x_scaler, y_scaler, _ = fitted
    model.eval()
    with torch.no_grad():
        scaled = model(torch.from_numpy(x_scaler.transform(X).astype("float32"))).numpy()
    predictions = y_scaler.inverse_transform(scaled).reshape(-1)
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    return predictions


def select_torch(X, y, groups, seed, args, curve_dir, prefix):
    return select_ann(X, y, groups, seed, args, curve_dir, prefix, fit_torch, predict_torch)


def main():
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise SystemExit("Install dependencies: python -m pip install -r requirements.txt") from exc
    run_comparison(args, "pytorch", torch.__version__, "PyTorch selected",
                   fit_torch, predict_torch, select_torch)


if __name__ == "__main__":
    main()
