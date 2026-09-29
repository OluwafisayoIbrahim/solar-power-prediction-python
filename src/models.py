from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_ann_model(
    hidden_layer_size: int = 10,
    alpha: float = 0.0001,
    random_state: int = 42,
) -> Pipeline:
    """
    Create the Python ANN approximation of the original MATLAB model.

    MATLAB:
        fitnet(10)
        tansig hidden activation
        purelin output activation
        trainlm optimizer

    Python:
        MLPRegressor
        10 hidden neurons
        tanh activation
        linear regression output
        L-BFGS optimizer
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ann",
                MLPRegressor(
                    hidden_layer_sizes=(hidden_layer_size,),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=alpha,
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )