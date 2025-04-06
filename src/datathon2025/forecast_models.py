from sklearn.linear_model import LinearRegression
import pandas as pd


def make_weight(datetime, a=100, b=10):
    weights = pd.DataFrame({"DATETIME": datetime, "WEIGHT": 1})

    weights.loc[
        (weights["DATETIME"] >= "2022-12-15")
        & (weights["DATETIME"] < "2023-01-16"),
        "WEIGHT",
    ] = a
    weights.loc[
        (weights["DATETIME"] >= "2022-08-01")
        & (weights["DATETIME"] < "2022-09-01"),
        "WEIGHT",
    ] = a
    weights.loc[
        (weights["DATETIME"] >= "2023-12-15")
        & (weights["DATETIME"] < "2024-01-16"),
        "WEIGHT",
    ] = a
    weights.loc[
        (weights["DATETIME"] >= "2023-08-01")
        & (weights["DATETIME"] < "2023-09-01"),
        "WEIGHT",
    ] = a
    weights.loc[
        (weights["DATETIME"] >= "2022-05-01")
        & (weights["DATETIME"] < "2022-08-01"),
        "WEIGHT",
    ] = b
    weights.loc[
        (weights["DATETIME"] >= "2023-05-01")
        & (weights["DATETIME"] < "2023-08-01"),
        "WEIGHT",
    ] = b
    weights.loc[
        (weights["DATETIME"] >= "2024-05-01")
        & (weights["DATETIME"] < "2024-08-01"),
        "WEIGHT",
    ] = b

    return weights


class SimpleModel:
    """
    This is a simple example of a model structure

    """

    def __init__(self):
        self.linear_regression = LinearRegression()

    def train(self, x, y, weight=None):
        self.linear_regression.fit(x, y, sample_weight=weight)

    def predict(self, x):
        return self.linear_regression.predict(x)
