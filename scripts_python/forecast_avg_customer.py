from pathlib import Path

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from datathon2025.data import (
    DataLoader,
    preprocess_target_data,
    preprocess_rollout,
)
from datathon2025.forecast_models import SimpleModel, make_weight

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def main(zone: str):
    """Train and evaluate the models for IT and ES"""

    # Inputs
    input_path = Path("../data")
    output_path = Path("../outputs")
    output_path.mkdir(exist_ok=True)

    team_name = "Placeholder"

    TRAIN_START = None  # "2023-05-01 00:00:00"
    FEATURES = ["rollout", "is_off_day", "hour", "spv", "day_of_week", "temp"]

    #   Step 1: Predict Portfolio
    # ----------------------------

    # Load Datasets
    loader = DataLoader(input_path)
    target, features, rollout, example_results = loader.load_data(zone)

    n_customer = len(target.columns)

    # Add rollout to features
    features["rollout"] = preprocess_rollout(
        rollout, country, level="portfolio", data_path=input_path
    )

    # Data Manipulation and Training
    training_beg = target.index.min() if TRAIN_START is None else TRAIN_START
    training_end = target.index.max()

    forecast_beg = example_results.index[0]
    forecast_end = example_results.index[-1]

    # Do split of past/future
    feature_training = features[training_beg:training_end]
    feature_forecast = features[forecast_beg:forecast_end]

    X_train = feature_training[FEATURES]
    y_train = preprocess_target_data(target, country, level="portfolio")
    y_train = y_train[training_beg:training_end]
    X_test = feature_forecast[FEATURES]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Get importance weihts
    weight = make_weight(feature_training.index)["WEIGHT"].values

    # X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Determine scale of y_train
    y_scale = y_train.std()
    y_train /= y_scale

    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=weight)

    # Get forcast on training data
    forecast_avg_train = pd.DataFrame(
        columns=["Avg"], index=feature_training.index
    )
    forecast_avg_train["Avg"] = y_scale * model.predict(X_train)

    # Vis model fit
    y_pred = model.predict(X_train)
    fig, axs = plt.subplots(1, 1)
    axs.plot(feature_training.index[:500], y_train[:500], label="True")
    axs.plot(feature_training.index[:500], y_pred[:500], label="Pred")
    axs.legend()
    fig.savefig(f"../figs/avg_customer_RegressionTree_{country}.pdf")

    # Predict
    forecast_avg = pd.DataFrame(
        columns=["Avg"],
        index=pd.date_range(start=forecast_beg, end=forecast_end, freq="1h"),
    )

    # Assign everyone to the mean
    forecast_avg["Avg"] = y_scale * model.predict(X_test)

    #   Step 2: Predict individuals
    # ----------------------------
    forecast = pd.DataFrame(
        columns=target.columns,
        index=pd.date_range(start=forecast_beg, end=forecast_end, freq="1h"),
    )
    for costumer in target.columns.values:
        print(costumer)
        costumer_id = costumer.split("_")[-1]

        #   Version 1: No individual prediction
        # ----------------------------
        # forecast[costumer] = forecast_avg["Avg"]  # / n_customer
        # continue

        #   Version 2: Simpleindividual prediction
        # ----------------------------
        features["rollout"] = preprocess_rollout(
            rollout, country, customer_id=costumer_id, level="idv"
        )

        # Data Manipulation and Training
        training_beg = (
            target.index.min() if TRAIN_START is None else TRAIN_START
        )
        training_end = target.index.max()

        forecast_beg = example_results.index[0]
        forecast_end = example_results.index[-1]

        # Do split of past/future
        feature_training = features[training_beg:training_end]
        feature_forecast = features[forecast_beg:forecast_end]

        # feature_training = feature_training["temp"].values.reshape(-1, 1)
        # feature_forecast = feature_forecast["temp"].values.reshape(-1, 1)

        feature_training = feature_training[FEATURES]
        feature_forecast = feature_forecast[FEATURES]

        # Get importance weihts
        weight = make_weight(feature_training.index)["WEIGHT"].values

        y_train = preprocess_target_data(
            target, country, level="idv", customer_id=costumer_id
        )
        y_train = y_train[training_beg:training_end]

        # Drop NaN
        na_mask = y_train.isna()
        feature_training = feature_training[~na_mask]
        y_train = y_train[~na_mask]

        if len(feature_forecast) == 0 or len(feature_training) == 0:
            forecast[costumer] = 0
            continue

        # Get target as deviation
        y_train = forecast_avg_train["Avg"][~na_mask] - y_train

        # Train
        # model = SimpleModel()
        # model.train(feature_training, y_train, weight=weight[~na_mask])
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        model.fit(feature_training, y_train)

        # Predict
        output = model.predict(feature_forecast)
        forecast[costumer] = forecast_avg["Avg"] - output

    """
    END OF THE MODIFIABLE PART.
    """

    # test to make sure that the output has the expected shape.
    dummy_error = np.abs(forecast - example_results).sum().sum()
    assert np.all(
        forecast.columns == example_results.columns
    ), "Wrong header or header order."
    assert np.all(
        forecast.index == example_results.index
    ), "Wrong index or index order."
    assert isinstance(dummy_error, np.float64), "Wrong dummy_error type."
    assert forecast.isna().sum().sum() == 0, "NaN in forecast."
    # Your solution will be evaluated using
    # forecast_error = np.abs(forecast - testing_set).sum().sum(),
    # and then doing a weighted sum the two portfolios:
    # score = forecast_error_IT + 5 * forecast_error_ES

    forecast.to_csv(output_path / f"students_results_{team_name}_{country}.csv")


if __name__ == "__main__":
    for country in ["ES", "IT"]:
        main(country)
