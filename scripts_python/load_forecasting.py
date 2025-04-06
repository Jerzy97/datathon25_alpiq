from pathlib import Path

import numpy as np
import pandas as pd

# depending on your IDE, you might need to add datathon_eth. in front of data
from datathon2025.data import DataLoader, SimpleEncoding

# depending on your IDE, you might need to add datathon_eth. in front of forecast_models
from datathon2025.forecast_models import SimpleModel


def main(zone: str):
    """Train and evaluate the models for IT and ES"""

    # Inputs
    input_path = Path("../data/split")
    output_path = Path("../outputs/split/baseline")
    output_path.mkdir(exist_ok=True, parents=True)

    # Load Datasets
    loader = DataLoader(input_path)
    training_set, features, _, example_results = loader.load_data(zone)

    team_name = "Placeholder"
    # Data Manipulation and Training
    start_training = training_set.index.min()
    end_training = training_set.index.max()
    start_forecast, end_forecast = (
        example_results.index[0],
        example_results.index[-1],
    )

    range_forecast = pd.date_range(
        start=start_forecast, end=end_forecast, freq="1h"
    )
    forecast = pd.DataFrame(columns=training_set.columns, index=range_forecast)
    for costumer in training_set.columns.values:
        print(costumer)
        consumption = training_set.loc[:, costumer]

        feature_dummy = features["temp"].loc[start_training:]

        encoding = SimpleEncoding(
            consumption,
            feature_dummy,
            end_training,
            start_forecast,
            end_forecast,
        )

        feature_past, feature_future, consumption_clean = (
            encoding.meta_encoding()
        )

        if len(feature_past) == 0:
            forecast[costumer] = 0
            continue

        # Train
        model = SimpleModel()
        model.train(feature_past, consumption_clean)

        # Predict
        output = model.predict(feature_future)
        forecast[costumer] = output

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
