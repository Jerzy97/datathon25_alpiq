from pathlib import Path

import pandas as pd


def preprocess_rollout(
    data: pd.DataFrame,
    suffix: str,
    customer_id: int = None,
    level: str = "portfolio",
    data_path: str = None,
):
    # Preprocess the rollout data
    # Portfolio level - take the median of all customers
    # Customer level - take the customer_id column and fillna with 0

    if level == "portfolio":
        # Load the better rollout data
        if suffix == "ES":
            rollout = pd.read_csv(
                data_path / f"highcorr_rollout_v2_ES.csv",
                index_col=0,
                parse_dates=True,
                date_format="%Y-%m-%d %H:%M:%S",
            )
            return rollout["0"]

        return data.median(axis=1)
    else:
        data_proc = data[f"INITIALROLLOUTVALUE_customer{suffix}_{customer_id}"]
        return data_proc.fillna(0)


def preprocess_target_data(
    data: pd.DataFrame,
    suffix: str,
    customer_id: int = None,
    level: str = "portfolio",
):
    # For portfolio level, sum the data across all customers
    # For customer level, filter by customer_id
    if level == "portfolio":

        return data.mean(axis=1)
    else:
        return data[f"VALUEMWHMETERINGDATA_customer{suffix}_{customer_id}"]


class DataLoader:
    def __init__(self, path: str):
        self.path = Path(path)

    def load_data(self, country: str):
        """
        Return
        """
        date_format = "%Y-%m-%d %H:%M:%S"

        consumptions = pd.read_csv(
            self.path / f"historical_metering_data_{country}.csv",
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )
        features = pd.read_excel(
            self.path / "spv_ec00_forecasts_es_it.xlsx",
            sheet_name=country,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )
        example_solution = pd.read_csv(
            self.path / f"example_set_{country}.csv",
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )

        # Load other data
        rollout = pd.read_csv(
            self.path / f"rollout_data_{country}.csv",
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )

        holiday = pd.read_excel(
            self.path / f"holiday_{country}.xlsx",
            parse_dates=True,
            date_format=date_format,
        )

        # Use imputed data for spain
        if country == "ES":
            consumptions_im = pd.read_csv(
                self.path / f"imputed_{country}.csv",
                index_col=0,
                parse_dates=True,
                date_format=date_format,
            )
            consumptions_im.columns = consumptions.columns
            consumptions_im = consumptions_im.loc[consumptions.index]
            consumptions = consumptions_im

        # # Add other data to features
        # features["rollout"] = rollout.mean(axis=1)
        # features["hour"] = features.index.hour
        # features["day"] = features.index.dayofweek
        # features["year"] = features.index.year
        # features["month"] = features.index.month
        # features.loc[holiday["holiday_ES"], "day"] = 7

        # Susie's feature eng
        features["hour"] = features.index.hour
        features["day_of_week"] = features.index.dayofweek
        features["month"] = features.index.month

        features["is_off_day"] = 0
        features.loc[holiday[f"holiday_{country}"], "is_off_day"] = 1
        features.loc[features["day_of_week"].isin([5, 6]), "is_off_day"] = 1

        # rollout_proc = preprocess_rollout(rollout, country, customer_id, level)
        # features = pd.concat([features, rollout_proc], axis=1)

        return consumptions, features, rollout, example_solution


# Encoding Part


class SimpleEncoding:
    """
    This class is an example of dataset encoding.

    """

    def __init__(
        self,
        consumption: pd.Series,
        features: pd.Series,
        end_training,
        start_forecast,
        end_forecast,
    ):
        self.consumption_mask = ~consumption.isna()
        self.consumption = consumption[self.consumption_mask]
        self.features = features
        self.end_training = end_training
        self.start_forecast = start_forecast
        self.end_forecast = end_forecast

    def meta_encoding(self):
        """
        This function returns the feature, split between past (for training) and future (for forecasting)),
        as well as the consumption, without missing values.
        :return: three numpy arrays

        """
        features_past = self.features[: self.end_training].values.reshape(-1, 1)
        features_future = self.features[
            self.start_forecast : self.end_forecast
        ].values.reshape(-1, 1)

        features_past = features_past[self.consumption_mask]

        return features_past, features_future, self.consumption


class Encoding:
    """Our feature encoding"""

    def __init__(
        self,
        consumption: pd.Series,
        features: pd.Series,
        training_beg,
        training_end,
        forecast_beg,
        forecast_end,
    ):
        # self.consumption_mask = ~consumption.isna()
        # self.consumption = consumption[self.consumption_mask]

        self.consumption = consumption
        self.features = features

        self.training_beg = training_beg
        self.training_end = training_end

        self.forecast_beg = forecast_beg
        self.forecast_end = forecast_end

    def meta_encoding(self):
        """
        This function returns the feature, split between
        past (for training) and future (for forecasting),
        as well as the consumption
        """
        features_past = self.features.loc[self.training_beg : self.training_end]
        features_future = self.features[self.forecast_beg : self.forecast_end]

        return features_past, features_future, self.consumption
