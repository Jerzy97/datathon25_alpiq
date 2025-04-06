"""Perform a train/test split"""

from pathlib import Path

import pandas as pd

if __name__ == "__main__":

    # Basic definitions
    date_format = "%Y-%m-%d %H:%M:%S"
    data_path = Path("../data")
    split = "split"

    split_path = data_path / split
    split_path.mkdir(exist_ok=True)

    # Define split
    if split == "split_1":
        train_beg = "2022-01-01 00:00:00"
        train_end = "2022-07-31 23:00:00"

        test_beg = "2022-08-01 00:00:00"
        test_end = "2022-08-31 23:00:00"
    elif split == "split_2":
        train_beg = "2023-01-01 00:00:00"
        train_end = "2023-07-31 23:00:00"

        test_beg = "2023-08-01 00:00:00"
        test_end = "2023-08-31 23:00:00"
    elif split == "split":
        train_beg = "2022-01-01 00:00:00"
        train_end = "2023-07-31 23:00:00"

        test_beg = "2023-08-01 00:00:00"
        test_end = "2023-08-31 23:00:00"
    else:
        raise ValueError("Undefined split")

    # Do split for ES & IT
    for country in ["ES", "IT"]:

        consumption_fname = f"historical_metering_data_{country}.csv"

        # Load data
        consumptions = pd.read_csv(
            data_path / consumption_fname,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )

        # Do split
        consumption_train = consumptions.loc[train_beg:train_end]
        consumption_test = consumptions.loc[test_beg:test_end]

        # Save split
        consumption_train.to_csv(split_path / consumption_fname)
        consumption_test.to_csv(split_path / f"example_set_{country}.csv")
