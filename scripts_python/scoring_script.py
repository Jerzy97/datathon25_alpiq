from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Basics
date_format = "%Y-%m-%d %H:%M:%S"
student_path = Path("../outputs/split/idv_pred_tree")
testing_set_path = Path("../data/split")

absolute_error = {}
portfolio_error = {}

for team_name in ["Placeholder"]:

    try:
        for country in ["ES", "IT"]:
            student_solution = pd.read_csv(
                student_path / f"students_results_{team_name}_{country}.csv",
                index_col=0,
                parse_dates=True,
                date_format=date_format,
            )
            testing_set = pd.read_csv(
                testing_set_path / f"example_set_{country}.csv",
                index_col=0,
                parse_dates=True,
                date_format=date_format,
            )

            country_error_vis = student_solution - testing_set
            portfolio_error_vis = (student_solution - testing_set).sum(axis=1)

            # Plot idv errors
            rng = np.random.default_rng(seed=1234)
            for customer in rng.choice(testing_set.columns, 10):
                fig, ax = plt.subplots(1, 1)
                ax.plot(country_error_vis.index, country_error_vis[customer])
                ax.tick_params(axis="x", rotation=45)
                ax.set_title(f"{customer}")
                fig.tight_layout()
                fig.savefig(f"../figs/{customer}_err.pdf")
                plt.close(fig)

            # Plot Portfolio error
            fig, ax = plt.subplots(1, 1)
            ax.plot(country_error_vis.index, portfolio_error_vis)
            ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"Portfolio {country}")
            fig.tight_layout()
            fig.savefig(f"../figs/portfolio_{country}_err.pdf")
            plt.close(fig)

            country_error = (student_solution - testing_set).abs().sum().sum()
            portfolio_country_error = (
                (student_solution - testing_set).sum(axis=1).abs().sum()
            )

            assert np.all(student_solution.columns == testing_set.columns), (
                "Wrong header or header order for team " + team_name
            )
            assert np.all(student_solution.index == testing_set.index), (
                "Wrong index or index order for team " + team_name
            )
            assert isinstance(country_error, np.float64), (
                "Wrong error type for team " + team_name
            )
            assert student_solution.isna().sum().sum() == 0, (
                "NaN in forecast for team " + team_name
            )

            absolute_error[country] = country_error
            portfolio_error[country] = portfolio_country_error

        forecast_score = (
            1.0 * absolute_error["IT"]
            + 5.0 * absolute_error["ES"]
            + 10.0 * portfolio_error["IT"]
            + 50.0 * portfolio_error["ES"]
        )

        print("Individual terms")
        print(f"Abs IT: {absolute_error["IT"]}")
        print(f"Abs ES: {absolute_error["ES"]}")
        print(f"Port IT: {portfolio_error["IT"]}")
        print(f"Port ES: {portfolio_error["ES"]}")
        print(
            "The team "
            + team_name
            + " reached a forecast score of "
            + str(np.round(forecast_score, 0))
        )
    except Exception as e:
        print("Error for team " + team_name)
        print(e)
print("=== End of the script, %s. ===" % (str(datetime.now())))
