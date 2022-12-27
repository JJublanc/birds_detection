from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.parser import parse
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_data_decomposition_to_file(
    df: pd.DataFrame, core_columns: list, file_path: str
):
    """
    Extracts trend and seasonality components from the dataset
    :param df: pandas data frame, main dataset
    :param core_columns: columns names for the data we want to decompose
    :return: data frame with trend and seasonality components for the
    core_columns
    """
    for c in range(0, len(core_columns), 2):
        columns = core_columns[c : c + 2]
        fig, ax = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(16, 8))
        for i, column in enumerate(columns):
            res = seasonal_decompose(
                df[column], model="additive", extrapolate_trend="freq"
            )

            ax[0, i].set_title(
                "Decomposition of {}".format(column), fontsize=16
            )
            res.observed.plot(ax=ax[0, i], legend=False, color="dodgerblue")
            ax[0, i].set_ylabel("Observed", fontsize=14)

            res.trend.plot(ax=ax[1, i], legend=False, color="dodgerblue")
            ax[1, i].set_ylabel("Trend", fontsize=14)

            res.seasonal.plot(ax=ax[2, i], legend=False, color="dodgerblue")
            ax[2, i].set_ylabel("Seasonal", fontsize=14)

            res.resid.plot(ax=ax[3, i], legend=False, color="dodgerblue")
            ax[3, i].set_ylabel("Residual", fontsize=14)

        fig.savefig(file_path + f"_{str(c // 2)}")


def get_clean_date_index(
    df: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    """
    Transform a string column into an datetime.datetime index
    :param df: dataframe to be used
    :param date_column: name of the column to use as date
    (must be a string column)
    :return:
    """
    df = df.drop_duplicates([date_column])
    df["date_parsed_to_datetime"] = df[date_column].map(parse)
    df.set_index("date_parsed_to_datetime", inplace=True)
    return df


def add_trend_and_season(df: pd.DataFrame, columns: list):
    """
    Extracts trend and seasonality components from the data set
    :param df: pandas data frame, main dataset
    :param columns: columns names for the data we want to decompose
    :return: data frame with added trend and seasonality components for
    the columns
    """
    cols = []
    for column in columns:
        decomp = seasonal_decompose(
            df[column], period=52, model="additive", extrapolate_trend="freq"
        )
        df[f"{column}_trend"] = decomp.trend
        cols.append(f"{column}_trend")
        df[f"{column}_seasonal"] = decomp.seasonal
        cols.append(f"{column}_seasonal")
    return df, cols


def add_lag_features(
    df: pd.DataFrame, lag_features: list, lag_intervals: list
) -> (pd.DataFrame, list):
    """
    Adds lag-based features generated with rolling statistics.
    :param df: pandas data frame, main dataset
    :param lag_features: names of columns to be shifted
    :param lag_intervals: time intervals for the data offset
    :return: data frame with lagged features
    """
    df.reset_index(drop=True, inplace=True)
    cols = []
    for window in lag_intervals:
        for feature in lag_features:
            df_rolled = df[feature].rolling(window=window, min_periods=0)
            df_mean = (
                df_rolled.mean().shift(1).reset_index().astype(np.float32)
            )
            df_std = df_rolled.std().shift(1).reset_index().astype(np.float32)

            df[f"{feature}_mean_lag_{window}"] = df_mean[feature]
            cols.append(f"{feature}_mean_lag_{window}")

            df[f"{feature}_std_lag_{window}"] = df_std[feature]
            cols.append(f"{feature}_mean_lag_{window}")

    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill NA/NaN
    return df, cols


def add_time_series_features(
    df: pd.DataFrame, datetime_column_name: str
) -> Tuple[pd.DataFrame, List[Union[str, Any]]]:
    """
    Creates time series features from datetime index.
    :param datetime_column_name: name of the column with dates and time
    :param df: pandas data frame, main dataset
    :return: data frame with time series features like day of week, year etc...
    """
    df = df.copy()
    df[datetime_column_name] = pd.to_datetime(df[datetime_column_name])
    df["hour"] = df[datetime_column_name].dt.hour
    df["dayofweek"] = df[datetime_column_name].dt.dayofweek
    df["month"] = df[datetime_column_name].dt.month
    df["year"] = df[datetime_column_name].dt.year
    df["dayofyear"] = df[datetime_column_name].dt.dayofyear
    df["dayofmonth"] = df[datetime_column_name].dt.day

    # encoding cyclical features as month and hour
    month_in_year = 12
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / month_in_year)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / month_in_year)
    hours_in_day = 24
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / hours_in_day)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / hours_in_day)
    dayofweek_in_week = 7
    df["dayofweek_sin"] = np.sin(
        2 * np.pi * df["dayofweek"] / dayofweek_in_week
    )
    df["dayofweek_cos"] = np.cos(
        2 * np.pi * df["dayofweek"] / dayofweek_in_week
    )
    dayofyear_in_year = 365
    df["dayofyear_sin"] = np.sin(
        2 * np.pi * df["dayofyear"] / dayofyear_in_year
    )
    df["dayofyear_cos"] = np.cos(
        2 * np.pi * df["dayofyear"] / dayofyear_in_year
    )
    dayofmonth_in_month = 30
    df["dayofmonth_sin"] = np.sin(
        2 * np.pi * df["dayofmonth"] / dayofmonth_in_month
    )
    df["dayofmonth_cos"] = np.cos(
        2 * np.pi * df["dayofmonth"] / dayofmonth_in_month
    )

    return df, [
        "hour",
        "dayofweek",
        "month",
        "year",
        "month",
        "dayofyear",
        "month_sin",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "dayofmonth_sin",
    ]


def show_data_decomposition(df: pd.DataFrame, core_columns: list):
    """
    Extracts trend and seasonality components from the dataset
    :param df: pandas data frame, main dataset
    :param core_columns: columns names for the data we want to decompose
    :return: data frame with trend and seasonality components for the
    core_columns
    """
    for c in range(0, len(core_columns), 2):
        columns = core_columns[c : c + 2]
        fig, ax = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(16, 8))
        for i, column in enumerate(columns):
            res = seasonal_decompose(
                df[column], model="additive", extrapolate_trend="freq"
            )

            ax[0, i].set_title(
                "Decomposition of {}".format(column), fontsize=16
            )
            res.observed.plot(ax=ax[0, i], legend=False, color="dodgerblue")
            ax[0, i].set_ylabel("Observed", fontsize=14)

            res.trend.plot(ax=ax[1, i], legend=False, color="dodgerblue")
            ax[1, i].set_ylabel("Trend", fontsize=14)

            res.seasonal.plot(ax=ax[2, i], legend=False, color="dodgerblue")
            ax[2, i].set_ylabel("Seasonal", fontsize=14)

            res.resid.plot(ax=ax[3, i], legend=False, color="dodgerblue")
            ax[3, i].set_ylabel("Residual", fontsize=14)

    plt.show()
