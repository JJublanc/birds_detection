import pandas as pd


class Preprocessor:
    def __init__(
        self,
        date_col,
        selected_cols,
        lag_intervals,
        train_data,
        func_lag_features,
        func_time_series_features,
        func_trend_and_season,
        func_clean_date_index,
    ):
        self.date_col = date_col
        self.selected_cols = selected_cols
        self.lag_intervals = lag_intervals
        self.train_data = train_data
        self.mean = train_data.mean(numeric_only=True)
        self.std = train_data.std(numeric_only=True)
        self.numeric_columns = list(train_data.describe().columns)
        self.add_lag_features = func_lag_features
        self.add_time_series_features = func_time_series_features
        self.add_trend_and_season = func_trend_and_season
        self.get_clean_date_index = func_clean_date_index

        #TODO: separate 3 type of preprocess :
        # 1/ cleaning : format transformation, dropping, etc. [often generic]
        # 2/ scaling
        # 3/ feature engineering : augmentation, embedding, etc. [often specific]

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend and season to data
        :param data: row data
        :return:
        """
        data, cols_lag = self.add_lag_features(
            df=data,
            lag_features=self.selected_cols,
            lag_intervals=self.lag_intervals,
        )
        data, cols_trend_season = self.add_trend_and_season(
            data, self.selected_cols
        )
        data, cols_ts_features = self.add_time_series_features(
            df=data, datetime_column_name=self.date_col
        )
        return data

    def standardize_data(self, data: pd.DataFrame):
        """
        Standardize data mean and std of the training data
        dataset.
        :param data:
        :return:
        """
        return (data - self.mean) / self.std

    def unstandardize_data(self, data: pd.DataFrame, column: str):
        """
        Standardize data mean and std of the training data
        dataset.
        :param data:
        :return:
        """
        return (data + self.mean[column].values[0]) * self.std[column].values[
            0
        ]

    def preprocess_data(self, data):
        """
        Preprocess training data. Preprocess params are compute from this
        dataset
        :param data:
        :return: preprocessed dataset
        """
        if self.date_col in data.columns:
            data = self.get_clean_date_index(data, self.date_col)
        data[self.numeric_columns] = self.standardize_data(
            data[self.numeric_columns]
        )
        data = self.augment_data(data)
        data.pop(self.date_col)
        return data
