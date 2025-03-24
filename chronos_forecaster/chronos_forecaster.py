from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from typing import Optional
from time import time
import pandas as pd
import shutil
import torch
import os


class ChronosForecaster:
    """
    ChronosForecaster provides time series forecasting using Foundation Model Chronos.
    """

    def __init__(
        self,
        forecast_horizon: int,
        datetime_col: str,
        target_col: str,
        item_id_col: Optional[str] = None,
        frequency: str = "h",
        random_state: Optional[int] = None,
        finetune: bool = False,
    ):
        """
        Initialize the ChronosForecaster.

        Parameters:
            forecast_horizon (int): The number of time steps to forecast into the future.
            datetime_col (str): The name of the datetime column in the input DataFrame.
            target_col (str): The name of the target column in the input DataFrame.
            item_id_col (str, optional): The name of the item identifier column. If not provided, a default column will be created.
            random_state (int, optional): The random seed for reproducibility.
            finetune (bool, optional): Whether to fine-tune the model.
        """
        if not isinstance(forecast_horizon, int):
            raise ValueError("forecast_horizon must be an integer.")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be a positive integer.")
        if forecast_horizon > 60:
            raise ValueError("Forecast horizon must be less than or equal to 60.")

        if not isinstance(datetime_col, str) or not isinstance(target_col, str):
            raise ValueError("datetime_col and target_col must be strings.")
        if datetime_col == target_col:
            raise ValueError("datetime_col and target_col must be different columns.")

        try:
            pd.date_range(start="2021-01-01", periods=1, freq=frequency)
        except ValueError:
            raise ValueError(f"Invalid frequency: {frequency}")


        self.forecast_horizon = forecast_horizon
        self.datetime_col = datetime_col
        self.target_col = target_col
        self.item_id_col = item_id_col
        self.frequency = frequency
        self.random_state = random_state
        self.finetune = finetune

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast future values based on the provided DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the time series data.

        Returns:
            pd.DataFrame: A DataFrame containing the forecasted values along with their timestamps
                        and confidence intervals.
        """
        if self.target_col not in df.columns:
            raise ValueError(f"{self.target_col} column not found in training DataFrame.")

        if self.item_id_col is None:
            if df[self.datetime_col].nunique() < len(df):
                raise ValueError(
                    "Multiple entries found for a single datetime. Please ensure that the datetime column is unique or "
                    "provide an item_id column to distinguish between different time series."
                )
            self.item_id_col = "item_id"
            df[self.item_id_col] = 0
        else:
            if self.item_id_col not in df.columns:
                raise ValueError(f"{self.item_id_col} column not found in training DataFrame.")
            if df.groupby(self.item_id_col)[self.datetime_col].nunique().min() < len(df[self.item_id_col].unique()):
                raise ValueError(
                    "Multiple entries found for a single datetime for at least one item_id. Please ensure that the "
                    "datetime column is unique for each item_id."
                )

        if self.datetime_col not in df.columns:
            raise ValueError(f"{self.datetime_col} column not found in DataFrame.")

        df_context = TimeSeriesDataFrame.from_data_frame(
            df, id_column=self.item_id_col, timestamp_column=self.datetime_col
        )

        temp_dir = f"/tmp/chronos_forecaster_{time()}"

        predictor = TimeSeriesPredictor(
            target=self.target_col,
            prediction_length=self.forecast_horizon,
            freq=self.frequency,
            path=temp_dir,
            cache_predictions=False,
            verbosity=0,
        )
        predictor.fit(
            train_data=df_context,
            presets="best_quality",
            hyperparameters={
                "Chronos": {
                    "context_length": len(df_context),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "model_path": "autogluon/chronos-bolt-base",
                    "fine_tune": self.finetune,
                }
            },
            random_seed=self.random_state,
            verbosity=0,
        )

        try:
            prediction = predictor.predict(
                data=df_context, random_seed=self.random_state, use_cache=False
            )
            pred_df = prediction.to_data_frame().reset_index()
            pred_df = pred_df[["timestamp", "0.1", "mean", "0.9"]]
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        return pred_df.rename(
            columns={
                "timestamp": self.datetime_col,
                "0.1": "lower_bound",
                "mean": self.target_col,
                "0.9": "upper_bound",
            }
        )
