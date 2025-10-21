"""Simple forecaster interface."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .engines import ChronosEngine, Chronos2Engine


class ChronosForecaster:
    """Simple forecaster using Chronos-Bolt or Chronos-2 engines."""

    def __init__(
        self,
        forecast_horizon: int,
        datetime_col: str,
        target_col: str,
        item_id_col: Optional[str] = None,
        frequency: str = "h",
        random_state: Optional[int] = None,
        engine: str = "chronos2",
    ) -> None:
        self.forecast_horizon = forecast_horizon
        self.datetime_col = datetime_col
        self.target_col = target_col
        self.item_id_col = item_id_col
        self.frequency = frequency

        # Create engine
        if engine.lower() == "chronos":
            self._engine = ChronosEngine(forecast_horizon, frequency, random_state)
        elif engine.lower() == "chronos2":
            self._engine = Chronos2Engine(forecast_horizon, frequency, random_state)
        else:
            raise ValueError(f"Unknown engine: {engine}. Use 'chronos' or 'chronos2'.")

    def predict(
        self,
        df: pd.DataFrame,
        past_covariates_df: Optional[pd.DataFrame] = None,
        future_covariates_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Make predictions.
        
        Returns DataFrame with columns: [datetime_col, {target_col}_predicted, lower_bound, upper_bound]
        """
        # Prepare data
        prepared_df = df.copy()
        prepared_df[self.datetime_col] = pd.to_datetime(prepared_df[self.datetime_col])
        
        # Handle item_id
        if self.item_id_col:
            item_id_col = self.item_id_col
            created_item_id = False
        else:
            item_id_col = "__item_id__"
            prepared_df[item_id_col] = 0
            created_item_id = True

        # Call engine
        predictions = self._engine.predict(
            prepared_df,
            datetime_col=self.datetime_col,
            target_col=self.target_col,
            item_id_col=item_id_col,
            past_covariates_df=past_covariates_df,
            future_covariates_df=future_covariates_df,
        )

        # Format output
        output = predictions.rename(columns={
            "timestamp": self.datetime_col,
            "point_forecast": f"{self.target_col}_predicted",
        })

        # Handle item_id column
        if not created_item_id and self.item_id_col:
            output = output.rename(columns={"item_id": self.item_id_col})
        else:
            output = output.drop(columns=["item_id"], errors="ignore")

        return output[[self.datetime_col, f"{self.target_col}_predicted", "lower_bound", "upper_bound"]]
