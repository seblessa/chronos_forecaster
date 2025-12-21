"""Chronos-2 engine with covariate support."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import torch
from chronos import Chronos2Pipeline


class Chronos2Engine:
    """Chronos-2 forecasting engine (supports covariates, zero-shot only)."""

    def __init__(
        self,
        forecast_horizon: int,
        frequency: str = "D",
        random_state: Optional[int] = None,
        model_uri: str = "amazon/chronos-2",
    ) -> None:
        self.forecast_horizon = forecast_horizon
        self.frequency = frequency
        self.random_state = random_state
        self.model_uri = model_uri
        self._pipeline = None

    def predict(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        target_col: str,
        item_id_col: str,
        past_covariates_df: Optional[pd.DataFrame] = None,
        future_covariates_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate forecasts using Chronos-2.
        
        Returns DataFrame with columns: 
        ["item_id", "timestamp", "lower_bound", "point_forecast", "upper_bound"]
        """
        
        # Load pipeline once
        if self._pipeline is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_uri,
                device_map=device,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )

        # Prepare context DataFrame
        context_df = df[[item_id_col, datetime_col, target_col]].rename(
            columns={item_id_col: "item_id", datetime_col: "timestamp", target_col: "target"}
        ).copy()
        context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])
        
        # Merge past covariates
        if past_covariates_df is not None:
            # Build rename mapping for past covariates
            past_rename = {datetime_col: "timestamp"}
            if item_id_col in past_covariates_df.columns:
                past_rename[item_id_col] = "item_id"
            
            past = past_covariates_df.rename(columns=past_rename).copy()
            past["timestamp"] = pd.to_datetime(past["timestamp"])
            
            # If no item_id in past covariates, add it to match context_df
            if "item_id" not in past.columns:
                past["item_id"] = 0
            
            context_df = context_df.merge(past, on=["item_id", "timestamp"], how="left")
        
        # Prepare future DataFrame
        future_df = None
        if future_covariates_df is not None:
            # Build rename mapping for future covariates
            future_rename = {datetime_col: "timestamp"}
            if item_id_col in future_covariates_df.columns:
                future_rename[item_id_col] = "item_id"
            
            future_df = future_covariates_df.rename(columns=future_rename).copy()
            future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])
            
            # If no item_id in future covariates, add it to match context_df
            if "item_id" not in future_df.columns:
                future_df["item_id"] = 0

        # Predict
        preds = self._pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=self.forecast_horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
        )

        # Format output
        output = preds.rename(columns={
            "id": "item_id",
            "0.1": "lower_bound",
            "0.5": "point_forecast",
            "0.9": "upper_bound",
        })
        
        return output[["item_id", "timestamp", "lower_bound", "point_forecast", "upper_bound"]]
