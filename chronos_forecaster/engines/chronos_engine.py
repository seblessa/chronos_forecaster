"""Chronos-Bolt engine for fast zero-shot forecasting."""
from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
import torch
from chronos import ChronosBoltPipeline


class ChronosEngine:
    """Chronos-Bolt forecasting engine (zero-shot only, no covariates)."""

    def __init__(
        self,
        forecast_horizon: int,
        frequency: str = "D",
        random_state: Optional[int] = None,
        model_uri: str = "amazon/chronos-bolt-base",
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
        """Generate forecasts using Chronos-Bolt.
        
        Returns DataFrame with columns: 
        ["item_id", "timestamp", "lower_bound", "point_forecast", "upper_bound"]
        """
        
        # Warnings for unsupported features
        if past_covariates_df is not None or future_covariates_df is not None:
            warnings.warn("Chronos-Bolt doesn't support covariates. Use engine='chronos2' instead.")

        # Load pipeline once
        if self._pipeline is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline = ChronosBoltPipeline.from_pretrained(
                self.model_uri,
                device_map=device,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )

        # Convert to tensors (one per series)
        context = [
            torch.tensor(df[df[item_id_col] == item_id][target_col].values, dtype=torch.float32)
            for item_id in df[item_id_col].unique()
        ]

        # Predict (shape: batch_size x num_quantiles x prediction_length)
        forecasts = self._pipeline.predict(
            inputs=context, 
            prediction_length=self.forecast_horizon
        )

        # Extract quantiles
        quantile_levels = self._pipeline.quantiles
        q10_idx = quantile_levels.index(0.1) if 0.1 in quantile_levels else 0
        q50_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else len(quantile_levels) // 2
        q90_idx = quantile_levels.index(0.9) if 0.9 in quantile_levels else -1

        # Build output dataframe
        results = []
        item_ids = df[item_id_col].unique()
        last_timestamp = pd.to_datetime(df[datetime_col].max())
        future_timestamps = pd.date_range(
            start=last_timestamp,
            periods=self.forecast_horizon + 1,
            freq=self.frequency
        )[1:]

        for i, item_id in enumerate(item_ids):
            results.append(pd.DataFrame({
                "item_id": item_id,
                "timestamp": future_timestamps,
                "lower_bound": forecasts[i, q10_idx, :].cpu().numpy(),
                "point_forecast": forecasts[i, q50_idx, :].cpu().numpy(),
                "upper_bound": forecasts[i, q90_idx, :].cpu().numpy(),
            }))

        return pd.concat(results, ignore_index=True)
