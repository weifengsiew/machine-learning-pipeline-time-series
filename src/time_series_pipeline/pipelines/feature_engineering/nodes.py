"""Nodes for the post-June-2012 feature table."""

import pandas as pd

from engineer_features import EngineerFeatures


def filter_start_date(
    cleaned_store_data: pd.DataFrame,
    start_date: str,
) -> pd.DataFrame:
    """Retain observations from the configured post-June-2012 boundary."""
    return cleaned_store_data.loc[
        cleaned_store_data.index >= pd.Timestamp(start_date)
    ].copy()


def engineer_features(
    filtered_cleaned_store_data: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Create leakage-safe lag, rolling, calendar, and target features."""
    return EngineerFeatures(horizon_days=horizon).engineer_features(
        filtered_cleaned_store_data
    )

