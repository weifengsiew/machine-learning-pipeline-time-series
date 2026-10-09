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
    features = add_event_indicator(filtered_cleaned_store_data)
    features = select_base_features(features)
    features = add_lag_features(features, horizon)
    features = add_cyclical_features(features)
    features = add_rolling_features(features, horizon)
    features = align_target_calendar_features(features, horizon)
    return add_target(features, horizon)


def add_event_indicator(
    filtered_cleaned_store_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add an indicator for whether a calendar event is present."""
    return EngineerFeatures().add_event_indicator(filtered_cleaned_store_data)


def select_base_features(event_features: pd.DataFrame) -> pd.DataFrame:
    """Select the observed sales, event, and SNAP columns."""
    return EngineerFeatures().select_base_features(event_features)


def add_lag_features(base_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add sales lag features relative to the forecast horizon."""
    return EngineerFeatures(horizon_days=horizon).add_lag_features(base_features)


def add_cyclical_features(lag_features: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical week-of-year and month features."""
    return EngineerFeatures().add_cyclical_features(lag_features)


def add_rolling_features(lag_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add past-only rolling sales features."""
    return EngineerFeatures(horizon_days=horizon).add_rolling_features(lag_features)


def align_target_calendar_features(
    rolling_features: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Align known calendar features with the date being forecast."""
    return EngineerFeatures(horizon_days=horizon).align_target_calendar_features(
        rolling_features
    )


def add_target(rolling_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add the future sales target and shift the feature index to its date."""
    return EngineerFeatures(horizon_days=horizon).add_target(rolling_features)

