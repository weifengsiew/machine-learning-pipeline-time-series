"""Nodes for the post-June-2012 feature table."""

import numpy as np
import pandas as pd

TARGET_LAG_DAYS = {
    "sales_lag_1_day": 1,
    "sales_lag_2_days": 2,
    "sales_lag_1_week": 7,
    "sales_lag_2_weeks": 14,
    "sales_lag_4_weeks": 28,
    "sales_lag_1_month": 30,
    "sales_lag_1_quarter": 90,
    "sales_lag_1_year": 365,
}
ROLLING_COLUMNS = [
    "sales_rolling_mean_7_past",
    "sales_rolling_mean_28_past",
]


def _validate_horizon(horizon: int) -> None:
    """Validate the configured forecast horizon."""
    if horizon < 1:
        raise ValueError("horizon_days must be at least 1")
    if horizon > min(TARGET_LAG_DAYS.values()):
        raise ValueError("horizon_days must not exceed the shortest target-relative lag")


def filter_start_date(
    cleaned_store_data: pd.DataFrame,
    start_date: str,
) -> pd.DataFrame:
    """Retain observations from the configured post-June-2012 boundary."""
    return cleaned_store_data.loc[
        cleaned_store_data.index >= pd.Timestamp(start_date)
    ].copy()


def add_event_indicator(
    filtered_cleaned_store_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add an indicator for whether a calendar event is present."""
    event_name_columns = [
        column
        for column in ("event_name_1", "event_name_2")
        if column in filtered_cleaned_store_data
    ]
    if not event_name_columns:
        raise KeyError("Missing event name columns")
    features = filtered_cleaned_store_data.copy()
    features["has_event"] = (
        features[event_name_columns].ne("No event").any(axis=1).astype(int)
    )
    return features


def select_base_features(event_features: pd.DataFrame) -> pd.DataFrame:
    """Select the observed sales, event, and SNAP columns."""
    if "snap" not in event_features.columns:
        raise KeyError("Missing SNAP column: snap")
    return event_features[
        ["sales", "snap", "has_event", "event_name_1", "event_type_1"]
    ].rename(
        columns={
            "event_name_1": "event_name",
            "event_type_1": "event_type",
        }
    ).copy()


def add_lag_features(base_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add sales lag features relative to the forecast horizon."""
    _validate_horizon(horizon)
    features = base_features.copy()
    for column, lag_days in TARGET_LAG_DAYS.items():
        features[column] = features["sales"].shift(lag_days - horizon)
    return features


def add_cyclical_features(lag_features: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical week-of-year and month features."""
    features = lag_features.copy()
    week_of_year = features.index.isocalendar().week.astype(int)
    month_of_year = features.index.month
    features["week_sin"] = np.sin(2 * np.pi * week_of_year / 52)
    features["week_cos"] = np.cos(2 * np.pi * week_of_year / 52)
    features["month_sin"] = np.sin(2 * np.pi * month_of_year / 12)
    features["month_cos"] = np.cos(2 * np.pi * month_of_year / 12)
    return features


def add_rolling_features(lag_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add past-only rolling sales features."""
    _validate_horizon(horizon)
    features = lag_features.copy()
    rolling_source = features["sales"].shift(horizon - 1)
    features[ROLLING_COLUMNS[0]] = rolling_source.rolling(7).mean()
    features[ROLLING_COLUMNS[1]] = rolling_source.rolling(28).mean()
    return features


def align_target_calendar_features(
    rolling_features: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Align known calendar features with the date being forecast."""
    _validate_horizon(horizon)
    calendar_columns = [
        "snap",
        "has_event",
        "event_name",
        "event_type",
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
    ]
    aligned_features = rolling_features.copy()
    aligned_features[calendar_columns] = aligned_features[calendar_columns].shift(-horizon)
    return aligned_features


def add_target(rolling_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add the future sales target and shift the feature index to its date."""
    _validate_horizon(horizon)
    feature_columns = [
        *TARGET_LAG_DAYS,
        "snap",
        "has_event",
        "event_name",
        "event_type",
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
        *ROLLING_COLUMNS,
    ]
    model_frame = pd.concat(
        [
            rolling_features[feature_columns],
            rolling_features["sales"].shift(-horizon).rename("target_next_day"),
        ],
        axis=1,
    )
    model_frame.index = model_frame.index + pd.to_timedelta(horizon, unit="D")
    return model_frame

