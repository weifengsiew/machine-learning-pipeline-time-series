"""Create leakage-safe features for one-day-ahead store-sales forecasting.

Inputs
------
pandas.DataFrame
    A cleaned daily table with sales, calendar, event, and SNAP columns.

Outputs
-------
pandas.DataFrame
    A model-ready table containing lag, rolling, cyclical, event, SNAP, and
    future-target columns. Missing values at history boundaries are retained.

Stages
------
1. Derive the event indicator and select observed base features.
2. Add lag, cyclical, and past-only rolling features.
3. Align known calendar features to the target date.
4. Add the future target and retain incomplete boundary rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class EngineerFeatures:
    """Own the configuration and actions used to engineer forecast features."""

    def __init__(self, horizon_days: int = 1, snap_column: str = "snap") -> None:
        """Initialize the feature engineer.

        Parameters
        ----------
        horizon_days : int, default=1
            Number of days ahead represented by the target.
        snap_column : str, default="snap"
            Name of the store-specific SNAP indicator column.
        """
        if horizon_days < 1:
            raise ValueError("horizon_days must be at least 1")
        self.horizon_days = horizon_days
        self.snap_column = snap_column
        target_lag_days = {
            "sales_lag_1_day": 1,
            "sales_lag_2_days": 2,
            "sales_lag_1_week": 7,
            "sales_lag_2_weeks": 14,
            "sales_lag_4_weeks": 28,
            "sales_lag_1_month": 30,
            "sales_lag_1_quarter": 90,
            "sales_lag_1_year": 365,
        }
        if horizon_days > min(target_lag_days.values()):
            raise ValueError(
                "horizon_days must not exceed the shortest target-relative lag"
            )
        self.lag_periods = {
            column: lag_days - horizon_days
            for column, lag_days in target_lag_days.items()
        }
        self.rolling_columns = [
            "sales_rolling_mean_7_past",
            "sales_rolling_mean_28_past",
        ]

    def select_base_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Select and rename the observed values used as features.

        Parameters
        ----------
        frame : pandas.DataFrame
            The cleaned daily table.

        Returns
        -------
        pandas.DataFrame
            The sales, SNAP, event, and event-indicator columns.
        """
        if self.snap_column not in frame.columns:
            raise KeyError(f"Missing SNAP column: {self.snap_column}")
        return frame[
            [
                "sales",
                self.snap_column,
                "has_event",
                "event_name_1",
                "event_type_1",
            ]
        ].rename(
            columns={
                "event_name_1": "event_name",
                "event_type_1": "event_type",
            }
        ).copy()

    def add_event_indicator(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Derive an event indicator from cleaned event name columns.

        Parameters
        ----------
        frame : pandas.DataFrame
            A cleaned table whose event labels use ``No event`` for expected
            missing values.

        Returns
        -------
        pandas.DataFrame
            A copy with the integer ``has_event`` feature.
        """
        event_name_columns = [
            column
            for column in ("event_name_1", "event_name_2")
            if column in frame
        ]
        if not event_name_columns:
            raise KeyError("Missing event name columns")
        features = frame.copy()
        features["has_event"] = (
            features[event_name_columns]
            .ne("No event")
            .any(axis=1)
            .astype(int)
        )
        return features

    def add_lag_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add recent, weekly, monthly, quarterly, and yearly lags.

        Parameters
        ----------
        features : pandas.DataFrame
            The selected base features indexed by date.

        Returns
        -------
        pandas.DataFrame
            A copy with the configured sales lag columns.
        """
        features = features.copy()
        for column, lag in self.lag_periods.items():
            features[column] = features["sales"].shift(lag)
        return features

    def add_cyclical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add sine and cosine encodings for week and month.

        Parameters
        ----------
        features : pandas.DataFrame
            Features indexed by datetime.

        Returns
        -------
        pandas.DataFrame
            A copy with week and month cyclical columns.
        """
        features = features.copy()
        week_of_year = features.index.isocalendar().week.astype(int)
        month_of_year = features.index.month
        features["week_sin"] = np.sin(2 * np.pi * week_of_year / 52)
        features["week_cos"] = np.cos(2 * np.pi * week_of_year / 52)
        features["month_sin"] = np.sin(2 * np.pi * month_of_year / 12)
        features["month_cos"] = np.cos(2 * np.pi * month_of_year / 12)
        return features

    def add_rolling_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add past-only rolling means without target leakage.

        Parameters
        ----------
        features : pandas.DataFrame
            Features containing the observed sales series.

        Returns
        -------
        pandas.DataFrame
            A copy with seven-day and 28-day past rolling means.
        """
        features = features.copy()
        rolling_source = features["sales"].shift(self.horizon_days - 1)
        features["sales_rolling_mean_7_past"] = rolling_source.rolling(7).mean()
        features["sales_rolling_mean_28_past"] = rolling_source.rolling(28).mean()
        return features

    def align_target_calendar_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Align known calendar features with the date being forecast.

        Sales history remains anchored to the forecast origin, while SNAP,
        event, and cyclical calendar values are shifted to the target date.
        These calendar values are known in advance and do not use future sales.
        """
        calendar_columns = [
            self.snap_column,
            "has_event",
            "event_name",
            "event_type",
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
        ]
        aligned_features = features.copy()
        aligned_features[calendar_columns] = aligned_features[calendar_columns].shift(
            -self.horizon_days
        )
        return aligned_features

    def add_target(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add the future sales target for the configured horizon.

        Parameters
        ----------
        features : pandas.DataFrame
            Features after lag, cyclical, and rolling transformations.

        Returns
        -------
        pandas.DataFrame
            The selected model features and ``target_next_day``.
        """
        feature_columns = [
            *self.lag_periods,
            self.snap_column,
            "has_event",
            "event_name",
            "event_type",
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
            *self.rolling_columns,
        ]
        model_frame = pd.concat(
            [
                features[feature_columns],
                features["sales"]
                .shift(-self.horizon_days)
                .rename("target_next_day"),
            ],
            axis=1,
        )
        model_frame.index = model_frame.index + pd.to_timedelta(
            self.horizon_days,
            unit="D",
        )
        return model_frame

    def engineer_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Run the feature engineering stages in order.

        Parameters
        ----------
        frame : pandas.DataFrame
            The cleaned daily store-sales table.

        Returns
        -------
        pandas.DataFrame
            The model-ready feature table with boundary missing values retained.
        """
        features = self.add_event_indicator(frame)
        features = self.select_base_features(features)
        features = self.add_lag_features(features)
        features = self.add_cyclical_features(features)
        features = self.add_rolling_features(features)
        features = self.align_target_calendar_features(features)
        return self.add_target(features)
