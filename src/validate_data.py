"""Validate daily store-sales data before and after cleaning.

Inputs
------
pandas.DataFrame
    A daily table with a date index, sales, calendar, and event columns.

Outputs
-------
pandas.Series
    Named boolean validation results. Invalid results raise an assertion when
    enforcement is enabled.

Stages
------
1. Validate the datetime index.
2. Validate calendar fields against the datetime index.
3. Validate sales values.
4. Validate event labels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_EVENT_COLUMNS = (
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
)
DEFAULT_CALENDAR_COLUMNS = ("wm_yr_wk", "wday", "month", "year")


class ValidateData:
    """Own the configuration and actions used to validate one daily time series."""

    def __init__(
        self,
        event_columns: Sequence[str] | None = None,
        calendar_columns: Sequence[str] | None = None,
    ) -> None:
        """Initialize the validator with event and calendar columns.

        Parameters
        ----------
        event_columns : sequence of str, optional
            Event name and event type columns. The M5 event columns are used
            when no value is supplied.
        calendar_columns : sequence of str, optional
            Calendar columns that must use the ``int64`` dtype.
        """
        self.event_columns = tuple(event_columns or DEFAULT_EVENT_COLUMNS)
        self.calendar_columns = tuple(calendar_columns or DEFAULT_CALENDAR_COLUMNS)

    def validate_dates(self, frame: pd.DataFrame) -> dict[str, bool]:
        """Validate datetime type, order, uniqueness, and daily coverage.

        Parameters
        ----------
        frame : pandas.DataFrame
            The daily table to inspect.

        Returns
        -------
        dict[str, bool]
            Date validation results.
        """
        date_values = pd.to_datetime(frame.index, errors="coerce")
        date_parseable = bool(date_values.notna().all())
        daily_coverage = False
        if date_parseable and not frame.empty:
            expected_dates = pd.date_range(
                date_values.min(),
                date_values.max(),
                freq="D",
            )
            daily_coverage = len(expected_dates.difference(date_values)) == 0
        return {
            "date_parseable": date_parseable,
            "date_is_datetime": pd.api.types.is_datetime64_any_dtype(frame.index),
            "date_unique": date_values.is_unique,
            "date_sorted": date_values.is_monotonic_increasing,
            "daily_coverage": daily_coverage,
        }

    def validate_calendar(self, frame: pd.DataFrame) -> dict[str, bool]:
        """Validate calendar fields against the datetime index.

        Parameters
        ----------
        frame : pandas.DataFrame
            The daily table to inspect.

        Returns
        -------
        dict[str, bool]
            Calendar validation results.
        """
        date_values = pd.to_datetime(frame.index, errors="coerce")
        required_columns = {"wday", "month", "year"}
        has_required_columns = required_columns.issubset(frame.columns)
        date_parseable = date_values.notna().all()
        calendar_matches_date = False
        if date_parseable and not frame.empty and has_required_columns:
            calendar_matches_date = (
                frame["wday"].eq((date_values.dayofweek + 2) % 7 + 1).all()
                and frame["month"].eq(date_values.month).all()
                and frame["year"].eq(date_values.year).all()
            )
        return {
            "calendar_matches_date": calendar_matches_date,
            "calendar_fields_are_int64": all(
                column in frame.columns and frame[column].dtype == "int64"
                for column in self.calendar_columns
            ),
        }

    def validate_sales(self, frame: pd.DataFrame) -> dict[str, bool]:
        """Validate sales completeness, sign, and integer-like values.

        Parameters
        ----------
        frame : pandas.DataFrame
            The daily table to inspect.

        Returns
        -------
        dict[str, bool]
            Sales validation results.
        """
        has_sales = "sales" in frame.columns
        return {
            "sales_has_no_missing": has_sales and frame["sales"].notna().all(),
            "sales_nonnegative": has_sales and (frame["sales"] >= 0).all(),
            "sales_integer_like": has_sales
            and np.isclose(frame["sales"] % 1, 0).all(),
        }

    def validate_events(self, frame: pd.DataFrame) -> dict[str, bool]:
        """Validate event label completeness and string types.

        Parameters
        ----------
        frame : pandas.DataFrame
            The daily table to inspect.

        Returns
        -------
        dict[str, bool]
            Event validation results.
        """
        has_event_columns = set(self.event_columns).issubset(frame.columns)
        return {
            "event_labels_are_complete": has_event_columns
            and frame[list(self.event_columns)].notna().all().all(),
            "event_labels_are_strings": has_event_columns
            and all(
                frame[column].dropna().map(type).eq(str).all()
                for column in self.event_columns
            ),
        }

    def validate_data(
        self,
        frame: pd.DataFrame,
        enforce: bool = True,
    ) -> pd.Series:
        """Run all validation stages and optionally enforce the rules.

        Parameters
        ----------
        frame : pandas.DataFrame
            The daily table to validate.
        enforce : bool, default=True
            Raise an assertion when any validation result is false.

        Returns
        -------
        pandas.Series
            Named boolean validation results.
        """
        validation = pd.Series(
            {
                **self.validate_dates(frame),
                **self.validate_calendar(frame),
                **self.validate_sales(frame),
                **self.validate_events(frame),
            }
        )

        if enforce:
            assert validation.all(), "Data validation failed."
        return validation
