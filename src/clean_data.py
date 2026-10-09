"""Clean daily store-sales data before validation and feature engineering.

Inputs
------
DataFrame
    A daily table with either a ``date`` column or a datetime-like index.

Outputs
-------
pandas.DataFrame
    A sorted table with one datetime index, completed event labels, and a
    consistent datetime representation.

Stages
------
1. Convert and sort the time axis.
2. Represent expected event nulls as ``No event``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

DEFAULT_EVENT_COLUMNS = (
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
)


class CleanData:
    """Own the configuration and actions used to clean one daily time series."""

    def __init__(self, event_columns: Sequence[str] | None = None) -> None:
        """Initialize the cleaner with the event columns to complete.

        Parameters
        ----------
        event_columns : sequence of str, optional
            Event name and event type columns. The M5 event columns are used
            when no value is supplied.
        """
        self.event_columns = tuple(event_columns or DEFAULT_EVENT_COLUMNS)

    def convert_datetime(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert the input date column or index to datetime.

        Parameters
        ----------
        frame : pandas.DataFrame
            The uncleaned daily table.

        Returns
        -------
        pandas.DataFrame
            A copy with a datetime date column or index.
        """
        cleaned = frame.copy()
        if "date" in cleaned.columns:
            cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        else:
            cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
        return cleaned

    def sort_by_datetime(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Sort the table and use one datetime index.

        Parameters
        ----------
        frame : pandas.DataFrame
            A table after datetime conversion.

        Returns
        -------
        pandas.DataFrame
            A chronologically sorted table indexed by ``datetime``.
        """
        if "date" in frame.columns:
            sorted_frame = frame.sort_values("date").set_index("date")
        else:
            sorted_frame = frame.sort_index()
        sorted_frame.index.name = "datetime"
        return sorted_frame

    def fill_event_nulls(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Represent expected missing event labels as ``No event``.

        Parameters
        ----------
        frame : pandas.DataFrame
            A table containing event name and event type columns.

        Returns
        -------
        pandas.DataFrame
            A copy with missing configured event labels filled.
        """
        cleaned = frame.copy()
        for column in self.event_columns:
            if column in cleaned.columns:
                cleaned[column] = cleaned[column].fillna("No event")
        return cleaned

    def clean_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Run the cleaning stages in order.

        Parameters
        ----------
        frame : pandas.DataFrame
            The raw daily store-sales table.

        Returns
        -------
        pandas.DataFrame
            The cleaned, sorted daily table.
        """
        cleaned = self.convert_datetime(frame)
        cleaned = self.sort_by_datetime(cleaned)
        return self.fill_event_nulls(cleaned)
