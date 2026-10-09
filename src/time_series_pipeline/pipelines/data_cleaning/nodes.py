"""Nodes for cleaning the aggregated store table."""

import pandas as pd

from clean_data import CleanData


def clean_store_data(raw_store_data: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's existing datetime and event-label cleaning."""
    return fill_event_nulls(
        sort_by_datetime(convert_datetime(raw_store_data))
    )


def convert_datetime(raw_store_data: pd.DataFrame) -> pd.DataFrame:
    """Convert the store date column or index to datetime."""
    return CleanData().convert_datetime(raw_store_data)


def sort_by_datetime(datetime_store_data: pd.DataFrame) -> pd.DataFrame:
    """Sort the store table and set its datetime index."""
    return CleanData().sort_by_datetime(datetime_store_data)


def fill_event_nulls(sorted_store_data: pd.DataFrame) -> pd.DataFrame:
    """Fill expected missing event labels with ``No event``."""
    return CleanData().fill_event_nulls(sorted_store_data)

