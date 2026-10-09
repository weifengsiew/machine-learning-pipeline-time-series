"""Nodes for cleaning and scoping the aggregated store table."""

import pandas as pd

DEFAULT_EVENT_COLUMNS = (
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
)


def convert_datetime(raw_store_data: pd.DataFrame) -> pd.DataFrame:
    """Convert the store date column or index to datetime."""
    cleaned = raw_store_data.copy()
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    else:
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
    return cleaned


def sort_by_datetime(datetime_store_data: pd.DataFrame) -> pd.DataFrame:
    """Sort the store table and set its datetime index."""
    if "date" in datetime_store_data.columns:
        sorted_data = datetime_store_data.sort_values("date").set_index("date")
    else:
        sorted_data = datetime_store_data.sort_index()
    sorted_data.index.name = "datetime"
    return sorted_data


def fill_event_nulls(sorted_store_data: pd.DataFrame) -> pd.DataFrame:
    """Fill expected missing event labels with ``No event``."""
    cleaned = sorted_store_data.copy()
    for column in DEFAULT_EVENT_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("No event")
    return cleaned


def filter_start_date(
    event_cleaned_store_data: pd.DataFrame,
    start_date: str,
) -> pd.DataFrame:
    """Retain observations from the configured post-June-2012 boundary."""
    return event_cleaned_store_data.loc[
        event_cleaned_store_data.index >= pd.Timestamp(start_date)
    ].copy()

