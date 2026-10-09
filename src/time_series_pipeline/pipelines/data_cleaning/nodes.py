"""Nodes for cleaning the aggregated store table."""

import pandas as pd

from clean_data import CleanData


def clean_store_data(raw_store_data: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's existing datetime and event-label cleaning."""
    return CleanData().clean_data(raw_store_data)

