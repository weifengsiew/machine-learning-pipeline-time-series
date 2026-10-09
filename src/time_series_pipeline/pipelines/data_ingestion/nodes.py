"""Nodes for aggregating the M5 files to one store-level daily table."""

from pathlib import Path

import pandas as pd

from prepare_store_data import prepare_store_data


def load_store_data(
    store_id: str,
    data_dir: str,
) -> pd.DataFrame:
    """Load and aggregate the configured store without writing a side effect."""
    return prepare_store_data(
        store_id=store_id,
        data_dir=Path(data_dir),
        save=False,
    )

