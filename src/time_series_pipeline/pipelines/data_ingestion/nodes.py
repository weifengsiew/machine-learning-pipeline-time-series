"""Nodes for aggregating the M5 files to one store-level daily table."""

from pathlib import Path

import pandas as pd

VALID_STORES = {
    "CA_1": "CA", "CA_2": "CA", "CA_3": "CA", "CA_4": "CA",
    "TX_1": "TX", "TX_2": "TX", "TX_3": "TX",
    "WI_1": "WI", "WI_2": "WI", "WI_3": "WI",
}


def load_store_data(
    store_id: str,
    data_dir: str,
) -> pd.DataFrame:
    """Load and aggregate one store without writing a side effect."""
    store_id = store_id.strip().upper()
    if store_id not in VALID_STORES:
        raise ValueError(
            f"Unknown store_id {store_id!r}. "
            f"Choose one of: {', '.join(VALID_STORES)}"
        )

    data_dir = Path(data_dir)
    sales_path = data_dir / "sales_train_evaluation.csv"
    calendar_path = data_dir / "calendar.csv"
    for path in (sales_path, calendar_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {path}. Point data_dir at the folder holding the M5 CSVs."
            )

    sales = pd.read_csv(sales_path)
    sales = sales[sales["store_id"] == store_id]
    if sales.empty:
        raise ValueError(f"No rows found for store {store_id!r} in {sales_path}.")

    day_columns = [column for column in sales.columns if column.startswith("d_")]
    store_total = sales[day_columns].sum(axis=0)
    store_total.name = "sales"
    store_total.index.name = "d"

    calendar = pd.read_csv(calendar_path)
    if "d" not in calendar.columns:
        calendar.insert(
            0,
            "d",
            [f"d_{day_number}" for day_number in range(1, len(calendar) + 1)],
        )
    snap_column = f"snap_{VALID_STORES[store_id]}"
    calendar = calendar[
        [
            "d",
            "date",
            "wm_yr_wk",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            snap_column,
        ]
    ].rename(columns={snap_column: "snap"})

    daily_data = calendar.merge(store_total, on="d", how="inner")
    daily_data["date"] = pd.to_datetime(daily_data["date"])
    return daily_data.sort_values("date").set_index("date").drop(columns="d")

