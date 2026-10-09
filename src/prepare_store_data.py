"""
prepare_store_data.py
======================

Slice the full Walmart **M5** dataset down to the daily series for **one store**.

The M5 competition data ships as several large CSVs (sales are ~128 MB, prices
~208 MB) covering 10 stores and 3,049 items each. For this exercise every group
only needs *their own* store, aggregated to a single **store-total daily** series
with a bit of calendar context. This script does exactly that:

    pass a store_id string  ->  get the small per-store table you should use.

What it produces (one row per day):
    date            the calendar date (use this as your DatetimeIndex)
    sales           store-total units sold that day  <-- forecasting target
    wday, month     day-of-week (1-7) and month (1-12) from the M5 calendar
    year
    event_name_1    named calendar event, if any (else empty)
    event_type_1    event category (Sporting / Cultural / National / Religious)
    event_name_2    a second event on the same day, if any
    event_type_2
    snap            1 if SNAP benefits were issued in THIS store's state that
                    day, else 0  (SNAP schedule differs by state -> CA/TX/WI)

Feature engineering (lags, rolling means, cyclical encodings, one-hot events...)
is deliberately left to the Kedro feature-engineering stage. This module only
hands the pipeline the clean raw material for the configured store.

--------------------------------------------------------------------------------
Command line
--------------------------------------------------------------------------------
    python src/prepare_store_data.py CA_1
    python src/prepare_store_data.py WI_2 --out-dir data/store_subsets

Importable
--------------------------------------------------------------------------------
    from src.prepare_store_data import prepare_store_data
    df = prepare_store_data("CA_3")          # returns a DataFrame, also cached
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# The 10 stores in M5, and the state each belongs to (state fixes the SNAP column).
VALID_STORES = {
    "CA_1": "CA", "CA_2": "CA", "CA_3": "CA", "CA_4": "CA",
    "TX_1": "TX", "TX_2": "TX", "TX_3": "TX",
    "WI_1": "WI", "WI_2": "WI", "WI_3": "WI",
}

# Default locations, relative to ts/ (the parent of src/).
DEFAULT_DATA_DIR = "data/m5-forecasting-accuracy"
DEFAULT_OUT_DIR = "data/store_subsets"


def prepare_store_data(
    store_id: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    save: bool = True,
) -> pd.DataFrame:
    """Build the store-total daily series for ``store_id`` and (optionally) cache it.

    Parameters
    ----------
    store_id : str
        One of CA_1..CA_4, TX_1..TX_3, WI_1..WI_3.
    data_dir : str or Path
        Folder holding the raw M5 CSVs (sales_train_evaluation.csv, calendar.csv).
    out_dir : str or Path
        Where to write the per-store CSV cache.
    save : bool
        If True, write ``{out_dir}/{store_id}_daily.csv``.

    Returns
    -------
    pandas.DataFrame
        Daily table described in the module docstring, indexed by ``date``.
    """
    store_id = store_id.strip().upper()
    if store_id not in VALID_STORES:
        raise ValueError(
            f"Unknown store_id {store_id!r}. "
            f"Choose one of: {', '.join(VALID_STORES)}"
        )
    state = VALID_STORES[store_id]

    data_dir = Path(data_dir)
    sales_path = data_dir / "sales_train_evaluation.csv"
    calendar_path = data_dir / "calendar.csv"
    for p in (sales_path, calendar_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected {p}. Point --data-dir at the folder holding the M5 CSVs."
            )

    # --- 1. Read sales, keep only this store's item rows -----------------------
    # The sales file is wide: one row per item, one column (d_1..d_1941) per day.
    # It is large (~128 MB); we read it once and filter to the store immediately.
    sales = pd.read_csv(sales_path)
    sales = sales[sales["store_id"] == store_id]
    if sales.empty:
        raise ValueError(f"No rows found for store {store_id!r} in {sales_path}.")

    # --- 2. Sum across all items -> one total per day (the d_* columns) --------
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    store_total = sales[day_cols].sum(axis=0)          # index = d_1..d_N, values = units
    store_total.name = "sales"
    store_total.index.name = "d"

    # --- 3. Join the calendar to turn d_* into real dates + context -----------
    calendar = pd.read_csv(calendar_path)
    if "d" not in calendar.columns:
        calendar.insert(
            0,
            "d",
            [f"d_{day_number}" for day_number in range(1, len(calendar) + 1)],
        )
    snap_col = f"snap_{state}"                          # e.g. snap_CA for a CA store
    keep = [
        "d", "date", "wm_yr_wk", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        snap_col,
    ]
    calendar = calendar[keep].rename(columns={snap_col: "snap"})

    df = calendar.merge(store_total, on="d", how="inner")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date").drop(columns="d")

    # --- 4. Cache and return ---------------------------------------------------
    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{store_id}_daily.csv"
        df.to_csv(out_path)
        print(f"[{store_id}] {len(df)} days "
              f"({df.index.min().date()} -> {df.index.max().date()}) "
              f"written to {out_path}")

    return df


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for store-data preparation."""
    parser = argparse.ArgumentParser(
        description="Slice the M5 dataset to one store's daily series."
    )
    parser.add_argument(
        "store_id",
        help="Store to extract, e.g. CA_1. One of: " + ", ".join(VALID_STORES),
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Folder with the raw M5 CSVs (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Where to write the cache (default: {DEFAULT_OUT_DIR})")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Resolve default relative paths against ts/ so it works
    # no matter which directory you run it from.
    here = Path(__file__).resolve().parent.parent
    data_dir = args.data_dir if Path(args.data_dir).is_absolute() else here / args.data_dir
    out_dir = args.out_dir if Path(args.out_dir).is_absolute() else here / args.out_dir
    prepare_store_data(args.store_id, data_dir=data_dir, out_dir=out_dir)
