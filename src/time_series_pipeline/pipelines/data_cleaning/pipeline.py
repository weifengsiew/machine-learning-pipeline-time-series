"""Kedro pipeline for cleaning and scoping the store-level table."""

from kedro.pipeline import Pipeline, node

from .nodes import convert_datetime, fill_event_nulls, filter_start_date, sort_by_datetime


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data-cleaning pipeline."""
    return Pipeline(
        [
            node(
                func=convert_datetime,
                inputs="raw_store_data",
                outputs="datetime_store_data",
                name="convert_datetime",
            ),
            node(
                func=sort_by_datetime,
                inputs="datetime_store_data",
                outputs="sorted_store_data",
                name="sort_by_datetime",
            ),
            node(
                func=fill_event_nulls,
                inputs="sorted_store_data",
                outputs="event_cleaned_store_data",
                name="fill_event_nulls",
            ),
            node(
                func=filter_start_date,
                inputs=["event_cleaned_store_data", "params:start_date"],
                outputs="cleaned_store_data",
                name="filter_start_date",
            ),
        ]
    )

