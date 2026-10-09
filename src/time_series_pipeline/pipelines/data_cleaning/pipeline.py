"""Kedro pipeline for cleaning the store-level table."""

from kedro.pipeline import Pipeline, node

from .nodes import clean_store_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data-cleaning pipeline."""
    return Pipeline(
        [
            node(
                func=clean_store_data,
                inputs="raw_store_data",
                outputs="cleaned_store_data",
                name="clean_store_data",
            )
        ]
    )

