"""Kedro pipeline for store-level data ingestion."""

from kedro.pipeline import Pipeline, node

from .nodes import load_store_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data-ingestion pipeline."""
    return Pipeline(
        [
            node(
                func=load_store_data,
                inputs=["params:store_id", "params:data_dir"],
                outputs="raw_store_data",
                name="load_store_data",
            )
        ]
    )

