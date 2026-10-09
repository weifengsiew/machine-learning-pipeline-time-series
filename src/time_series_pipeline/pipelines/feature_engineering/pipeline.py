"""Kedro pipeline for post-June-2012 feature engineering."""

from kedro.pipeline import Pipeline, node

from .nodes import engineer_features, filter_start_date


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature-engineering pipeline."""
    return Pipeline(
        [
            node(
                func=filter_start_date,
                inputs=["cleaned_store_data", "params:start_date"],
                outputs="filtered_cleaned_store_data",
                name="filter_start_date",
            ),
            node(
                func=engineer_features,
                inputs=["filtered_cleaned_store_data", "params:horizon"],
                outputs="engineered_data",
                name="engineer_features",
            ),
        ]
    )

