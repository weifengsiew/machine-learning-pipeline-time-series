"""Kedro pipeline for validation evidence."""

from kedro.pipeline import Pipeline, node

from .nodes import validate_store_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data-validation pipeline."""
    return Pipeline(
        [
            node(
                func=validate_store_data,
                inputs=["raw_store_data", "cleaned_store_data"],
                outputs=["validation_before_cleaning", "validation_after_cleaning"],
                name="validate_store_data",
            )
        ]
    )

