"""Kedro pipeline for Great Expectations validation evidence."""

from kedro.pipeline import Pipeline, node

from .expectations import build_expectation_suite
from .nodes import validate_cleaned_data, validate_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    """Apply one shared suite to raw data and cleaned data."""
    return Pipeline(
        [
            node(
                func=build_expectation_suite,
                inputs=None,
                outputs="expectation_suite",
                name="build_expectation_suite",
            ),
            node(
                func=validate_raw_data,
                inputs=["raw_store_data", "expectation_suite"],
                outputs=["raw_validation_result", "validation_before_cleaning"],
                name="validate_raw_data",
            ),
            node(
                func=validate_cleaned_data,
                inputs=["cleaned_store_data", "expectation_suite"],
                outputs=["cleaned_validation_result", "validation_after_cleaning"],
                name="validate_cleaned_data",
            ),
        ]
    )

