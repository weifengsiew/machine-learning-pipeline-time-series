"""Kedro pipeline for fitting and applying model preprocessing."""

from kedro.pipeline import Pipeline, node

from .nodes import (
    build_training_preprocessor,
    fit_transform_training_data,
    transform_test_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model preprocessing pipeline."""
    return Pipeline(
        [
            node(
                func=build_training_preprocessor,
                inputs="raw_X_train",
                outputs="unfitted_preprocessor",
                name="build_training_preprocessor",
            ),
            node(
                func=fit_transform_training_data,
                inputs=["raw_X_train", "unfitted_preprocessor"],
                outputs=["X_train", "fitted_preprocessor"],
                name="fit_transform_training_data",
            ),
            node(
                func=transform_test_data,
                inputs=["raw_X_test", "fitted_preprocessor"],
                outputs="X_test",
                name="transform_test_data",
            ),
        ]
    )
