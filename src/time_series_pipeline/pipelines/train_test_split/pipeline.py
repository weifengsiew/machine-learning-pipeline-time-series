"""Kedro pipeline for chronological model-data preparation."""

from kedro.pipeline import Pipeline, node

from .nodes import (
    build_training_preprocessor,
    create_train_test_split,
    fit_transform_training_data,
    split_features_and_target,
    transform_test_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the train/test and preprocessing pipeline."""
    return Pipeline(
        [
            node(
                func=split_features_and_target,
                inputs="engineered_data",
                outputs=["features", "target"],
                name="split_features_and_target",
            ),
            node(
                func=create_train_test_split,
                inputs=["features", "target", "params:test_start_date"],
                outputs=["raw_X_train", "raw_X_test", "y_train", "y_test"],
                name="create_train_test_split",
            ),
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

