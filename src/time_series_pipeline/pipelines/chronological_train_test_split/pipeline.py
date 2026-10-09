"""Kedro pipeline for chronological model-data splitting."""

from kedro.pipeline import Pipeline, node

from .nodes import create_train_test_split, split_features_and_target


def create_pipeline(**kwargs) -> Pipeline:
    """Create the chronological train/test split pipeline."""
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
        ]
    )
