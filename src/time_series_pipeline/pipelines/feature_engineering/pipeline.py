"""Kedro pipeline for post-June-2012 feature engineering."""

from kedro.pipeline import Pipeline, node

from .nodes import (
    add_cyclical_features,
    add_event_indicator,
    add_lag_features,
    add_rolling_features,
    add_target,
    align_target_calendar_features,
    filter_start_date,
    select_base_features,
)


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
                func=add_event_indicator,
                inputs="filtered_cleaned_store_data",
                outputs="event_features",
                name="add_event_indicator",
            ),
            node(
                func=select_base_features,
                inputs="event_features",
                outputs="base_features",
                name="select_base_features",
            ),
            node(
                func=add_lag_features,
                inputs=["base_features", "params:horizon"],
                outputs="lag_features",
                name="add_lag_features",
            ),
            node(
                func=add_cyclical_features,
                inputs="lag_features",
                outputs="cyclical_features",
                name="add_cyclical_features",
            ),
            node(
                func=add_rolling_features,
                inputs=["cyclical_features", "params:horizon"],
                outputs="rolling_features",
                name="add_rolling_features",
            ),
            node(
                func=align_target_calendar_features,
                inputs=["rolling_features", "params:horizon"],
                outputs="aligned_features",
                name="align_target_calendar_features",
            ),
            node(
                func=add_target,
                inputs=["aligned_features", "params:horizon"],
                outputs="engineered_data",
                name="add_target",
            ),
        ]
    )

