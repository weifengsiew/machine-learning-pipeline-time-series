"""Kedro pipeline for model tuning and model selection."""

from kedro.pipeline import Pipeline, node

from .nodes import (
    build_candidate_grids,
    build_candidate_models,
    compare_models,
    fit_baselines,
    select_best_model,
    select_complete_rows,
    tune_candidate_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model-training pipeline."""
    return Pipeline(
        [
            node(
                func=build_candidate_models,
                inputs=None,
                outputs="candidate_models",
                name="build_candidate_models",
            ),
            node(
                func=build_candidate_grids,
                inputs="candidate_models",
                outputs="candidate_grids",
                name="build_candidate_grids",
            ),
            node(
                func=tune_candidate_models,
                inputs=[
                    "X_train",
                    "y_train",
                    "candidate_models",
                    "candidate_grids",
                    "params:cv_splits",
                ],
                outputs=["tuned_models", "tuning_results"],
                name="tune_candidate_models",
            ),
            node(
                func=select_complete_rows,
                inputs=[
                    "X_train",
                    "y_train",
                    "raw_X_train",
                    "X_test",
                    "y_test",
                    "raw_X_test",
                ],
                outputs=[
                    "complete_X_train",
                    "complete_y_train",
                    "complete_raw_X_train",
                    "complete_X_test",
                    "complete_y_test",
                    "complete_raw_X_test",
                ],
                name="select_complete_rows",
            ),
            node(
                func=compare_models,
                inputs=[
                    "tuned_models",
                    "complete_X_train",
                    "complete_y_train",
                    "complete_raw_X_train",
                    "complete_X_test",
                    "complete_y_test",
                    "complete_raw_X_test",
                ],
                outputs="model_comparison",
                name="compare_models",
            ),
            node(
                func=select_best_model,
                inputs=["tuned_models", "model_comparison"],
                outputs=["selected_model", "selected_model_name"],
                name="select_best_model",
            ),
            node(
                func=fit_baselines,
                inputs=None,
                outputs=["baseline_one_day", "baseline_one_week"],
                name="fit_baselines",
            ),
        ]
    )

