"""Kedro pipeline for holdout forecasts and reporting."""

from kedro.pipeline import Pipeline, node

from .nodes import build_metrics, create_forecast_errors, create_plots


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model-evaluation pipeline."""
    return Pipeline(
        [
            node(
                func=create_forecast_errors,
                inputs=[
                    "selected_model",
                    "baseline_one_day",
                    "baseline_one_week",
                    "complete_X_test",
                    "complete_y_test",
                    "complete_raw_X_test",
                ],
                outputs="forecast_errors",
                name="create_forecast_errors",
            ),
            node(
                func=create_plots,
                inputs=["forecast_errors", "model_comparison", "selected_model_name"],
                outputs=[
                    "forecast_plot",
                    "predicted_vs_actual_plot",
                    "model_comparison_plot",
                ],
                name="create_plots",
            ),
            node(
                func=build_metrics,
                inputs=["model_comparison", "tuning_results", "selected_model_name"],
                outputs="metrics",
                name="build_metrics",
            ),
        ]
    )

