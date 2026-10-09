"""Nodes for forecasts, metrics, and plots."""

import pandas as pd

from ml_model import (
    Baseline,
    ForecastModel,
    create_forecast_frame,
    plot_forecasts,
    plot_model_comparison,
    plot_predicted_vs_actual,
)


def create_forecast_errors(
    selected_model: ForecastModel,
    baseline_one_day: Baseline,
    baseline_one_week: Baseline,
    complete_X_test: pd.DataFrame,
    complete_y_test: pd.Series,
    complete_raw_X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Create selected-model and baseline forecasts in original sales units."""
    predictions = selected_model.predict(complete_X_test)
    return create_forecast_frame(
        complete_y_test,
        predictions,
        {
            "naive_1_day": baseline_one_day.predict(complete_raw_X_test),
            "naive_1_week": baseline_one_week.predict(complete_raw_X_test),
        },
    )


def create_plots(
    forecast_errors: pd.DataFrame,
    model_comparison: pd.DataFrame,
    selected_model_name: str,
) -> tuple[object, object, object]:
    """Create the same three report plots produced by the existing experiment."""
    forecast_plot, _ = plot_forecasts(
        forecast_errors,
        title=f"{selected_model_name} test-period forecast",
        last_n_days=90,
    )
    predicted_vs_actual_plot, _ = plot_predicted_vs_actual(
        forecast_errors["actual"],
        forecast_errors["prediction"],
    )
    model_comparison_plot = plot_model_comparison(model_comparison)
    return forecast_plot, predicted_vs_actual_plot, model_comparison_plot


def build_metrics(
    model_comparison: pd.DataFrame,
    tuning_results: pd.DataFrame,
    selected_model_name: str,
) -> dict[str, float | str]:
    """Build JSON-compatible metrics for the post-June-2012 experiment."""
    metrics: dict[str, float | str] = {"selected_model": selected_model_name}
    for row in model_comparison.itertuples(index=False):
        model_name = row.model.replace(" ", "_")
        metrics[f"{model_name}_train_rmse"] = float(row.train_rmse)
        metrics[f"{model_name}_test_rmse"] = float(row.test_rmse)
    best_cv_rmse = tuning_results.loc[
        tuning_results["model"] == selected_model_name,
        "cv_rmse",
    ].iloc[0]
    metrics["selected_model_cv_rmse"] = float(best_cv_rmse)
    return metrics

