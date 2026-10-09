"""Nodes for forecasts, metrics, and plots."""

from collections.abc import Sequence

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_forecast_frame(
    actual: pd.Series,
    predictions: Sequence[float] | np.ndarray,
    baseline_predictions: dict[str, Sequence[float] | np.ndarray] | None = None,
) -> pd.DataFrame:
    """Create an indexed table of forecasts and forecast errors."""
    dates = pd.to_datetime(actual.index)
    if len(actual) != len(predictions):
        raise ValueError("actual and predictions must have the same length")

    frame = pd.DataFrame(
        {
            "actual": actual.to_numpy(),
            "prediction": np.asarray(predictions),
        },
        index=dates,
    )
    frame.index.name = actual.index.name or "datetime"
    frame["error"] = frame["prediction"] - frame["actual"]
    if baseline_predictions is not None:
        for baseline_name, baseline_values in baseline_predictions.items():
            if len(actual) != len(baseline_values):
                raise ValueError(
                    f"actual and {baseline_name} baseline must have the same length"
                )
            prediction_column = f"baseline_{baseline_name}_prediction"
            error_column = f"baseline_{baseline_name}_error"
            frame[prediction_column] = np.asarray(baseline_values)
            frame[error_column] = frame[prediction_column] - frame["actual"]
    return frame


def plot_forecasts(
    forecast_frame: pd.DataFrame,
    title: str = "Test-period sales forecast",
    last_n_days: int | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot actual sales against each forecast for a time window."""
    required_columns = {"actual", "prediction"}
    missing_columns = required_columns - set(forecast_frame.columns)
    if missing_columns:
        raise KeyError(f"Missing forecast columns: {sorted(missing_columns)}")
    if last_n_days is not None and last_n_days < 1:
        raise ValueError("last_n_days must be at least 1")

    plot_frame = (
        forecast_frame.tail(last_n_days)
        if last_n_days is not None
        else forecast_frame
    )
    forecast_columns = [
        ("prediction", "Model"),
        ("baseline_naive_1_day_prediction", "Naive 1 Day"),
        ("baseline_naive_1_week_prediction", "Naive 1 Week"),
    ]
    available_forecasts = [
        (column, label) for column, label in forecast_columns if column in plot_frame
    ]
    figure, axes = plt.subplots(
        len(available_forecasts),
        1,
        figsize=(14, 4 * len(available_forecasts)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, (forecast_column, forecast_label) in zip(axes, available_forecasts):
        axis.plot(plot_frame.index, plot_frame["actual"], label="Actual")
        axis.plot(plot_frame.index, plot_frame[forecast_column], label=forecast_label)
        axis.set_title(f"Actual vs {forecast_label}")
        axis.set_ylabel("Sales")
        axis.legend()
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Date")
    figure.suptitle(title)
    figure.tight_layout()
    figure.autofmt_xdate()
    return figure, axes


def plot_predicted_vs_actual(
    actual: pd.Series,
    predictions: Sequence[float] | np.ndarray,
    title: str = "Predicted sales vs actual sales",
) -> tuple[Figure, Axes]:
    """Plot predicted sales against actual sales with error guides."""
    if len(actual) != len(predictions):
        raise ValueError("actual and predictions must have the same length")

    actual_values = actual.to_numpy(dtype=float)
    predicted_values = np.asarray(predictions, dtype=float)
    finite = np.isfinite(actual_values) & np.isfinite(predicted_values)
    if not finite.any():
        raise ValueError("actual and predictions must contain finite values")

    actual_values = actual_values[finite]
    predicted_values = predicted_values[finite]
    errors = actual_values - predicted_values
    lower_error, upper_error = np.percentile(errors, [5, 95])
    correlation = np.corrcoef(predicted_values, actual_values)[0, 1]
    plot_min = min(predicted_values.min(), actual_values.min())
    plot_max = max(predicted_values.max(), actual_values.max())
    line_values = np.array([plot_min, plot_max])

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(
        predicted_values,
        actual_values,
        alpha=0.35,
        color="#77a9d4",
        edgecolors="none",
        label="Observed sales",
    )
    axis.plot(line_values, line_values, "--", color="#2f6da8", label="Perfect prediction")
    axis.plot(
        line_values,
        line_values + lower_error,
        "--",
        color="#f28e2b",
        label="5th percentile error",
    )
    axis.plot(
        line_values,
        line_values + upper_error,
        "--",
        color="#e15759",
        label="95th percentile error",
    )
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel("Predicted sales")
    axis.set_ylabel("Actual sales")
    axis.set_xlim(plot_min, plot_max)
    axis.set_ylim(plot_min, plot_max)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend(loc="upper left")
    axis.text(
        0.70,
        0.97,
        f"Pearson r = {correlation:.4f}\n5th: {lower_error:.0f}  95th: {upper_error:.0f}",
        transform=axis.transAxes,
        va="top",
    )
    figure.tight_layout()
    return figure, axis


def plot_model_comparison(
    comparison: pd.DataFrame,
    title: str = "Model and baseline comparison",
) -> Figure:
    """Plot test RMSE for models and naive baselines."""
    required_columns = {"model", "test_rmse"}
    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        raise KeyError(f"Missing comparison columns: {sorted(missing_columns)}")

    ordered_comparison = comparison.sort_values("test_rmse")
    labels = ordered_comparison["model"].str.replace("_", " ").str.title()
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels, ordered_comparison["test_rmse"])
    axis.set_title("Test RMSE")
    axis.set_xlabel("RMSE")
    axis.set_ylabel("Model")
    axis.grid(axis="x", alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def create_forecast_errors(
    selected_model: BaseEstimator,
    baseline_one_day: str,
    baseline_one_week: str,
    complete_X_test: pd.DataFrame,
    complete_y_test: pd.Series,
    complete_raw_X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Create selected-model and baseline forecasts in original sales units."""
    return create_forecast_frame(
        complete_y_test,
        selected_model.predict(complete_X_test),
        {
            "naive_1_day": complete_raw_X_test[baseline_one_day].to_numpy(),
            "naive_1_week": complete_raw_X_test[baseline_one_week].to_numpy(),
        },
    )


def create_plots(
    forecast_errors: pd.DataFrame,
    model_comparison: pd.DataFrame,
    selected_model_name: str,
) -> tuple[object, object, object]:
    """Create the three report plots for the selected model."""
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
