"""Define, train, and evaluate the time-series forecasting models."""

import pickle
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ModelFactory = Callable[[], BaseEstimator]

MODEL_FACTORIES: dict[str, ModelFactory] = {
    "linear_regression": LinearRegression,
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=100,
        n_jobs=-1,
        random_state=0,
    ),
    "neural_network": lambda: MLPRegressor(
        max_iter=1000,
        random_state=0,
    ),
    "knn": KNeighborsRegressor,
}

MODEL_PARAM_GRIDS: dict[str, dict[str, list[object]]] = {
    "linear_regression": {
        "fit_intercept": [True, False],
        "positive": [False, True],
        "tol": [1e-6, 1e-4],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [None, 20],
    },
    "neural_network": {
        "hidden_layer_sizes": [
            (32,),
            (64,),
            (32, 16),
            (64, 32),
            (64, 32, 16),
            (128, 64, 32),
        ],
        "alpha": [0.0001, 0.001],
    },
    "knn": {
        "n_neighbors": [1, 3, 7, 14, 28],
        "weights": ["uniform", "distance"],
    },
}


def build_model(model_name: str) -> BaseEstimator:
    """Build a configured regression estimator by registry name.

    Parameters
    ----------
    model_name : str
        Name of a model registered in ``MODEL_FACTORIES``.

    Returns
    -------
    sklearn.base.BaseEstimator
        A new unfitted estimator.

    Raises
    ------
    ValueError
        If ``model_name`` is not registered.
    """
    if model_name not in MODEL_FACTORIES:
        available_models = ", ".join(sorted(MODEL_FACTORIES))
        raise ValueError(f"Unknown model {model_name!r}. Choose from {available_models}.")
    return MODEL_FACTORIES[model_name]()


def calculate_metrics(
    y_true_train: pd.Series,
    predictions_train: np.ndarray,
    y_true_test: pd.Series,
    predictions_test: np.ndarray,
) -> dict[str, float]:
    """Calculate train and test RMSE values for one model.

    Parameters
    ----------
    y_true_train, y_true_test : pandas.Series
        Observed target values for the training and test partitions.
    predictions_train, predictions_test : numpy.ndarray
        Model predictions aligned with the corresponding target values.

    Returns
    -------
    dict[str, float]
        ``train_rmse`` and ``test_rmse`` values.
    """
    return {
        "train_rmse": mean_squared_error(y_true_train, predictions_train) ** 0.5,
        "test_rmse": mean_squared_error(y_true_test, predictions_test) ** 0.5,
    }


class Baseline:
    """Forecast using a configured lagged-sales feature."""

    def __init__(self, lag_feature: str = "sales_lag_1_week") -> None:
        """Configure the lagged-sales column used as the forecast."""
        self.lag_feature = lag_feature

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Baseline":
        """Validate the baseline feature and return the fitted baseline."""
        del y
        if self.lag_feature not in X.columns:
            raise KeyError(f"Missing baseline feature: {self.lag_feature}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the configured lagged-sales feature as the forecast."""
        if self.lag_feature not in X.columns:
            raise KeyError(f"Missing baseline feature: {self.lag_feature}")
        return X[self.lag_feature].to_numpy()

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Return RMSE for the baseline on train and test data."""
        return calculate_metrics(
            y_train,
            self.predict(X_train),
            y_test,
            self.predict(X_test),
        )

    def save(self, path: Path) -> None:
        """Save the fitted baseline to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as model_file:
            pickle.dump(self, model_file)

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        """Load a saved baseline model."""
        with path.open("rb") as model_file:
            model = pickle.load(model_file)
        if not isinstance(model, cls):
            raise TypeError(f"Expected a {cls.__name__} artifact, got {type(model).__name__}")
        return model


class ForecastModel:
    """Wrap a scikit-learn-compatible regression estimator."""

    def __init__(self, estimator: BaseEstimator | None = None) -> None:
        """Initialize the wrapper with an estimator or the default forest."""
        self.model = estimator if estimator is not None else RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=0
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecastModel":
        """Fit the wrapped estimator and return this model wrapper.

        Parameters
        ----------
        X : pandas.DataFrame
            Training feature rows.
        y : pandas.Series
            Training target values aligned with ``X``.

        Returns
        -------
        ForecastModel
            This fitted wrapper.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions for feature rows."""
        return self.model.predict(X)

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Return train and test RMSE values.

        Parameters
        ----------
        X_train, X_test : pandas.DataFrame
            Complete feature rows for training and evaluation.
        y_train, y_test : pandas.Series
            Targets aligned with the feature rows.

        Returns
        -------
        dict[str, float]
            Train and test RMSE values.
        """
        train_predictions = self.predict(X_train)
        test_predictions = self.predict(X_test)
        return calculate_metrics(
            y_train,
            train_predictions,
            y_test,
            test_predictions,
        )

    def save(self, path: Path) -> None:
        """Save the fitted forecasting model to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as model_file:
            pickle.dump(self, model_file)

    @classmethod
    def load(cls, path: Path) -> "ForecastModel":
        """Load a saved forecasting model."""
        with path.open("rb") as model_file:
            model = pickle.load(model_file)
        if not isinstance(model, cls):
            raise TypeError(f"Expected a {cls.__name__} artifact, got {type(model).__name__}")
        return model


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
        (
            "baseline_naive_1_week_prediction",
            "Naive 1 Week",
        ),
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
    axis.plot(
        line_values,
        line_values,
        "--",
        color="#2f6da8",
        label="Perfect prediction",
    )
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


class ModelTraining:
    """Tune, compare, and evaluate the configured regression models."""

    def __init__(
        self,
        cv_splits: int = 5,
        scoring: str = "neg_root_mean_squared_error",
    ) -> None:
        """Initialize time-series cross-validation settings.

        Parameters
        ----------
        cv_splits : int, default=5
            Number of expanding-window validation splits.
        scoring : str, default="neg_root_mean_squared_error"
            Scikit-learn scoring name used during grid search.
        """
        if cv_splits < 2:
            raise ValueError("cv_splits must be at least 2")
        self.cv_splits = cv_splits
        self.scoring = scoring

    def select_complete_rows(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Keep feature rows with complete inputs and known targets.

        Parameters
        ----------
        features : pandas.DataFrame
            Candidate model features.
        target : pandas.Series
            Target values aligned with ``features``.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.Series]
            Feature rows and targets retained after dropping incomplete rows.
        """
        complete_rows = features.notna().all(axis=1) & target.notna()
        return features.loc[complete_rows], target.loc[complete_rows]

    def tune_models(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[dict[str, ForecastModel], pd.DataFrame]:
        """Tune all registered models with expanding time-series validation.

        Parameters
        ----------
        features : pandas.DataFrame
            Candidate model features.
        target : pandas.Series
            Target values aligned with ``features``.

        Returns
        -------
        tuple[dict[str, ForecastModel], pandas.DataFrame]
            Tuned model wrappers and a table of best cross-validation RMSE
            values and parameters.
        """
        complete_features, complete_target = self.select_complete_rows(features, target)
        time_series_split = TimeSeriesSplit(n_splits=self.cv_splits)
        tuned_models = {}
        tuning_rows = []

        for model_name in MODEL_FACTORIES:
            print(f"  Tuning {model_name}...")
            grid_search = GridSearchCV(
                estimator=build_model(model_name),
                param_grid=MODEL_PARAM_GRIDS[model_name],
                scoring=self.scoring,
                cv=time_series_split,
                n_jobs=-1,
                refit=True,
            )
            grid_search.fit(complete_features, complete_target)
            tuned_models[model_name] = ForecastModel(
                estimator=grid_search.best_estimator_
            )
            tuning_rows.append(
                {
                    "model": model_name,
                    "cv_rmse": -grid_search.best_score_,
                    "best_params": grid_search.best_params_,
                }
            )
            print(f"    Best CV RMSE: {-grid_search.best_score_:.2f}")

        tuning_results = pd.DataFrame(tuning_rows).sort_values("cv_rmse")
        return tuned_models, tuning_results

    def compare_models(
        self,
        models: dict[str, ForecastModel],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        *,
        baseline_train_features: pd.DataFrame | None = None,
        baseline_test_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compare trained models with one-day and one-week baselines.

        Parameters
        ----------
        models : dict[str, ForecastModel]
            Fitted model wrappers keyed by model name.
        X_train, X_test : pandas.DataFrame
            Training and test features.
        y_train, y_test : pandas.Series
            Targets aligned with the corresponding feature partitions.
        baseline_train_features, baseline_test_features : pandas.DataFrame, optional
            Unscaled feature partitions for naive baselines. If omitted, the
            model feature partitions are used.

        Returns
        -------
        pandas.DataFrame
            One RMSE row for every learned model and baseline, sorted by test
            RMSE.
        """
        if (baseline_train_features is None) != (baseline_test_features is None):
            raise ValueError(
                "baseline_train_features and baseline_test_features must be supplied together"
            )
        if baseline_train_features is None:
            baseline_train_features = X_train
            baseline_test_features = X_test

        baselines = {
            "naive_1_day": Baseline("sales_lag_1_day"),
            "naive_1_week": Baseline("sales_lag_1_week"),
        }
        comparison_rows = []
        for baseline_name, baseline in baselines.items():
            baseline.fit(baseline_train_features, y_train)
            comparison_rows.append(
                {
                    "model": baseline_name,
                    **baseline.evaluate(
                        baseline_train_features,
                        y_train,
                        baseline_test_features,
                        y_test,
                    ),
                }
            )
        for model_name, model in models.items():
            comparison_rows.append(
                {
                    "model": model_name,
                    **model.evaluate(X_train, y_train, X_test, y_test),
                }
            )
        return pd.DataFrame(comparison_rows).sort_values("test_rmse")
