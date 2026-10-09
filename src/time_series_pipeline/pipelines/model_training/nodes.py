"""Nodes for model tuning and learned-model selection."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

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
    """Build a configured regression estimator by registry name."""
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
    """Calculate train and test RMSE values for one model."""
    return {
        "train_rmse": mean_squared_error(y_true_train, predictions_train) ** 0.5,
        "test_rmse": mean_squared_error(y_true_test, predictions_test) ** 0.5,
    }


def build_candidate_models() -> list[str]:
    """Return the configured model registry names for this experiment."""
    return list(MODEL_FACTORIES)


def build_candidate_grids(candidate_models: list[str]) -> dict[str, dict]:
    """Return parameter grids for the configured candidate models."""
    return {
        model_name: MODEL_PARAM_GRIDS[model_name]
        for model_name in candidate_models
    }


def tune_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_models: list[str],
    candidate_grids: dict[str, dict],
    cv_splits: int,
) -> tuple[dict[str, BaseEstimator], pd.DataFrame]:
    """Tune candidate regressors using expanding time-series folds."""
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")

    complete_X_train, complete_y_train = _select_complete_model_rows(X_train, y_train)
    time_series_split = TimeSeriesSplit(n_splits=cv_splits)
    tuned_models: dict[str, BaseEstimator] = {}
    tuning_rows = []
    for model_name in candidate_models:
        print(f"  Tuning {model_name}...")
        grid_search = GridSearchCV(
            estimator=build_model(model_name),
            param_grid=candidate_grids[model_name],
            scoring="neg_root_mean_squared_error",
            cv=time_series_split,
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(complete_X_train, complete_y_train)
        tuned_models[model_name] = grid_search.best_estimator_
        tuning_rows.append(
            {
                "model": model_name,
                "cv_rmse": -grid_search.best_score_,
                "best_params": grid_search.best_params_,
            }
        )
        print(f"    Best CV RMSE: {-grid_search.best_score_:.2f}")

    return tuned_models, pd.DataFrame(tuning_rows).sort_values("cv_rmse")


def select_complete_rows(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    raw_X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Align complete rows for models and unscaled naive baselines."""
    complete_X_train, complete_y_train = _select_complete_model_rows(X_train, y_train)
    complete_X_test, complete_y_test = _select_complete_model_rows(X_test, y_test)
    complete_raw_X_train = raw_X_train.loc[complete_X_train.index]
    complete_raw_X_test = raw_X_test.loc[complete_X_test.index]
    return (
        complete_X_train,
        complete_y_train,
        complete_raw_X_train,
        complete_X_test,
        complete_y_test,
        complete_raw_X_test,
    )


def _select_complete_model_rows(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep feature rows with complete inputs and known targets."""
    complete_rows = features.notna().all(axis=1) & target.notna()
    return features.loc[complete_rows], target.loc[complete_rows]


def compare_models(
    tuned_models: dict[str, BaseEstimator],
    complete_X_train: pd.DataFrame,
    complete_y_train: pd.Series,
    complete_raw_X_train: pd.DataFrame,
    complete_X_test: pd.DataFrame,
    complete_y_test: pd.Series,
    complete_raw_X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Compare tuned models with one-day and one-week naive baselines."""
    comparison_rows = []
    for baseline_name, feature_name in {
        "naive_1_day": "sales_lag_1_day",
        "naive_1_week": "sales_lag_1_week",
    }.items():
        comparison_rows.append(
            {
                "model": baseline_name,
                **calculate_metrics(
                    complete_y_train,
                    complete_raw_X_train[feature_name].to_numpy(),
                    complete_y_test,
                    complete_raw_X_test[feature_name].to_numpy(),
                ),
            }
        )
    for model_name, model in tuned_models.items():
        comparison_rows.append(
            {
                "model": model_name,
                **calculate_metrics(
                    complete_y_train,
                    model.predict(complete_X_train),
                    complete_y_test,
                    model.predict(complete_X_test),
                ),
            }
        )
    return pd.DataFrame(comparison_rows).sort_values("test_rmse")


def select_best_model(
    tuned_models: dict[str, BaseEstimator],
    model_comparison: pd.DataFrame,
) -> tuple[BaseEstimator, str]:
    """Select the best learned model by holdout test RMSE."""
    learned = model_comparison.loc[
        ~model_comparison["model"].isin({"naive_1_day", "naive_1_week"})
    ]
    model_name = learned.iloc[0]["model"]
    return tuned_models[model_name], model_name


def fit_baselines() -> tuple[str, str]:
    """Return the lag columns used by the naive baseline forecasts."""
    return "sales_lag_1_day", "sales_lag_1_week"
