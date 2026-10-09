"""Nodes for model tuning and learned-model selection."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from ml_model import (
    MODEL_FACTORIES,
    MODEL_PARAM_GRIDS,
    Baseline,
    ForecastModel,
    build_model,
)


def build_candidate_models() -> list[str]:
    """Return the configured model registry names for this experiment."""
    return list(MODEL_FACTORIES)


def build_candidate_grids(candidate_models: list[str]) -> dict[str, dict]:
    """Return parameter grids for the configured candidate models."""
    return {
        model_name: MODEL_PARAM_GRIDS[model_name]
        for model_name in candidate_models
    }


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
) -> tuple[dict, pd.DataFrame]:
    """Tune the configured regressors using expanding time-series folds."""
    candidate_models = build_candidate_models()
    return tune_candidate_models(
        X_train,
        y_train,
        candidate_models,
        build_candidate_grids(candidate_models),
        cv_splits,
    )


def tune_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_models: list[str],
    candidate_grids: dict[str, dict],
    cv_splits: int,
) -> tuple[dict, pd.DataFrame]:
    """Tune candidate regressors using expanding time-series folds."""
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")

    complete_X_train, complete_y_train = _select_complete_model_rows(X_train, y_train)
    time_series_split = TimeSeriesSplit(n_splits=cv_splits)
    tuned_models = {}
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
    tuned_models: dict,
    complete_X_train: pd.DataFrame,
    complete_y_train: pd.Series,
    complete_raw_X_train: pd.DataFrame,
    complete_X_test: pd.DataFrame,
    complete_y_test: pd.Series,
    complete_raw_X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Compare tuned models with one-day and one-week naive baselines."""
    baselines = {
        "naive_1_day": Baseline("sales_lag_1_day"),
        "naive_1_week": Baseline("sales_lag_1_week"),
    }
    comparison_rows = []
    for baseline_name, baseline in baselines.items():
        baseline.fit(complete_raw_X_train, complete_y_train)
        comparison_rows.append(
            {
                "model": baseline_name,
                **baseline.evaluate(
                    complete_raw_X_train,
                    complete_y_train,
                    complete_raw_X_test,
                    complete_y_test,
                ),
            }
        )
    for model_name, model in tuned_models.items():
        comparison_rows.append(
            {
                "model": model_name,
                **model.evaluate(
                    complete_X_train,
                    complete_y_train,
                    complete_X_test,
                    complete_y_test,
                ),
            }
        )
    return pd.DataFrame(comparison_rows).sort_values("test_rmse")


def select_best_model(
    tuned_models: dict,
    model_comparison: pd.DataFrame,
) -> tuple[object, str]:
    """Select the best learned model by holdout test RMSE."""
    learned = model_comparison.loc[
        ~model_comparison["model"].isin({"naive_1_day", "naive_1_week"})
    ]
    model_name = learned.iloc[0]["model"]
    return tuned_models[model_name], model_name


def fit_baselines(
    complete_raw_X_train: pd.DataFrame,
    complete_y_train: pd.Series,
) -> tuple[Baseline, Baseline]:
    """Fit the one-day and one-week naive baselines on raw features."""
    one_day = Baseline("sales_lag_1_day").fit(complete_raw_X_train, complete_y_train)
    one_week = Baseline("sales_lag_1_week").fit(complete_raw_X_train, complete_y_train)
    return one_day, one_week

