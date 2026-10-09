"""Nodes for model tuning and learned-model selection."""

import pandas as pd

from ml_model import Baseline, ModelTraining


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
) -> tuple[dict, pd.DataFrame]:
    """Tune the configured regressors using expanding time-series folds."""
    return ModelTraining(cv_splits=cv_splits).tune_models(X_train, y_train)


def select_complete_rows(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    raw_X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Align complete rows for models and unscaled naive baselines."""
    training = ModelTraining()
    complete_X_train, complete_y_train = training.select_complete_rows(X_train, y_train)
    complete_X_test, complete_y_test = training.select_complete_rows(X_test, y_test)
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
    return ModelTraining().compare_models(
        tuned_models,
        complete_X_train,
        complete_y_train,
        complete_X_test,
        complete_y_test,
        baseline_train_features=complete_raw_X_train,
        baseline_test_features=complete_raw_X_test,
    )


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

