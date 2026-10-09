"""Nodes for fitting and applying model preprocessing."""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from time_series_pipeline.pipelines.feature_engineering.nodes import TARGET_LAG_DAYS

DEFAULT_CATEGORICAL_FEATURES = ("event_name", "event_type")
DEFAULT_NUMERIC_FEATURES = (
    *TARGET_LAG_DAYS,
    "snap",
    "has_event",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "sales_rolling_mean_7_past",
    "sales_rolling_mean_28_past",
)


def build_preprocessor(
    features: pd.DataFrame,
    categorical_features: Sequence[str] = DEFAULT_CATEGORICAL_FEATURES,
    numeric_features: Sequence[str] = DEFAULT_NUMERIC_FEATURES,
) -> ColumnTransformer:
    """Build an unfitted preprocessor from the available feature columns."""
    categorical_columns = [
        column for column in categorical_features if column in features.columns
    ]
    numeric_columns = [
        column for column in numeric_features if column in features.columns
    ]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("numeric", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_training_preprocessor(raw_X_train: pd.DataFrame):
    """Build the unfitted project preprocessor from training columns."""
    return build_preprocessor(raw_X_train)


def fit_transform_training_data(
    raw_X_train: pd.DataFrame,
    preprocessor,
) -> tuple[pd.DataFrame, object]:
    """Fit preprocessing only on training data and return its fitted state."""
    transformed = preprocessor.fit_transform(raw_X_train)
    columns = preprocessor.get_feature_names_out()
    X_train = pd.DataFrame(transformed, columns=columns, index=raw_X_train.index)
    return X_train, preprocessor


def transform_test_data(
    raw_X_test: pd.DataFrame,
    fitted_preprocessor,
) -> pd.DataFrame:
    """Transform the test partition using training-fitted preprocessing."""
    transformed = fitted_preprocessor.transform(raw_X_test)
    columns = fitted_preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=columns, index=raw_X_test.index)
