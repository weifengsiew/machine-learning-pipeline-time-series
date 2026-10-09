"""Nodes for target separation, chronological splitting, and preprocessing."""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_CATEGORICAL_FEATURES = ("event_name", "event_type")
DEFAULT_NUMERIC_FEATURES = (
    "sales_lag_1_day",
    "sales_lag_2_days",
    "sales_lag_1_week",
    "sales_lag_2_weeks",
    "sales_lag_4_weeks",
    "sales_lag_1_month",
    "sales_lag_1_quarter",
    "sales_lag_1_year",
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


def chronological_split_by_date(
    X: pd.DataFrame,
    y: pd.Series,
    test_start_date: pd.Timestamp | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split aligned feature and target rows at a calendar boundary."""
    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical indexes")
    if not X.index.is_monotonic_increasing:
        raise ValueError("X must be sorted chronologically")

    test_start = pd.Timestamp(test_start_date)
    train_mask = X.index < test_start
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise ValueError("test_start_date must leave non-empty train and test sets")
    return X.loc[train_mask], X.loc[test_mask], y.loc[train_mask], y.loc[test_mask]


def split_features_and_target(
    engineered_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the future target from the model feature table."""
    model_data = engineered_data.copy()
    target = model_data.pop("target_next_day")
    return model_data, target


def create_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create non-shuffled train and test partitions at a calendar boundary."""
    return chronological_split_by_date(features, target, test_start_date)


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

