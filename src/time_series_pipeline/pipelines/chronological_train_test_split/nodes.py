"""Nodes for target separation and chronological splitting."""

import pandas as pd


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
    target = model_data.pop("target_sales")
    return model_data, target


def create_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create non-shuffled train and test partitions at a calendar boundary."""
    return chronological_split_by_date(features, target, test_start_date)
