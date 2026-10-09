"""Nodes for recording validation evidence before and after cleaning."""

import pandas as pd

from validate_data import ValidateData


def _validation_frame(frame: pd.DataFrame, enforce: bool) -> pd.DataFrame:
    """Return validation results in a catalog-friendly tabular form."""
    validation = ValidateData().validate_data(frame, enforce=enforce)
    return validation.rename("passed").to_frame()


def validate_store_data(
    raw_store_data: pd.DataFrame,
    cleaned_store_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate raw evidence and enforce the cleaned-data contract."""
    raw_validation = _validation_frame(raw_store_data, enforce=False)
    cleaned_validation = _validation_frame(cleaned_store_data, enforce=False)
    if not cleaned_validation["passed"].all():
        failed_checks = cleaned_validation.index[~cleaned_validation["passed"]].tolist()
        raise AssertionError(f"Data validation failed: {failed_checks}")
    return raw_validation, cleaned_validation

