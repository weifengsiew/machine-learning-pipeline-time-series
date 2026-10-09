"""Nodes for running the shared Great Expectations suite twice."""

from __future__ import annotations

from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations import ExpectationSuite

from .reporting import build_validation_report


def _run_validation(
    frame: pd.DataFrame,
    expectation_suite: ExpectationSuite,
    table_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one in-memory GX validation and build its tabular report."""
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(f"{table_name}_source")
    data_asset = data_source.add_dataframe_asset(name=table_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe("whole_table")
    batch_request = batch_definition.build_batch_request(
        batch_parameters={"dataframe": frame}
    )
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite=expectation_suite,
    )
    validation_result = validator.validate().to_json_dict()
    return validation_result, build_validation_report(validation_result)


def validate_raw_data(
    raw_store_data: pd.DataFrame,
    expectation_suite: ExpectationSuite,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validate raw data and retain failures as evidence for the report."""
    return _run_validation(raw_store_data, expectation_suite, "raw_store_data")


def validate_cleaned_data(
    cleaned_store_data: pd.DataFrame,
    expectation_suite: ExpectationSuite,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validate cleaned data and fail the pipeline if any expectation fails."""
    validation_result, report = _run_validation(
        cleaned_store_data,
        expectation_suite,
        "cleaned_store_data",
    )
    if not validation_result["success"]:
        failed_expectations = report.loc[~report["success"], "expectation_type"].tolist()
        raise AssertionError(
            "Cleaned data failed Great Expectations: "
            f"{failed_expectations}"
        )
    return validation_result, report

