"""Helpers for converting Great Expectations results to Kedro tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

REPORT_COLUMNS = [
    "success",
    "expectation_type",
    "affected_field",
    "unexpected_count",
    "unexpected_percent",
    "examples",
]


def _affected_field(expectation_kwargs: dict[str, Any]) -> str:
    """Return the column or table affected by an expectation."""
    if "column" in expectation_kwargs:
        return str(expectation_kwargs["column"])
    if "column_list" in expectation_kwargs:
        return ", ".join(expectation_kwargs["column_list"])
    return "table"


def _format_examples(examples: list[Any]) -> str:
    """Format representative unexpected values for a CSV report."""
    unique_examples = pd.Series(examples, dtype="object").drop_duplicates().head(30)
    return "; ".join(str(value) for value in unique_examples)


def build_validation_report(validation_result: dict[str, Any]) -> pd.DataFrame:
    """Convert every GX expectation result to a catalog-friendly table."""
    rows = []
    for expectation_result in validation_result["results"]:
        config = expectation_result["expectation_config"]
        result = expectation_result.get("result", {})
        rows.append(
            {
                "success": bool(expectation_result["success"]),
                "expectation_type": config["type"],
                "affected_field": _affected_field(config["kwargs"]),
                "unexpected_count": result.get("unexpected_count"),
                "unexpected_percent": result.get("unexpected_percent_total"),
                "examples": _format_examples(
                    result.get("partial_unexpected_list", [])
                ),
            }
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)
