"""Great Expectations rules for the M5 store-level daily table."""

from __future__ import annotations

from great_expectations import ExpectationSuite
from great_expectations.expectations import (
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToBeOfType,
    ExpectColumnValuesToNotBeNull,
    ExpectTableColumnsToMatchOrderedList,
)

STORE_COLUMNS = [
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
    "sales",
]
INTEGER_COLUMNS = ["wm_yr_wk", "wday", "month", "year", "snap", "sales"]
EVENT_COLUMNS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


def _add_expectation(suite: ExpectationSuite, expectation: object) -> None:
    """Append one expectation to the code-defined suite."""
    suite.expectations.append(expectation)


def add_basic_expectations(suite: ExpectationSuite) -> None:
    """Require the expected columns and complete values."""
    _add_expectation(
        suite,
        ExpectTableColumnsToMatchOrderedList(column_list=STORE_COLUMNS),
    )
    for column in STORE_COLUMNS:
        _add_expectation(suite, ExpectColumnValuesToNotBeNull(column=column))


def add_type_expectations(suite: ExpectationSuite) -> None:
    """Require integer calendar/sales fields and string event labels."""
    for column in INTEGER_COLUMNS:
        _add_expectation(
            suite,
            ExpectColumnValuesToBeOfType(column=column, type_="int64"),
        )
    for column in EVENT_COLUMNS:
        _add_expectation(
            suite,
            ExpectColumnValuesToBeOfType(column=column, type_="str"),
        )


def add_range_expectations(suite: ExpectationSuite) -> None:
    """Require valid M5 calendar, SNAP, and sales ranges."""
    for column, minimum, maximum in [
        ("wday", 1, 7),
        ("month", 1, 12),
        ("year", 2011, 2016),
        ("sales", 0, None),
    ]:
        _add_expectation(
            suite,
            ExpectColumnValuesToBeBetween(
                column=column,
                min_value=minimum,
                max_value=maximum,
            ),
        )
    _add_expectation(
        suite,
        ExpectColumnValuesToBeInSet(column="snap", value_set=[0, 1]),
    )


def build_expectation_suite() -> ExpectationSuite:
    """Build the shared raw/cleaned M5 validation suite."""
    suite = ExpectationSuite(name="m5_store_daily_validation_suite")
    add_basic_expectations(suite)
    add_type_expectations(suite)
    add_range_expectations(suite)
    return suite
