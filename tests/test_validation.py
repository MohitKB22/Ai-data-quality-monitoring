import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.great_expectations import (ExpectationSuite, generate_html_report,  # noqa: E402
                                            validate_dataframe)


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "CustomerID": [f"CUST{i}" for i in range(20)],
        "Age": [30 + i % 10 for i in range(20)],
        "Salary": [50000 + i * 100 for i in range(20)],
        "PurchaseAmount": [100 + i for i in range(20)],
        "City": ["Paris"] * 20,
        "Country": ["France"] * 20,
        "TransactionDate": ["2026-01-01"] * 20,
    })


@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "CustomerID": ["BADID"] * 5 + [f"CUST{i}" for i in range(5)],
        "Age": [None] * 5 + [30] * 5,
        "Salary": [-100] * 5 + [50000] * 5,
        "PurchaseAmount": [50] * 10,
        "City": ["???"] * 5 + ["Paris"] * 5,
        "Country": ["France"] * 10,
        "TransactionDate": ["2099-01-01"] * 5 + ["2026-01-01"] * 5,
    })


def test_not_null_expectation_passes_on_clean_data(clean_df):
    suite = ExpectationSuite("test").expect_column_values_to_not_be_null("Age", mostly=1.0)
    result = suite.run(clean_df)[0]
    assert result.success


def test_not_null_expectation_fails_on_dirty_data(dirty_df):
    suite = ExpectationSuite("test").expect_column_values_to_not_be_null("Age", mostly=0.95)
    result = suite.run(dirty_df)[0]
    assert not result.success
    assert result.details["null_percent"] == 50.0


def test_range_expectation_catches_negative_salary(dirty_df):
    suite = ExpectationSuite("test").expect_column_values_to_be_between("Salary", min_value=0, mostly=0.9)
    result = suite.run(dirty_df)[0]
    assert not result.success


def test_regex_expectation_on_customer_id(dirty_df):
    import re
    suite = ExpectationSuite("test").expect_column_values_to_match_regex("CustomerID", re.compile(r"^CUST\d+$"), mostly=0.9)
    result = suite.run(dirty_df)[0]
    assert not result.success  # "BADID" x5 doesn't match, 50% < 90% threshold


def test_duplicate_expectation(clean_df):
    dup_df = pd.concat([clean_df, clean_df.iloc[:5]], ignore_index=True)
    suite = ExpectationSuite("test").expect_compound_row_uniqueness(
        ["CustomerID", "Age", "Salary", "PurchaseAmount", "City", "Country", "TransactionDate"], mostly=0.95)
    result = suite.run(dup_df)[0]
    assert not result.success


def test_validate_dataframe_returns_expected_summary_shape(dirty_df):
    summary = validate_dataframe(dirty_df, dataset_name="unit_test")
    assert "expectations_passed" in summary
    assert "expectations_total" in summary
    assert summary["expectations_total"] > 0
    assert 0 <= summary["success_rate"] <= 1
    assert isinstance(summary["results"], list)


def test_validate_dataframe_detects_drift():
    import numpy as np
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({
        "CustomerID": [f"C{i}" for i in range(200)], "Age": rng.normal(40, 5, 200),
        "Salary": rng.normal(50000, 5000, 200), "PurchaseAmount": rng.normal(100, 10, 200),
        "City": ["Paris"] * 200, "Country": ["France"] * 200, "TransactionDate": ["2026-01-01"] * 200,
    })
    current = reference.copy()
    current["Salary"] = rng.normal(150000, 20000, 200)  # dramatic shift

    summary = validate_dataframe(current, reference_df=reference, dataset_name="drift_test")
    drift_result = next(r for r in summary["results"] if r["expectation_type"] == "expect_distribution_to_match_reference" and r["column"] == "Salary")
    assert drift_result["success"] is False


def test_html_report_generation(dirty_df, tmp_path):
    summary = validate_dataframe(dirty_df, dataset_name="html_test")
    out_path = tmp_path / "report.html"
    generate_html_report(summary, str(out_path))
    assert out_path.exists()
    html = out_path.read_text()
    assert "<!DOCTYPE html>" in html
    assert "{dataset_name}" not in html  # no unresolved template placeholders
    assert "html_test" in html
