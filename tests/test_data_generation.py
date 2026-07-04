import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.generate_synthetic_data import generate_dataset  # noqa: E402


@pytest.fixture(scope="module")
def small_dataset():
    # A small n_base keeps the test suite fast; the shipped dataset itself
    # uses n_base=100_000 (see data/generate_synthetic_data.py main()).
    return generate_dataset(n_base=5_000, seed=1)


def test_expected_columns(small_dataset):
    expected = {"CustomerID", "Age", "Salary", "PurchaseAmount", "City", "Country",
                "TransactionDate", "QualityIssue", "Label"}
    assert expected.issubset(set(small_dataset.columns))


def test_row_count_includes_duplicates(small_dataset):
    # n_base=5000 + ~3% duplicates appended
    assert len(small_dataset) > 5_000


def test_label_values_are_valid(small_dataset):
    assert set(small_dataset["Label"].unique()) <= {"Good Data", "Bad Data"}


def test_label_distribution_is_reasonable(small_dataset):
    bad_rate = (small_dataset["Label"] == "Bad Data").mean()
    # With ~8-10% probabilistic issue injection across several independent
    # issue types plus label noise, Bad Data should land well under 50%
    # but comfortably above 0 -- a wide, deliberately loose sanity band.
    assert 0.05 < bad_rate < 0.60


def test_quality_issue_sentinel_survives_csv_roundtrip(small_dataset, tmp_path):
    # Regression test: "No Issue" must NOT collide with pandas' default NA
    # tokens (unlike the literal string "None", which does -- see the note
    # in data/generate_synthetic_data.py).
    path = tmp_path / "roundtrip.csv"
    small_dataset.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert reloaded["QualityIssue"].isna().sum() == 0
    assert "No Issue" in reloaded["QualityIssue"].unique()


def test_injected_issues_present(small_dataset):
    issues = set(small_dataset["QualityIssue"].unique())
    for expected_issue in ["Missing Value", "Duplicate Row", "Outlier", "Invalid Date Format",
                            "Future Date", "Negative Salary", "Negative Purchase Amount", "Invalid Category"]:
        assert expected_issue in issues, f"Expected issue type '{expected_issue}' was not generated"


def test_negative_salary_tag_implies_negative_value(small_dataset):
    # The reverse doesn't always hold: a row can receive a negative Salary
    # value while being tagged with a *different* issue if it was already
    # claimed by an earlier-processed issue type (see the `tag()` helper's
    # first-come-first-tagged docstring in data/generate_synthetic_data.py).
    # What IS guaranteed: every row actually TAGGED "Negative Salary" has Salary < 0.
    tagged_rows = small_dataset[small_dataset["QualityIssue"] == "Negative Salary"]
    assert len(tagged_rows) > 0
    assert (tagged_rows["Salary"] < 0).all()


def test_reproducible_with_same_seed():
    df1 = generate_dataset(n_base=500, seed=99)
    df2 = generate_dataset(n_base=500, seed=99)
    pd.testing.assert_frame_equal(df1, df2)
