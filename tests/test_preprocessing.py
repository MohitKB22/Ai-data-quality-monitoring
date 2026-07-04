import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.preprocess import (FEATURE_COLUMNS, compute_reference_stats,  # noqa: E402
                                       engineer_features, preprocess_dataset)


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "CustomerID": ["CUST1", "CUST2", "CUST3", "CUST4", "CUST1"],  # CUST4 duplicate of none, last row dup of row 0
        "Age": [30, 45, np.nan, 200, 30],
        "Salary": [50000, 60000, 55000, -1000, 50000],
        "PurchaseAmount": [100, 150, 120, 90, 100],
        "City": ["Paris", "Tokyo", "InvalidCityXYZ", "Paris", "Paris"],
        "Country": ["France", "Japan", "France", "France", "France"],
        "TransactionDate": ["2026-01-01", "2026-01-02", "2026-01-03", "2099-01-01", "2026-01-01"],
    })


def test_compute_reference_stats_structure(sample_raw_df):
    stats = compute_reference_stats(sample_raw_df, min_category_count=1)
    assert "numeric" in stats and "categorical" in stats
    for col in ["Age", "Salary", "PurchaseAmount"]:
        assert col in stats["numeric"]
        assert "mean" in stats["numeric"][col] and "lower_bound" in stats["numeric"][col]
    for col in ["City", "Country"]:
        assert col in stats["categorical"]


def test_engineer_features_has_no_nans(sample_raw_df):
    stats = compute_reference_stats(sample_raw_df, min_category_count=1)
    features = engineer_features(sample_raw_df, stats)
    assert features.isna().sum().sum() == 0


def test_engineer_features_returns_expected_columns(sample_raw_df):
    stats = compute_reference_stats(sample_raw_df, min_category_count=1)
    features = engineer_features(sample_raw_df, stats)
    assert list(features.columns) == FEATURE_COLUMNS


def test_missing_age_is_flagged(sample_raw_df):
    stats = compute_reference_stats(sample_raw_df, min_category_count=1)
    features = engineer_features(sample_raw_df, stats)
    assert features.loc[2, "age_missing"] == 1
    assert features.loc[0, "age_missing"] == 0


def test_duplicate_row_is_flagged(sample_raw_df):
    stats = compute_reference_stats(sample_raw_df, min_category_count=1)
    features = engineer_features(sample_raw_df, stats)
    # row 4 is an exact duplicate of row 0 across the core columns
    assert features.loc[4, "is_duplicate"] == 1
    assert features.loc[0, "is_duplicate"] == 0


def test_negative_salary_flagged(sample_raw_df):
    stats = compute_reference_stats(sample_raw_df, min_category_count=1)
    features = engineer_features(sample_raw_df, stats)
    assert features.loc[3, "negative_salary_flag"] == 1


def test_future_date_flagged(sample_raw_df):
    stats = compute_reference_stats(sample_raw_df, min_category_count=1)
    features = engineer_features(sample_raw_df, stats)
    assert features.loc[3, "is_future_date"] == 1
    assert features.loc[0, "is_future_date"] == 0


def test_unknown_category_flagged_invalid(sample_raw_df):
    stats = compute_reference_stats(sample_raw_df, min_category_count=3)  # Paris appears 3x, others less
    features = engineer_features(sample_raw_df, stats)
    assert features.loc[2, "city_valid"] == 0  # "InvalidCityXYZ" never repeats


def test_preprocess_dataset_convenience_wrapper(sample_raw_df):
    features, stats = preprocess_dataset(sample_raw_df, fit_reference=True)
    assert features.shape[0] == len(sample_raw_df)
    assert stats["fitted_on_rows"] == len(sample_raw_df)


def test_engineer_features_reuses_provided_reference_stats(sample_raw_df):
    # Leak-safety check: engineer_features must not recompute stats internally --
    # passing an obviously-wrong reference should change the output deterministically.
    real_stats = compute_reference_stats(sample_raw_df, min_category_count=1)
    fake_stats = compute_reference_stats(sample_raw_df, min_category_count=1)
    fake_stats["numeric"]["Salary"]["mean"] = 999_999
    fake_stats["numeric"]["Salary"]["std"] = 1.0

    real_features = engineer_features(sample_raw_df, real_stats)
    fake_features = engineer_features(sample_raw_df, fake_stats)
    assert not real_features["salary_zscore"].equals(fake_features["salary_zscore"])
