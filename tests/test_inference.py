import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.predict import CLASSIFIER_PATH, DataQualityPredictor  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(CLASSIFIER_PATH),
    reason="Models not trained yet -- run `python training/train.py` before this test suite.",
)


@pytest.fixture(scope="module")
def predictor():
    return DataQualityPredictor()


@pytest.fixture
def sample_batch():
    return pd.DataFrame({
        "CustomerID": ["CUSTA", "CUSTB"],
        "Age": [35, 40],
        "Salary": [-50000, 65000],       # first row deliberately bad (negative salary)
        "PurchaseAmount": [80, 120],
        "City": ["Paris", "Tokyo"],
        "Country": ["France", "Japan"],
        "TransactionDate": ["2026-01-15", "2025-11-02"],
    })


def test_predict_returns_expected_columns(predictor, sample_batch):
    result = predictor.predict(sample_batch)
    for col in ["predicted_label", "bad_data_probability", "confidence"]:
        assert col in result.columns
    assert len(result) == len(sample_batch)


def test_predict_flags_negative_salary_as_bad(predictor, sample_batch):
    result = predictor.predict(sample_batch)
    assert result.iloc[0]["predicted_label"] == "Bad Data"


def test_predict_probability_in_valid_range(predictor, sample_batch):
    result = predictor.predict(sample_batch)
    assert result["bad_data_probability"].between(0, 1).all()


def test_detect_anomalies_returns_expected_columns(predictor, sample_batch):
    result = predictor.detect_anomalies(sample_batch)
    assert "is_anomaly" in result.columns
    assert "anomaly_score" in result.columns


def test_predict_missing_required_column_raises(predictor):
    bad_df = pd.DataFrame({"CustomerID": ["X"], "Age": [30]})  # missing most required columns
    with pytest.raises(ValueError):
        predictor.predict(bad_df)


def test_predict_full_combines_classifier_and_anomaly(predictor, sample_batch):
    result = predictor.predict_full(sample_batch)
    assert "predicted_label" in result.columns
    assert "is_anomaly" in result.columns
