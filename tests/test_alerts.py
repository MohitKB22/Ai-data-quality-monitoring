import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monitoring.alerts import evaluate_alerts  # noqa: E402


def test_no_alerts_when_all_metrics_within_thresholds():
    metrics = {
        "dataset_name": "ok_batch", "missing_percent": 2.0, "duplicate_percent": 1.0,
        "outlier_percent": 0.5, "drift_detected": False, "drift_details": {},
        "avg_prediction_confidence": 0.95,
    }
    assert evaluate_alerts(metrics) == []


def test_missing_values_alert_fires_above_threshold():
    metrics = {"dataset_name": "x", "missing_percent": 15.0, "duplicate_percent": 0, "outlier_percent": 0,
               "drift_detected": False, "drift_details": {}, "avg_prediction_confidence": 0.9}
    alerts = evaluate_alerts(metrics)
    assert len(alerts) == 1
    assert alerts[0]["alert_type"] == "high_missing_values"
    assert alerts[0]["severity"] == "warning"


def test_missing_values_alert_is_critical_above_20_percent():
    metrics = {"dataset_name": "x", "missing_percent": 25.0, "duplicate_percent": 0, "outlier_percent": 0,
               "drift_detected": False, "drift_details": {}, "avg_prediction_confidence": 0.9}
    alerts = evaluate_alerts(metrics)
    assert alerts[0]["severity"] == "critical"


def test_duplicate_alert_fires_above_threshold():
    metrics = {"dataset_name": "x", "missing_percent": 0, "duplicate_percent": 8.0, "outlier_percent": 0,
               "drift_detected": False, "drift_details": {}, "avg_prediction_confidence": 0.9}
    alerts = evaluate_alerts(metrics)
    assert any(a["alert_type"] == "high_duplicate_rate" for a in alerts)


def test_outlier_alert_fires_above_threshold():
    metrics = {"dataset_name": "x", "missing_percent": 0, "duplicate_percent": 0, "outlier_percent": 5.0,
               "drift_detected": False, "drift_details": {}, "avg_prediction_confidence": 0.9}
    alerts = evaluate_alerts(metrics)
    assert any(a["alert_type"] == "high_outlier_rate" for a in alerts)


def test_drift_alert_fires_when_detected():
    metrics = {"dataset_name": "x", "missing_percent": 0, "duplicate_percent": 0, "outlier_percent": 0,
               "drift_detected": True, "drift_details": {"Salary": {"drift_detected": True}},
               "avg_prediction_confidence": 0.9}
    alerts = evaluate_alerts(metrics)
    assert any(a["alert_type"] == "data_drift_detected" and a["severity"] == "critical" for a in alerts)


def test_low_confidence_alert_fires_below_threshold():
    metrics = {"dataset_name": "x", "missing_percent": 0, "duplicate_percent": 0, "outlier_percent": 0,
               "drift_detected": False, "drift_details": {}, "avg_prediction_confidence": 0.3}
    alerts = evaluate_alerts(metrics)
    assert any(a["alert_type"] == "low_prediction_confidence" for a in alerts)


def test_multiple_alerts_can_fire_together():
    metrics = {"dataset_name": "x", "missing_percent": 20.0, "duplicate_percent": 10.0, "outlier_percent": 10.0,
               "drift_detected": True, "drift_details": {"Salary": {"drift_detected": True}},
               "avg_prediction_confidence": 0.2}
    alerts = evaluate_alerts(metrics)
    assert len(alerts) == 5


def test_alert_missing_optional_confidence_key_does_not_crash():
    metrics = {"dataset_name": "x", "missing_percent": 0, "duplicate_percent": 0, "outlier_percent": 0,
               "drift_detected": False, "drift_details": {}}  # no avg_prediction_confidence key at all
    assert evaluate_alerts(metrics) == []
