"""
Monitoring
===========
Computes the KPIs the dashboard's "Live Monitoring" view and the alerting
system are built on: missing-value %, duplicate %, outlier %, distribution
drift vs. a reference period, average model prediction confidence, and a
composite 0-100 data quality score. Each run is persisted to SQLite via
utils.db so the dashboard can show trends over time.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.preprocess import CORE_COLUMNS, engineer_features  # noqa: E402
from utils import config, db  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("monitoring")

DRIFT_COLUMNS = ["Salary", "PurchaseAmount"]


def compute_drift(current_df: pd.DataFrame, reference_df: pd.DataFrame,
                   columns: list = DRIFT_COLUMNS, max_ks_statistic: float = config.THRESHOLD_DRIFT_KS_STATISTIC) -> dict:
    details = {}
    any_drift = False
    for col in columns:
        cur = pd.to_numeric(current_df[col], errors="coerce").dropna()
        ref = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        if len(cur) < 10 or len(ref) < 10:
            details[col] = {"ks_statistic": None, "p_value": None, "drift_detected": False, "note": "insufficient data"}
            continue
        ks_stat, p_value = scipy_stats.ks_2samp(cur, ref)
        drifted = bool(ks_stat > max_ks_statistic)
        any_drift = any_drift or drifted
        details[col] = {"ks_statistic": round(float(ks_stat), 4), "p_value": round(float(p_value), 6),
                         "drift_detected": drifted, "threshold": max_ks_statistic}
    return {"drift_detected": any_drift, "columns": details}


def compute_monitoring_metrics(current_df: pd.DataFrame, reference_stats: dict,
                                reference_df: pd.DataFrame = None, predictor=None,
                                dataset_name: str = "current_batch") -> dict:
    features = engineer_features(current_df, reference_stats)
    n = len(current_df)

    missing_percent = round(float((features["missing_ratio"] > 0).mean() * 100), 2)
    duplicate_percent = round(float(features["is_duplicate"].mean() * 100), 2)
    outlier_percent = round(float(
        ((features["age_iqr_outlier"] + features["salary_iqr_outlier"] + features["purchase_iqr_outlier"]) > 0).mean() * 100
    ), 2)

    drift = {"drift_detected": False, "columns": {}}
    if reference_df is not None and len(reference_df) >= 10:
        drift = compute_drift(current_df, reference_df)

    avg_confidence = None
    if predictor is not None:
        try:
            preds = predictor.predict(current_df)
            avg_confidence = round(float(preds["confidence"].mean()), 4)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not compute prediction confidence: {e}")

    # Composite quality score: start at 100, subtract weighted penalties.
    # Weights reflect the alert severities used in monitoring/alerts.py.
    score = 100.0
    score -= min(missing_percent, 100) * 0.5
    score -= min(duplicate_percent, 100) * 0.6
    score -= min(outlier_percent, 100) * 0.8
    score -= 8.0 if drift["drift_detected"] else 0.0
    if avg_confidence is not None and avg_confidence < config.THRESHOLD_LOW_CONFIDENCE:
        score -= 5.0
    quality_score = round(float(np.clip(score, 0, 100)), 2)

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "row_count": n,
        "missing_percent": missing_percent,
        "duplicate_percent": duplicate_percent,
        "outlier_percent": outlier_percent,
        "drift_detected": drift["drift_detected"],
        "drift_details": drift["columns"],
        "avg_prediction_confidence": avg_confidence,
        "quality_score": quality_score,
    }


def run_monitoring_cycle(current_df: pd.DataFrame, reference_stats: dict, reference_df: pd.DataFrame = None,
                          predictor=None, dataset_name: str = "current_batch", db_path: str = db.DEFAULT_DB_PATH,
                          fire_alerts: bool = True) -> dict:
    """Compute metrics, persist to SQLite, and (optionally) evaluate alert rules.
    Returns the metrics dict plus `run_id` and `alerts` (list of fired alerts)."""
    from monitoring.alerts import evaluate_and_send_alerts  # local import avoids a circular import at module load

    metrics = compute_monitoring_metrics(current_df, reference_stats, reference_df, predictor, dataset_name)
    conn = db.get_connection(db_path)

    alerts = []
    if fire_alerts:
        alerts = evaluate_and_send_alerts(metrics)
    metrics["alert_count"] = len(alerts)

    run_id = db.insert_monitoring_run(conn, metrics)
    for alert in alerts:
        db.insert_alert(conn, run_id, alert)
    conn.close()

    metrics["run_id"] = run_id
    metrics["alerts"] = alerts
    logger.info(f"Monitoring run #{run_id} on '{dataset_name}': quality_score={metrics['quality_score']}, "
                f"missing={metrics['missing_percent']}%, duplicate={metrics['duplicate_percent']}%, "
                f"outlier={metrics['outlier_percent']}%, drift={metrics['drift_detected']}, "
                f"alerts_fired={len(alerts)}")
    return metrics


def main():
    import json
    from preprocessing.preprocess import compute_reference_stats

    df = pd.read_csv(config.RAW_DATA_PATH)
    parsed_dates = pd.to_datetime(df["TransactionDate"], errors="coerce", format="mixed")
    cutoff = parsed_dates.quantile(0.70)
    reference_df = df[parsed_dates <= cutoff].copy()
    current_df = df[parsed_dates > cutoff].copy()
    reference_stats = compute_reference_stats(reference_df)

    predictor = None
    try:
        from inference.predict import get_predictor
        predictor = get_predictor()
    except Exception as e:
        logger.warning(f"Predictor unavailable (train models first for confidence tracking): {e}")

    metrics = run_monitoring_cycle(current_df, reference_stats, reference_df, predictor,
                                    dataset_name="synthetic_data.csv (current period)")

    print("\nMonitoring run summary:")
    print(json.dumps({k: v for k, v in metrics.items() if k != "alerts"}, indent=2, default=str))
    print(f"\nAlerts fired: {len(metrics['alerts'])}")
    for a in metrics["alerts"]:
        print(f"  [{a['severity'].upper()}] {a['alert_type']}: {a['message']}")


if __name__ == "__main__":
    main()
