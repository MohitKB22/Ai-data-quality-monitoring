"""
SQLite persistence layer for monitoring history and alert logs.
Used by monitoring/monitor.py, monitoring/alerts.py, the FastAPI service, and
the Streamlit dashboard's "Live Monitoring" view.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "monitoring", "quality_monitoring.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS monitoring_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    dataset_name TEXT,
    row_count INTEGER,
    missing_percent REAL,
    duplicate_percent REAL,
    outlier_percent REAL,
    drift_detected INTEGER,
    drift_details TEXT,
    avg_prediction_confidence REAL,
    quality_score REAL,
    alert_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS alerts_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    triggered_at TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    channel TEXT NOT NULL,
    observed_value REAL,
    threshold_value REAL,
    FOREIGN KEY (run_id) REFERENCES monitoring_runs (id)
);
"""


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def insert_monitoring_run(conn: sqlite3.Connection, record: dict) -> int:
    cur = conn.execute(
        """INSERT INTO monitoring_runs
           (run_at, dataset_name, row_count, missing_percent, duplicate_percent,
            outlier_percent, drift_detected, drift_details, avg_prediction_confidence,
            quality_score, alert_count)
           VALUES (:run_at, :dataset_name, :row_count, :missing_percent, :duplicate_percent,
                   :outlier_percent, :drift_detected, :drift_details, :avg_prediction_confidence,
                   :quality_score, :alert_count)""",
        {
            "run_at": record.get("run_at", datetime.now(timezone.utc).isoformat()),
            "dataset_name": record.get("dataset_name", "unknown"),
            "row_count": record.get("row_count", 0),
            "missing_percent": record.get("missing_percent", 0.0),
            "duplicate_percent": record.get("duplicate_percent", 0.0),
            "outlier_percent": record.get("outlier_percent", 0.0),
            "drift_detected": int(bool(record.get("drift_detected", False))),
            "drift_details": json.dumps(record.get("drift_details", {})),
            "avg_prediction_confidence": record.get("avg_prediction_confidence"),
            "quality_score": record.get("quality_score", 0.0),
            "alert_count": record.get("alert_count", 0),
        },
    )
    conn.commit()
    return cur.lastrowid


def insert_alert(conn: sqlite3.Connection, run_id: Optional[int], alert: dict) -> int:
    cur = conn.execute(
        """INSERT INTO alerts_log
           (run_id, triggered_at, alert_type, severity, message, channel, observed_value, threshold_value)
           VALUES (:run_id, :triggered_at, :alert_type, :severity, :message, :channel, :observed_value, :threshold_value)""",
        {
            "run_id": run_id,
            "triggered_at": alert.get("triggered_at", datetime.now(timezone.utc).isoformat()),
            "alert_type": alert["alert_type"],
            "severity": alert.get("severity", "warning"),
            "message": alert["message"],
            "channel": alert.get("channel", "console"),
            "observed_value": alert.get("observed_value"),
            "threshold_value": alert.get("threshold_value"),
        },
    )
    conn.commit()
    return cur.lastrowid


def fetch_recent_runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute("SELECT * FROM monitoring_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows][::-1]  # chronological order


def fetch_recent_alerts(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute("SELECT * FROM alerts_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def clear_all(conn: sqlite3.Connection):
    """Utility for tests: wipe both tables."""
    conn.execute("DELETE FROM alerts_log")
    conn.execute("DELETE FROM monitoring_runs")
    conn.commit()
