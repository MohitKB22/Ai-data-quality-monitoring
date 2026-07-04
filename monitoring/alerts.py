"""
Alerting
=========
Threshold-based alert rules evaluated against a monitoring run's metrics
(see monitoring/monitor.py). Three channels are implemented:

  - console : logged immediately via the standard logger
  - email   : MOCKED -- formats a real email (subject + body) and appends it
              to logs/mock_emails.log instead of calling an SMTP server
  - slack   : MOCKED -- formats a real Slack Block Kit payload and appends it
              to logs/mock_slack.log instead of POSTing to a webhook URL

This project has no real mail server or Slack workspace to talk to, so both
are implemented as realistic, drop-in-ready payload builders. To go live,
replace `_send_mock_email` with an smtplib/SES/SendGrid call and
`_send_mock_slack` with a `requests.post(SLACK_WEBHOOK_URL, json=payload)`
call -- the alert-evaluation logic above them does not need to change.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("alerts")

MOCK_EMAIL_LOG = os.path.join(config.LOGS_DIR, "mock_emails.log")
MOCK_SLACK_LOG = os.path.join(config.LOGS_DIR, "mock_slack.log")
ALERT_RECIPIENT_EMAIL = "data-quality-team@example.com"
ALERT_SLACK_CHANNEL = "#data-quality-alerts"


def _make_alert(alert_type: str, severity: str, message: str, observed_value=None, threshold_value=None) -> dict:
    return {
        "alert_type": alert_type,
        "severity": severity,
        "message": message,
        "observed_value": observed_value,
        "threshold_value": threshold_value,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_alerts(metrics: dict) -> list[dict]:
    """Pure rule evaluation (no I/O) -- kept separate from dispatch for easy unit testing."""
    alerts = []

    if metrics.get("missing_percent", 0) > config.THRESHOLD_MISSING_PERCENT:
        alerts.append(_make_alert(
            "high_missing_values", "critical" if metrics["missing_percent"] > 20 else "warning",
            f"{metrics['missing_percent']}% of rows in '{metrics.get('dataset_name','dataset')}' have at least "
            f"one missing field (threshold: {config.THRESHOLD_MISSING_PERCENT}%).",
            metrics["missing_percent"], config.THRESHOLD_MISSING_PERCENT,
        ))

    if metrics.get("duplicate_percent", 0) > config.THRESHOLD_DUPLICATE_PERCENT:
        alerts.append(_make_alert(
            "high_duplicate_rate", "critical" if metrics["duplicate_percent"] > 15 else "warning",
            f"{metrics['duplicate_percent']}% of rows in '{metrics.get('dataset_name','dataset')}' are duplicates "
            f"(threshold: {config.THRESHOLD_DUPLICATE_PERCENT}%).",
            metrics["duplicate_percent"], config.THRESHOLD_DUPLICATE_PERCENT,
        ))

    if metrics.get("outlier_percent", 0) > config.THRESHOLD_OUTLIER_PERCENT:
        alerts.append(_make_alert(
            "high_outlier_rate", "warning",
            f"{metrics['outlier_percent']}% of rows in '{metrics.get('dataset_name','dataset')}' are statistical "
            f"outliers (threshold: {config.THRESHOLD_OUTLIER_PERCENT}%).",
            metrics["outlier_percent"], config.THRESHOLD_OUTLIER_PERCENT,
        ))

    if metrics.get("drift_detected"):
        drifted_cols = [c for c, d in metrics.get("drift_details", {}).items() if d.get("drift_detected")]
        alerts.append(_make_alert(
            "data_drift_detected", "critical",
            f"Statistically significant distribution drift detected in column(s) {drifted_cols} "
            f"for '{metrics.get('dataset_name','dataset')}' (KS statistic > {config.THRESHOLD_DRIFT_KS_STATISTIC}).",
            None, config.THRESHOLD_DRIFT_KS_STATISTIC,
        ))

    conf = metrics.get("avg_prediction_confidence")
    if conf is not None and conf < config.THRESHOLD_LOW_CONFIDENCE:
        alerts.append(_make_alert(
            "low_prediction_confidence", "warning",
            f"Average model confidence ({conf}) on '{metrics.get('dataset_name','dataset')}' fell below "
            f"the {config.THRESHOLD_LOW_CONFIDENCE} threshold -- the incoming data may look unlike training data.",
            conf, config.THRESHOLD_LOW_CONFIDENCE,
        ))

    return alerts


# --------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------
def _send_console(alert: dict):
    log_fn = logger.critical if alert["severity"] == "critical" else logger.warning
    log_fn(f"[{alert['alert_type']}] {alert['message']}")


def _send_mock_email(alert: dict):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    subject = f"[Data Quality Alert - {alert['severity'].upper()}] {alert['alert_type'].replace('_', ' ').title()}"
    body = (
        f"To: {ALERT_RECIPIENT_EMAIL}\n"
        f"Subject: {subject}\n"
        f"Sent: {alert['triggered_at']}\n\n"
        f"{alert['message']}\n\n"
        f"Observed value: {alert['observed_value']}\n"
        f"Threshold: {alert['threshold_value']}\n"
        f"-- AI-Based Data Quality Monitoring System\n"
        + "-" * 70 + "\n"
    )
    with open(MOCK_EMAIL_LOG, "a") as f:
        f.write(body)
    logger.info(f"(mock) email queued to {ALERT_RECIPIENT_EMAIL}: {subject}")


def _send_mock_slack(alert: dict):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    emoji = ":rotating_light:" if alert["severity"] == "critical" else ":warning:"
    payload = {
        "channel": ALERT_SLACK_CHANNEL,
        "text": f"{emoji} *{alert['alert_type'].replace('_', ' ').title()}* ({alert['severity']})",
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn",
             "text": f"{emoji} *{alert['alert_type'].replace('_', ' ').title()}*\n{alert['message']}"}},
            {"type": "context", "elements": [{"type": "mrkdwn",
             "text": f"Observed: `{alert['observed_value']}` | Threshold: `{alert['threshold_value']}` | "
                     f"{alert['triggered_at']}"}]},
        ],
    }
    with open(MOCK_SLACK_LOG, "a") as f:
        f.write(json.dumps(payload) + "\n")
    logger.info(f"(mock) Slack message posted to {ALERT_SLACK_CHANNEL}: {alert['alert_type']}")


def dispatch_alert(alert: dict, channels=("console", "email", "slack")):
    if "console" in channels:
        _send_console(alert)
    if "email" in channels:
        _send_mock_email(alert)
    if "slack" in channels:
        _send_mock_slack(alert)


def evaluate_and_send_alerts(metrics: dict, channels=("console", "email", "slack")) -> list[dict]:
    alerts = evaluate_alerts(metrics)
    for alert in alerts:
        alert["channel"] = "+".join(channels)
        dispatch_alert(alert, channels)
    if not alerts:
        logger.info(f"No alert thresholds breached for '{metrics.get('dataset_name', 'dataset')}'.")
    return alerts


if __name__ == "__main__":
    # Demo: fabricate a metrics dict that breaches every threshold, to show all channels firing.
    demo_metrics = {
        "dataset_name": "demo_batch.csv",
        "missing_percent": 14.2,
        "duplicate_percent": 7.8,
        "outlier_percent": 4.1,
        "drift_detected": True,
        "drift_details": {"Salary": {"drift_detected": True, "ks_statistic": 0.27}},
        "avg_prediction_confidence": 0.52,
    }
    fired = evaluate_and_send_alerts(demo_metrics)
    print(f"\n{len(fired)} alert(s) fired. See logs/alerts.log, logs/mock_emails.log, logs/mock_slack.log")
