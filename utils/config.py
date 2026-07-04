"""
Central configuration: filesystem paths and alert thresholds shared across
preprocessing, validation, training, monitoring, the API, and the dashboard.
Keeping these in one place means the alert thresholds documented in the
README are the actual thresholds the code enforces -- not just a comment
that can drift out of sync with the implementation.
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---- Paths ----
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DATA_PATH = os.path.join(DATA_RAW_DIR, "synthetic_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOCS_IMAGES_DIR = os.path.join(PROJECT_ROOT, "docs", "images")
VALIDATION_REPORTS_DIR = os.path.join(PROJECT_ROOT, "validation", "reports")
MONITORING_DB_PATH = os.path.join(PROJECT_ROOT, "monitoring", "quality_monitoring.db")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
ALERTS_LOG_PATH = os.path.join(LOGS_DIR, "alerts.log")

# ---- Alert thresholds (see monitoring/alerts.py) ----
THRESHOLD_MISSING_PERCENT = 10.0        # alert if > 10% of rows have a missing field
THRESHOLD_DUPLICATE_PERCENT = 5.0       # alert if > 5% of rows are duplicates
THRESHOLD_OUTLIER_PERCENT = 3.0         # alert if > 3% of rows are statistical outliers
THRESHOLD_DRIFT_KS_STATISTIC = 0.15     # alert if KS statistic exceeds this for a monitored column
THRESHOLD_LOW_CONFIDENCE = 0.60         # alert if average model confidence drops below this

# ---- Misc ----
RANDOM_STATE = 42
TODAY_OVERRIDE = "2026-07-01"  # fixed "current date" so the synthetic dataset stays reproducible
