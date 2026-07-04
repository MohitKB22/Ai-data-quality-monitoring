"""
Preprocessing / Feature Engineering
====================================
Turns raw, messy records into a clean, purely-numeric feature matrix that
downstream ML models (RandomForest, XGBoost, IsolationForest) can consume.

Design principle: engineered features never contain NaN. Every "unknown" or
"missing" situation is represented explicitly (a *_missing flag, a neutral
z-score of 0, a sentinel value alongside a validity flag) rather than left as
NaN, because that missingness is itself the signal we are trying to detect.

Leak-safety: `compute_reference_stats` must only ever be fit on a TRAINING
split (never on validation/test/production data). The resulting stats
(means, stds, IQR bounds, "known-good" category sets) are persisted to disk
and reused unchanged by inference and monitoring, exactly like a fitted
sklearn scaler would be.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

TODAY = datetime(2026, 7, 1)

CORE_COLUMNS = ["CustomerID", "Age", "Salary", "PurchaseAmount", "City", "Country", "TransactionDate"]
NUMERIC_COLUMNS = ["Age", "Salary", "PurchaseAmount"]
CATEGORICAL_COLUMNS = ["City", "Country"]

FEATURE_COLUMNS = [
    "missing_ratio", "age_missing", "salary_missing", "purchase_missing",
    "city_missing", "country_missing", "date_missing",
    "is_duplicate",
    "age_zscore", "salary_zscore", "purchase_zscore",
    "age_iqr_outlier", "salary_iqr_outlier", "purchase_iqr_outlier",
    "outlier_score",
    "is_valid_date", "is_future_date", "days_since_transaction",
    "rolling_avg_purchase_7",
    "schema_violation_count",
    "city_valid", "country_valid",
    "negative_salary_flag", "negative_purchase_flag",
]


# --------------------------------------------------------------------------
# Reference statistics (fit once on training data, reused everywhere else)
# --------------------------------------------------------------------------
def compute_reference_stats(df: pd.DataFrame, min_category_count: Optional[int] = None) -> dict:
    """Fit population statistics used to compute z-scores, IQR bounds and
    'known-good' category membership. Must be fit on a TRAIN split only."""
    stats_out = {"numeric": {}, "categorical": {}, "fitted_on_rows": int(len(df)), "fitted_at": datetime.now(timezone.utc).isoformat()}

    for col in NUMERIC_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        std = series.std()
        stats_out["numeric"][col] = {
            "mean": float(series.mean()),
            "std": float(std) if std and std > 0 else 1.0,
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "lower_bound": float(q1 - 1.5 * iqr),
            "upper_bound": float(q3 + 1.5 * iqr),
            "median": float(series.median()),
        }

    if min_category_count is None:
        min_category_count = max(20, int(0.001 * len(df)))
    for col in CATEGORICAL_COLUMNS:
        counts = df[col].value_counts()
        known_good = counts[counts >= min_category_count].index.tolist()
        stats_out["categorical"][col] = {
            "known_values": sorted(str(v) for v in known_good),
            "min_category_count": int(min_category_count),
        }

    return stats_out


def save_reference_stats(stats_dict: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats_dict, f, indent=2)


def load_reference_stats(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def engineer_features(raw_df: pd.DataFrame, reference_stats: dict) -> pd.DataFrame:
    """Compute the full engineered feature matrix for `raw_df` using a
    previously-fit `reference_stats` dict. Returns a DataFrame indexed like
    `raw_df` with exactly FEATURE_COLUMNS, no NaNs."""
    df = raw_df.copy()
    n = len(df)
    feats = pd.DataFrame(index=df.index)

    # ---- Missing-value flags ----
    for col, feat_name in [("Age", "age_missing"), ("Salary", "salary_missing"),
                            ("PurchaseAmount", "purchase_missing"), ("City", "city_missing"),
                            ("Country", "country_missing")]:
        feats[feat_name] = df[col].isna().astype(int)
    feats["date_missing"] = df["TransactionDate"].isna().astype(int)
    feats["missing_ratio"] = feats[["age_missing", "salary_missing", "purchase_missing",
                                     "city_missing", "country_missing", "date_missing"]].mean(axis=1)

    # ---- Duplicate flag (exact duplicate of an earlier row in this batch) ----
    dedup_cols = [c for c in CORE_COLUMNS if c in df.columns]
    feats["is_duplicate"] = df.duplicated(subset=dedup_cols, keep="first").astype(int)

    # ---- Z-scores (neutral 0 where the raw value is missing/non-numeric) ----
    numeric_parsed = {}
    zscore_feature_name = {"Age": "age_zscore", "Salary": "salary_zscore", "PurchaseAmount": "purchase_zscore"}
    for col in NUMERIC_COLUMNS:
        s = pd.to_numeric(df[col], errors="coerce")
        numeric_parsed[col] = s
        rs = reference_stats["numeric"][col]
        z = (s - rs["mean"]) / (rs["std"] if rs["std"] else 1.0)
        feats[zscore_feature_name[col]] = z.fillna(0.0)

    # ---- IQR-based outlier flags ----
    for col, feat_name in [("Age", "age_iqr_outlier"), ("Salary", "salary_iqr_outlier"),
                            ("PurchaseAmount", "purchase_iqr_outlier")]:
        s = numeric_parsed[col]
        rs = reference_stats["numeric"][col]
        flag = (s < rs["lower_bound"]) | (s > rs["upper_bound"])
        feats[feat_name] = flag.fillna(False).astype(int)

    # ---- Aggregate outlier score: worst absolute z-score across numeric cols ----
    z_cols = ["age_zscore", "salary_zscore", "purchase_zscore"]
    feats["outlier_score"] = feats[z_cols].abs().max(axis=1)

    # ---- Date validity / freshness ----
    parsed_date = pd.to_datetime(df["TransactionDate"], errors="coerce", format="mixed")
    feats["is_valid_date"] = parsed_date.notna().astype(int)
    feats["is_future_date"] = (parsed_date > TODAY).fillna(False).astype(int)
    days_since = (TODAY - parsed_date).dt.days
    feats["days_since_transaction"] = days_since.fillna(-9999).astype(float)

    # ---- Rolling average purchase amount (temporal context, ordered by date) ----
    purchase_numeric = numeric_parsed["PurchaseAmount"]
    order_df = pd.DataFrame({"date": parsed_date, "purchase": purchase_numeric}, index=df.index)
    order_df_sorted = order_df.sort_values("date", na_position="last")
    rolling = order_df_sorted["purchase"].rolling(window=7, min_periods=1).mean()
    rolling_aligned = rolling.reindex(df.index)  # restore original row order
    fallback_mean = reference_stats["numeric"]["PurchaseAmount"]["mean"]
    feats["rolling_avg_purchase_7"] = rolling_aligned.fillna(fallback_mean)

    # ---- Categorical validity ----
    for col, feat_name in [("City", "city_valid"), ("Country", "country_valid")]:
        known_values = set(reference_stats["categorical"][col]["known_values"])
        feats[feat_name] = df[col].astype(str).isin(known_values).astype(int)
        # explicit NaN -> invalid (isin already returns False for NaN, this just documents intent)

    # ---- Negative value flags ----
    feats["negative_salary_flag"] = (numeric_parsed["Salary"] < 0).fillna(False).astype(int)
    feats["negative_purchase_flag"] = (numeric_parsed["PurchaseAmount"] < 0).fillna(False).astype(int)

    # ---- Schema violation count (business-rule aggregate) ----
    age_valid = numeric_parsed["Age"].between(0, 120)
    violations = (
        feats["age_missing"] + feats["salary_missing"] + feats["purchase_missing"]
        + feats["city_missing"] + feats["country_missing"] + feats["date_missing"]
        + (~age_valid).fillna(True).astype(int)
        + feats["negative_salary_flag"] + feats["negative_purchase_flag"]
        + (1 - feats["city_valid"]) + (1 - feats["country_valid"])
        + (1 - feats["is_valid_date"]) + feats["is_future_date"]
    )
    feats["schema_violation_count"] = violations.astype(int)

    feats = feats[FEATURE_COLUMNS]
    assert feats.isna().sum().sum() == 0, "Engineered features must never contain NaN"
    return feats


def preprocess_dataset(raw_df: pd.DataFrame, reference_stats: Optional[dict] = None,
                        fit_reference: bool = False):
    """Convenience wrapper: optionally fit reference stats on `raw_df`, engineer
    features, and return (features_df, reference_stats)."""
    if fit_reference or reference_stats is None:
        reference_stats = compute_reference_stats(raw_df)
    features = engineer_features(raw_df, reference_stats)
    return features, reference_stats


# --------------------------------------------------------------------------
# CLI / demo entry point
# --------------------------------------------------------------------------
def main():
    here = os.path.dirname(__file__)
    raw_path = os.path.join(here, "..", "data", "raw", "synthetic_data.csv")
    out_path = os.path.join(here, "..", "data", "processed", "processed_data.csv")
    stats_path = os.path.join(here, "..", "data", "processed", "demo_reference_stats.json")

    raw_df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset: {raw_df.shape}")

    # NOTE: this whole-dataset fit is for standalone EDA/demo purposes only.
    # training/train.py fits its own reference stats on the TRAIN split alone
    # (see leak-safety note in the module docstring) and saves the production
    # copy to models/feature_reference_stats.json.
    features, ref_stats = preprocess_dataset(raw_df, fit_reference=True)
    save_reference_stats(ref_stats, stats_path)

    processed = pd.concat([raw_df[["CustomerID"]].reset_index(drop=True),
                            features.reset_index(drop=True),
                            raw_df[["Label"]].reset_index(drop=True)], axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    processed.to_csv(out_path, index=False)

    print(f"Engineered {features.shape[1]} features for {features.shape[0]:,} rows")
    print(f"Saved processed dataset -> {out_path}")
    print(f"Saved demo reference stats -> {stats_path}")
    print("\nFeature summary:")
    print(features.describe().T[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
