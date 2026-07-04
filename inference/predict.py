"""
Inference
==========
Loads the persisted classifier, anomaly detector, and reference statistics to
score new (raw, unprocessed) records. This is the module the FastAPI `/predict`
and `/anomaly` endpoints call, and it's also usable standalone or from the
Streamlit dashboard.
"""

from __future__ import annotations

import os
import sys

import joblib
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.preprocess import CORE_COLUMNS, engineer_features, load_reference_stats  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "quality_classifier.pkl")
ANOMALY_MODEL_PATH = os.path.join(MODELS_DIR, "anomaly_model.pkl")
REFERENCE_STATS_PATH = os.path.join(MODELS_DIR, "feature_reference_stats.json")
LABEL_MAP_INV = {0: "Good Data", 1: "Bad Data"}


class DataQualityPredictor:
    """Loads all artifacts once and exposes predict / anomaly-detect methods."""

    def __init__(self, classifier_path: str = CLASSIFIER_PATH, anomaly_model_path: str = ANOMALY_MODEL_PATH,
                 reference_stats_path: str = REFERENCE_STATS_PATH):
        missing = [p for p in [classifier_path, anomaly_model_path, reference_stats_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifact(s): {missing}. Run `python training/train.py` first."
            )
        self.classifier = joblib.load(classifier_path)
        self.anomaly_model = joblib.load(anomaly_model_path)
        self.reference_stats = load_reference_stats(reference_stats_path)

    def _validate_columns(self, df: pd.DataFrame):
        missing_cols = [c for c in CORE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Input data is missing required column(s): {missing_cols}. "
                              f"Expected columns: {CORE_COLUMNS}")

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Return per-row Good/Bad Data predictions with class probability."""
        self._validate_columns(raw_df)
        features = engineer_features(raw_df, self.reference_stats)
        proba_bad = self.classifier.predict_proba(features)[:, 1]
        pred = (proba_bad >= 0.5).astype(int)

        result = raw_df.copy().reset_index(drop=True)
        result["predicted_label"] = [LABEL_MAP_INV[p] for p in pred]
        result["bad_data_probability"] = proba_bad.round(4)
        result["confidence"] = [max(p, 1 - p) for p in proba_bad.round(4)]
        return result

    def detect_anomalies(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Return per-row anomaly flags from the (unsupervised) Isolation Forest."""
        self._validate_columns(raw_df)
        features = engineer_features(raw_df, self.reference_stats)
        raw_pred = self.anomaly_model.predict(features)               # -1 = anomaly, 1 = normal
        anomaly_score = -self.anomaly_model.decision_function(features)  # higher = more anomalous

        result = raw_df.copy().reset_index(drop=True)
        result["is_anomaly"] = raw_pred == -1
        result["anomaly_score"] = anomaly_score.round(4)
        return result

    def predict_full(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Combined classifier + anomaly-detector view, used by the dashboard."""
        preds = self.predict(raw_df)
        anomalies = self.detect_anomalies(raw_df)
        preds["is_anomaly"] = anomalies["is_anomaly"]
        preds["anomaly_score"] = anomalies["anomaly_score"]
        return preds


_predictor_singleton: DataQualityPredictor | None = None


def get_predictor() -> DataQualityPredictor:
    """Lazily-instantiated singleton so the API/dashboard only load models once."""
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = DataQualityPredictor()
    return _predictor_singleton


def main():
    here = os.path.dirname(__file__)
    sample_path = os.path.join(PROJECT_ROOT, "data", "raw", "synthetic_data.csv")
    df = pd.read_csv(sample_path).sample(10, random_state=7)[CORE_COLUMNS]

    predictor = get_predictor()
    result = predictor.predict_full(df)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("Sample predictions on 10 random raw records:\n")
    print(result[["CustomerID", "Age", "Salary", "PurchaseAmount", "predicted_label",
                   "bad_data_probability", "is_anomaly", "anomaly_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
