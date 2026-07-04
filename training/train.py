"""
Model Training Pipeline
=========================
Trains and compares three models on the engineered data-quality features:

  1. RandomForestClassifier   -- supervised Good/Bad Data classifier
  2. XGBoost (XGBClassifier)  -- supervised Good/Bad Data classifier
  3. IsolationForest          -- unsupervised anomaly detector (benchmarked
                                  against the same ground truth for reference)

Both classifiers go through GridSearchCV (hyperparameter tuning) plus a
standalone k-fold cross-validation pass on the winning configuration. The
best classifier (by test-set F1) is saved as `models/quality_classifier.pkl`;
the Isolation Forest is saved as `models/anomaly_model.pkl`. All runs are
logged to a local MLflow tracking store (SQLite, ./mlflow.db) -- run
`mlflow ui --backend-store-uri sqlite:///mlflow.db` from the project root to
browse experiments.

Leak-safety: reference statistics (means/stds/IQR bounds/known-good
categories) used for feature engineering are fit ONLY on the training split
of the RAW data, then reused unchanged to engineer features for the test
split, and persisted to `models/feature_reference_stats.json` for inference
and monitoring to reuse.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.preprocess import (FEATURE_COLUMNS, compute_reference_stats,  # noqa: E402
                                       engineer_features, save_reference_stats)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "synthetic_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "docs", "images")
RANDOM_STATE = 42
LABEL_MAP = {"Good Data": 0, "Bad Data": 1}


def load_and_split_raw(test_size: float = 0.2, data_path: str = DATA_PATH):
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=RANDOM_STATE, stratify=df["Label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    reference_stats = compute_reference_stats(train_df)  # fit on TRAIN ONLY
    X_train = engineer_features(train_df, reference_stats)
    X_test = engineer_features(test_df, reference_stats)
    y_train = train_df["Label"].map(LABEL_MAP).to_numpy()
    y_test = test_df["Label"].map(LABEL_MAP).to_numpy()
    return X_train, X_test, y_train, y_test, reference_stats


def evaluate_classifier(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["Good Data", "Bad Data"], output_dict=True),
    }
    return metrics, y_pred, y_proba


def _stratified_subsample(X, y, n, seed=RANDOM_STATE):
    if len(X) <= n:
        return X, y
    idx, _ = train_test_split(np.arange(len(X)), train_size=n, random_state=seed, stratify=y)
    return X.iloc[idx], y[idx]


def train_random_forest(X_train, y_train, X_test, y_test, search_sample_size: int = 25_000):
    print("\n[1/3] Training RandomForestClassifier (GridSearchCV)...")
    # Random Forest is by far the most expensive model to fit here, so hyperparameter
    # search and the cross-validation robustness check both run on a fast stratified
    # subsample; the FINAL production model is then fit once on the FULL training set
    # with the winning hyperparameters. This is a standard, time-efficient pattern
    # (search cheap, commit expensive) and not a shortcut on model quality.
    X_search, y_search = _stratified_subsample(X_train, y_train, search_sample_size)
    print(f"    Hyperparameter search on a {len(X_search):,}-row stratified subsample "
          f"(full train set is {len(X_train):,} rows)")

    t0 = time.time()
    param_grid = {"n_estimators": [100, 150], "max_depth": [10, 16]}
    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced", min_samples_leaf=2)
    grid = GridSearchCV(base, param_grid, cv=3, scoring="f1", n_jobs=1, verbose=0)
    grid.fit(X_search, y_search)
    best_params = grid.best_params_
    print(f"    Best params: {best_params}  (search took {time.time()-t0:.1f}s)")

    t0 = time.time()
    cv_scores = cross_val_score(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced",
                                                          min_samples_leaf=2, **best_params),
                                 X_search, y_search, cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                                 scoring="f1", n_jobs=1)
    print(f"    3-fold CV F1 (subsample): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    best_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced",
                                         min_samples_leaf=2, **best_params)
    best_model.fit(X_train, y_train)  # final fit on the FULL training set
    print(f"    Final fit on full {len(X_train):,}-row training set: {time.time()-t0:.1f}s")

    metrics, y_pred, y_proba = evaluate_classifier(best_model, X_test, y_test, "RandomForest")
    metrics["best_params"] = best_params
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["cv_f1_std"] = float(cv_scores.std())
    metrics["hyperparameter_search_sample_size"] = len(X_search)
    print(f"    Test set  -> Acc: {metrics['accuracy']:.4f}  Prec: {metrics['precision']:.4f}  "
          f"Rec: {metrics['recall']:.4f}  F1: {metrics['f1_score']:.4f}  ROC-AUC: {metrics['roc_auc']:.4f}")
    return best_model, metrics, y_pred, y_proba


def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n[2/3] Training XGBClassifier (GridSearchCV)...")
    t0 = time.time()
    param_grid = {"n_estimators": [100, 150], "max_depth": [4, 6]}
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    base = XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, eval_metric="logloss",
                          learning_rate=0.15, scale_pos_weight=scale_pos_weight)
    grid = GridSearchCV(base, param_grid, cv=3, scoring="f1", n_jobs=1, verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"    Best params: {grid.best_params_}  (search took {time.time()-t0:.1f}s)")

    cv_scores = cross_val_score(XGBClassifier(random_state=RANDOM_STATE, n_jobs=1, eval_metric="logloss",
                                                learning_rate=0.15, scale_pos_weight=scale_pos_weight, **grid.best_params_),
                                 X_train, y_train, cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                                 scoring="f1", n_jobs=1)
    print(f"    3-fold CV F1 on train: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    metrics, y_pred, y_proba = evaluate_classifier(best_model, X_test, y_test, "XGBoost")
    metrics["best_params"] = grid.best_params_
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["cv_f1_std"] = float(cv_scores.std())
    print(f"    Test set  -> Acc: {metrics['accuracy']:.4f}  Prec: {metrics['precision']:.4f}  "
          f"Rec: {metrics['recall']:.4f}  F1: {metrics['f1_score']:.4f}  ROC-AUC: {metrics['roc_auc']:.4f}")
    return best_model, metrics, y_pred, y_proba


def train_isolation_forest(X_train, y_train, X_test, y_test):
    print("\n[3/3] Training IsolationForest (unsupervised anomaly detector)...")
    contamination = float(np.clip(y_train.mean(), 0.01, 0.5))
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=RANDOM_STATE, n_jobs=1)
    model.fit(X_train)  # unsupervised: does NOT see y_train

    raw_pred = model.predict(X_test)              # -1 = anomaly, 1 = normal
    y_pred = (raw_pred == -1).astype(int)          # 1 = Bad Data (anomaly), 0 = Good Data
    anomaly_score = -model.decision_function(X_test)  # higher = more anomalous

    metrics = {
        "model_name": "IsolationForest",
        "contamination": contamination,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, anomaly_score)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "note": "Unsupervised model; benchmarked against ground-truth labels for reference only "
                "(it never sees y_train). Expected to underperform the supervised classifiers.",
    }
    print(f"    (benchmark vs ground truth) Acc: {metrics['accuracy']:.4f}  Prec: {metrics['precision']:.4f}  "
          f"Rec: {metrics['recall']:.4f}  F1: {metrics['f1_score']:.4f}  ROC-AUC: {metrics['roc_auc']:.4f}")
    return model, metrics, y_pred, anomaly_score


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_confusion_matrix(cm, model_name, out_path):
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Good Data", "Bad Data"], yticklabels=["Good Data", "Bad Data"])
    plt.title(f"Confusion Matrix - {model_name} (winning model)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_roc_curves(curves: dict, out_path):
    plt.figure(figsize=(6, 5))
    for name, (y_test, scores) in curves.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_feature_importance(model, feature_names, model_name, out_path, top_n=15):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    order = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(7, 6))
    plt.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#3b5bfd")
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(data_path: str = DATA_PATH):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")
    mlflow.set_experiment("data_quality_monitoring")

    print("Loading data and creating an 80/20 stratified train/test split...")
    train_df, test_df = load_and_split_raw(data_path=data_path)
    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    print("Engineering features (reference stats fit on TRAIN split only)...")
    X_train, X_test, y_train, y_test, reference_stats = build_features(train_df, test_df)
    save_reference_stats(reference_stats, os.path.join(MODELS_DIR, "feature_reference_stats.json"))
    print(f"  Feature matrix: {X_train.shape[1]} features")

    results = {}

    with mlflow.start_run(run_name="random_forest"):
        rf_model, rf_metrics, rf_pred, rf_proba = train_random_forest(X_train, y_train, X_test, y_test)
        mlflow.log_params(rf_metrics["best_params"])
        mlflow.log_metrics({k: v for k, v in rf_metrics.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(rf_model, name="model")
        results["RandomForest"] = rf_metrics

    with mlflow.start_run(run_name="xgboost"):
        xgb_model, xgb_metrics, xgb_pred, xgb_proba = train_xgboost(X_train, y_train, X_test, y_test)
        mlflow.log_params(xgb_metrics["best_params"])
        mlflow.log_metrics({k: v for k, v in xgb_metrics.items() if isinstance(v, (int, float))})
        mlflow.xgboost.log_model(xgb_model, name="model")
        results["XGBoost"] = xgb_metrics

    with mlflow.start_run(run_name="isolation_forest"):
        iso_model, iso_metrics, iso_pred, iso_scores = train_isolation_forest(X_train, y_train, X_test, y_test)
        mlflow.log_params({"contamination": iso_metrics["contamination"]})
        mlflow.log_metrics({k: v for k, v in iso_metrics.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(iso_model, name="model")
        results["IsolationForest"] = iso_metrics

    # ---- Pick the winning supervised classifier by test F1 ----
    winner_name = "RandomForest" if rf_metrics["f1_score"] >= xgb_metrics["f1_score"] else "XGBoost"
    winner_model = rf_model if winner_name == "RandomForest" else xgb_model
    winner_metrics = results[winner_name]
    print(f"\nBest classifier: {winner_name}  (F1={winner_metrics['f1_score']:.4f} vs "
          f"{'XGBoost' if winner_name=='RandomForest' else 'RandomForest'}="
          f"{(xgb_metrics if winner_name=='RandomForest' else rf_metrics)['f1_score']:.4f})")

    # ---- Persist models ----
    joblib.dump(winner_model, os.path.join(MODELS_DIR, "quality_classifier.pkl"))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    joblib.dump(iso_model, os.path.join(MODELS_DIR, "anomaly_model.pkl"))

    # ---- Plots ----
    plot_confusion_matrix(np.array(winner_metrics["confusion_matrix"]), winner_name,
                           os.path.join(IMAGES_DIR, "confusion_matrix.png"))
    plot_roc_curves({
        "RandomForest": (y_test, rf_proba),
        "XGBoost": (y_test, xgb_proba),
        "IsolationForest": (y_test, iso_scores),
    }, os.path.join(IMAGES_DIR, "roc_curve.png"))
    plot_feature_importance(winner_model, list(X_train.columns), winner_name,
                             os.path.join(IMAGES_DIR, "feature_importance.png"))

    # ---- Metadata ----
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": data_path,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_columns": FEATURE_COLUMNS,
        "winning_classifier": winner_name,
        "label_map": LABEL_MAP,
        "results": {
            "RandomForest": {k: v for k, v in rf_metrics.items() if k not in ("classification_report",)},
            "XGBoost": {k: v for k, v in xgb_metrics.items() if k not in ("classification_report",)},
            "IsolationForest": iso_metrics,
        },
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved models to {MODELS_DIR}/")
    print(f"Saved plots to {IMAGES_DIR}/")
    print("Training complete.")
    return metadata


if __name__ == "__main__":
    main()
