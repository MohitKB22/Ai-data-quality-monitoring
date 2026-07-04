#!/usr/bin/env python3
"""
AI-Based Data Quality Monitoring System -- Pipeline Orchestrator
====================================================================
Runs the full local pipeline end-to-end:

    1. Generate the synthetic dataset (if it doesn't already exist)
    2. Run the validation suite -> HTML + JSON reports
    3. Engineer features -> processed dataset
    4. Train RandomForest / XGBoost / IsolationForest (skipped if models
       already exist, unless --retrain is passed -- training takes ~2 minutes)
    5. Run one monitoring cycle (KPIs + alert evaluation) on a data slice
    6. Print a summary and next-step commands for the API and dashboard

Usage:
    python run.py                 # first run: generates data, trains models
    python run.py --retrain       # force retraining even if models exist
    python run.py --skip-training # generate/validate/monitor only, no ML training
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("run")


def _step(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Run the AI Data Quality Monitoring pipeline end-to-end.")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if models already exist")
    parser.add_argument("--skip-training", action="store_true", help="Skip the ML training step entirely")
    args = parser.parse_args()

    t_start = time.time()

    # ---- 1. Data ----
    _step("STEP 1/5  Synthetic dataset")
    if not os.path.exists(config.RAW_DATA_PATH):
        print("No dataset found -- generating synthetic_data.csv (this takes a few seconds)...")
        from data.generate_synthetic_data import main as generate_main
        generate_main()
    else:
        import pandas as pd
        n_rows = len(pd.read_csv(config.RAW_DATA_PATH, usecols=[0]))
        print(f"Found existing dataset at {config.RAW_DATA_PATH} ({n_rows:,} rows). Delete it to regenerate.")

    # ---- 2. Validation ----
    _step("STEP 2/5  Data validation")
    import pandas as pd
    from preprocessing.preprocess import compute_reference_stats
    from validation.great_expectations import generate_html_report, validate_dataframe

    df = pd.read_csv(config.RAW_DATA_PATH)
    reference_stats_full = compute_reference_stats(df)
    validation_summary = validate_dataframe(df, reference_stats=reference_stats_full, reference_df=df,
                                             dataset_name="synthetic_data.csv")
    generate_html_report(validation_summary, os.path.join(config.VALIDATION_REPORTS_DIR, "validation_report.html"))
    print(f"Validation: {validation_summary['expectations_passed']}/{validation_summary['expectations_total']} "
          f"expectations passed. Report -> {config.VALIDATION_REPORTS_DIR}/validation_report.html")

    # ---- 3. Feature engineering (demo/analytics artifact) ----
    _step("STEP 3/5  Feature engineering")
    from preprocessing.preprocess import engineer_features, save_reference_stats
    features = engineer_features(df, reference_stats_full)
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(config.DATA_PROCESSED_DIR, "processed_data.csv")
    pd.concat([df[["CustomerID"]], features, df[["Label"]]], axis=1).to_csv(processed_path, index=False)
    save_reference_stats(reference_stats_full, os.path.join(config.DATA_PROCESSED_DIR, "demo_reference_stats.json"))
    print(f"Engineered {features.shape[1]} features for {len(features):,} rows -> {processed_path}")

    # ---- 4. Training ----
    _step("STEP 4/5  Model training")
    models_exist = os.path.exists(os.path.join(config.MODELS_DIR, "quality_classifier.pkl"))
    if args.skip_training:
        print("--skip-training passed -- skipping model training.")
    elif models_exist and not args.retrain:
        print("Trained models already exist in models/ -- skipping (pass --retrain to force).")
    else:
        print("Training RandomForest, XGBoost, and Isolation Forest (~1-2 minutes on a single core)...")
        from training.train import main as train_main
        train_main()

    # ---- 5. Monitoring ----
    _step("STEP 5/5  Monitoring cycle")
    from monitoring.monitor import run_monitoring_cycle
    predictor = None
    try:
        from inference.predict import get_predictor
        predictor = get_predictor()
    except FileNotFoundError:
        print("(No trained models available -- monitoring will run without prediction-confidence tracking.)")

    metrics = run_monitoring_cycle(df, reference_stats_full, reference_df=df, predictor=predictor,
                                    dataset_name="synthetic_data.csv (full dataset)")
    print(f"Quality score: {metrics['quality_score']}/100 | Missing: {metrics['missing_percent']}% | "
          f"Duplicate: {metrics['duplicate_percent']}% | Outlier: {metrics['outlier_percent']}% | "
          f"Drift: {metrics['drift_detected']} | Alerts fired: {metrics['alert_count']}")

    elapsed = time.time() - t_start
    _step(f"Pipeline complete in {elapsed:.1f}s")
    print("""Next steps:
  Start the REST API:        uvicorn api.app:app --reload --port 8000   (docs at /docs)
  Start the dashboard:       streamlit run dashboard/dashboard.py       (opens at :8501)
  Re-run with fresh models:  python run.py --retrain
  Run the test suite:        pytest tests/ -v
""")


if __name__ == "__main__":
    main()
