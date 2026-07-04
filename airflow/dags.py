"""
Airflow DAG: AI-Based Data Quality Monitoring Pipeline
=========================================================
Orchestrates the daily batch pipeline: Ingest -> Validate -> Preprocess ->
Train -> Evaluate -> Generate Report -> Deploy.

Targets Airflow 3.x. Airflow 3 moved the stable Dag-authoring interface to
the `airflow.sdk` namespace (`from airflow.sdk import dag, task`) -- the
legacy `from airflow import DAG` / `from airflow.decorators import task`
imports still work on many 3.x installs but are deprecated and slated for
removal, so this file uses the current interface.

This DAG is NOT executed as part of the project's own test/CI suite (running
a full Airflow scheduler is out of scope for a portfolio project's CI), but
it is written to drop directly into an Airflow `dags/` folder:

    cp airflow/dags.py $AIRFLOW_HOME/dags/data_quality_pipeline.py

Each task is a thin wrapper around the project's real modules (the same
`preprocessing`, `training`, `validation`, `monitoring` packages used by the
API/CLI/dashboard) -- Airflow is only responsible for scheduling and
dependency ordering, not for containing business logic.
"""

from __future__ import annotations

import datetime
import os
import sys

import pendulum

from airflow.sdk import dag, task

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_ARGS = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": datetime.timedelta(minutes=5),
}

# Minimum test-set F1 the newly-trained classifier must reach for the
# "deploy" task to approve it -- a simple, explicit quality gate.
MIN_DEPLOYABLE_F1 = 0.60


@dag(
    dag_id="ai_data_quality_monitoring_pipeline",
    description="Ingest -> Validate -> Preprocess -> Train -> Evaluate -> Report -> Deploy",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["data-quality", "machine-learning", "monitoring"],
)
def ai_data_quality_monitoring_pipeline():

    @task()
    def ingest_data() -> str:
        """Stand-in for pulling a fresh batch from a real source system
        (warehouse, event stream, vendor feed, ...). Here it just ensures the
        synthetic dataset exists and returns its path."""
        from utils import config

        if not os.path.exists(config.RAW_DATA_PATH):
            from data.generate_synthetic_data import main as generate_main
            generate_main()
        return config.RAW_DATA_PATH

    @task()
    def validate_data(data_path: str) -> dict:
        """Run the validation suite and persist the HTML/JSON reports."""
        import json

        import pandas as pd

        from preprocessing.preprocess import compute_reference_stats
        from utils import config
        from validation.great_expectations import generate_html_report, validate_dataframe

        df = pd.read_csv(data_path)
        reference_stats = compute_reference_stats(df)
        summary = validate_dataframe(df, reference_stats=reference_stats, reference_df=df,
                                      dataset_name=os.path.basename(data_path))
        generate_html_report(summary, os.path.join(config.VALIDATION_REPORTS_DIR, "validation_report.html"))
        with open(os.path.join(config.VALIDATION_REPORTS_DIR, "validation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Keep the XCom payload small: only headline numbers, not the full per-expectation list.
        return {
            "expectations_passed": summary["expectations_passed"],
            "expectations_total": summary["expectations_total"],
            "null_percent_overall": summary["null_percent_overall"],
            "duplicate_percent": summary["duplicate_percent"],
        }

    @task()
    def preprocess_data(data_path: str) -> str:
        """Fit reference stats + engineer features, save the processed dataset."""
        import pandas as pd

        from preprocessing.preprocess import preprocess_dataset, save_reference_stats
        from utils import config

        df = pd.read_csv(data_path)
        features, reference_stats = preprocess_dataset(df, fit_reference=True)
        save_reference_stats(reference_stats, os.path.join(config.DATA_PROCESSED_DIR, "airflow_reference_stats.json"))

        out_path = os.path.join(config.DATA_PROCESSED_DIR, "airflow_processed_data.csv")
        pd.concat([df[["CustomerID"]].reset_index(drop=True), features.reset_index(drop=True),
                   df[["Label"]].reset_index(drop=True)], axis=1).to_csv(out_path, index=False)
        return out_path

    @task()
    def train_model(data_path: str) -> dict:
        """Retrain RandomForest / XGBoost / IsolationForest on the ingested batch."""
        from training.train import main as train_main

        metadata = train_main(data_path=data_path)
        winner = metadata["winning_classifier"]
        return {
            "winning_classifier": winner,
            "f1_score": metadata["results"][winner]["f1_score"],
            "roc_auc": metadata["results"][winner]["roc_auc"],
            "accuracy": metadata["results"][winner]["accuracy"],
        }

    @task()
    def evaluate_model(train_result: dict) -> dict:
        """Decide whether the freshly-trained model clears the deployability bar."""
        approved = train_result["f1_score"] >= MIN_DEPLOYABLE_F1
        return {**train_result, "approved_for_deployment": approved, "min_required_f1": MIN_DEPLOYABLE_F1}

    @task()
    def generate_report(validation_result: dict, eval_result: dict) -> str:
        """Write a small timestamped run summary alongside the HTML validation report."""
        import json
        from datetime import datetime, timezone

        from utils import config

        run_summary = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "validation": validation_result,
            "model_evaluation": eval_result,
        }
        report_path = os.path.join(config.VALIDATION_REPORTS_DIR, "pipeline_run_summary.json")
        with open(report_path, "w") as f:
            json.dump(run_summary, f, indent=2)
        return report_path

    @task()
    def deploy_model(eval_result: dict) -> None:
        """Record the deploy/reject decision. `train_model` already wrote the
        artifacts directly to models/*.pkl; a fuller setup would train to a
        staging path and only copy-to-production here on gate pass (see
        README > Future Improvements)."""
        import json
        from datetime import datetime, timezone

        from utils import config

        decision = "DEPLOYED" if eval_result["approved_for_deployment"] else "REJECTED"
        log_entry = {"decided_at": datetime.now(timezone.utc).isoformat(), "decision": decision, **eval_result}
        log_path = os.path.join(config.MODELS_DIR, "deployment_log.json")

        history = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                history = json.load(f)
        history.append(log_entry)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        if decision == "REJECTED":
            raise ValueError(
                f"Model F1={eval_result['f1_score']:.3f} is below the "
                f"{eval_result['min_required_f1']} deployment gate -- failing task to alert on-call."
            )

    # ---- Dependency graph ----
    raw_path = ingest_data()
    validation_result = validate_data(raw_path)
    # preprocess_data materializes engineered features for downstream/analytics consumers.
    # train_model deliberately reuses raw_path (not processed_path): training.train fits its
    # own leak-safe train/test split and reference statistics internally (see training/train.py
    # module docstring), so it must start from raw data rather than an already-engineered file.
    processed_path = preprocess_data(raw_path)
    train_result = train_model(raw_path)
    eval_result = evaluate_model(train_result)
    report_path = generate_report(validation_result, eval_result)
    deploy_model(eval_result)


ai_data_quality_monitoring_pipeline()
