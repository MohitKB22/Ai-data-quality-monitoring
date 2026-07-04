"""
FastAPI Service
=================
REST API for the AI-Based Data Quality Monitoring System.

Endpoints (as specified):
  POST /upload     - upload a CSV dataset
  POST /validate   - run the validation suite against a dataset
  POST /train      - (re)train the models on a dataset
  POST /predict    - classify records as Good Data / Bad Data
  GET  /report     - fetch the latest validation report (json or html)
  POST /anomaly    - run Isolation Forest anomaly detection on records
  GET  /dashboard  - info on how to reach the Streamlit dashboard

Plus a couple of small operational bonuses: GET /health, GET /,
POST /monitoring/run, GET /monitoring/history.

Run with:  uvicorn api.app:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.preprocess import CORE_COLUMNS, compute_reference_stats  # noqa: E402
from utils import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from validation.great_expectations import generate_html_report, validate_dataframe  # noqa: E402

logger = get_logger("api")

UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(
    title="AI-Based Data Quality Monitoring System",
    description="Validate, monitor, and classify data quality with ML.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory registry of uploaded datasets: dataset_id -> filepath. Fine for a
# single-process demo API; swap for Redis/DB-backed storage in production.
_UPLOADED_DATASETS: dict[str, str] = {}


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class TransactionRecord(BaseModel):
    CustomerID: Optional[str] = None
    Age: Optional[float] = None
    Salary: Optional[float] = None
    PurchaseAmount: Optional[float] = None
    City: Optional[str] = None
    Country: Optional[str] = None
    TransactionDate: Optional[str] = None


class PredictRequest(BaseModel):
    records: list[TransactionRecord] = Field(..., description="Raw transaction records to score")


class TrainRequest(BaseModel):
    dataset_id: Optional[str] = Field(None, description="Previously-uploaded dataset id; omit to use the default synthetic dataset")


def _resolve_dataset_path(dataset_id: Optional[str]) -> str:
    if dataset_id is None:
        return config.RAW_DATA_PATH
    path = _UPLOADED_DATASETS.get(dataset_id)
    if path is None or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Unknown dataset_id '{dataset_id}'. Upload a file via /upload first.")
    return path


def _records_to_df(records: Optional[list[TransactionRecord]]) -> pd.DataFrame:
    if not records:
        raise HTTPException(status_code=400, detail="No records provided.")
    df = pd.DataFrame([r.model_dump() for r in records])
    for col in CORE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[CORE_COLUMNS]


def _get_predictor():
    from inference.predict import get_predictor
    try:
        return get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Models not trained yet: {e}. Call POST /train first.")


# --------------------------------------------------------------------------
# Root / health
# --------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "AI-Based Data Quality Monitoring System",
        "version": "1.0.0",
        "endpoints": ["/upload", "/validate", "/train", "/predict", "/report", "/anomaly", "/dashboard",
                      "/monitoring/run", "/monitoring/history", "/health", "/docs"],
    }


@app.get("/health")
def health():
    models_ready = os.path.exists(os.path.join(config.MODELS_DIR, "quality_classifier.pkl"))
    return {"status": "ok", "models_ready": models_ready, "time": datetime.now(timezone.utc).isoformat()}


# --------------------------------------------------------------------------
# /upload
# --------------------------------------------------------------------------
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    dataset_id = uuid.uuid4().hex[:12]
    dest_path = os.path.join(UPLOADS_DIR, f"{dataset_id}.csv")
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    try:
        df = pd.read_csv(dest_path)
    except Exception as e:
        os.remove(dest_path)
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    _UPLOADED_DATASETS[dataset_id] = dest_path
    logger.info(f"Uploaded dataset '{file.filename}' -> id={dataset_id} ({len(df):,} rows)")

    return {
        "dataset_id": dataset_id,
        "original_filename": file.filename,
        "row_count": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# /validate
# --------------------------------------------------------------------------
@app.post("/validate")
def validate_dataset(dataset_id: Optional[str] = Query(None, description="Dataset id from /upload; omit for the default dataset")):
    path = _resolve_dataset_path(dataset_id)
    df = pd.read_csv(path)

    # Use the current dataset itself as its own reference for category/drift
    # baselines when no separate reference is available (standalone validation).
    reference_stats = compute_reference_stats(df)
    summary = validate_dataframe(df, reference_stats=reference_stats, reference_df=df,
                                  dataset_name=os.path.basename(path))
    generate_html_report(summary, os.path.join(config.VALIDATION_REPORTS_DIR, "validation_report.html"))

    import json
    with open(os.path.join(config.VALIDATION_REPORTS_DIR, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Validated '{path}': {summary['expectations_passed']}/{summary['expectations_total']} passed")
    return summary


# --------------------------------------------------------------------------
# /train
# --------------------------------------------------------------------------
@app.post("/train")
def train_models(request: TrainRequest = TrainRequest()):
    from training.train import main as train_main
    path = _resolve_dataset_path(request.dataset_id)
    logger.info(f"Training triggered on '{path}' (this can take ~1-2 minutes)...")
    try:
        metadata = train_main(data_path=path)
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    # Force the predictor singleton to reload the freshly-trained models.
    import inference.predict as predict_module
    predict_module._predictor_singleton = None

    return {
        "status": "success",
        "winning_classifier": metadata["winning_classifier"],
        "train_rows": metadata["train_rows"],
        "test_rows": metadata["test_rows"],
        "results": {k: {m: v for m, v in metrics.items() if m in
                         ("accuracy", "precision", "recall", "f1_score", "roc_auc")}
                    for k, metrics in metadata["results"].items()},
    }


# --------------------------------------------------------------------------
# /predict
# --------------------------------------------------------------------------
@app.post("/predict")
def predict(request: Optional[PredictRequest] = None,
            dataset_id: Optional[str] = Query(None, description="Alternative to `records`: score a previously-uploaded dataset")):
    predictor = _get_predictor()

    if dataset_id is not None:
        df = pd.read_csv(_resolve_dataset_path(dataset_id))[CORE_COLUMNS]
    else:
        df = _records_to_df(request.records if request else None)

    result = predictor.predict(df)
    return {
        "row_count": len(result),
        "bad_data_count": int((result["predicted_label"] == "Bad Data").sum()),
        "predictions": result[["CustomerID", "predicted_label", "bad_data_probability", "confidence"]].to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# /report
# --------------------------------------------------------------------------
@app.get("/report")
def get_report(format: str = Query("json", pattern="^(json|html)$")):
    json_path = os.path.join(config.VALIDATION_REPORTS_DIR, "validation_summary.json")
    html_path = os.path.join(config.VALIDATION_REPORTS_DIR, "validation_report.html")

    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="No validation report yet. Call POST /validate first.")

    if format == "html":
        with open(html_path) as f:
            return HTMLResponse(f.read())

    import json
    with open(json_path) as f:
        return JSONResponse(json.load(f))


# --------------------------------------------------------------------------
# /anomaly
# --------------------------------------------------------------------------
@app.post("/anomaly")
def detect_anomalies(request: Optional[PredictRequest] = None,
                      dataset_id: Optional[str] = Query(None, description="Alternative to `records`: score a previously-uploaded dataset")):
    predictor = _get_predictor()

    if dataset_id is not None:
        df = pd.read_csv(_resolve_dataset_path(dataset_id))[CORE_COLUMNS]
    else:
        df = _records_to_df(request.records if request else None)

    result = predictor.detect_anomalies(df)
    return {
        "row_count": len(result),
        "anomaly_count": int(result["is_anomaly"].sum()),
        "anomaly_rate_percent": round(float(result["is_anomaly"].mean() * 100), 2),
        "results": result[["CustomerID", "is_anomaly", "anomaly_score"]].to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# /dashboard
# --------------------------------------------------------------------------
@app.get("/dashboard")
def dashboard_info():
    return {
        "message": "The Streamlit dashboard runs as a separate process (Streamlit apps are not served through FastAPI).",
        "launch_command": "streamlit run dashboard/dashboard.py",
        "default_url": "http://localhost:8501",
    }


# --------------------------------------------------------------------------
# Bonus: monitoring endpoints
# --------------------------------------------------------------------------
@app.post("/monitoring/run")
def trigger_monitoring_run(dataset_id: Optional[str] = Query(None)):
    from monitoring.monitor import run_monitoring_cycle
    from preprocessing.preprocess import load_reference_stats

    path = _resolve_dataset_path(dataset_id)
    df = pd.read_csv(path)
    ref_stats_path = os.path.join(config.MODELS_DIR, "feature_reference_stats.json")
    if not os.path.exists(ref_stats_path):
        raise HTTPException(status_code=503, detail="No trained reference stats yet. Call POST /train first.")
    reference_stats = load_reference_stats(ref_stats_path)

    predictor = None
    try:
        predictor = _get_predictor()
    except HTTPException:
        pass

    metrics = run_monitoring_cycle(df, reference_stats, reference_df=df, predictor=predictor,
                                    dataset_name=os.path.basename(path))
    return metrics


@app.get("/monitoring/history")
def monitoring_history(limit: int = 50):
    from utils import db
    conn = db.get_connection()
    runs = db.fetch_recent_runs(conn, limit=limit)
    conn.close()
    return {"runs": runs}
