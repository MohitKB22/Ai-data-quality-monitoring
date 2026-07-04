"""
API tests.

NOTE: `/train` is deliberately NOT covered here. It's fully functional (see
the manual end-to-end run captured in README > API Documentation), but
calling it -- even against a tiny dataset -- overwrites the real
models/*.pkl files that ship with this repo, since training.train.main()
always saves to the fixed MODELS_DIR regardless of which input dataset it
trained on. Exercising it automatically on every `pytest` run would silently
replace the pre-trained, README-documented models with whatever the test
happened to train on. If you want to test it yourself, monkeypatch
`training.train.MODELS_DIR` / `IMAGES_DIR` to a tmp_path first.
"""

import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.app import app  # noqa: E402
from inference.predict import CLASSIFIER_PATH  # noqa: E402

client = TestClient(app)
MODELS_READY = os.path.exists(CLASSIFIER_PATH)


def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_dashboard_info_endpoint():
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "streamlit" in r.json()["launch_command"].lower()


def test_validate_endpoint_default_dataset():
    r = client.post("/validate")
    assert r.status_code == 200
    body = r.json()
    assert body["expectations_total"] > 0


def test_report_endpoint_json_after_validate():
    client.post("/validate")  # ensure a report exists
    r = client.get("/report")
    assert r.status_code == 200
    assert "row_count" in r.json()


def test_report_endpoint_html_format():
    client.post("/validate")
    r = client.get("/report?format=html")
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text


def test_upload_rejects_non_csv():
    r = client.post("/upload", files={"file": ("data.txt.exe", b"not a csv", "application/octet-stream")})
    assert r.status_code == 400


def test_upload_accepts_valid_csv():
    small = pd.DataFrame({
        "CustomerID": ["C1", "C2"], "Age": [30, 40], "Salary": [50000, 60000],
        "PurchaseAmount": [100, 200], "City": ["Paris", "Tokyo"], "Country": ["France", "Japan"],
        "TransactionDate": ["2026-01-01", "2026-01-02"],
    })
    csv_bytes = small.to_csv(index=False).encode()
    r = client.post("/upload", files={"file": ("test_upload.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["row_count"] == 2
    assert "dataset_id" in body


def test_validate_unknown_dataset_id_returns_404():
    r = client.post("/validate?dataset_id=does-not-exist")
    assert r.status_code == 404


@pytest.mark.skipif(not MODELS_READY, reason="Models not trained yet")
def test_predict_endpoint_with_inline_records():
    payload = {"records": [
        {"CustomerID": "C1", "Age": 35, "Salary": -1000, "PurchaseAmount": 50,
         "City": "Paris", "Country": "France", "TransactionDate": "2026-01-01"},
    ]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["row_count"] == 1


@pytest.mark.skipif(not MODELS_READY, reason="Models not trained yet")
def test_predict_endpoint_requires_input():
    r = client.post("/predict", json={"records": []})
    assert r.status_code == 400


@pytest.mark.skipif(not MODELS_READY, reason="Models not trained yet")
def test_anomaly_endpoint_with_inline_records():
    payload = {"records": [
        {"CustomerID": "C1", "Age": 35, "Salary": 55000, "PurchaseAmount": 90,
         "City": "Paris", "Country": "France", "TransactionDate": "2026-01-01"},
    ]}
    r = client.post("/anomaly", json=payload)
    assert r.status_code == 200
    assert "anomaly_count" in r.json()


@pytest.mark.skipif(not MODELS_READY, reason="Models not trained yet")
def test_predict_without_trained_models_returns_503(monkeypatch):
    import inference.predict as predict_module

    def _raise_not_found():
        raise FileNotFoundError("Missing model artifact(s) (simulated for this test)")

    # api/app.py's _get_predictor() does `from inference.predict import get_predictor`
    # freshly inside the function body on every call, so patching the module-level
    # name here is picked up on the next request without needing a server restart.
    monkeypatch.setattr(predict_module, "_predictor_singleton", None)
    monkeypatch.setattr(predict_module, "get_predictor", _raise_not_found)

    payload = {"records": [{"CustomerID": "C1", "Age": 35}]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 503
