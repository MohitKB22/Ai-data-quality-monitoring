# AI-Based Data Quality Monitoring System

An end-to-end system that continuously monitors datasets, validates them against a Great-Expectations-style rule suite, detects anomalies and data-quality issues with machine learning, tracks distribution drift over time, and fires alerts when quality degrades -- with a REST API, a live Streamlit dashboard, and an Airflow DAG to orchestrate it all on a schedule.

Built as a complete, runnable reference project: a real 103,000-row synthetic dataset with deliberately injected data-quality issues, real trained/evaluated models, real validation reports, and a real (if intentionally lightweight) SQLite-backed monitoring history -- not a scaffold of empty functions.

---

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Quickstart](#installation--quickstart)
- [The Dataset](#the-dataset)
- [Data Validation](#data-validation)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [Monitoring & Alerting](#monitoring--alerting)
- [REST API](#rest-api)
- [Dashboard](#dashboard)
- [Airflow Orchestration](#airflow-orchestration)
- [Docker](#docker)
- [Testing](#testing)
- [Database Schema](#database-schema)
- [Design Decisions & Trade-offs](#design-decisions--trade-offs)
- [Future Improvements](#future-improvements)

---

## Features

- **Synthetic data generator** producing 100K+ rows with realistic, controllable data-quality issues: missing values, duplicate rows, statistical outliers, malformed/future dates, negative monetary values, invalid categories, and a genuine distribution-drift period.
- **Validation engine** (`validation/great_expectations.py`) -- a Great-Expectations-style expectation suite (null %, duplicate %, range checks, regex, date validity, categorical validity, KS-test distribution drift) that renders a polished, self-contained HTML report.
- **Feature engineering** that turns raw/dirty records into 24 clean, always-non-null numeric quality-signal features (z-scores, IQR outlier flags, missing ratios, schema-violation counts, rolling averages, date freshness).
- **Three trained, evaluated models**: RandomForest + XGBoost classifiers (GridSearchCV + cross-validation) and an Isolation Forest anomaly detector, with real metrics, confusion matrices, and ROC curves.
- **MLOps plumbing**: leak-safe reference-statistic fitting (train-split only), MLflow experiment tracking, joblib model persistence, versioned model metadata.
- **Monitoring & alerting**: a composite 0-100 quality score, SQLite-backed run history, and threshold-based alerts across console / mocked-email / mocked-Slack channels.
- **FastAPI service** with 7+ endpoints, OpenAPI docs, and an in-memory upload registry.
- **Streamlit dashboard**: KPIs, validation report browser, model performance, an interactive prediction/anomaly tool, live monitoring trends, and an alerts feed.
- **Airflow 3.x DAG** (TaskFlow API) orchestrating Ingest -> Validate -> Preprocess -> Train -> Evaluate -> Report -> Deploy, with an explicit F1 quality gate.
- **54 passing pytest tests**, a GitHub Actions CI workflow, and a production Dockerfile.

## Architecture

<p align="center"><img src="docs/images/architecture_diagram.png" width="620" alt="System architecture diagram"></p>

Data flows top to bottom: raw records are ingested (via the synthetic generator, a CSV upload, or the Airflow DAG), validated against the expectation suite, and converted into engineered features. Those features train three models tracked in MLflow and persisted as `models/*.pkl`. The FastAPI service and Streamlit dashboard both sit on top of the same saved models and feature-engineering code, so predictions are identical regardless of which surface you use. The dashboard's monitoring runs write to SQLite and can trigger the (mocked) alert channels.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data | Pandas, NumPy |
| ML | Scikit-learn (RandomForest, IsolationForest, GridSearchCV), XGBoost |
| Validation | Custom Great-Expectations-style engine (see [Design Decisions](#design-decisions--trade-offs)) |
| Experiment tracking | MLflow (local SQLite backend) |
| Persistence | Joblib (models), SQLite (monitoring history), JSON (reference stats, reports) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Orchestration | Apache Airflow 3.x (TaskFlow API) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Testing | Pytest (54 tests), GitHub Actions |
| Deployment | Docker |

## Project Structure

```
AI-Data-Quality-Monitoring/
├── data/
│   ├── raw/synthetic_data.csv        # generated dataset (103,000 rows)
│   ├── processed/                    # engineered-feature exports
│   ├── uploads/                      # files posted to /upload
│   └── generate_synthetic_data.py
├── models/                           # quality_classifier.pkl, anomaly_model.pkl,
│                                      # feature_reference_stats.json, model_metadata.json
├── notebooks/01_EDA_and_Prototyping.ipynb
├── airflow/dags.py                   # Airflow 3.x TaskFlow DAG
├── api/app.py                        # FastAPI service
├── validation/great_expectations.py  # validation engine + HTML report renderer
├── preprocessing/preprocess.py       # feature engineering (leak-safe)
├── training/train.py                 # GridSearchCV + CV + 3 models + plots
├── inference/predict.py              # loads saved models, scores new data
├── monitoring/
│   ├── monitor.py                    # KPIs, drift, composite quality score
│   └── alerts.py                     # threshold rules + console/email*/slack* channels
├── dashboard/dashboard.py            # Streamlit app
├── utils/                            # config, logging, SQLite helpers
├── tests/                            # 54 pytest tests
├── docs/images/                      # diagrams + generated plots
├── .github/workflows/ci.yml
├── requirements.txt
├── Dockerfile
└── run.py                            # end-to-end pipeline orchestrator
```

## Installation & Quickstart

```bash
pip install -r requirements.txt
python run.py
```

`run.py` will (1) generate the synthetic dataset if it's not already present, (2) run validation, (3) engineer features, (4) train all three models (skipped automatically if `models/quality_classifier.pkl` already exists -- pass `--retrain` to force it), and (5) run one monitoring cycle. First run takes roughly 2 minutes (model training); subsequent runs take a few seconds.

Then, in separate terminals:

```bash
uvicorn api.app:app --reload --port 8000     # REST API + docs at http://localhost:8000/docs
streamlit run dashboard/dashboard.py          # Dashboard at http://localhost:8501
```

Run the test suite:

```bash
pytest tests/ -v
```

## The Dataset

`data/generate_synthetic_data.py` builds a 100,000-row base population of customer transactions (`CustomerID, Age, Salary, PurchaseAmount, City, Country, TransactionDate`) and deliberately injects, at controlled rates: missing values (~8% of rows), duplicate rows (~3%, appended), statistical outliers (~3%), malformed date strings (~2%), future-dated transactions (~1.5%), negative salaries/purchase amounts (~1.5% each), and invalid categories (~2%). The most recent ~30% of the time window also gets a shifted Salary/PurchaseAmount distribution to simulate real data drift. Each row is labeled `Good Data` / `Bad Data` from the injected issues, with 4% random label noise added for realism -- **103,000 rows total**, **23.0% Bad Data**.

## Data Validation

`validation/great_expectations.py` runs 17 expectations (null %, ranges, regex, date validity, categorical membership, row uniqueness, and two KS-test drift checks) and renders both a JSON summary and a styled standalone HTML report (`validation/reports/validation_report.html`). On the shipped dataset it currently reports **15/17 expectations passing**, correctly failing the Salary range check (injected negative salaries) and the date-validity check (injected malformed/future dates) -- exactly the issues the generator was designed to inject.

## Feature Engineering

`preprocessing/preprocess.py` fits reference statistics (means, stds, IQR bounds, "known-good" category sets) on a **training split only**, then converts raw records into **24 always-non-null numeric features**: per-column missing flags + overall missing ratio, an exact-duplicate flag, z-scores and IQR-outlier flags per numeric column, an aggregate outlier score, date validity/freshness/future flags, a 7-transaction rolling purchase average, categorical validity flags, negative-value flags, and a schema-violation count. These are the features the models actually train on -- not the raw City/Country strings.

## Model Performance

Trained on an 80/20 stratified split (82,400 train / 20,600 test rows) with `GridSearchCV` hyperparameter search + k-fold cross-validation, evaluated on the held-out test set:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| **RandomForest** (winner) | 92.51% | 95.07% | 71.18% | **81.41%** | 85.57% |
| XGBoost | 92.51% | 95.05% | 71.18% | 81.40% | 85.77% |
| Isolation Forest (unsupervised) | 88.21% | 75.40% | 72.44% | 73.89% | 85.11% |

RandomForest and XGBoost finished essentially tied (F1 differs by 0.0001) -- RandomForest was saved as `models/quality_classifier.pkl` by the tie-break rule, but either is a reasonable production choice. The Isolation Forest never sees the labels during training and is benchmarked against them only as a sanity check, so its lower scores are expected, not a bug.

Precision noticeably exceeds recall for both classifiers (~95% vs ~71%): the models are conservative -- when they flag a record as Bad Data they're usually right, but they miss some real issues. That gap is largely explained by the 4% random label noise injected into the ground truth (some rows are unlearnable by design) and is a realistic pattern for this kind of problem; see [Future Improvements](#future-improvements) for ways to tune the precision/recall trade-off.

<p align="center">
<img src="docs/images/confusion_matrix.png" width="360" alt="Confusion matrix">
<img src="docs/images/roc_curve.png" width="360" alt="ROC curve">
</p>
<p align="center"><img src="docs/images/feature_importance.png" width="500" alt="Feature importance"></p>

`schema_violation_count`, `missing_ratio`, and `outlier_score` dominate feature importance -- sensible, since those are direct, low-noise aggregates of the underlying issues, while individual z-scores and per-column flags carry redundant, more diluted signal.

Re-run training any time with `python training/train.py` (or `python run.py --retrain`); it regenerates all three models, `models/model_metadata.json`, and the three plots above.

## Monitoring & Alerting

`monitoring/monitor.py` computes, for any data slice: missing %, duplicate %, outlier %, KS-test drift vs. a reference period, average model prediction confidence, and a composite 0-100 quality score -- then persists the run to SQLite (`monitoring/quality_monitoring.db`). `monitoring/alerts.py` evaluates five threshold rules against those metrics:

| Alert | Threshold |
|---|---|
| High missing values | > 10% of rows |
| High duplicate rate | > 5% of rows |
| High outlier rate | > 3% of rows |
| Data drift detected | KS statistic > 0.15 |
| Low prediction confidence | average confidence < 0.60 |

Fired alerts are logged to the console, and formatted, ready-to-send payloads are appended to `logs/mock_emails.log` and `logs/mock_slack.log` -- see [Design Decisions](#design-decisions--trade-offs) for why these are mocked rather than live.

On the shipped dataset's drifted "current period," the system correctly fires a **critical `data_drift_detected`** alert (KS statistic 0.254 on Salary, vs. the 0.15 threshold) and a **`high_outlier_rate`** alert -- because outlier detection is calibrated against the pre-drift reference period, drift itself causes a spike in apparent outliers. That's a real, well-known production phenomenon (a model/monitor calibrated on stale data starts misfiring once the input distribution moves), not an artifact of this demo.

## REST API

`api/app.py`, run with `uvicorn api.app:app --reload --port 8000`. Interactive docs at `/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload a CSV, get back a `dataset_id` + preview |
| `/validate` | POST | Run the validation suite (`?dataset_id=` optional; defaults to the shipped dataset) |
| `/train` | POST | Retrain all three models on a dataset (~1-2 min) |
| `/predict` | POST | Classify records as Good/Bad Data (JSON body or `?dataset_id=`) |
| `/report` | GET | Latest validation report (`?format=json\|html`) |
| `/anomaly` | POST | Isolation Forest anomaly detection on records |
| `/dashboard` | GET | How to reach the Streamlit dashboard |
| `/monitoring/run` | POST | Trigger a monitoring cycle + alert evaluation *(bonus)* |
| `/monitoring/history` | GET | Recent monitoring runs from SQLite *(bonus)* |
| `/health` | GET | Health check *(bonus)* |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"CustomerID":"C1","Age":35,"Salary":-5000,"PurchaseAmount":80,"City":"Paris","Country":"France","TransactionDate":"2026-01-15"}]}'
# -> {"row_count":1,"bad_data_count":1,"predictions":[{"CustomerID":"C1","predicted_label":"Bad Data","bad_data_probability":0.88,"confidence":0.88}]}
```

> `/train` is fully functional but intentionally excluded from the automated test suite, since it overwrites the shipped `models/*.pkl` -- see `tests/test_api.py` for details.

## Dashboard

`streamlit run dashboard/dashboard.py` opens six views: **Overview** (KPIs + label/issue breakdown), **Data Quality Report** (live validation results + drift histograms), **Model Performance** (metrics table + the plots above), **Predict / Anomaly Check** (single-record form or batch CSV upload), **Live Monitoring** (run a monitoring cycle on demand, watch the quality-score trend build in SQLite), and **Alerts** (recent fired alerts with severity).

## Airflow Orchestration

<p align="center"><img src="docs/images/workflow_diagram.png" width="620" alt="Airflow DAG workflow diagram"></p>

`airflow/dags.py` is a TaskFlow-API DAG (`ai_data_quality_monitoring_pipeline`, `@daily`) with an explicit deployment gate: `evaluate_model` only approves the freshly-trained classifier if test F1 >= 0.60, and `deploy_model` fails the task (alerting on-call) if it doesn't. It targets **Airflow 3.x**, importing from the modern `airflow.sdk` namespace rather than the legacy `airflow.models.dag` path. To use it:

```bash
cp airflow/dags.py $AIRFLOW_HOME/dags/data_quality_pipeline.py
```

Airflow itself isn't installed in this project's dependencies (it's a heavy install and isn't required to run the API, dashboard, or `run.py`) -- see [Design Decisions](#design-decisions--trade-offs).

## Docker

```bash
docker build -t data-quality-monitor .
docker run -p 8000:8000 data-quality-monitor                 # FastAPI (default)
docker run -p 8501:8501 data-quality-monitor \
  streamlit run dashboard/dashboard.py --server.address 0.0.0.0   # dashboard variant
```

## Testing

```bash
pytest tests/ -v      # 54 tests across data generation, feature engineering,
                       # validation, alerting, inference, and the API
```

CI (`.github/workflows/ci.yml`) runs the same suite on Python 3.11 and 3.12 on every push.

## Database Schema

<p align="center"><img src="docs/images/er_diagram.png" width="560" alt="SQLite ER diagram"></p>

`monitoring/quality_monitoring.db` has two tables: `monitoring_runs` (one row per monitoring cycle, with the KPIs and quality score) and `alerts_log` (one row per fired alert, foreign-keyed to the run that triggered it).

## Design Decisions & Trade-offs

A few pragmatic calls worth calling out explicitly rather than glossing over:

- **Validation engine is a custom Great-Expectations-*style* implementation, not the `great_expectations` package.** The real library's modern API requires a Data Context, Datasource, Data Asset, Batch Definition, and Checkpoint just to run a handful of checks, and its API has shifted significantly across recent major versions -- a lot of fragile surface area for one module in a larger system. `validation/great_expectations.py`'s module docstring includes a verified, current integration snippet for swapping in the real library.
- **Airflow is not installed as a dependency.** `apache-airflow` is a notably heavy install (many transitive dependencies, a metadata DB, a scheduler process) that isn't needed to run the API, dashboard, or `run.py`. The DAG is written and syntax-checked against the current Airflow 3.x API but not executed against a live scheduler in this repo.
- **Email and Slack alerts are mocked.** There's no real SMTP server or Slack workspace to send to here. Both channels build the real payload (formatted email text; a Slack Block Kit JSON payload) and write it to `logs/`, so swapping in `smtplib`/`requests.post(webhook_url, ...)` is a one-function change -- see `monitoring/alerts.py`.
- **Hyperparameter search runs on a stratified subsample, the final model fits on full data.** RandomForest is the expensive model to grid-search; searching on a 25K-row subsample and doing a single final fit on the full 82,400-row training set keeps training under ~2 minutes without touching final-model data volume.
- **Docker image is written and structurally sound but not build-tested in this environment** (no Docker daemon available here). Standard `python:3.12-slim` + pip install + non-root user pattern.

## Future Improvements

- Tune the classification decision threshold (currently 0.5) to trade precision for recall based on the real business cost of missing a bad record vs. a false alarm
- Swap the in-memory `/upload` registry for Redis/S3-backed storage for multi-worker deployments
- Add a model registry stage (MLflow Model Registry or similar) with staging/production aliases, so Airflow's `deploy_model` gate actually promotes/demotes a served model rather than just logging a decision
- Real email/Slack integration (SES/SendGrid, Slack incoming webhooks)
- Online/incremental drift detection (e.g., a sliding-window PSI) instead of the current batch-vs-reference KS test
- Authentication/rate-limiting on the FastAPI service before any real deployment
