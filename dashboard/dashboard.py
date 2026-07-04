"""
Streamlit Dashboard
=====================
Run with:  streamlit run dashboard/dashboard.py
Opens at:  http://localhost:8501
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.preprocess import CORE_COLUMNS, compute_reference_stats, engineer_features, load_reference_stats  # noqa: E402
from utils import config, db  # noqa: E402
from validation.great_expectations import validate_dataframe  # noqa: E402

st.set_page_config(page_title="AI Data Quality Monitor", page_icon="\U0001F50D", layout="wide")

MODELS_READY = os.path.exists(os.path.join(config.MODELS_DIR, "quality_classifier.pkl"))


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset(path=config.RAW_DATA_PATH):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_model_metadata():
    path = os.path.join(config.MODELS_DIR, "model_metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_predictor():
    from inference.predict import DataQualityPredictor
    return DataQualityPredictor()


@st.cache_data(show_spinner=False)
def get_reference_split(_df):
    parsed = pd.to_datetime(_df["TransactionDate"], errors="coerce", format="mixed")
    cutoff = parsed.quantile(0.70)
    reference_df = _df[parsed <= cutoff]
    current_df = _df[parsed > cutoff]
    return reference_df, current_df


def quality_badge(score: float) -> str:
    if score >= 85:
        return "\U0001F7E2 Excellent"
    if score >= 70:
        return "\U0001F7E1 Needs attention"
    return "\U0001F534 Critical"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("\U0001F50D Data Quality Monitor")
page = st.sidebar.radio("Navigate", [
    "Overview (KPIs)", "Data Quality Report", "Model Performance",
    "Predict / Anomaly Check", "Live Monitoring", "Alerts",
])
st.sidebar.markdown("---")
st.sidebar.caption("AI-Based Data Quality Monitoring System")
st.sidebar.caption(f"Models ready: {'✅' if MODELS_READY else '❌ run `python training/train.py`'}")

df = load_dataset()

# ==========================================================================
# PAGE: Overview
# ==========================================================================
if page == "Overview (KPIs)":
    st.title("Overview")
    st.caption("Headline data-quality KPIs for the current dataset.")

    reference_stats = compute_reference_stats(df)
    features = engineer_features(df, reference_stats)

    missing_pct = round((features["missing_ratio"] > 0).mean() * 100, 2)
    dup_pct = round(features["is_duplicate"].mean() * 100, 2)
    outlier_pct = round(((features[["age_iqr_outlier", "salary_iqr_outlier", "purchase_iqr_outlier"]].sum(axis=1)) > 0).mean() * 100, 2)
    quality_score = round(100 - missing_pct * 0.5 - dup_pct * 0.6 - outlier_pct * 0.8, 1)
    quality_score = max(0, min(100, quality_score))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Missing Data", f"{missing_pct}%", help="Rows with at least one missing field")
    c3.metric("Duplicates", f"{dup_pct}%")
    c4.metric("Outliers", f"{outlier_pct}%")
    c5.metric("Quality Score", f"{quality_score}/100", quality_badge(quality_score))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Label Distribution")
        label_counts = df["Label"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]
        fig = px.pie(label_counts, names="Label", values="Count", hole=0.45,
                     color="Label", color_discrete_map={"Good Data": "#1e8e5a", "Bad Data": "#c53030"})
        st.plotly_chart(fig, width='stretch')
    with col_b:
        st.subheader("Quality Issue Breakdown")
        if "QualityIssue" in df.columns:
            issue_counts = df["QualityIssue"].value_counts().reset_index()
            issue_counts.columns = ["Issue", "Count"]
            fig = px.bar(issue_counts, x="Count", y="Issue", orientation="h", color="Count",
                         color_continuous_scale="Reds")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width='stretch')

# ==========================================================================
# PAGE: Data Quality Report
# ==========================================================================
elif page == "Data Quality Report":
    st.title("Data Quality Report")
    st.caption("Great-Expectations-style validation suite, plus missing/duplicate/outlier/drift breakdowns.")

    reference_df, current_df = get_reference_split(df)
    reference_stats = compute_reference_stats(reference_df)

    with st.spinner("Running validation suite..."):
        summary = validate_dataframe(current_df, reference_stats=reference_stats, reference_df=reference_df,
                                      dataset_name="current batch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expectations Passed", f"{summary['expectations_passed']}/{summary['expectations_total']}")
    c2.metric("Success Rate", f"{summary['success_rate']:.1%}")
    c3.metric("Missing Rows", f"{summary['null_percent_overall']}%")
    c4.metric("Duplicate Rows", f"{summary['duplicate_percent']}%")

    st.subheader("Expectation Results")
    results_df = pd.DataFrame(summary["results"])
    results_df["status"] = results_df["success"].map({True: "✅ PASS", False: "❌ FAIL"})
    # observed_value/threshold can be a float, bool, or list (e.g. row-count's [min,max]) depending
    # on the expectation type -- cast to string so the mixed-type column serializes cleanly to Arrow.
    results_df["observed_value"] = results_df["observed_value"].astype(str)
    results_df["threshold"] = results_df["threshold"].astype(str)
    st.dataframe(results_df[["status", "expectation_type", "column", "description", "observed_value", "threshold"]],
                 width='stretch', hide_index=True)

    st.subheader("Missing Values by Column")
    miss = df[["Age", "Salary", "PurchaseAmount", "City", "Country", "TransactionDate"]].isna().mean() * 100
    fig = px.bar(x=miss.index, y=miss.values, labels={"x": "Column", "y": "% Missing"}, color=miss.values,
                 color_continuous_scale="Oranges")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Distribution Drift: Reference vs Current Period")
    drift_col = st.selectbox("Column", ["Salary", "PurchaseAmount"])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pd.to_numeric(reference_df[drift_col], errors="coerce"), name="Reference", opacity=0.6, histnorm="probability density"))
    fig.add_trace(go.Histogram(x=pd.to_numeric(current_df[drift_col], errors="coerce"), name="Current", opacity=0.6, histnorm="probability density"))
    fig.update_layout(barmode="overlay", xaxis_title=drift_col, yaxis_title="Density")
    st.plotly_chart(fig, width='stretch')
    drift_check = next((r for r in summary["results"] if r["expectation_type"] == "expect_distribution_to_match_reference" and r["column"] == drift_col), None)
    if drift_check:
        st.info(f"KS statistic: **{drift_check['observed_value']}** (threshold {drift_check['threshold']}) -> "
                f"{'⚠️ Drift detected' if not drift_check['success'] else '✅ No significant drift'}")

    st.download_button("Download full HTML report", data=open(os.path.join(config.VALIDATION_REPORTS_DIR, "validation_report.html")).read()
                        if os.path.exists(os.path.join(config.VALIDATION_REPORTS_DIR, "validation_report.html")) else "No report generated yet",
                        file_name="validation_report.html", mime="text/html")

# ==========================================================================
# PAGE: Model Performance
# ==========================================================================
elif page == "Model Performance":
    st.title("Model Performance")
    metadata = load_model_metadata()
    if metadata is None:
        st.warning("No trained models yet. Run `python training/train.py` first.")
    else:
        st.caption(f"Trained {metadata['trained_at']} on {metadata['train_rows']:,} train / {metadata['test_rows']:,} test rows. "
                   f"Winning classifier: **{metadata['winning_classifier']}**")

        rows = []
        for name, m in metadata["results"].items():
            rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                         "Recall": m["recall"], "F1 Score": m["f1_score"], "ROC AUC": m["roc_auc"]})
        metrics_df = pd.DataFrame(rows).set_index("Model").round(4)
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="#d3f8d3"), width='stretch')

        fig = px.bar(metrics_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score"),
                     x="Metric", y="Score", color="Model", barmode="group")
        st.plotly_chart(fig, width='stretch')

        st.subheader("Diagnostic Plots (from the last training run)")
        img_cols = st.columns(3)
        for col, fname, caption in zip(img_cols,
                                        ["confusion_matrix.png", "roc_curve.png", "feature_importance.png"],
                                        ["Confusion Matrix", "ROC Curve", "Feature Importance"]):
            path = os.path.join(config.DOCS_IMAGES_DIR, fname)
            if os.path.exists(path):
                col.image(path, caption=caption, width='stretch')

# ==========================================================================
# PAGE: Predict / Anomaly Check
# ==========================================================================
elif page == "Predict / Anomaly Check":
    st.title("Predict / Anomaly Check")
    if not MODELS_READY:
        st.warning("No trained models yet. Run `python training/train.py` first.")
    else:
        predictor = load_predictor()
        tab1, tab2 = st.tabs(["Single record", "Batch (CSV upload)"])

        with tab1:
            st.caption("Enter a record manually, or load a random sample from the dataset.")
            if st.button("\U0001F3B2 Load random sample row"):
                st.session_state["sample_row"] = df.sample(1).iloc[0]
            sample = st.session_state.get("sample_row", df.iloc[0])

            c1, c2, c3 = st.columns(3)
            customer_id = c1.text_input("CustomerID", str(sample["CustomerID"]))
            age = c1.number_input("Age", value=float(sample["Age"]) if pd.notna(sample["Age"]) else 0.0)
            salary = c2.number_input("Salary", value=float(sample["Salary"]) if pd.notna(sample["Salary"]) else 0.0)
            purchase = c2.number_input("PurchaseAmount", value=float(sample["PurchaseAmount"]) if pd.notna(sample["PurchaseAmount"]) else 0.0)
            city = c3.text_input("City", str(sample["City"]) if pd.notna(sample["City"]) else "")
            country = c3.text_input("Country", str(sample["Country"]) if pd.notna(sample["Country"]) else "")
            tx_date = st.text_input("TransactionDate (YYYY-MM-DD)", str(sample["TransactionDate"]) if pd.notna(sample["TransactionDate"]) else "")

            if st.button("Run Prediction", type="primary"):
                record = pd.DataFrame([{"CustomerID": customer_id, "Age": age, "Salary": salary,
                                         "PurchaseAmount": purchase, "City": city, "Country": country,
                                         "TransactionDate": tx_date}])
                result = predictor.predict_full(record).iloc[0]
                label = result["predicted_label"]
                color = "green" if label == "Good Data" else "red"
                st.markdown(f"### Prediction: :{color}[{label}]")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Bad Data Probability", f"{result['bad_data_probability']:.1%}")
                mc2.metric("Confidence", f"{result['confidence']:.1%}")
                mc3.metric("Anomaly (Isolation Forest)", "Yes ⚠️" if result["is_anomaly"] else "No ✅")

        with tab2:
            uploaded = st.file_uploader("Upload a CSV with columns: " + ", ".join(CORE_COLUMNS), type="csv")
            if uploaded is not None:
                batch_df = pd.read_csv(uploaded)
                missing_cols = [c for c in CORE_COLUMNS if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    results = predictor.predict_full(batch_df[CORE_COLUMNS])
                    st.success(f"Scored {len(results):,} rows -- "
                               f"{(results['predicted_label']=='Bad Data').sum():,} flagged as Bad Data, "
                               f"{results['is_anomaly'].sum():,} flagged as anomalies.")
                    st.dataframe(results, width='stretch')
                    st.download_button("Download predictions CSV", results.to_csv(index=False),
                                        file_name="predictions.csv", mime="text/csv")

# ==========================================================================
# PAGE: Live Monitoring
# ==========================================================================
elif page == "Live Monitoring":
    st.title("Live Monitoring")
    st.caption("Each run computes fresh KPIs on a data slice and stores them in SQLite, so you can watch trends over time.")

    if not MODELS_READY:
        st.warning("No trained models yet -- monitoring will still run, but without prediction-confidence tracking.")

    c1, c2 = st.columns([1, 3])
    with c1:
        batch_choice = st.selectbox("Data slice to monitor", ["Random 5,000-row sample", "Reference period", "Current (drifted) period", "Full dataset"])
        run_clicked = st.button("\u25B6 Run Monitoring Cycle", type="primary")

    if run_clicked:
        from monitoring.monitor import run_monitoring_cycle
        reference_df, current_df = get_reference_split(df)
        reference_stats = compute_reference_stats(reference_df)
        if batch_choice == "Random 5,000-row sample":
            batch = df.sample(min(5000, len(df)))
        elif batch_choice == "Reference period":
            batch = reference_df
        elif batch_choice == "Current (drifted) period":
            batch = current_df
        else:
            batch = df

        predictor = load_predictor() if MODELS_READY else None
        with st.spinner("Running monitoring cycle..."):
            metrics = run_monitoring_cycle(batch, reference_stats, reference_df=reference_df, predictor=predictor,
                                            dataset_name=batch_choice)
        st.success(f"Run #{metrics['run_id']} complete -- quality score {metrics['quality_score']}/100, "
                   f"{metrics['alert_count']} alert(s) fired.")

    conn = db.get_connection()
    history = db.fetch_recent_runs(conn, limit=100)
    conn.close()

    if not history:
        st.info("No monitoring runs yet -- click **Run Monitoring Cycle** above.")
    else:
        hist_df = pd.DataFrame(history)
        hist_df["run_at"] = pd.to_datetime(hist_df["run_at"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["run_at"], y=hist_df["quality_score"], mode="lines+markers", name="Quality Score"))
        fig.update_layout(yaxis_title="Quality Score (0-100)", xaxis_title="Run time", yaxis_range=[0, 100])
        st.plotly_chart(fig, width='stretch')

        fig2 = go.Figure()
        for col, label in [("missing_percent", "Missing %"), ("duplicate_percent", "Duplicate %"), ("outlier_percent", "Outlier %")]:
            fig2.add_trace(go.Scatter(x=hist_df["run_at"], y=hist_df[col], mode="lines+markers", name=label))
        fig2.update_layout(yaxis_title="%", xaxis_title="Run time")
        st.plotly_chart(fig2, width='stretch')

        st.subheader("Run History")
        st.dataframe(hist_df[["id", "run_at", "dataset_name", "row_count", "missing_percent", "duplicate_percent",
                               "outlier_percent", "drift_detected", "quality_score", "alert_count"]].sort_values("id", ascending=False),
                     width='stretch', hide_index=True)

# ==========================================================================
# PAGE: Alerts
# ==========================================================================
elif page == "Alerts":
    st.title("Alerts")
    st.caption(f"Thresholds -- Missing: >{config.THRESHOLD_MISSING_PERCENT}% · Duplicate: >{config.THRESHOLD_DUPLICATE_PERCENT}% · "
               f"Outlier: >{config.THRESHOLD_OUTLIER_PERCENT}% · Drift KS: >{config.THRESHOLD_DRIFT_KS_STATISTIC} · "
               f"Low confidence: <{config.THRESHOLD_LOW_CONFIDENCE}")

    conn = db.get_connection()
    alerts = db.fetch_recent_alerts(conn, limit=100)
    conn.close()

    if not alerts:
        st.info("No alerts fired yet -- run a monitoring cycle on the **Live Monitoring** page.")
    else:
        alerts_df = pd.DataFrame(alerts)
        sev_counts = alerts_df["severity"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Alerts", len(alerts_df))
        c2.metric("Critical", int(sev_counts.get("critical", 0)))
        c3.metric("Warning", int(sev_counts.get("warning", 0)))

        def sev_icon(s):
            return "\U0001F534" if s == "critical" else "\U0001F7E1"
        alerts_df["severity_icon"] = alerts_df["severity"].apply(sev_icon)
        st.dataframe(alerts_df[["severity_icon", "triggered_at", "alert_type", "message", "channel"]],
                     width='stretch', hide_index=True)
