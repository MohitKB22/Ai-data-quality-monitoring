"""
Data Validation Engine
=======================
A Great-Expectations-style validation engine: the same mental model (a suite
of named "expectations" run against a dataframe, each producing a pass/fail
verdict plus an observed value, rolled up into an HTML "Data Docs"-style
report) implemented directly on pandas so the project has zero fragile,
version-sensitive external dependencies.

WHY NOT THE REAL `great_expectations` PACKAGE?
The real library is excellent, but its modern (1.x) API requires a Data
Context, a Pandas Datasource, a Data Asset, a Batch Definition and a
Checkpoint just to run a handful of checks, and its exact API has shifted
significantly across recent major versions. For a self-contained, offline,
CI-friendly portfolio project, that's a lot of moving parts for what is
fundamentally "run these rules against a dataframe and render a report". This
module keeps the *concepts* (expectations, suites, "mostly" thresholds,
validation results, Data Docs) but implements them directly.

If you want to swap in the real library in your own environment, the
integration point below is exactly where you'd do it:

    # pip install great_expectations
    import great_expectations as gx
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="dq_source")
    data_asset = data_source.add_dataframe_asset(name="dq_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    suite = context.suites.add(gx.ExpectationSuite(name="dq_suite"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Age"))
    result = batch.validate(suite)

Everything downstream of `validate_dataframe()` in this file (results schema,
HTML rendering, alert thresholds) is intentionally shaped so that swapping the
internals for real GX later would not require touching the rest of the
project.
"""

from __future__ import annotations

import os
import sys
import re
from dataclasses import dataclass, field

# Make the project root importable regardless of how/where this script is invoked
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

CUSTOMER_ID_REGEX = re.compile(r"^CUST\d+$")
TODAY = datetime(2026, 7, 1)


# --------------------------------------------------------------------------
# Core expectation framework
# --------------------------------------------------------------------------
@dataclass
class ExpectationResult:
    expectation_type: str
    column: Optional[str]
    description: str
    success: bool
    observed_value: Any
    threshold: Any = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "expectation_type": self.expectation_type,
            "column": self.column,
            "description": self.description,
            "success": bool(self.success),
            "observed_value": _jsonable(self.observed_value),
            "threshold": _jsonable(self.threshold),
            "details": {k: _jsonable(v) for k, v in self.details.items()},
        }


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class ExpectationSuite:
    """A named, ordered collection of expectations to run against a dataframe."""

    def __init__(self, name: str):
        self.name = name
        self._checks: list[Callable[[pd.DataFrame], ExpectationResult]] = []

    # ---- expectation registration (Great-Expectations-style method names) ----
    def expect_column_values_to_not_be_null(self, column: str, mostly: float = 1.0):
        def check(df):
            not_null_ratio = df[column].notna().mean() if len(df) else 1.0
            return ExpectationResult(
                "expect_column_values_to_not_be_null", column,
                f"At least {mostly:.0%} of '{column}' values should be non-null",
                success=not_null_ratio >= mostly,
                observed_value=round(not_null_ratio, 4), threshold=mostly,
                details={"null_percent": round((1 - not_null_ratio) * 100, 2)},
            )
        self._checks.append(check)
        return self

    def expect_column_values_to_be_between(self, column: str, min_value=None, max_value=None, mostly: float = 1.0):
        def check(df):
            s = pd.to_numeric(df[column], errors="coerce")
            in_range = pd.Series(True, index=s.index)
            if min_value is not None:
                in_range &= (s >= min_value)
            if max_value is not None:
                in_range &= (s <= max_value)
            in_range &= s.notna()
            ratio = in_range.mean() if len(df) else 1.0
            return ExpectationResult(
                "expect_column_values_to_be_between", column,
                f"At least {mostly:.0%} of '{column}' values should be between {min_value} and {max_value}",
                success=ratio >= mostly, observed_value=round(ratio, 4), threshold=mostly,
                details={"violations": int((~in_range).sum())},
            )
        self._checks.append(check)
        return self

    def expect_column_values_to_match_regex(self, column: str, pattern: re.Pattern, mostly: float = 1.0):
        def check(df):
            matches = df[column].astype(str).apply(lambda v: bool(pattern.match(v)))
            ratio = matches.mean() if len(df) else 1.0
            return ExpectationResult(
                "expect_column_values_to_match_regex", column,
                f"At least {mostly:.0%} of '{column}' values should match pattern {pattern.pattern}",
                success=ratio >= mostly, observed_value=round(ratio, 4), threshold=mostly,
                details={"violations": int((~matches).sum())},
            )
        self._checks.append(check)
        return self

    def expect_column_values_to_be_in_set(self, column: str, value_set: set, mostly: float = 1.0):
        def check(df):
            in_set = df[column].astype(str).isin(value_set)
            ratio = in_set.mean() if len(df) else 1.0
            return ExpectationResult(
                "expect_column_values_to_be_in_set", column,
                f"At least {mostly:.0%} of '{column}' values should be a recognized category",
                success=ratio >= mostly, observed_value=round(ratio, 4), threshold=mostly,
                details={"violations": int((~in_set).sum()), "known_category_count": len(value_set)},
            )
        self._checks.append(check)
        return self

    def expect_column_values_to_be_valid_dates(self, column: str, mostly: float = 1.0,
                                                 disallow_future: bool = True):
        def check(df):
            parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")
            valid = parsed.notna()
            if disallow_future:
                valid &= (parsed <= TODAY) | parsed.isna()
                valid = valid & parsed.notna()
            ratio = valid.mean() if len(df) else 1.0
            return ExpectationResult(
                "expect_column_values_to_be_valid_dates", column,
                f"At least {mostly:.0%} of '{column}' values should be valid, non-future dates",
                success=ratio >= mostly, observed_value=round(ratio, 4), threshold=mostly,
                details={"unparsable_or_future": int((~valid).sum())},
            )
        self._checks.append(check)
        return self

    def expect_table_row_count_to_be_between(self, min_value=None, max_value=None):
        def check(df):
            n = len(df)
            ok = True
            if min_value is not None:
                ok &= n >= min_value
            if max_value is not None:
                ok &= n <= max_value
            return ExpectationResult(
                "expect_table_row_count_to_be_between", None,
                f"Row count should be between {min_value} and {max_value}",
                success=ok, observed_value=n, threshold=[min_value, max_value],
            )
        self._checks.append(check)
        return self

    def expect_compound_row_uniqueness(self, columns: list, mostly: float = 1.0):
        def check(df):
            dup_ratio = df.duplicated(subset=[c for c in columns if c in df.columns], keep="first").mean() if len(df) else 0.0
            ratio = 1 - dup_ratio
            return ExpectationResult(
                "expect_compound_row_uniqueness", ", ".join(columns),
                f"At least {mostly:.0%} of rows should be unique across {columns}",
                success=ratio >= mostly, observed_value=round(ratio, 4), threshold=mostly,
                details={"duplicate_percent": round(dup_ratio * 100, 2)},
            )
        self._checks.append(check)
        return self

    def expect_distribution_to_match_reference(self, column: str, reference_values: np.ndarray,
                                                 max_ks_statistic: float = 0.15):
        """Custom expectation: two-sample Kolmogorov-Smirnov test against a
        reference distribution, used as the statistical-drift check."""
        def check(df):
            current = pd.to_numeric(df[column], errors="coerce").dropna()
            ref = pd.Series(reference_values).dropna()
            if len(current) < 10 or len(ref) < 10:
                return ExpectationResult(
                    "expect_distribution_to_match_reference", column,
                    f"'{column}' distribution should match the reference distribution (KS test)",
                    success=True, observed_value=None, threshold=max_ks_statistic,
                    details={"note": "insufficient data for KS test"},
                )
            ks_stat, p_value = scipy_stats.ks_2samp(current, ref)
            return ExpectationResult(
                "expect_distribution_to_match_reference", column,
                f"'{column}' distribution should match the reference distribution (KS statistic <= {max_ks_statistic})",
                success=ks_stat <= max_ks_statistic, observed_value=round(float(ks_stat), 4),
                threshold=max_ks_statistic, details={"p_value": round(float(p_value), 6)},
            )
        self._checks.append(check)
        return self

    def run(self, df: pd.DataFrame) -> list[ExpectationResult]:
        return [check(df) for check in self._checks]


# --------------------------------------------------------------------------
# Default suite matching the project's validation requirements
# --------------------------------------------------------------------------
def build_default_suite(reference_stats: Optional[dict] = None,
                         reference_df: Optional[pd.DataFrame] = None) -> ExpectationSuite:
    suite = ExpectationSuite("customer_transaction_quality_suite")

    # Null checks
    for col, mostly in [("Age", 0.95), ("Salary", 0.95), ("PurchaseAmount", 0.97),
                         ("City", 0.97), ("Country", 0.97), ("TransactionDate", 0.97)]:
        suite.expect_column_values_to_not_be_null(col, mostly=mostly)

    # Range checks
    suite.expect_column_values_to_be_between("Age", min_value=0, max_value=120, mostly=0.97)
    suite.expect_column_values_to_be_between("Salary", min_value=0, max_value=None, mostly=0.97)
    suite.expect_column_values_to_be_between("PurchaseAmount", min_value=0, max_value=None, mostly=0.97)

    # Regex check
    suite.expect_column_values_to_match_regex("CustomerID", CUSTOMER_ID_REGEX, mostly=0.99)

    # Date validation
    suite.expect_column_values_to_be_valid_dates("TransactionDate", mostly=0.97, disallow_future=True)

    # Categorical validation (uses reference_stats known-good categories if available)
    if reference_stats is not None:
        city_set = set(reference_stats["categorical"]["City"]["known_values"])
        country_set = set(reference_stats["categorical"]["Country"]["known_values"])
    else:
        city_set, country_set = set(), set()
    suite.expect_column_values_to_be_in_set("City", city_set, mostly=0.97)
    suite.expect_column_values_to_be_in_set("Country", country_set, mostly=0.97)

    # Duplicate / uniqueness check
    suite.expect_compound_row_uniqueness(
        ["CustomerID", "Age", "Salary", "PurchaseAmount", "City", "Country", "TransactionDate"], mostly=0.95
    )

    # Row-count sanity check
    suite.expect_table_row_count_to_be_between(min_value=1, max_value=None)

    # Distribution drift checks
    if reference_df is not None:
        for col in ["Salary", "PurchaseAmount"]:
            ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().to_numpy()
            suite.expect_distribution_to_match_reference(col, ref_vals, max_ks_statistic=0.15)

    return suite


# --------------------------------------------------------------------------
# High-level API
# --------------------------------------------------------------------------
def validate_dataframe(df: pd.DataFrame, reference_stats: Optional[dict] = None,
                        reference_df: Optional[pd.DataFrame] = None, dataset_name: str = "dataset") -> dict:
    suite = build_default_suite(reference_stats=reference_stats, reference_df=reference_df)
    results = suite.run(df)

    passed = sum(1 for r in results if r.success)
    total = len(results)
    summary = {
        "dataset_name": dataset_name,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "expectations_passed": passed,
        "expectations_total": total,
        "success_rate": round(passed / total, 4) if total else 1.0,
        "overall_success": passed == total,
        "null_percent_overall": round(df[[c for c in ["Age", "Salary", "PurchaseAmount", "City", "Country", "TransactionDate"] if c in df.columns]]
                                       .isna().any(axis=1).mean() * 100, 2) if len(df) else 0.0,
        "duplicate_percent": round(df.duplicated(subset=[c for c in df.columns if c in
                                    ["CustomerID", "Age", "Salary", "PurchaseAmount", "City", "Country", "TransactionDate"]]).mean() * 100, 2) if len(df) else 0.0,
        "results": [r.to_dict() for r in results],
    }
    return summary


# --------------------------------------------------------------------------
# HTML "Data Docs"-style report
# --------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data Validation Report - {dataset_name}</title>
<style>
  :root {{ --pass:#1e8e5a; --fail:#c53030; --bg:#f6f7fb; --card:#ffffff; --ink:#1a1d29; --muted:#697086; --accent:#3b5bfd; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background:var(--bg); color:var(--ink); }}
  header {{ background: linear-gradient(135deg, #1a1d29, #2f3550); color:#fff; padding: 32px 40px; }}
  header h1 {{ margin:0 0 6px 0; font-size: 24px; }}
  header p {{ margin:0; color:#c7cbe0; font-size: 14px; }}
  .container {{ max-width: 1000px; margin: -28px auto 40px auto; padding: 0 24px; }}
  .summary-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap:16px; margin-bottom: 24px; }}
  .stat-card {{ background:var(--card); border-radius:12px; padding:18px 20px; box-shadow: 0 2px 10px rgba(20,20,50,0.08); }}
  .stat-card .value {{ font-size: 26px; font-weight:700; }}
  .stat-card .label {{ font-size: 12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; margin-top:4px;}}
  .stat-card.ok .value {{ color: var(--pass); }}
  .stat-card.warn .value {{ color: var(--fail); }}
  table {{ width:100%; border-collapse: collapse; background:var(--card); border-radius:12px; overflow:hidden; box-shadow: 0 2px 10px rgba(20,20,50,0.08);}}
  th, td {{ text-align:left; padding: 12px 16px; border-bottom:1px solid #eef0f6; font-size: 13.5px; }}
  th {{ background:#f0f1f8; color:var(--muted); text-transform:uppercase; font-size:11px; letter-spacing:.04em; }}
  tr:last-child td {{ border-bottom:none; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }}
  .badge.pass {{ background:#e5f7ee; color:var(--pass); }}
  .badge.fail {{ background:#fdeaea; color:var(--fail); }}
  .col-name {{ font-family: 'SFMono-Regular', Consolas, monospace; color:var(--accent); font-weight:600; }}
  footer {{ text-align:center; color:var(--muted); font-size:12px; padding: 24px; }}
</style>
</head>
<body>
<header>
  <h1>&#128269; Data Validation Report</h1>
  <p>Dataset: <strong>{dataset_name}</strong> &nbsp;|&nbsp; Generated {run_at} &nbsp;|&nbsp; {row_count:,} rows evaluated</p>
</header>
<div class="container">
  <div class="summary-grid">
    <div class="stat-card {overall_class}"><div class="value">{passed}/{total}</div><div class="label">Expectations Passed</div></div>
    <div class="stat-card"><div class="value">{success_rate:.1%}</div><div class="label">Success Rate</div></div>
    <div class="stat-card {null_class}"><div class="value">{null_percent}%</div><div class="label">Rows with Missing Fields</div></div>
    <div class="stat-card {dup_class}"><div class="value">{duplicate_percent}%</div><div class="label">Duplicate Rows</div></div>
  </div>
  <table>
    <thead><tr><th>Status</th><th>Expectation</th><th>Column</th><th>Description</th><th>Observed</th><th>Threshold</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</div>
<footer>Generated by the AI-Based Data Quality Monitoring System &mdash; Great-Expectations-style validation engine</footer>
</body>
</html>
"""

_ROW_TEMPLATE = """<tr>
  <td><span class="badge {badge_class}">{badge_text}</span></td>
  <td>{expectation_type}</td>
  <td class="col-name">{column}</td>
  <td>{description}</td>
  <td>{observed_value}</td>
  <td>{threshold}</td>
</tr>"""


def generate_html_report(summary: dict, output_path: str) -> str:
    rows_html = []
    for r in summary["results"]:
        rows_html.append(_ROW_TEMPLATE.format(
            badge_class="pass" if r["success"] else "fail",
            badge_text="PASS" if r["success"] else "FAIL",
            expectation_type=r["expectation_type"],
            column=r["column"] or "-",
            description=r["description"],
            observed_value=r["observed_value"],
            threshold=r["threshold"],
        ))

    html = _HTML_TEMPLATE.format(
        dataset_name=summary["dataset_name"],
        run_at=summary["run_at"],
        row_count=summary["row_count"],
        overall_class="ok" if summary["overall_success"] else "warn",
        passed=summary["expectations_passed"],
        total=summary["expectations_total"],
        success_rate=summary["success_rate"],
        null_class="ok" if summary["null_percent_overall"] <= 10 else "warn",
        null_percent=summary["null_percent_overall"],
        dup_class="ok" if summary["duplicate_percent"] <= 5 else "warn",
        duplicate_percent=summary["duplicate_percent"],
        rows="\n".join(rows_html),
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


# --------------------------------------------------------------------------
# CLI / demo entry point
# --------------------------------------------------------------------------
def main():
    import json
    from preprocessing.preprocess import compute_reference_stats

    here = os.path.dirname(__file__)
    raw_path = os.path.join(here, "..", "data", "raw", "synthetic_data.csv")
    report_path = os.path.join(here, "reports", "validation_report.html")
    json_path = os.path.join(here, "reports", "validation_summary.json")

    df = pd.read_csv(raw_path)

    # Split chronologically: the older ~70% acts as the "reference" baseline
    # (what a training/reference dataset would look like) and the most recent
    # ~30% acts as the "current" incoming batch being validated. This mirrors
    # the drift period injected by the data generator, so the drift check
    # below has something real to detect instead of comparing data to itself.
    parsed_dates = pd.to_datetime(df["TransactionDate"], errors="coerce", format="mixed")
    cutoff = parsed_dates.quantile(0.70)
    reference_df = df[parsed_dates <= cutoff].copy()
    current_df = df[parsed_dates > cutoff].copy()
    print(f"Reference period: {len(reference_df):,} rows (<= {cutoff.date()}) | "
          f"Current period: {len(current_df):,} rows (> {cutoff.date()})")

    reference_stats = compute_reference_stats(reference_df)

    summary = validate_dataframe(current_df, reference_stats=reference_stats,
                                  reference_df=reference_df, dataset_name="synthetic_data.csv (current batch)")
    generate_html_report(summary, report_path)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Validation complete: {summary['expectations_passed']}/{summary['expectations_total']} expectations passed")
    print(f"Missing-field rows: {summary['null_percent_overall']}% | Duplicate rows: {summary['duplicate_percent']}%")
    print(f"HTML report -> {report_path}")
    print(f"JSON summary -> {json_path}")


if __name__ == "__main__":
    main()
