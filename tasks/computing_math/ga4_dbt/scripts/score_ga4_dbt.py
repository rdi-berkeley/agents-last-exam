"""Repo-owned evaluator for the GA4 dbt analytics task.

The scorer expects an agent-produced dbt project directory and the hidden
reference_outputs directory. It prints a JSON report to stdout and exits 0 so
the CUA wrapper can return partial scores instead of treating sub-70% outputs
as infrastructure failures.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

CURRENCY_TOL = 0.01
RATE_TOL = 0.0001

EVALUATOR_QUERIES = {
    1: """
    select
        session_date as metric_date, channel,
        count(*) as session_count,
        sum(is_conversion) as conversion_count,
        round(sum(is_conversion)::float / count(*), 6) as conversion_rate
    from fct_sessions
    group by session_date, channel
    order by metric_date asc, channel asc
    """,
    2: """
    with first_pageview as (
        select
            user_pseudo_id || '-' || ga_session_id as session_id,
            page_location as landing_page,
            row_number() over (
                partition by user_pseudo_id, ga_session_id
                order by event_timestamp
            ) as pv_rank
        from stg_events
        where event_name = 'page_view'
    )
    select
        fp.landing_page,
        count(distinct s.session_id) as session_count,
        sum(s.is_conversion) as conversion_count,
        round(sum(s.is_conversion)::float / count(distinct s.session_id), 6) as conversion_rate
    from fct_sessions s
    join first_pageview fp on s.session_id = fp.session_id and fp.pv_rank = 1
    group by fp.landing_page
    having count(distinct s.session_id) >= 100
    order by conversion_rate desc, landing_page asc
    limit 10
    """,
    3: """
    select
        channel,
        count(*) as total_sessions,
        sum(is_conversion) as total_conversions,
        round(sum(revenue), 2) as total_revenue,
        round(avg(case when not is_bounce then session_duration_seconds end), 2) as avg_session_duration_seconds,
        round(sum(case when is_bounce then 1 else 0 end)::float / count(*), 6) as bounce_rate
    from fct_sessions
    group by channel
    order by total_revenue desc
    """,
    4: """
    select
        session_date as metric_date,
        count(*) as session_count,
        round(sum(is_conversion)::float / count(*), 6) as conversion_rate,
        round(avg(case when not is_bounce then session_duration_seconds end), 2) as avg_session_duration_seconds,
        round(sum(case when is_bounce then 1 else 0 end)::float / count(*), 6) as bounce_rate,
        round(sum(revenue), 2) as total_revenue,
        round(sum(revenue) / count(*), 4) as revenue_per_session
    from fct_sessions
    where session_date between '2025-10-15' and '2025-10-21'
    group by session_date
    order by metric_date asc
    """,
    5: """
    select
        user_id,
        first_session_date,
        total_sessions,
        round(total_revenue, 2) as total_revenue,
        round(total_revenue, 2) as ltv,
        days_active
    from dim_users
    order by total_revenue desc, user_id asc
    limit 20
    """,
}


def _detail(ok: bool, message: str) -> str:
    return f"{'PASS' if ok else 'FAIL'}: {message}"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series)


def compare_dataframes(agent_df: pd.DataFrame, ref_df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"score": 0.0, "details": []}
    if list(agent_df.columns) != list(ref_df.columns):
        result["details"].append(
            f"FAIL: columns mismatch. Expected {list(ref_df.columns)}, got {list(agent_df.columns)}"
        )
        return result
    if len(agent_df) != len(ref_df):
        result["details"].append(
            f"FAIL: row count mismatch. Expected {len(ref_df)}, got {len(agent_df)}"
        )
        return result

    all_match = True
    for col in ref_df.columns:
        ref_col = ref_df[col]
        agent_col = agent_df[col]
        if _numeric_series(ref_col):
            tol = CURRENCY_TOL if "revenue" in col.lower() or "ltv" in col.lower() else RATE_TOL
            ref_values = ref_col.fillna(0).to_numpy()
            agent_values = agent_col.fillna(0).to_numpy()
            if not np.allclose(agent_values, ref_values, atol=tol, equal_nan=True):
                mismatches = (~np.isclose(agent_values, ref_values, atol=tol, equal_nan=True)).sum()
                result["details"].append(f"FAIL: {col} — {int(mismatches)} values outside tolerance {tol}")
                all_match = False
        else:
            ref_norm = ref_col.astype(str).str.strip()
            agent_norm = agent_col.astype(str).str.strip()
            if not (ref_norm == agent_norm).all():
                mismatches = int((ref_norm != agent_norm).sum())
                result["details"].append(f"FAIL: {col} — {mismatches} string mismatches")
                all_match = False

    if all_match:
        result["score"] = 1.0
        result["details"].append("PASS: all values match within tolerance")
    return result


def score_pipeline_health(project_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"score": 0.0, "details": []}
    manifest_path = project_dir / "target" / "manifest.json"
    if not manifest_path.exists():
        result["details"].append("FAIL: target/manifest.json not found")
        return result

    manifest = _read_json(manifest_path)
    nodes = manifest.get("nodes", {})
    models = [v for v in nodes.values() if v.get("resource_type") == "model"]
    tests = [v for v in nodes.values() if v.get("resource_type") == "test"]

    required_models = {"stg_events", "fct_sessions", "dim_users", "fct_daily_metrics"}
    found_models = {v.get("name", "") for v in models}
    missing_models = required_models - found_models

    required_metrics = {
        "conversion_rate": {"ratio"},
        "avg_session_duration": {"derived", "ratio"},
        "bounce_rate": {"ratio"},
        "revenue_per_session": {"derived"},
        "new_user_ratio": {"ratio"},
    }
    found_metrics = set()
    metric_type_issues = []
    for key, metric in manifest.get("metrics", {}).items():
        name = key.split(".")[-1]
        found_metrics.add(name)
        if name in required_metrics and metric.get("type", "unknown") not in required_metrics[name]:
            metric_type_issues.append(
                f"{name}: type={metric.get('type', 'unknown')}, expected one of {sorted(required_metrics[name])}"
            )
    missing_metrics = set(required_metrics) - found_metrics

    placeholder_measures = []
    for semantic_model in manifest.get("semantic_models", {}).values():
        for measure in semantic_model.get("measures", []):
            expr = str(measure.get("expr", "")).strip().strip("\"'")
            if expr in {"", "0", "0.0", "null"}:
                placeholder_measures.append(f"{measure.get('name', '?')} (expr={expr!r})")

    run_results_path = project_dir / "target" / "run_results.json"
    build_success = False
    if run_results_path.exists():
        run_results = _read_json(run_results_path)
        results = run_results.get("results", [])
        failed = [item for item in results if item.get("status") not in {"success", "pass"}]
        build_success = bool(results) and not failed

    checks = [
        (len(models) >= 5, f"Models: {len(models)} (need >=5)"),
        (len(tests) >= 12, f"Tests: {len(tests)} (need >=12)"),
        (
            not missing_metrics,
            "Required metrics: all 5 present" if not missing_metrics else "Required metrics: missing " + ", ".join(sorted(missing_metrics)),
        ),
        (
            not missing_models,
            "Required models: all present" if not missing_models else "Required models: missing " + ", ".join(sorted(missing_models)),
        ),
        (build_success, "Pipeline: build succeeded" if build_success else "Pipeline: build not verified"),
        (
            not metric_type_issues,
            "Metric types: all valid" if not metric_type_issues else "Metric types: issues: " + "; ".join(metric_type_issues),
        ),
        (
            not placeholder_measures,
            "Semantic measures: no placeholders" if not placeholder_measures else "Semantic measures: placeholders found: " + ", ".join(placeholder_measures),
        ),
    ]
    result["score"] = sum(1 for ok, _ in checks if ok) / len(checks)
    result["details"] = [_detail(ok, message) for ok, message in checks]
    return result


def score_query_correctness(project_dir: Path, ref_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"score": 0.0, "details": []}
    query_scores: list[float] = []
    con = None
    db_path = project_dir / "ga4_analytics.duckdb"
    if db_path.exists():
        try:
            con = duckdb.connect(str(db_path), read_only=True)
        except Exception as exc:
            result["details"].append(f"WARN: could not open DuckDB — {exc}")

    try:
        for idx in range(1, 6):
            ref_path = ref_dir / f"query_{idx}.csv"
            ref_df = pd.read_csv(ref_path)
            agent_df = None
            source = "evaluator query"
            if con is not None:
                try:
                    agent_df = con.execute(EVALUATOR_QUERIES[idx]).df()
                except Exception as exc:
                    result["details"].append(f"Query {idx}: WARN — evaluator query failed ({exc}), falling back to CSV")
            if agent_df is None:
                csv_path = project_dir / "query_results" / f"query_{idx}.csv"
                if csv_path.exists():
                    try:
                        agent_df = pd.read_csv(csv_path)
                        source = "agent CSV (capped at 50%)"
                    except Exception as exc:
                        result["details"].append(f"Query {idx}: WARN — CSV read failed ({exc})")
            if agent_df is None:
                query_scores.append(0.0)
                result["details"].append(f"Query {idx}: 0% — no data source available")
                continue

            query_result = compare_dataframes(agent_df, ref_df)
            score = float(query_result["score"])
            if source.startswith("agent CSV"):
                score = min(score, 0.5)
            query_scores.append(score)
            result["details"].append(
                f"Query {idx} ({source}): {score:.0%} — {'; '.join(query_result['details'])}"
            )

        missing_csvs = [
            f"query_{idx}.csv"
            for idx in range(1, 6)
            if not (project_dir / "query_results" / f"query_{idx}.csv").exists()
        ]
        if missing_csvs:
            result["details"].append("WARN: missing required CSV deliverables: " + ", ".join(missing_csvs))
            penalty = 0.1 * len(missing_csvs) / 5
            query_scores = [max(0.0, score - penalty) for score in query_scores]
    finally:
        if con is not None:
            con.close()

    result["score"] = sum(query_scores) / len(query_scores) if query_scores else 0.0
    return result


def score_incremental_parity(project_dir: Path, ref_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"score": 0.0, "details": []}
    sub_scores: list[float] = []
    db_path = project_dir / "ga4_analytics.duckdb"
    if not db_path.exists():
        result["details"].append("FAIL: ga4_analytics.duckdb not found")
        return result

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        parity_ref_path = ref_dir / "parity_nov15_21.csv"
        if parity_ref_path.exists():
            ref_df = pd.read_csv(parity_ref_path)
            try:
                agent_df = con.execute(
                    """
                    select * from fct_daily_metrics
                    where metric_date between '2025-11-15' and '2025-11-21'
                    order by metric_date, channel, device_category, stream_id
                    """
                ).df()
                parity_result = compare_dataframes(agent_df, ref_df) if len(agent_df) else {"score": 0.0, "details": ["FAIL: no data in fct_daily_metrics for Nov 15-21"]}
                sub_scores.append(float(parity_result["score"]))
                result["details"].append(
                    f"Parity data check: {float(parity_result['score']):.0%} — {'; '.join(parity_result['details'])}"
                )
            except Exception as exc:
                result["details"].append(f"FAIL: parity query error — {exc}")
                sub_scores.append(0.0)
        else:
            result["details"].append("FAIL: parity_nov15_21.csv not found")
            sub_scores.append(0.0)
    finally:
        con.close()

    fct_sessions_found = False
    for root, _, files in os.walk(project_dir / "models"):
        if "fct_sessions.sql" not in files:
            continue
        fct_sessions_found = True
        content = Path(root, "fct_sessions.sql").read_text(encoding="utf-8")
        has_incremental = "is_incremental()" in content
        lookback_match = re.search(r"interval\s+'(\d+)\s+days?'", content)
        adequate_lookback = bool(lookback_match and int(lookback_match.group(1)) >= 5)
        if has_incremental and adequate_lookback:
            sub_scores.append(1.0)
            result["details"].append(f"PASS: incremental config with {lookback_match.group(1)}-day lookback")
        elif has_incremental:
            sub_scores.append(0.5)
            result["details"].append("PARTIAL: incremental config found but lookback may be too narrow")
        else:
            sub_scores.append(0.0)
            result["details"].append("FAIL: no is_incremental() block found in fct_sessions.sql")
        break

    if not fct_sessions_found:
        sub_scores.append(0.0)
        result["details"].append("FAIL: fct_sessions.sql not found")

    if len(sub_scores) >= 2:
        result["score"] = 0.6 * sub_scores[0] + 0.4 * sub_scores[1]
    elif sub_scores:
        result["score"] = sub_scores[0]
    return result


def _fetchone(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    return con.execute(sql).fetchone()


def score_edge_cases(project_dir: Path, ref_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"score": 0.0, "details": []}
    fixtures_path = ref_dir / "fixtures.json"
    if not fixtures_path.exists():
        result["details"].append("FAIL: fixtures.json not found")
        return result
    db_path = project_dir / "ga4_analytics.duckdb"
    if not db_path.exists():
        result["details"].append("FAIL: ga4_analytics.duckdb not found")
        return result

    fixtures = _read_json(fixtures_path)
    passed = 0.0
    total = 6.0
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        fix = fixtures.get("cross_midnight")
        if fix:
            row = _fetchone(con, f"select session_date from fct_sessions where session_id = '{fix['session_id']}'")
            if row and str(row[0]) == fix["expected_session_date"]:
                passed += 1
                result["details"].append(f"PASS: cross-midnight session {fix['session_id']} attributed to {fix['expected_session_date']}")
            else:
                result["details"].append(f"FAIL: cross-midnight session {fix['session_id']} not attributed to {fix['expected_session_date']}")

        fix = fixtures.get("duplicate_purchase")
        if fix:
            row = _fetchone(con, f"select is_conversion, revenue from fct_sessions where session_id = '{fix['session_id']}'")
            if row and row[0] == fix["expected_is_conversion"] and math.isclose(float(row[1]), float(fix["expected_revenue"]), abs_tol=CURRENCY_TOL):
                passed += 1
                result["details"].append(f"PASS: duplicate purchase deduped for {fix['session_id']}")
            elif row:
                result["details"].append(
                    f"FAIL: duplicate purchase {fix['session_id']} produced is_conversion={row[0]}, revenue={float(row[1]):.2f}"
                )
            else:
                result["details"].append(f"FAIL: duplicate purchase session {fix['session_id']} not found")

        fix = fixtures.get("bot_ip_only")
        if fix:
            row = _fetchone(con, f"select count(*) from fct_sessions where session_id = '{fix['session_id']}'")
            if row and row[0] == 0:
                passed += 1
                result["details"].append(f"PASS: stealth bot {fix['session_id']} excluded")
            else:
                result["details"].append(f"FAIL: stealth bot {fix['session_id']} found in fct_sessions")

        fix = fixtures.get("conflicting_identity")
        if fix:
            upid = fix["user_pseudo_id"]
            session_count = _fetchone(con, f"select count(*) from fct_sessions where user_pseudo_id = '{upid}'")[0]
            first_exists = _fetchone(con, f"select count(*) from dim_users where user_id = '{fix['first_auth_id']}'")[0] > 0
            second_exists = _fetchone(con, f"select count(*) from dim_users where user_id = '{fix['second_auth_id']}'")[0] > 0
            stitched_total = _fetchone(
                con,
                f"select coalesce(sum(total_sessions), 0) from dim_users where user_id in ('{fix['first_auth_id']}', '{fix['second_auth_id']}')",
            )[0]
            if first_exists and second_exists and stitched_total <= session_count + 2:
                passed += 1
                result["details"].append("PASS: conflicting identity links resolved without session duplication")
            else:
                result["details"].append("FAIL: conflicting identity links missing auth IDs or duplicating sessions")

            expected_auth = fix.get("expected_overlap_auth_id")
            overlap_dates = fix.get("overlap_session_dates", [])
            if expected_auth and overlap_dates:
                overlap_date = overlap_dates[0]
                formatted = f"{overlap_date[:4]}-{overlap_date[4:6]}-{overlap_date[6:8]}"
                try:
                    row = _fetchone(
                        con,
                        f"""
                        select resolved_user_id from fct_sessions
                        where user_pseudo_id = '{upid}'
                          and session_date = '{formatted}'
                        limit 1
                        """,
                    )
                    if row and row[0] == expected_auth:
                        passed += 1
                        result["details"].append(f"PASS: overlap session on {formatted} resolved to {expected_auth}")
                    elif row:
                        result["details"].append(f"FAIL: overlap session resolved to {row[0]}, expected {expected_auth}")
                    else:
                        result["details"].append(f"FAIL: no overlap session found for {upid} on {formatted}")
                except Exception:
                    passed += 0.5
                    result["details"].append("PARTIAL: resolved_user_id column missing; overlap assignment cannot be fully verified")

        fix = fixtures.get("late_arrival")
        if fix:
            expected_count = fix.get("expected_event_count", 2)
            row = _fetchone(
                con,
                f"select event_count, session_duration_seconds from fct_sessions where session_id = '{fix['session_id']}'",
            )
            if row and row[0] >= expected_count and row[1] > 0:
                passed += 1
                result["details"].append(f"PASS: late-arrival session {fix['session_id']} retained {row[0]} events")
            elif row:
                result["details"].append(f"FAIL: late-arrival session has event_count={row[0]}, duration={float(row[1]):.1f}")
            else:
                result["details"].append(f"FAIL: late-arrival session {fix['session_id']} not found")
    except Exception as exc:
        result["details"].append(f"FAIL: edge-case validation error — {exc}")
    finally:
        con.close()

    result["score"] = passed / total
    return result


def score(project_dir: Path, reference_dir: Path) -> dict[str, Any]:
    if not project_dir.exists():
        return {
            "score": 0.0,
            "passed": False,
            "hard_fail_reason": f"project directory not found: {project_dir}",
            "components": {},
        }
    if not reference_dir.exists():
        return {
            "score": 0.0,
            "passed": False,
            "hard_fail_reason": f"reference directory not found: {reference_dir}",
            "components": {},
        }

    components = {
        "pipeline_health": score_pipeline_health(project_dir),
        "query_correctness": score_query_correctness(project_dir, reference_dir),
        "incremental_parity": score_incremental_parity(project_dir, reference_dir),
        "edge_cases": score_edge_cases(project_dir, reference_dir),
    }
    total = (
        0.15 * components["pipeline_health"]["score"]
        + 0.40 * components["query_correctness"]["score"]
        + 0.20 * components["incremental_parity"]["score"]
        + 0.25 * components["edge_cases"]["score"]
    )
    return {
        "score": float(total),
        "passed": bool(total >= 0.7),
        "hard_fail_reason": None,
        "weights": {
            "pipeline_health": 0.15,
            "query_correctness": 0.40,
            "incremental_parity": 0.20,
            "edge_cases": 0.25,
        },
        "components": components,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = score(Path(args.project_dir), Path(args.reference_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
