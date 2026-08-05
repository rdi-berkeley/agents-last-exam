from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import NormalDist

PRIMARY_METRIC = "opened_rate"
SECONDARY_METRICS = ["clicked_rate", "converted_rate", "unsubscribed_rate"]
METRIC_COLUMN_MAP = {
    "opened_rate": "opened",
    "clicked_rate": "clicked",
    "converted_rate": "converted",
    "unsubscribed_rate": "unsubscribed",
}
Z_975 = NormalDist().inv_cdf(0.975)
POWER_SAMPLE_SIZE_LOWER_TOLERANCE = 2
ASSIGNMENT_WEIGHT = 0.2
RESULT_METRIC_WEIGHT = 0.15
REPORT_WEIGHT = 0.2
UNDEFINED_NUMERIC_VALUES = {"", "na", "n/a", "nan", "none", "null", "undefined"}
RECOMMENDATION_HEADING_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*|__)?recommendation\s*:?(?:\*\*|__)?\s*(.*)$",
    re.IGNORECASE,
)
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
RECOMMENDATION_TOKEN_RE = re.compile(r"\b(ship|hold)\b", re.IGNORECASE)
NEGATED_TOKEN_RE = re.compile(
    r"(?:\b(?:not|never|avoid|without)\b|\brather\s+than\b|\binstead\s+of\b)"
    r"(?:\s+\w+){0,3}\s*$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def to_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_csv_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def sample_size_required(historical_metrics: Path) -> int:
    rows = read_csv_rows(historical_metrics)
    baseline = sum(float(row["open_rate"]) for row in rows) / len(rows)
    mde = 0.03
    alpha = 0.05
    power = 0.80
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_beta = NormalDist().inv_cdf(power)
    p1 = baseline
    p2 = baseline + mde
    p_bar = (p1 + p2) / 2
    n = (
        (
            z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
            + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        )
        ** 2
    ) / (mde**2)
    return math.ceil(n)


def metric_stats(results_raw: Path) -> dict[str, dict[str, float | bool]]:
    rows = read_csv_rows(results_raw)
    stats: dict[str, dict[str, float | bool]] = {}
    for metric_name, column_name in METRIC_COLUMN_MAP.items():
        control = [
            int(r[column_name]) for r in rows if r["variant"] == "control" and r["delivered"] == "1"
        ]
        treatment = [
            int(r[column_name])
            for r in rows
            if r["variant"] == "treatment" and r["delivered"] == "1"
        ]
        n_c = len(control)
        n_t = len(treatment)
        x_c = sum(control)
        x_t = sum(treatment)
        p_c = x_c / n_c
        p_t = x_t / n_t
        diff = p_t - p_c
        rel_lift = 0.0 if p_c == 0 and p_t == 0 else (diff / p_c * 100.0 if p_c else float("inf"))
        se_unpooled = math.sqrt((p_c * (1 - p_c) / n_c) + (p_t * (1 - p_t) / n_t))
        ci_lower = diff - Z_975 * se_unpooled
        ci_upper = diff + Z_975 * se_unpooled
        pooled = (x_c + x_t) / (n_c + n_t)
        se_pooled = (
            math.sqrt(pooled * (1 - pooled) * ((1 / n_c) + (1 / n_t)))
            if pooled not in {0.0, 1.0}
            else 0.0
        )
        z_stat = diff / se_pooled if se_pooled else 0.0
        p_val = 2 * (1 - NormalDist().cdf(abs(z_stat)))
        stats[metric_name] = {
            "control_rate": p_c,
            "treatment_rate": p_t,
            "absolute_lift": diff,
            "relative_lift_pct": 0.0 if math.isinf(rel_lift) else rel_lift,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "z_statistic": z_stat,
            "p_value_raw": p_val,
            "significant_at_05": p_val < 0.05,
            "is_primary": metric_name == PRIMARY_METRIC,
        }
    return stats


def bh_correct(p_values: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    out: dict[str, dict[str, float | bool]] = {}
    for idx, (metric, p_val) in enumerate(ordered, start=1):
        threshold = idx / m * 0.05
        out[metric] = {
            "bh_rank": float(idx),
            "bh_threshold": threshold,
            "bh_significant": p_val <= threshold,
        }
    return out


def floats_close(a: float, b: float, tol: float = 0.001) -> bool:
    return abs(a - b) <= tol


def numeric_value_matches(
    value: str | None,
    expected: float,
    *,
    tol: float = 0.001,
    allow_undefined: bool = False,
) -> bool:
    text = str(value or "").strip().lower()
    if text in UNDEFINED_NUMERIC_VALUES:
        return allow_undefined
    try:
        return floats_close(float(text), expected, tol=tol)
    except ValueError:
        return False


def report_has_acceptable_sample_size(text: str, required_n: int) -> bool:
    """Allow the small Cohen's h vs. closed-form power-analysis difference."""
    minimum_n = max(0, required_n - POWER_SAMPLE_SIZE_LOWER_TOLERANCE)
    return any(
        str(candidate) in text or f"{candidate:,}" in text
        for candidate in range(minimum_n, required_n + 1)
    )


def extract_recommendation(text: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        heading = RECOMMENDATION_HEADING_RE.match(line)
        if heading is None:
            continue

        section_lines = []
        inline_text = heading.group(1).strip()
        if inline_text:
            section_lines.append(inline_text)
        for following in lines[index + 1 :]:
            if MARKDOWN_HEADING_RE.match(following):
                break
            if not following.strip():
                if section_lines:
                    break
                continue
            section_lines.append(following.strip())

        section = " ".join(section_lines)
        decisions = set()
        for token in RECOMMENDATION_TOKEN_RE.finditer(section):
            decision = token.group(1).lower()
            prefix = section[max(0, token.start() - 48) : token.start()]
            if NEGATED_TOKEN_RE.search(prefix):
                decision = "hold" if decision == "ship" else "ship"
            decisions.add(decision)
        return decisions.pop() if len(decisions) == 1 else None
    return None


def validate_metric_row(
    metric: str,
    row: dict[str, str],
    expected: dict[str, float | bool],
    bh_expected: dict[str, dict[str, float | bool]],
) -> bool:
    try:
        if to_bool(row["is_primary"]) != bool(expected["is_primary"]):
            return False
        for key in (
            "control_rate",
            "treatment_rate",
            "absolute_lift",
            "ci_lower_95",
            "ci_upper_95",
        ):
            if not numeric_value_matches(row[key], float(expected[key])):
                return False

        control_rate = float(expected["control_rate"])
        treatment_rate = float(expected["treatment_rate"])
        degenerate = control_rate == treatment_rate and control_rate in {0.0, 1.0}
        if not numeric_value_matches(
            row["z_statistic"],
            float(expected["z_statistic"]),
            tol=0.01,
            allow_undefined=degenerate,
        ):
            return False
        if not numeric_value_matches(
            row["p_value_raw"],
            float(expected["p_value_raw"]),
            allow_undefined=degenerate,
        ):
            return False
        if to_bool(row["significant_at_05"]) != bool(expected["significant_at_05"]):
            return False
        if metric == PRIMARY_METRIC:
            return not any(
                str(row[key]).strip() for key in ["bh_rank", "bh_threshold", "bh_significant"]
            )

        bh = bh_expected[metric]
        if not numeric_value_matches(row["bh_rank"], float(bh["bh_rank"])):
            return False
        if not numeric_value_matches(row["bh_threshold"], float(bh["bh_threshold"])):
            return False
        return to_bool(row["bh_significant"]) == bool(bh["bh_significant"])
    except (KeyError, TypeError, ValueError):
        return False


def validate_result_rows(
    output_path: Path, expected_stats: dict[str, dict[str, float | bool]]
) -> dict[str, bool]:
    checks = dict.fromkeys(METRIC_COLUMN_MAP, False)
    try:
        rows = read_csv_rows(output_path, delimiter="\t")
    except (OSError, csv.Error):
        return checks
    if len(rows) != 4:
        return checks
    by_metric = {row.get("metric"): row for row in rows}
    if set(by_metric) != set(METRIC_COLUMN_MAP):
        return checks

    bh_expected = bh_correct(
        {metric: expected_stats[metric]["p_value_raw"] for metric in SECONDARY_METRICS}
    )
    for metric, expected in expected_stats.items():
        checks[metric] = validate_metric_row(metric, by_metric[metric], expected, bh_expected)
    return checks


def validate_results_tsv(
    output_path: Path, expected_stats: dict[str, dict[str, float | bool]]
) -> bool:
    return all(validate_result_rows(output_path, expected_stats).values())


def validate_assignment_csv(output_path: Path) -> bool:
    try:
        rows = read_csv_rows(output_path)
    except (OSError, csv.Error):
        return False
    if not rows:
        return False
    try:
        values = {row["metric"]: row["value"] for row in rows}
    except (KeyError, TypeError):
        return False
    required = {"n_control", "n_treatment", "ratio", "srm_chi2", "srm_pvalue", "srm_pass"}
    if set(values) != required:
        return False
    try:
        if int(float(values["n_control"])) <= 0 or int(float(values["n_treatment"])) <= 0:
            return False
        if not floats_close(float(values["ratio"]), 1.0):
            return False
        if float(values["srm_pvalue"]) <= 0.01:
            return False
    except (TypeError, ValueError):
        return False
    return to_bool(values["srm_pass"])


def validate_report_md(
    output_path: Path, expected_stats: dict[str, dict[str, float | bool]], required_n: int
) -> bool:
    try:
        text = output_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    guardrail_lift_pp = expected_stats["unsubscribed_rate"]["absolute_lift"] * 100
    guardrail_pass = guardrail_lift_pp < 0.5
    primary_sig = bool(expected_stats[PRIMARY_METRIC]["significant_at_05"])
    expected_recommendation = "ship" if primary_sig and guardrail_pass else "hold"
    if extract_recommendation(text) != expected_recommendation:
        return False
    if not report_has_acceptable_sample_size(text, required_n):
        return False
    return "3.4" in text or "3.40" in text


def component_scores(
    assignment_ok: bool, result_checks: dict[str, bool], report_ok: bool
) -> dict[str, float]:
    scores = {"assignment": ASSIGNMENT_WEIGHT if assignment_ok else 0.0}
    scores.update(
        {
            f"results_{metric}": RESULT_METRIC_WEIGHT if ok else 0.0
            for metric, ok in result_checks.items()
        }
    )
    scores["report"] = REPORT_WEIGHT if report_ok else 0.0
    return scores


def compare_to_reference(output_path: Path, reference_path: Path) -> bool:
    return (
        output_path.read_text(encoding="utf-8", errors="replace").strip()
        == reference_path.read_text(encoding="utf-8", errors="replace").strip()
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)
    payload = {"score": 0.0}

    required_files = {
        "assignment": output_dir / "randomization_assignment.csv",
        "results": output_dir / "experiment_results.tsv",
        "report": output_dir / "experiment_report.md",
    }
    payload["missing_files"] = [name for name, path in required_files.items() if not path.exists()]

    expected_stats = metric_stats(input_dir / "experiment_results_raw.csv")
    required_n = sample_size_required(input_dir / "historical_metrics.csv")
    assignment_ok = required_files["assignment"].exists() and validate_assignment_csv(
        required_files["assignment"]
    )
    if required_files["results"].exists():
        result_checks = validate_result_rows(required_files["results"], expected_stats)
    else:
        result_checks = dict.fromkeys(METRIC_COLUMN_MAP, False)
    results_ok = all(result_checks.values())
    report_ok = required_files["report"].exists() and validate_report_md(
        required_files["report"], expected_stats, required_n
    )
    reference_match = not payload["missing_files"] and all(
        compare_to_reference(required_files[key], reference_dir / required_files[key].name)
        for key in required_files
    )
    scores = component_scores(assignment_ok, result_checks, report_ok)

    payload.update(
        {
            "assignment_ok": assignment_ok,
            "results_ok": results_ok,
            "result_checks": result_checks,
            "report_ok": report_ok,
            "reference_match": reference_match,
            "component_scores": scores,
            "required_n": required_n,
            "open_rate_lift_pp": round(expected_stats[PRIMARY_METRIC]["absolute_lift"] * 100, 3),
            "unsubscribe_lift_pp": round(
                expected_stats["unsubscribed_rate"]["absolute_lift"] * 100, 3
            ),
        }
    )
    payload["score"] = round(sum(scores.values()), 10)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
