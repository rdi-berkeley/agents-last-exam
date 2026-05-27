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
        control = [int(r[column_name]) for r in rows if r["variant"] == "control" and r["delivered"] == "1"]
        treatment = [int(r[column_name]) for r in rows if r["variant"] == "treatment" and r["delivered"] == "1"]
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
        se_pooled = math.sqrt(pooled * (1 - pooled) * ((1 / n_c) + (1 / n_t))) if pooled not in {0.0, 1.0} else 0.0
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


def validate_results_tsv(output_path: Path, expected_stats: dict[str, dict[str, float | bool]]) -> bool:
    rows = read_csv_rows(output_path, delimiter="\t")
    if len(rows) != 4:
        return False
    by_metric = {row["metric"]: row for row in rows}
    if set(by_metric) != set(METRIC_COLUMN_MAP):
        return False

    bh_expected = bh_correct({metric: expected_stats[metric]["p_value_raw"] for metric in SECONDARY_METRICS})
    for metric, expected in expected_stats.items():
        row = by_metric[metric]
        if to_bool(row["is_primary"]) != bool(expected["is_primary"]):
            return False
        if not floats_close(float(row["control_rate"]), float(expected["control_rate"])):
            return False
        if not floats_close(float(row["treatment_rate"]), float(expected["treatment_rate"])):
            return False
        if not floats_close(float(row["absolute_lift"]), float(expected["absolute_lift"])):
            return False
        if not floats_close(float(row["ci_lower_95"]), float(expected["ci_lower_95"])):
            return False
        if not floats_close(float(row["ci_upper_95"]), float(expected["ci_upper_95"])):
            return False
        if not floats_close(float(row["z_statistic"]), float(expected["z_statistic"]), tol=0.01):
            return False
        if not floats_close(float(row["p_value_raw"]), float(expected["p_value_raw"]), tol=0.001):
            return False
        if to_bool(row["significant_at_05"]) != bool(expected["significant_at_05"]):
            return False
        if metric == PRIMARY_METRIC:
            if any(str(row[key]).strip() for key in ["bh_rank", "bh_threshold", "bh_significant"]):
                return False
        else:
            bh = bh_expected[metric]
            if not floats_close(float(row["bh_rank"]), float(bh["bh_rank"]), tol=0.001):
                return False
            if not floats_close(float(row["bh_threshold"]), float(bh["bh_threshold"]), tol=0.001):
                return False
            if to_bool(row["bh_significant"]) != bool(bh["bh_significant"]):
                return False
    return True


def validate_assignment_csv(output_path: Path) -> bool:
    rows = read_csv_rows(output_path)
    if not rows:
        return False
    values = {row["metric"]: row["value"] for row in rows}
    required = {"n_control", "n_treatment", "ratio", "srm_chi2", "srm_pvalue", "srm_pass"}
    if set(values) != required:
        return False
    if int(float(values["n_control"])) <= 0 or int(float(values["n_treatment"])) <= 0:
        return False
    if not floats_close(float(values["ratio"]), 1.0):
        return False
    if float(values["srm_pvalue"]) <= 0.01:
        return False
    return to_bool(values["srm_pass"])


def validate_report_md(output_path: Path, expected_stats: dict[str, dict[str, float | bool]], required_n: int) -> bool:
    text = output_path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    if "recommendation" not in lower or ("ship" not in lower and "hold" not in lower):
        return False
    guardrail_lift_pp = expected_stats["unsubscribed_rate"]["absolute_lift"] * 100
    guardrail_pass = guardrail_lift_pp < 0.5
    primary_sig = bool(expected_stats[PRIMARY_METRIC]["significant_at_05"])
    expected_recommendation = "ship" if primary_sig and guardrail_pass else "hold"
    if expected_recommendation not in lower:
        return False
    if str(required_n) not in text and f"{required_n:,}" not in text:
        return False
    if "3.4" not in text and "3.40" not in text:
        return False
    return True


def compare_to_reference(output_path: Path, reference_path: Path) -> bool:
    return output_path.read_text(encoding="utf-8", errors="replace").strip() == reference_path.read_text(
        encoding="utf-8", errors="replace"
    ).strip()


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
    if payload["missing_files"]:
        print(json.dumps(payload, indent=2))
        return 0

    expected_stats = metric_stats(input_dir / "experiment_results_raw.csv")
    required_n = sample_size_required(input_dir / "historical_metrics.csv")
    assignment_ok = validate_assignment_csv(required_files["assignment"])
    results_ok = validate_results_tsv(required_files["results"], expected_stats)
    report_ok = validate_report_md(required_files["report"], expected_stats, required_n)
    reference_match = all(
        compare_to_reference(required_files[key], reference_dir / required_files[key].name)
        for key in required_files
    )

    payload.update(
        {
            "assignment_ok": assignment_ok,
            "results_ok": results_ok,
            "report_ok": report_ok,
            "reference_match": reference_match,
            "required_n": required_n,
            "open_rate_lift_pp": round(expected_stats[PRIMARY_METRIC]["absolute_lift"] * 100, 3),
            "unsubscribe_lift_pp": round(expected_stats["unsubscribed_rate"]["absolute_lift"] * 100, 3),
        }
    )
    if assignment_ok and results_ok and report_ok:
        payload["score"] = 1.0
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
