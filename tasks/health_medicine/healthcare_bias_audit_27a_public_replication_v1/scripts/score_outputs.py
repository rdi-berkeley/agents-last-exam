"""Deterministic scorer for healthcare_bias_audit_27a_public_replication_v1."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NUMERIC_TOLERANCE = 1e-6
REQUIRED_OUTPUT_FILES = [
    "audit_answers.json",
    "audit_memo.md",
    "results/figure1b.csv",
    "results/model_lasso_predictors.csv",
    "results/model_r2.csv",
    "results/table2_concentration_metric.csv",
    "results/table3.csv",
]
REQUIRED_JSON_KEYS = [
    "dataset_context",
    "figure1b_percentile_97_before",
    "figure1b_percentile_97_after",
    "figure1b_percentile_97_ratio",
    "table2_race_black_total_costs",
    "table2_race_black_avoidable_costs",
    "table2_race_black_active_chronic_conditions",
    "table2_race_black_best_worst_difference",
    "model_r2_gagne_on_risk_score",
    "model_r2_gagne_on_gagne_hat",
    "table3_observed_program_frac_black",
    "table3_predicted_health_in_cost_bin_frac_black",
    "table3_highest_predicted_cost_frac_black",
    "table3_worst_predicted_health_frac_black",
    "diagnosis",
    "recommended_action",
]
EXACT_JSON_STRING_FIELDS = {
    "dataset_context": "public_synthetic_replication",
    "diagnosis": "cost_based_ranking_understates_need_for_black_patients",
    "recommended_action": "prefer_need_based_label_for_allocation",
}


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hard_fail(reason: str, details: dict[str, Any] | None = None) -> ScoreResult:
    return ScoreResult(0.0, False, reason, reason, details or {})


def _read_csv(text: str) -> tuple[list[str], list[list[str]]]:
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    return rows[0], rows[1:]


def _parse_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        stripped = str(value).strip()
        if stripped == "":
            return None
        number = float(stripped)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value {value!r}")
    return number


def _values_match(candidate: Any, reference: Any) -> bool:
    if candidate == reference:
        return True
    try:
        candidate_num = _parse_float(candidate)
        reference_num = _parse_float(reference)
    except Exception:
        return False
    if candidate_num is None or reference_num is None:
        return candidate_num is None and reference_num is None
    return math.isclose(candidate_num, reference_num, rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE)


def _compare_csv(name: str, candidate_text: str, reference_text: str) -> ScoreResult | None:
    try:
        candidate_header, candidate_rows = _read_csv(candidate_text)
        reference_header, reference_rows = _read_csv(reference_text)
    except Exception as exc:
        return _hard_fail(f"{name}: csv_parse_error", {"error": str(exc)})

    if candidate_header != reference_header:
        return _hard_fail(
            f"{name}: header_mismatch",
            {"candidate": candidate_header, "reference": reference_header},
        )
    if len(candidate_rows) != len(reference_rows):
        return _hard_fail(
            f"{name}: row_count_mismatch",
            {"candidate": len(candidate_rows), "reference": len(reference_rows)},
        )

    for row_index, (candidate_row, reference_row) in enumerate(
        zip(candidate_rows, reference_rows),
        start=1,
    ):
        if len(candidate_row) != len(reference_header):
            return _hard_fail(
                f"{name}: candidate_row_width_mismatch",
                {"row_index": row_index, "candidate_width": len(candidate_row)},
            )
        if len(reference_row) != len(reference_header):
            return _hard_fail(
                f"{name}: reference_row_width_mismatch",
                {"row_index": row_index, "reference_width": len(reference_row)},
            )
        for column_index, column_name in enumerate(reference_header):
            candidate_value = candidate_row[column_index]
            reference_value = reference_row[column_index]
            if _values_match(candidate_value, reference_value):
                continue
            return _hard_fail(
                f"{name}: value_mismatch",
                {
                    "row_index": row_index,
                    "column": column_name,
                    "candidate": candidate_value,
                    "reference": reference_value,
                },
            )
    return None


def _compare_json(candidate_text: str, reference_text: str) -> ScoreResult | None:
    try:
        candidate = json.loads(candidate_text)
        reference = json.loads(reference_text)
    except Exception as exc:
        return _hard_fail("audit_answers.json: json_parse_error", {"error": str(exc)})

    if not isinstance(candidate, dict):
        return _hard_fail("audit_answers.json: not_an_object")
    if list(reference.keys()) != REQUIRED_JSON_KEYS:
        return _hard_fail(
            "audit_answers.json: reference_key_order_mismatch",
            {"reference_keys": list(reference.keys())},
        )
    if set(candidate.keys()) != set(REQUIRED_JSON_KEYS):
        return _hard_fail(
            "audit_answers.json: key_set_mismatch",
            {"candidate_keys": sorted(candidate.keys())},
        )

    for key in REQUIRED_JSON_KEYS:
        candidate_value = candidate[key]
        reference_value = reference[key]
        if key in EXACT_JSON_STRING_FIELDS:
            if candidate_value != EXACT_JSON_STRING_FIELDS[key]:
                return _hard_fail(
                    "audit_answers.json: exact_string_mismatch",
                    {"key": key, "candidate": candidate_value},
                )
            continue
        if not _values_match(candidate_value, reference_value):
            return _hard_fail(
                "audit_answers.json: value_mismatch",
                {"key": key, "candidate": candidate_value, "reference": reference_value},
            )

    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_numbers(text: str) -> list[float]:
    numbers = []
    for match in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text):
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    return numbers


def _contains_anchor(numbers: list[float], anchors: list[float]) -> bool:
    for number in numbers:
        for anchor in anchors:
            tolerance = max(0.02, abs(anchor) * 0.02)
            if math.isclose(number, anchor, rel_tol=0.02, abs_tol=tolerance):
                return True
    return False


def _check_forbidden_paper_values(lowered: str) -> str | None:
    sentences = re.split(r"(?<=[.!?])\s+", lowered)
    for sentence in sentences:
        if "17.7" not in sentence and "46.5" not in sentence and "59%" not in sentence and "59 percent" not in sentence:
            continue
        allowed_context = (
            "do not" in sentence
            or "don't" in sentence
            or "not " in sentence
            or "private data" in sentence
            or "private-data" in sentence
            or "original paper" in sentence
            or "graded answer" in sentence
            or "public synthetic" in sentence
        )
        if not allowed_context:
            return sentence
    return None


def _check_memo(candidate_text: str, reference_answers_text: str) -> ScoreResult | None:
    lowered = _normalize(candidate_text)
    if not lowered:
        return _hard_fail("audit_memo.md: empty")

    try:
        reference_answers = json.loads(reference_answers_text)
    except Exception as exc:
        return _hard_fail("audit_memo.md: bad_reference_answers", {"error": str(exc)})

    forbidden_sentence = _check_forbidden_paper_values(lowered)
    if forbidden_sentence is not None:
        return _hard_fail(
            "audit_memo.md: private_paper_value_present_without_context",
            {"sentence": forbidden_sentence},
        )

    if re.search(r"does not .*understate|do not .*understate|not support .*understate", lowered):
        return _hard_fail("audit_memo.md: incorrect_conclusion")
    if "keep the cost-based label" in lowered or "keep cost-based label" in lowered:
        return _hard_fail("audit_memo.md: incorrect_recommendation")

    required_checks = {
        "public_synthetic_context": "public synthetic" in lowered,
        "private_data_distinction": (
            "private data" in lowered or "private-data" in lowered or "original paper" in lowered
        ),
        "understates_need": (
            "understate" in lowered and "black" in lowered and "need" in lowered
        ),
        "cost_need_mechanism": (
            "cost" in lowered
            and "need" in lowered
            and ("proxy" in lowered or "diverge" in lowered or "imperfect" in lowered)
        ),
        "diagnosis_not_just_retraining": (
            ("diagnos" in lowered or "mechanism" in lowered)
            and ("retrain" in lowered or "retraining" in lowered)
        ),
        "need_based_recommendation": (
            ("need-based" in lowered or "need based" in lowered)
            and "allocation" in lowered
            and ("prefer" in lowered or "better" in lowered or "appropriate" in lowered)
        ),
    }
    missing = [name for name, passed in required_checks.items() if not passed]
    if missing:
        return _hard_fail("audit_memo.md: missing_required_content", {"missing": missing})

    numbers = _extract_numbers(lowered)
    figure_anchors = [
        float(reference_answers["figure1b_percentile_97_before"]),
        float(reference_answers["figure1b_percentile_97_after"]),
        float(reference_answers["figure1b_percentile_97_ratio"]),
        float(reference_answers["figure1b_percentile_97_before"]) * 100,
        float(reference_answers["figure1b_percentile_97_after"]) * 100,
        float(reference_answers["figure1b_percentile_97_ratio"]) * 100,
    ]
    table_anchors = [
        float(reference_answers["table2_race_black_total_costs"]),
        float(reference_answers["table2_race_black_active_chronic_conditions"]),
        float(reference_answers["table2_race_black_best_worst_difference"]),
        float(reference_answers["table3_observed_program_frac_black"]),
        float(reference_answers["table3_predicted_health_in_cost_bin_frac_black"]),
        float(reference_answers["table3_highest_predicted_cost_frac_black"]),
        float(reference_answers["table3_worst_predicted_health_frac_black"]),
        float(reference_answers["table2_race_black_total_costs"]) * 100,
        float(reference_answers["table2_race_black_active_chronic_conditions"]) * 100,
        float(reference_answers["table2_race_black_best_worst_difference"]) * 100,
        float(reference_answers["table3_observed_program_frac_black"]) * 100,
        float(reference_answers["table3_predicted_health_in_cost_bin_frac_black"]) * 100,
        float(reference_answers["table3_highest_predicted_cost_frac_black"]) * 100,
        float(reference_answers["table3_worst_predicted_health_frac_black"]) * 100,
    ]
    has_figure_reference = (
        ("figure 1b" in lowered or "figure1b" in lowered or "97th percentile" in lowered or "percentile 97" in lowered)
        and _contains_anchor(numbers, figure_anchors)
    )
    has_table_reference = (
        (
            "table 2" in lowered
            or "table2" in lowered
            or "table 3" in lowered
            or "table3" in lowered
            or "observed program enrollment" in lowered
            or "worst predicted health" in lowered
        )
        and _contains_anchor(numbers, table_anchors)
    )
    if not has_figure_reference or not has_table_reference:
        return _hard_fail(
            "audit_memo.md: missing_output_grounding",
            {
                "has_figure_reference": has_figure_reference,
                "has_table_reference": has_table_reference,
            },
        )

    return None


def score_output_bundle(
    *,
    candidate_files: dict[str, str],
    reference_files: dict[str, str],
) -> ScoreResult:
    if set(candidate_files.keys()) != set(REQUIRED_OUTPUT_FILES):
        return _hard_fail(
            "candidate_bundle: unexpected_file_set",
            {"files": sorted(candidate_files.keys())},
        )
    if set(reference_files.keys()) != set(REQUIRED_OUTPUT_FILES):
        return _hard_fail(
            "reference_bundle: unexpected_file_set",
            {"files": sorted(reference_files.keys())},
        )

    for rel_path in REQUIRED_OUTPUT_FILES:
        if rel_path.endswith(".csv"):
            failure = _compare_csv(rel_path, candidate_files[rel_path], reference_files[rel_path])
            if failure is not None:
                return failure

    failure = _compare_json(
        candidate_files["audit_answers.json"],
        reference_files["audit_answers.json"],
    )
    if failure is not None:
        return failure

    failure = _check_memo(
        candidate_files["audit_memo.md"],
        reference_files["audit_answers.json"],
    )
    if failure is not None:
        return failure

    return ScoreResult(1.0, True, "passed", None, {})


def _load_bundle_from_dir(root: Path) -> dict[str, str]:
    bundle = {}
    for rel_path in REQUIRED_OUTPUT_FILES:
        path = root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"missing required file: {path}")
        bundle[rel_path] = path.read_text(encoding="utf-8")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()

    result = score_output_bundle(
        candidate_files=_load_bundle_from_dir(args.candidate_dir),
        reference_files=_load_bundle_from_dir(args.reference_dir),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
