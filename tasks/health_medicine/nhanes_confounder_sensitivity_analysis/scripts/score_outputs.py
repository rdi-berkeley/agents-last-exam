"""Weighted partial scorer for nhanes_confounder_sensitivity_analysis.

Scoring uses six weighted gates instead of all-or-nothing:

  Gate 1 (0.15) \u2014 structure: analysis labels + formula strings
  Gate 2 (0.10) \u2014 subset row count match
  Gate 3 (0.30) \u2014 subset numeric match on shared IDs
  Gate 4 (0.10) \u2014 summary n / events match
  Gate 5 (0.25) \u2014 summary estimate / CI / p / or_change match
  Gate 6 (0.10) \u2014 metadata semantic match (keyword, not exact string)

Schema (column names + order) is still a hard prerequisite \u2014 without correct
columns, nothing downstream can be compared.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import asdict, dataclass
from typing import Any


SUBSET_COLUMNS = [
    "id",
    "age",
    "gender",
    "education",
    "bmi",
    "ulcer",
    "MM_count",
    "LBXHP1",
    "hpyl_lbxhp1_log1p",
    "nsaid_current",
    "n_meds_records",
]

SUMMARY_COLUMNS = [
    "analysis",
    "formula",
    "estimate",
    "ci_low",
    "ci_high",
    "p_value",
    "n",
    "events",
    "or_change_pct_vs_modelA_subset",
    "sddsrvyr_values",
    "hpylori_variable",
    "hpylori_transform",
    "nsaid_rule",
    "survey_weights_note",
]

SUMMARY_ANALYSES = [
    "Model A (base; full sample)",
    "Model A (base; LBXHP1 non-missing subset)",
    "Model B (+H. pylori index, NSAID)",
]

SUMMARY_FORMULAS = {
    "Model A (base; full sample)": "ulcer ~ MM_count + age + gender + education + bmi",
    "Model A (base; LBXHP1 non-missing subset)": "ulcer ~ MM_count + age + gender + education + bmi",
    "Model B (+H. pylori index, NSAID)": (
        "ulcer ~ MM_count + age + gender + education + bmi + log1p(LBXHP1) + nsaid_current"
    ),
}

SUBSET_NUMERIC_COLUMNS = [column for column in SUBSET_COLUMNS if column != "id"]

SUMMARY_COUNT_COLUMNS = ["n", "events"]
SUMMARY_STAT_COLUMNS = [
    "estimate",
    "ci_low",
    "ci_high",
    "p_value",
    "or_change_pct_vs_modelA_subset",
    "sddsrvyr_values",
]
SUMMARY_META_COLUMNS = [
    "hpylori_variable",
    "hpylori_transform",
    "nsaid_rule",
    "survey_weights_note",
]

NUMERIC_TOLERANCE = 1e-9

WEIGHT_STRUCTURE = 0.15
WEIGHT_SUBSET_ROWCOUNT = 0.10
WEIGHT_SUBSET_NUMERIC = 0.30
WEIGHT_SUMMARY_COUNTS = 0.10
WEIGHT_SUMMARY_STATS = 0.25
WEIGHT_METADATA = 0.10

METADATA_KEYWORDS: dict[str, list[str]] = {
    "hpylori_variable": ["lbxhp1"],
    "hpylori_transform": ["log1p"],
    "nsaid_rule": ["ibuprofen"],
    "survey_weights_note": ["unweighted"],
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


def _read_csv(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if reader.fieldnames is None:
        raise ValueError("CSV has no header row")
    return list(reader.fieldnames), list(reader)


def _parse_float(value: str) -> float | None:
    stripped = value.strip()
    if stripped == "":
        return None
    number = float(stripped)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value {value!r}")
    return number


def _numeric_match(left_raw: str, right_raw: str) -> bool:
    left = _parse_float(left_raw)
    right = _parse_float(right_raw)
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, rel_tol=NUMERIC_TOLERANCE, abs_tol=NUMERIC_TOLERANCE)


def _metadata_semantic_match(column: str, candidate_value: str) -> bool:
    keywords = METADATA_KEYWORDS.get(column, [])
    lower = candidate_value.lower()
    return all(kw in lower for kw in keywords)


def _score_subset_shared_ids(
    candidate_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
) -> tuple[float, dict[str, Any]]:
    cand_by_id = {row["id"].strip(): row for row in candidate_rows}
    ref_by_id = {row["id"].strip(): row for row in reference_rows}
    shared_ids = sorted(set(cand_by_id) & set(ref_by_id))

    if not shared_ids:
        return 0.0, {"shared_ids": 0, "total_comparisons": 0, "matches": 0}

    total = 0
    matches = 0
    first_mismatch = None
    for sid in shared_ids:
        cand_row = cand_by_id[sid]
        ref_row = ref_by_id[sid]
        for col in SUBSET_NUMERIC_COLUMNS:
            total += 1
            if _numeric_match(cand_row[col], ref_row[col]):
                matches += 1
            elif first_mismatch is None:
                first_mismatch = {
                    "id": sid,
                    "column": col,
                    "candidate": cand_row[col],
                    "reference": ref_row[col],
                }

    frac = matches / total if total > 0 else 0.0
    details: dict[str, Any] = {
        "shared_ids": len(shared_ids),
        "reference_ids": len(ref_by_id),
        "candidate_ids": len(cand_by_id),
        "total_comparisons": total,
        "matches": matches,
        "fraction": frac,
    }
    if first_mismatch:
        details["first_mismatch"] = first_mismatch
    return frac, details


def _score_summary_numeric(
    candidate_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    columns: list[str],
) -> tuple[float, dict[str, Any]]:
    cand_by_label = {row["analysis"]: row for row in candidate_rows}
    ref_by_label = {row["analysis"]: row for row in reference_rows}
    shared = [a for a in SUMMARY_ANALYSES if a in cand_by_label and a in ref_by_label]

    if not shared:
        return 0.0, {"shared_analyses": 0, "matches": 0, "total": 0}

    total = 0
    matches = 0
    first_mismatch = None
    for analysis in shared:
        cand_row = cand_by_label[analysis]
        ref_row = ref_by_label[analysis]
        for col in columns:
            ref_val = ref_row.get(col, "").strip()
            cand_val = cand_row.get(col, "").strip()
            if ref_val == "":
                continue
            total += 1
            if _numeric_match(cand_val, ref_val):
                matches += 1
            elif first_mismatch is None:
                first_mismatch = {
                    "analysis": analysis,
                    "column": col,
                    "candidate": cand_val,
                    "reference": ref_val,
                }

    frac = matches / total if total > 0 else 1.0
    details: dict[str, Any] = {"total": total, "matches": matches, "fraction": frac}
    if first_mismatch:
        details["first_mismatch"] = first_mismatch
    return frac, details


def _score_metadata_semantic(
    candidate_rows: list[dict[str, str]],
) -> tuple[float, dict[str, Any]]:
    total = 0
    matches = 0
    mismatches: list[dict[str, Any]] = []
    for row in candidate_rows:
        for col in SUMMARY_META_COLUMNS:
            total += 1
            if _metadata_semantic_match(col, row.get(col, "")):
                matches += 1
            else:
                mismatches.append(
                    {
                        "analysis": row.get("analysis", ""),
                        "column": col,
                        "candidate": row.get(col, ""),
                        "required_keywords": METADATA_KEYWORDS.get(col, []),
                    }
                )

    frac = matches / total if total > 0 else 0.0
    details: dict[str, Any] = {"total": total, "matches": matches, "fraction": frac}
    if mismatches:
        details["mismatches"] = mismatches[:3]
    return frac, details


def score_output_bundle(
    *,
    candidate_subset_csv: str,
    candidate_summary_csv: str,
    reference_subset_csv: str,
    reference_summary_csv: str,
) -> ScoreResult:
    # Hard prerequisite: parse + schema ------------------------------------------
    try:
        cand_sub_cols, cand_sub_rows = _read_csv(candidate_subset_csv)
        cand_sum_cols, cand_sum_rows = _read_csv(candidate_summary_csv)
        ref_sub_cols, ref_sub_rows = _read_csv(reference_subset_csv)
        ref_sum_cols, ref_sum_rows = _read_csv(reference_summary_csv)
    except Exception as exc:
        return ScoreResult(0.0, False, f"csv_parse_error: {exc}", "csv_parse", {})

    for label, observed, expected in [
        ("subset_schema", cand_sub_cols, SUBSET_COLUMNS),
        ("summary_schema", cand_sum_cols, SUMMARY_COLUMNS),
        ("reference_subset_schema", ref_sub_cols, SUBSET_COLUMNS),
        ("reference_summary_schema", ref_sum_cols, SUMMARY_COLUMNS),
    ]:
        if observed != expected:
            return ScoreResult(
                0.0, False, f"{label}_mismatch", label,
                {"expected": expected, "observed": observed},
            )

    gate_details: dict[str, Any] = {}

    # Gate 1 \u2014 structure (labels + formulas) ------------------------------------
    cand_analyses = [row["analysis"] for row in cand_sum_rows]
    labels_ok = cand_analyses == SUMMARY_ANALYSES
    formulas_ok = labels_ok and all(
        row["formula"] == SUMMARY_FORMULAS.get(row["analysis"], "")
        for row in cand_sum_rows
    )
    structure_score = (0.5 if labels_ok else 0.0) + (0.5 if formulas_ok else 0.0)
    gate_details["structure"] = {
        "labels_ok": labels_ok,
        "formulas_ok": formulas_ok,
        "gate_score": structure_score,
    }

    # Gate 2 \u2014 subset row count -------------------------------------------------
    rowcount_score = 1.0 if len(cand_sub_rows) == len(ref_sub_rows) else 0.0
    gate_details["subset_rowcount"] = {
        "candidate": len(cand_sub_rows),
        "reference": len(ref_sub_rows),
        "gate_score": rowcount_score,
    }

    # Gate 3 \u2014 subset numeric on shared IDs -------------------------------------
    subset_score, subset_det = _score_subset_shared_ids(cand_sub_rows, ref_sub_rows)
    subset_det["gate_score"] = subset_score
    gate_details["subset_numeric"] = subset_det

    # Gate 4 \u2014 summary n / events -----------------------------------------------
    if labels_ok:
        counts_score, counts_det = _score_summary_numeric(
            cand_sum_rows, ref_sum_rows, SUMMARY_COUNT_COLUMNS,
        )
    else:
        counts_score, counts_det = 0.0, {"skipped": "labels_mismatch"}
    counts_det["gate_score"] = counts_score
    gate_details["summary_counts"] = counts_det

    # Gate 5 \u2014 summary estimate / CI / p / or_change ----------------------------
    if labels_ok:
        stats_score, stats_det = _score_summary_numeric(
            cand_sum_rows, ref_sum_rows, SUMMARY_STAT_COLUMNS,
        )
    else:
        stats_score, stats_det = 0.0, {"skipped": "labels_mismatch"}
    stats_det["gate_score"] = stats_score
    gate_details["summary_stats"] = stats_det

    # Gate 6 \u2014 metadata semantic ------------------------------------------------
    if labels_ok:
        meta_score, meta_det = _score_metadata_semantic(cand_sum_rows)
    else:
        meta_score, meta_det = 0.0, {"skipped": "labels_mismatch"}
    meta_det["gate_score"] = meta_score
    gate_details["metadata"] = meta_det

    # Final weighted sum --------------------------------------------------------
    final_score = round(
        WEIGHT_STRUCTURE * structure_score
        + WEIGHT_SUBSET_ROWCOUNT * rowcount_score
        + WEIGHT_SUBSET_NUMERIC * subset_score
        + WEIGHT_SUMMARY_COUNTS * counts_score
        + WEIGHT_SUMMARY_STATS * stats_score
        + WEIGHT_METADATA * meta_score,
        6,
    )

    passed = final_score >= 1.0 - 1e-9
    if passed:
        reason = "exact_match_within_tolerance"
        hard_gate = None
    else:
        worst = min(gate_details, key=lambda k: gate_details[k].get("gate_score", 0))
        reason = f"partial_match (worst_gate={worst})"
        hard_gate = reason

    return ScoreResult(
        score=final_score,
        passed=passed,
        reason=reason,
        hard_gate=hard_gate,
        details=gate_details,
    )
