"""Score CRF-to-SDTM mapping CSV outputs for crf_sdtm_mapping_4."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

COMMON_COLUMNS = [
    "crf_form",
    "crf_field_label",
    "crf_item_or_placeholder",
    "sdtm_dataset",
    "sdtm_variable",
    "role",
    "origin",
    "mapping_rule",
    "controlled_terms_or_expected_values",
]

VARIANT_SPECS = {
    "base": {
        "output_file": "ae_mapping.csv",
        "columns": COMMON_COLUMNS + ["goes_to_suppqual", "notes"],
        "allowed_datasets": ("AE", "SUPPAE"),
        "flag_column": "goes_to_suppqual",
        "allowed_flags": ("YES", "NO"),
    },
    "dm": {
        "output_file": "dm_mapping.csv",
        "columns": COMMON_COLUMNS + ["derived_or_assigned", "notes"],
        "allowed_datasets": ("DM", "SUPPDM"),
        "flag_column": "derived_or_assigned",
        "allowed_flags": ("CRF", "DERIVED", "ASSIGNED"),
    },
}

KEY_COLUMNS = [
    "crf_item_or_placeholder",
    "sdtm_dataset",
    "sdtm_variable",
]


@dataclass
class ScoreResult:
    score: float
    strict_score: float = 0.0
    row_coverage: float = 0.0
    avg_column_accuracy: float = 0.0
    errors: list[str] = field(default_factory=list)
    missing_keys: list[list[str]] = field(default_factory=list)
    extra_keys: list[list[str]] = field(default_factory=list)
    mismatches: list[dict[str, str]] = field(default_factory=list)
    relaxed_matches: list[dict] = field(default_factory=list)
    compared_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "strict_score": self.strict_score,
            "row_coverage": self.row_coverage,
            "avg_column_accuracy": self.avg_column_accuracy,
            "errors": self.errors,
            "missing_keys": self.missing_keys,
            "extra_keys": self.extra_keys,
            "mismatches": self.mismatches,
            "relaxed_matches": self.relaxed_matches[:25],
            "compared_rows": self.compared_rows,
        }


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _format_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(normalize_cell(row[column]) for column in KEY_COLUMNS)


def _parse_csv(
    text: str, *, label: str, columns: list[str]
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    errors: list[str] = []
    try:
        reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    except csv.Error as exc:
        return [], [], [f"{label}: CSV parse error: {exc}"]

    if fieldnames != columns:
        errors.append(f"{label}: columns must exactly match expected order; got {fieldnames!r}")

    normalized_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(rows, start=2):
        if not any(normalize_cell(value) for value in row.values()):
            continue
        if None in row:
            errors.append(f"{label}: row {row_index} has extra unheaded values")
            continue
        normalized_rows.append({column: normalize_cell(row.get(column, "")) for column in columns})

    if not normalized_rows:
        errors.append(f"{label}: CSV has no data rows")

    return fieldnames, normalized_rows, errors


def _index_rows(
    rows: list[dict[str, str]], *, label: str, spec: dict[str, object]
) -> tuple[dict[tuple[str, str, str], dict[str, str]], list[str]]:
    errors: list[str] = []
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    allowed_datasets = tuple(spec["allowed_datasets"])
    flag_column = str(spec["flag_column"])
    allowed_flags = set(spec["allowed_flags"])
    primary, suppqual = allowed_datasets

    for row_index, row in enumerate(rows, start=2):
        dataset = row["sdtm_dataset"]
        flag_value = row[flag_column]
        if dataset not in allowed_datasets:
            errors.append(
                f"{label}: row {row_index} has sdtm_dataset={dataset!r}; "
                f"expected one of {allowed_datasets!r}"
            )
        if flag_value not in allowed_flags:
            errors.append(
                f"{label}: row {row_index} has {flag_column}={flag_value!r}; "
                f"expected one of {tuple(sorted(allowed_flags))!r}"
            )
        if flag_column == "goes_to_suppqual":
            if dataset == primary and flag_value != "NO":
                errors.append(
                    f"{label}: row {row_index} maps primary dataset {primary} "
                    f"with goes_to_suppqual={flag_value!r}"
                )
            if dataset == suppqual and flag_value != "YES":
                errors.append(
                    f"{label}: row {row_index} maps supplemental dataset {suppqual} "
                    f"with goes_to_suppqual={flag_value!r}"
                )

        key = _format_key(row)
        if not all(key):
            errors.append(f"{label}: row {row_index} has an empty composite-key cell: {key!r}")
            continue
        if key in indexed:
            errors.append(f"{label}: duplicate composite key {key!r}")
            continue
        indexed[key] = row

    return indexed, errors


# ---------------------------------------------------------------------------
# Relaxed matching: match by (sdtm_dataset, sdtm_variable), with fuzzy
# fallback for supplemental rows where the agent may have used QVAL instead
# of the QNAM as sdtm_variable.
# ---------------------------------------------------------------------------


def _relaxed_key(row: dict[str, str]) -> tuple[str, str]:
    return (normalize_cell(row["sdtm_dataset"]), normalize_cell(row["sdtm_variable"]))


def _relaxed_match(
    agent_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    spec: dict[str, object],
) -> list[tuple[dict[str, str], dict[str, str] | None]]:
    """Match each reference row to an agent row using relaxed criteria.

    Primary: exact (sdtm_dataset, sdtm_variable) match.
    Fallback for SUPP* rows: reference sdtm_variable found as substring in
    agent's crf_item_or_placeholder within the same sdtm_dataset.

    Tracks used agent rows by list index so that supplemental rows sharing the
    same (dataset, QVAL) relaxed key can each be claimed independently.
    """
    agent_by_relaxed: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(agent_rows):
        key = _relaxed_key(row)
        agent_by_relaxed.setdefault(key, []).append(idx)

    _, supp_dataset = spec["allowed_datasets"]
    agent_supp_indices = [
        i for i, r in enumerate(agent_rows) if normalize_cell(r["sdtm_dataset"]) == supp_dataset
    ]

    matches: list[tuple[dict[str, str], dict[str, str] | None]] = []
    used_agent_indices: set[int] = set()

    for ref_row in reference_rows:
        ref_relaxed = _relaxed_key(ref_row)

        candidates = agent_by_relaxed.get(ref_relaxed, [])
        matched = False
        for idx in candidates:
            if idx not in used_agent_indices:
                matches.append((ref_row, agent_rows[idx]))
                used_agent_indices.add(idx)
                matched = True
                break

        if matched:
            continue

        ref_dataset = normalize_cell(ref_row["sdtm_dataset"])
        ref_variable = normalize_cell(ref_row["sdtm_variable"])
        if ref_dataset == supp_dataset and ref_variable:
            found = False
            for idx in agent_supp_indices:
                if idx in used_agent_indices:
                    continue
                agent_placeholder = normalize_cell(agent_rows[idx]["crf_item_or_placeholder"])
                if ref_variable in agent_placeholder:
                    matches.append((ref_row, agent_rows[idx]))
                    used_agent_indices.add(idx)
                    found = True
                    break
            if not found:
                matches.append((ref_row, None))
        else:
            matches.append((ref_row, None))

    return matches


def _compare_columns(
    ref_row: dict[str, str],
    agent_row: dict[str, str],
    columns: list[str],
) -> tuple[int, int, list[str]]:
    correct = 0
    mismatched: list[str] = []
    for col in columns:
        if normalize_cell(ref_row.get(col, "")) == normalize_cell(agent_row.get(col, "")):
            correct += 1
        else:
            mismatched.append(col)
    return correct, len(columns), mismatched


def score_mapping_csv(agent_csv: str, reference_csv: str, *, variant: str) -> ScoreResult:
    """Return a fine-grained score for a submitted mapping CSV.

    Scoring has two tiers:
    1. **Strict** — original binary logic on the full composite key
       (crf_item_or_placeholder, sdtm_dataset, sdtm_variable).  If every
       reference row matches perfectly, strict_score = score = 1.0.
    2. **Relaxed** — rows are matched by (sdtm_dataset, sdtm_variable) with
       a fuzzy fallback for supplemental rows.  For each matched pair every
       column is compared.  The final score is row_coverage *
       avg_column_accuracy, giving partial credit for identifying the right
       variable set even when formatting differs.
    """

    if variant not in VARIANT_SPECS:
        return ScoreResult(score=0.0, errors=[f"unknown variant {variant!r}"])

    spec = VARIANT_SPECS[variant]
    columns = list(spec["columns"])

    # ---- parse ----
    _, agent_rows, agent_errors = _parse_csv(agent_csv, label="agent", columns=columns)
    _, reference_rows, reference_errors = _parse_csv(reference_csv, label="reference", columns=columns)
    errors = agent_errors + reference_errors
    if errors:
        return ScoreResult(score=0.0, errors=errors)

    # ---- strict index ----
    agent_index, agent_index_errors = _index_rows(agent_rows, label="agent", spec=spec)
    reference_index, reference_index_errors = _index_rows(reference_rows, label="reference", spec=spec)
    errors.extend(agent_index_errors)
    errors.extend(reference_index_errors)
    if errors:
        return ScoreResult(score=0.0, errors=errors)

    # ---- strict scoring (original binary logic) ----
    agent_keys = set(agent_index)
    reference_keys = set(reference_index)
    missing_keys = sorted(reference_keys - agent_keys)
    extra_keys = sorted(agent_keys - reference_keys)

    strict_mismatches: list[dict[str, str]] = []
    for key in sorted(reference_keys & agent_keys):
        expected = reference_index[key]
        observed = agent_index[key]
        for column in columns:
            if observed[column] != expected[column]:
                strict_mismatches.append(
                    {
                        "key": " | ".join(key),
                        "column": column,
                        "expected": expected[column],
                        "observed": observed[column],
                    }
                )
                break

    strict_passed = not missing_keys and not extra_keys and not strict_mismatches
    strict_score = 1.0 if strict_passed else 0.0

    if strict_passed:
        return ScoreResult(
            score=1.0,
            strict_score=1.0,
            row_coverage=1.0,
            avg_column_accuracy=1.0,
            compared_rows=len(reference_index),
        )

    # ---- relaxed scoring ----
    matches = _relaxed_match(agent_rows, reference_rows, spec)

    num_matched = sum(1 for _, a in matches if a is not None)
    row_coverage = num_matched / len(reference_rows) if reference_rows else 0.0

    row_scores: list[float] = []
    relaxed_match_details: list[dict] = []
    for ref_row, agent_row in matches:
        if agent_row is not None:
            correct, total, mismatched_cols = _compare_columns(ref_row, agent_row, columns)
            row_score = correct / total if total > 0 else 0.0
            row_scores.append(row_score)
            relaxed_match_details.append(
                {
                    "ref_key": [ref_row.get(c, "") for c in KEY_COLUMNS],
                    "agent_key": [agent_row.get(c, "") for c in KEY_COLUMNS],
                    "correct_columns": correct,
                    "total_columns": total,
                    "mismatched_columns": mismatched_cols,
                }
            )

    avg_column_accuracy = sum(row_scores) / len(row_scores) if row_scores else 0.0
    score = row_coverage * avg_column_accuracy

    return ScoreResult(
        score=round(score, 4),
        strict_score=strict_score,
        row_coverage=round(row_coverage, 4),
        avg_column_accuracy=round(avg_column_accuracy, 4),
        errors=errors,
        missing_keys=[list(k) for k in missing_keys],
        extra_keys=[list(k) for k in extra_keys],
        mismatches=strict_mismatches[:25],
        relaxed_matches=relaxed_match_details,
        compared_rows=len(reference_index),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_SPECS))
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = score_mapping_csv(
        args.agent.read_text(encoding="utf-8-sig"),
        args.reference.read_text(encoding="utf-8-sig"),
        variant=args.variant,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    raise SystemExit(0 if result.strict_score == 1.0 else 1)


if __name__ == "__main__":
    main()
