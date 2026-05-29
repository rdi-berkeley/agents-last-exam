"""Local scorer for healthcare_icu_calculator_audit."""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TOLERANCES: dict[str, tuple[str, float]] = {
    "apache2_score": ("int", 1.0),
    "sofa_total": ("int", 1.0),
    "cha2ds2vasc": ("int", 0.0),
    "wells_pe": ("float", 0.0),
    "meld": ("int", 1.0),
    "child_pugh_class": ("str", 0.0),
    "egfr": ("pct", 0.05),
}
EXPECTED_SCORE_COLUMNS = [
    "patient_id",
    "apache2_score",
    "sofa_total",
    "cha2ds2vasc",
    "cha2ds2vasc_risk",
    "wells_pe",
    "wells_risk",
    "meld",
    "child_pugh_score",
    "child_pugh_class",
    "egfr",
    "ckd_stage",
]
PASS_THRESHOLD = 0.90
REQUIRED_AUXILIARY_FILES = ("summary.json", "quality_report.csv")


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
    return ScoreResult(
        score=0.0,
        passed=False,
        reason=reason,
        hard_gate=reason,
        details=details or {},
    )


def _read_csv_rows(payload: str | bytes, *, label: str) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except csv.Error as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc
    if not rows:
        raise ValueError(f"{label} has no data rows")
    return rows


def _normalize_cell(value: str | None) -> str:
    return "" if value is None else str(value).strip()


def _ids_to_rows(rows: list[dict[str, str]], *, label: str) -> dict[str, dict[str, str]]:
    if "patient_id" not in rows[0]:
        raise ValueError(f"{label} missing patient_id column")
    by_id: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        patient_id = _normalize_cell(row.get("patient_id"))
        if not patient_id:
            raise ValueError(f"{label} contains empty patient_id")
        if patient_id in by_id:
            duplicates.append(patient_id)
        by_id[patient_id] = row
    if duplicates:
        raise ValueError(f"{label} contains duplicate patient_id values: {duplicates[:10]}")
    return by_id


def _matches(predicted: str, gold: str, dtype: str, tolerance: float) -> bool:
    p = _normalize_cell(predicted)
    g = _normalize_cell(gold)

    if g in ("", "incomplete"):
        return p in ("", "incomplete")
    if p in ("", "incomplete"):
        return False

    if dtype == "int":
        return abs(int(float(p)) - int(float(g))) <= tolerance
    if dtype == "float":
        return abs(float(p) - float(g)) <= tolerance
    if dtype == "pct":
        return abs(float(p) - float(g)) / max(abs(float(g)), 0.01) <= tolerance
    if dtype == "str":
        return p.upper() == g.upper()
    return p == g


def score_scores_csv(
    *,
    predicted_scores_csv: str | bytes,
    reference_scores_csv: str | bytes,
    auxiliary_files_present: dict[str, bool] | None = None,
) -> ScoreResult:
    try:
        predicted_rows = _read_csv_rows(predicted_scores_csv, label="scores.csv")
        reference_rows = _read_csv_rows(reference_scores_csv, label="reference/scores.csv")
        predicted_by_id = _ids_to_rows(predicted_rows, label="scores.csv")
        reference_by_id = _ids_to_rows(reference_rows, label="reference/scores.csv")
    except ValueError as exc:
        return _hard_fail(str(exc))

    missing_columns = [column for column in EXPECTED_SCORE_COLUMNS if column not in predicted_rows[0]]
    if missing_columns:
        return _hard_fail("missing_required_score_columns", {"missing_columns": missing_columns})

    if auxiliary_files_present is not None:
        missing_auxiliary = [
            name for name in REQUIRED_AUXILIARY_FILES if not auxiliary_files_present.get(name, False)
        ]
        if missing_auxiliary:
            return _hard_fail("missing_required_auxiliary_files", {"missing_files": missing_auxiliary})

    missing_patients = sorted(set(reference_by_id) - set(predicted_by_id))
    if missing_patients:
        return _hard_fail("missing_patient_rows", {"missing_patient_ids": missing_patients[:20]})

    unknown_patients = sorted(set(predicted_by_id) - set(reference_by_id))
    if unknown_patients:
        return _hard_fail("unknown_patient_rows", {"unknown_patient_ids": unknown_patients[:20]})

    total = 0
    matches = 0
    mismatches: list[dict[str, str]] = []
    parse_errors: list[dict[str, str]] = []

    for patient_id in sorted(reference_by_id):
        predicted = predicted_by_id[patient_id]
        reference = reference_by_id[patient_id]
        for column, (dtype, tolerance) in TOLERANCES.items():
            total += 1
            pred_value = _normalize_cell(predicted.get(column))
            gold_value = _normalize_cell(reference.get(column))
            try:
                ok = _matches(pred_value, gold_value, dtype, tolerance)
            except (TypeError, ValueError, ZeroDivisionError):
                parse_errors.append(
                    {"patient_id": patient_id, "column": column, "predicted": pred_value, "gold": gold_value}
                )
                continue
            if ok:
                matches += 1
            else:
                mismatches.append(
                    {"patient_id": patient_id, "column": column, "predicted": pred_value, "gold": gold_value}
                )

    if total == 0:
        return _hard_fail("no_scored_values")

    match_rate = matches / total
    passed = match_rate >= PASS_THRESHOLD
    return ScoreResult(
        score=round(match_rate, 6),
        passed=passed,
        reason="passed" if passed else "match_rate_below_threshold",
        hard_gate=None,
        details={
            "matches": matches,
            "total": total,
            "match_rate": round(match_rate, 6),
            "pass_threshold": PASS_THRESHOLD,
            "mismatches": mismatches[:20],
            "parse_errors": parse_errors[:20],
            "auxiliary_files_present": auxiliary_files_present or {},
        },
    )


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ICU calculator audit output.")
    parser.add_argument("--predicted-scores", required=True)
    parser.add_argument("--reference-scores", required=True)
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    auxiliary = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        auxiliary = {
            name: (output_dir / name).exists()
            for name in REQUIRED_AUXILIARY_FILES
        }
    result = score_scores_csv(
        predicted_scores_csv=_read_text(args.predicted_scores),
        reference_scores_csv=_read_text(args.reference_scores),
        auxiliary_files_present=auxiliary,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
