"""Scorer for the PK parameter filtering task."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_CSV_COLUMNS = [
    "Compound_ID",
    "CL_pred",
    "VD_pred",
    "T_half_pred",
    "Score",
    "PK_Risk",
    "Decision",
    "Rank",
]
NUMERIC_REL_TOLERANCE_COLUMNS = ("CL_pred", "VD_pred", "T_half_pred")
SCORE_ABS_TOLERANCE = 0.01
RELATIVE_TOLERANCE = 0.05


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fail(reason: str, hard_gate: str, **details: Any) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        passed=False,
        reason=reason,
        hard_gate=hard_gate,
        details=details,
    )


def _read_csv_rows(text: str, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None:
            raise ValueError("missing header")
        return list(reader.fieldnames), list(reader)
    except csv.Error as exc:
        raise ValueError(f"{label} is not parseable CSV: {exc}") from exc


def _index_by_compound(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        compound_id = (row.get("Compound_ID") or "").strip()
        if not compound_id:
            raise ValueError(f"{label} has a row with empty Compound_ID")
        if compound_id in indexed:
            duplicates.append(compound_id)
        indexed[compound_id] = row
    if duplicates:
        raise ValueError(f"{label} has duplicate Compound_ID values: {sorted(set(duplicates))}")
    return indexed


def _parse_float(value: str, field: str, compound_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{compound_id} field {field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{compound_id} field {field} is not finite: {value!r}")
    return parsed


def _parse_rank(value: str, compound_id: str) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{compound_id} Rank is not numeric: {value!r}") from exc
    if not parsed.is_integer():
        raise ValueError(f"{compound_id} Rank is not an integer: {value!r}")
    return int(parsed)


def _within_relative_tolerance(candidate: float, reference: float) -> bool:
    if reference == 0:
        return abs(candidate - reference) <= RELATIVE_TOLERANCE
    return abs(candidate - reference) <= abs(reference) * RELATIVE_TOLERANCE


def _normalize_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def score_pk_outputs(
    *,
    output_csv: str,
    reference_csv: str,
    output_representatives: str,
    reference_representatives: str,
    output_instability: str,
    reference_instability: str,
) -> ScoreResult:
    try:
        output_columns, output_rows = _read_csv_rows(output_csv, "agent output")
        reference_columns, reference_rows = _read_csv_rows(reference_csv, "reference")
    except ValueError as exc:
        return _fail(str(exc), "csv_parse_error")

    missing_columns = [col for col in REQUIRED_CSV_COLUMNS if col not in output_columns]
    if missing_columns:
        return _fail(
            "agent output CSV is missing required columns",
            "missing_columns",
            missing_columns=missing_columns,
            output_columns=output_columns,
        )

    missing_reference_columns = [
        col for col in REQUIRED_CSV_COLUMNS if col not in reference_columns
    ]
    if missing_reference_columns:
        return _fail(
            "reference CSV is missing required columns",
            "bad_reference_columns",
            missing_columns=missing_reference_columns,
        )

    try:
        output_by_id = _index_by_compound(output_rows, "agent output")
        reference_by_id = _index_by_compound(reference_rows, "reference")
    except ValueError as exc:
        return _fail(str(exc), "bad_compound_ids")

    output_ids = set(output_by_id)
    reference_ids = set(reference_by_id)
    if output_ids != reference_ids:
        return _fail(
            "agent output has missing or extra compounds",
            "compound_set_mismatch",
            missing=sorted(reference_ids - output_ids),
            extra=sorted(output_ids - reference_ids),
        )

    for compound_id in sorted(reference_ids):
        out_row = output_by_id[compound_id]
        ref_row = reference_by_id[compound_id]

        for field_name in NUMERIC_REL_TOLERANCE_COLUMNS:
            try:
                out_value = _parse_float(out_row.get(field_name, ""), field_name, compound_id)
                ref_value = _parse_float(ref_row.get(field_name, ""), field_name, compound_id)
            except ValueError as exc:
                return _fail(str(exc), "numeric_parse_error")
            if not _within_relative_tolerance(out_value, ref_value):
                return _fail(
                    f"{compound_id} {field_name} is outside +/-5% tolerance",
                    "numeric_tolerance_mismatch",
                    compound_id=compound_id,
                    field=field_name,
                    output=out_value,
                    reference=ref_value,
                )

        try:
            output_score = _parse_float(out_row.get("Score", ""), "Score", compound_id)
            reference_score = _parse_float(ref_row.get("Score", ""), "Score", compound_id)
        except ValueError as exc:
            return _fail(str(exc), "score_parse_error")
        if abs(output_score - reference_score) > SCORE_ABS_TOLERANCE:
            return _fail(
                f"{compound_id} Score is outside +/-0.01 tolerance",
                "score_tolerance_mismatch",
                compound_id=compound_id,
                output=output_score,
                reference=reference_score,
            )

        for field_name in ("PK_Risk", "Decision"):
            if (out_row.get(field_name) or "").strip() != (ref_row.get(field_name) or "").strip():
                return _fail(
                    f"{compound_id} {field_name} does not match reference",
                    "label_mismatch",
                    compound_id=compound_id,
                    field=field_name,
                    output=(out_row.get(field_name) or "").strip(),
                    reference=(ref_row.get(field_name) or "").strip(),
                )

        try:
            output_rank = _parse_rank(out_row.get("Rank", ""), compound_id)
            reference_rank = _parse_rank(ref_row.get("Rank", ""), compound_id)
        except ValueError as exc:
            return _fail(str(exc), "rank_parse_error")
        if output_rank != reference_rank:
            return _fail(
                f"{compound_id} Rank does not match reference",
                "rank_mismatch",
                compound_id=compound_id,
                output=output_rank,
                reference=reference_rank,
            )

    output_rep_lines = _normalize_lines(output_representatives)
    reference_rep_lines = _normalize_lines(reference_representatives)
    if output_rep_lines != reference_rep_lines:
        return _fail(
            "representative_compounds.txt does not match reference",
            "representatives_mismatch",
            output=output_rep_lines,
            reference=reference_rep_lines,
        )

    output_instability_lines = _normalize_lines(output_instability)
    reference_instability_lines = _normalize_lines(reference_instability)
    if output_instability_lines != reference_instability_lines:
        return _fail(
            "ranking_instability.txt does not match reference",
            "instability_mismatch",
            output=output_instability_lines,
            reference=reference_instability_lines,
        )

    return ScoreResult(
        score=1.0,
        passed=True,
        reason="passed",
        details={
            "compound_count": len(reference_ids),
            "representative_count": len(reference_rep_lines),
        },
    )


def _read_required_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def score_directories(output_dir: Path, reference_dir: Path) -> ScoreResult:
    required_names = (
        "pk_final_output_100.csv",
        "representative_compounds.txt",
        "ranking_instability.txt",
    )
    missing = [name for name in required_names if not (output_dir / name).exists()]
    if missing:
        return _fail("agent output is missing required files", "missing_files", missing=missing)

    try:
        return score_pk_outputs(
            output_csv=_read_required_file(output_dir / "pk_final_output_100.csv"),
            reference_csv=_read_required_file(reference_dir / "pk_final_output_100.csv"),
            output_representatives=_read_required_file(output_dir / "representative_compounds.txt"),
            reference_representatives=_read_required_file(
                reference_dir / "representative_compounds.txt"
            ),
            output_instability=_read_required_file(output_dir / "ranking_instability.txt"),
            reference_instability=_read_required_file(reference_dir / "ranking_instability.txt"),
        )
    except FileNotFoundError as exc:
        return _fail("reference data is missing required files", "missing_reference", path=str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score PK task output against the hidden reference."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    args = parser.parse_args()

    result = score_directories(args.output_dir, args.reference_dir)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
