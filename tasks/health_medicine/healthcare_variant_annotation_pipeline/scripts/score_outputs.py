"""Scoring helpers for healthcare_variant_annotation_pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any

EXPECTED_FIELDS = [
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene",
    "consequence",
    "max_population_af",
    "clinvar_significance",
    "is_reportable",
]
EXPECTED_VARIANT_COUNT = 50
PASS_THRESHOLD = 0.85

PIPELINE_POINTS = 2.0
RUN_LOG_POINTS = 3.0
STRUCTURE_POINTS = 10.0
GENE_POINTS = 25.0
CONSEQUENCE_POINTS = 25.0
AF_POINTS = 10.0
CLINVAR_POINTS = 5.0
REPORTABLE_POINTS = 20.0


@dataclass
class ScoreResult:
    score: float
    total_points: float
    max_points: float
    passed: bool
    valid: bool
    reason: str
    pipeline_points: float
    run_log_points: float
    structure_points: float
    gene_points: float
    consequence_points: float
    af_points: float
    clinvar_points: float
    reportable_points: float
    gene_matches: int
    consequence_matches: int
    af_matches: int
    clinvar_matches: int
    reportable_flag_matches: int
    expected_reportable_count: int
    reported_reportable_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_tsv_rows(raw_text: str, *, label: str) -> list[dict[str, str]]:
    reader = csv.DictReader(StringIO(raw_text), delimiter="\t")
    fieldnames = reader.fieldnames or []
    if fieldnames != EXPECTED_FIELDS:
        raise ValueError(
            f"{label} must have exact header {EXPECTED_FIELDS}, got {fieldnames}"
        )
    cleaned_rows: list[dict[str, str]] = []
    for row in reader:
        if None in row:
            raise ValueError(f"{label} contains ragged rows or extra tab-delimited fields")
        cleaned_rows.append({key: (value or "").strip() for key, value in row.items()})
    return cleaned_rows


def _variant_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (row["variant_id"], row["chrom"], row["pos"], row["ref"], row["alt"])


def _reportable_ids(rows: list[dict[str, str]]) -> set[str]:
    return {row["variant_id"] for row in rows}


_MISSING_SENTINELS = {"", ".", "na", "none", "not_reported"}


def _is_missing(value: str) -> bool:
    return value.strip().lower() in _MISSING_SENTINELS


def _parse_af(value: str) -> float | None:
    value = value.strip()
    if _is_missing(value):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite AF value: {value}")
    return parsed


def _af_matches(left: str, right: str) -> bool:
    left_af = _parse_af(left)
    right_af = _parse_af(right)
    if left_af is None or right_af is None:
        return left_af is None and right_af is None
    return abs(left_af - right_af) <= 1e-6


def _clinvar_matches(left: str, right: str) -> bool:
    left_v = left.strip()
    right_v = right.strip()
    if _is_missing(left_v) and _is_missing(right_v):
        return True
    return left_v == right_v


def _f1_score(predicted: set[str], reference: set[str]) -> float:
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0
    tp = len(predicted & reference)
    precision = tp / len(predicted)
    recall = tp / len(reference)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _threshold_points(
    matches: int,
    *,
    full_threshold: int,
    partial_floor: int,
    full_points: float,
    partial_points: float,
) -> float:
    if matches >= full_threshold:
        return full_points
    if matches < partial_floor:
        return 0.0
    span = full_threshold - partial_floor
    if span <= 0:
        return 0.0
    fraction = (matches - partial_floor) / span
    return fraction * partial_points


def _pipeline_points(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    required_tokens = [
        "annotated_variants.tsv",
        "reportable_variants.tsv",
        "run_log.json",
        "annotation_snapshots",
        "variants_to_annotate.tsv",
    ]
    hits = sum(1 for token in required_tokens if token in stripped)
    if hits == len(required_tokens):
        return PIPELINE_POINTS
    return PIPELINE_POINTS * (hits / len(required_tokens))


def _deep_get(obj: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


def _run_log_points(
    raw_text: str,
    *,
    annotated_rows: list[dict[str, str]],
    reportable_rows: list[dict[str, str]],
) -> float:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(payload, dict):
        return 0.0
    has_status = bool(
        payload.get("status")
        or payload.get("pipeline")
        or payload.get("run_start_utc")
        or payload.get("run_end_utc")
    )
    points = 1.0 if has_status else 0.0

    def _first_defined(*candidates: Any) -> Any:
        for c in candidates:
            if c is not None:
                return c
        return None

    ann_count = _first_defined(
        payload.get("annotated_variants_written"),
        _deep_get(payload, "counts", "annotated_variants"),
        _deep_get(payload, "counts", "total_input_variants"),
    )
    if ann_count == len(annotated_rows):
        points += 1.0
    rep_count = _first_defined(
        payload.get("reportable_variants_written"),
        _deep_get(payload, "counts", "reportable_variants"),
    )
    if rep_count == len(reportable_rows):
        points += 1.0
    return min(points, RUN_LOG_POINTS)


def score_submission_texts(
    *,
    annotated_tsv_text: str,
    reportable_tsv_text: str,
    pipeline_py_text: str,
    run_log_text: str,
    reference_annotated_tsv_text: str,
    reference_reportable_tsv_text: str,
) -> ScoreResult:
    try:
        reference_rows = _load_tsv_rows(reference_annotated_tsv_text, label="reference annotated_variants.tsv")
        reference_reportable_rows = _load_tsv_rows(
            reference_reportable_tsv_text,
            label="reference reportable_variants.tsv",
        )
        submitted_rows = _load_tsv_rows(annotated_tsv_text, label="submitted annotated_variants.tsv")
        submitted_reportable_rows = _load_tsv_rows(
            reportable_tsv_text,
            label="submitted reportable_variants.tsv",
        )
    except Exception as exc:
        return ScoreResult(
            score=0.0,
            total_points=0.0,
            max_points=100.0,
            passed=False,
            valid=False,
            reason=str(exc),
            pipeline_points=0.0,
            run_log_points=0.0,
            structure_points=0.0,
            gene_points=0.0,
            consequence_points=0.0,
            af_points=0.0,
            clinvar_points=0.0,
            reportable_points=0.0,
            gene_matches=0,
            consequence_matches=0,
            af_matches=0,
            clinvar_matches=0,
            reportable_flag_matches=0,
            expected_reportable_count=0,
            reported_reportable_count=0,
        )

    reference_keys = [_variant_key(row) for row in reference_rows]
    submitted_keys = [_variant_key(row) for row in submitted_rows]
    canonical_alignment = submitted_keys == reference_keys

    structure_points = 0.0
    if len(submitted_rows) == EXPECTED_VARIANT_COUNT:
        structure_points += 5.0
    if canonical_alignment:
        structure_points += 5.0

    reference_by_id = {row["variant_id"]: row for row in reference_rows}
    submitted_by_id = {row["variant_id"]: row for row in submitted_rows}
    gene_matches = 0
    consequence_matches = 0
    af_matches = 0
    clinvar_matches = 0
    for row in submitted_rows:
        reference = reference_by_id.get(row["variant_id"])
        if reference is None:
            continue
        if row["gene"] == reference["gene"]:
            gene_matches += 1
        if row["consequence"] == reference["consequence"]:
            consequence_matches += 1
        if _af_matches(row["max_population_af"], reference["max_population_af"]):
            af_matches += 1
        if _clinvar_matches(row["clinvar_significance"], reference["clinvar_significance"]):
            clinvar_matches += 1

    reportable_flag_matches = 0
    if canonical_alignment:
        for reference in reference_rows:
            variant_id = reference["variant_id"]
            if submitted_by_id[variant_id]["is_reportable"] == reference["is_reportable"]:
                reportable_flag_matches += 1

    gene_points = _threshold_points(
        gene_matches,
        full_threshold=45,
        partial_floor=35,
        full_points=GENE_POINTS,
        partial_points=8.0,
    )
    consequence_points = _threshold_points(
        consequence_matches,
        full_threshold=43,
        partial_floor=30,
        full_points=CONSEQUENCE_POINTS,
        partial_points=8.0,
    )
    af_points = (af_matches / EXPECTED_VARIANT_COUNT) * AF_POINTS
    clinvar_points = (clinvar_matches / EXPECTED_VARIANT_COUNT) * CLINVAR_POINTS

    expected_reportable = _reportable_ids(reference_reportable_rows)
    submitted_reportable = _reportable_ids(submitted_reportable_rows)
    reportable_points = 0.0

    def _normalize_row(row: dict[str, str]) -> dict[str, str]:
        out = dict(row)
        if _is_missing(out.get("max_population_af", "")):
            out["max_population_af"] = "NA"
        if _is_missing(out.get("clinvar_significance", "")):
            out["clinvar_significance"] = "not_reported"
        return out

    if (
        canonical_alignment
        and
        [_normalize_row(r) for r in submitted_reportable_rows]
        == [_normalize_row(r) for r in reference_reportable_rows]
        and all(row["is_reportable"] == "yes" for row in submitted_reportable_rows)
        and submitted_reportable == expected_reportable
        and reportable_flag_matches == EXPECTED_VARIANT_COUNT
    ):
        reportable_points = REPORTABLE_POINTS

    pipeline_points = _pipeline_points(pipeline_py_text)
    run_log_points = _run_log_points(
        run_log_text,
        annotated_rows=submitted_rows,
        reportable_rows=submitted_reportable_rows,
    )

    total_points = (
        pipeline_points
        + run_log_points
        + structure_points
        + gene_points
        + consequence_points
        + af_points
        + clinvar_points
        + reportable_points
    )
    normalized = total_points / 100.0
    return ScoreResult(
        score=normalized,
        total_points=total_points,
        max_points=100.0,
        passed=normalized >= PASS_THRESHOLD,
        valid=True,
        reason="scored successfully",
        pipeline_points=pipeline_points,
        run_log_points=run_log_points,
        structure_points=structure_points,
        gene_points=gene_points,
        consequence_points=consequence_points,
        af_points=af_points,
        clinvar_points=clinvar_points,
        reportable_points=reportable_points,
        gene_matches=gene_matches,
        consequence_matches=consequence_matches,
        af_matches=af_matches,
        clinvar_matches=clinvar_matches,
        reportable_flag_matches=reportable_flag_matches,
        expected_reportable_count=len(expected_reportable),
        reported_reportable_count=len(submitted_reportable),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir)
    reference_dir = Path(args.reference_dir)
    result = score_submission_texts(
        annotated_tsv_text=(submission_dir / "annotated_variants.tsv").read_text(encoding="utf-8"),
        reportable_tsv_text=(submission_dir / "reportable_variants.tsv").read_text(encoding="utf-8"),
        pipeline_py_text=(submission_dir / "pipeline.py").read_text(encoding="utf-8"),
        run_log_text=(submission_dir / "run_log.json").read_text(encoding="utf-8"),
        reference_annotated_tsv_text=(reference_dir / "annotated_variants.tsv").read_text(encoding="utf-8"),
        reference_reportable_tsv_text=(reference_dir / "reportable_variants.tsv").read_text(encoding="utf-8"),
    )
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
