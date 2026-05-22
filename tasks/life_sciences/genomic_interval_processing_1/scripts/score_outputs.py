"""Deterministic scorer for the genomic interval union peak task."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_FILES = ["union_peaks.bed", "commands.sh", "summary.json"]
INPUT_BED_FILES = ["ENCFF483KVM.bed", "ENCFF511NNV.bed", "ENCFF758CQW.bed"]


@dataclass
class ScoreReport:
    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reasons": self.reasons,
            "details": self.details,
        }


def _decode(payload: bytes, label: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8 text") from exc


def _normalize_bed_text(payload: bytes, label: str) -> str:
    text = _decode(payload, label).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def _parse_bed3(payload: bytes, label: str) -> tuple[list[tuple[str, int, int]], list[str]]:
    rows: list[tuple[str, int, int]] = []
    issues: list[str] = []
    text = _normalize_bed_text(payload, label)
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            issues.append(f"{label}:{line_number} must have exactly 3 tab-delimited columns")
            continue
        chrom, start_raw, end_raw = parts
        try:
            start = int(start_raw)
            end = int(end_raw)
        except ValueError:
            issues.append(f"{label}:{line_number} has non-integer coordinates")
            continue
        if not chrom:
            issues.append(f"{label}:{line_number} has blank chromosome")
        if start < 0 or end < 0:
            issues.append(f"{label}:{line_number} has negative coordinates")
        if start >= end:
            issues.append(f"{label}:{line_number} has start >= end")
        rows.append((chrom, start, end))
    return rows, issues


def _load_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(_decode(payload, label))
    except Exception as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _bed_is_sorted(rows: list[tuple[str, int, int]]) -> bool:
    return rows == sorted(rows, key=lambda row: (row[0], row[1], row[2]))


def _bed_is_non_overlapping(rows: list[tuple[str, int, int]]) -> bool:
    previous: tuple[str, int, int] | None = None
    for row in rows:
        if previous is not None and row[0] == previous[0] and row[1] < previous[2]:
            return False
        previous = row
    return True


def _summary_score(
    summary: Any,
    *,
    input_counts: dict[str, int],
    output_rows: int,
    reasons: list[str],
    details: dict[str, Any],
) -> float:
    if not isinstance(summary, dict):
        reasons.append("summary.json must contain a JSON object")
        return 0.0

    expected_total = sum(input_counts.values())
    observed_input_counts = summary.get("input_interval_counts")
    observed_total = summary.get("total_input_intervals")
    observed_output = summary.get("output_intervals")
    observed_output_file = summary.get("output_file")

    details["summary_output_intervals"] = observed_output
    details["summary_total_input_intervals"] = observed_total

    score = 0.0
    if observed_input_counts == input_counts and observed_total == expected_total:
        score += 0.04
    else:
        reasons.append("summary.json input counts do not match staged inputs")
    if observed_output == output_rows and observed_output_file == "union_peaks.bed":
        score += 0.06
    else:
        reasons.append("summary.json output count or output filename is inconsistent")
    return score


def score_submission(
    outputs: dict[str, bytes],
    *,
    reference_bed: bytes,
    input_counts: dict[str, int],
) -> ScoreReport:
    reasons: list[str] = []
    details: dict[str, Any] = {}
    missing = [name for name in REQUIRED_FILES if name not in outputs or not outputs[name]]
    if missing:
        return ScoreReport(0.0, False, [f"missing or empty required files: {missing}"], details)

    score = 0.0
    rows, bed_issues = _parse_bed3(outputs["union_peaks.bed"], "union_peaks.bed")
    details["output_intervals"] = len(rows)
    if bed_issues:
        reasons.extend(bed_issues[:10])
    else:
        score += 0.04
        is_sorted = _bed_is_sorted(rows)
        if is_sorted:
            score += 0.03
        else:
            reasons.append("union_peaks.bed is not sorted by chromosome, start, and end")
        if is_sorted and _bed_is_non_overlapping(rows):
            score += 0.03
        elif not is_sorted:
            reasons.append("union_peaks.bed must be sorted before non-overlap can be credited")
        else:
            reasons.append("union_peaks.bed contains overlapping intervals")

    observed_bed = _normalize_bed_text(outputs["union_peaks.bed"], "union_peaks.bed")
    expected_bed = _normalize_bed_text(reference_bed, "reference union_ref.bed")
    if observed_bed == expected_bed:
        score += 0.75
        details["exact_reference_match"] = True
    else:
        details["exact_reference_match"] = False
        reasons.append("union_peaks.bed does not exactly match the expected union peak set")

    commands = _decode(outputs["commands.sh"], "commands.sh").strip()
    details["commands_chars"] = len(commands)
    if len(commands) >= 40 and any(token in commands.lower() for token in ["bedtools", "sort", "awk", "python"]):
        score += 0.05
    else:
        reasons.append("commands.sh is too short or does not record a plausible interval workflow")

    try:
        summary = _load_json(outputs["summary.json"], "summary.json")
        score += _summary_score(
            summary,
            input_counts=input_counts,
            output_rows=len(rows),
            reasons=reasons,
            details=details,
        )
    except Exception as exc:
        reasons.append(str(exc))

    final_score = min(1.0, round(score, 6))
    return ScoreReport(final_score, final_score >= 0.999, reasons, details)


def _load_outputs(output_dir: Path) -> dict[str, bytes]:
    return {name: (output_dir / name).read_bytes() for name in REQUIRED_FILES if (output_dir / name).exists()}


def _count_input_rows(input_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for filename in INPUT_BED_FILES:
        path = input_dir / filename
        counts[filename] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a genomic interval union output directory.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()

    report = score_submission(
        _load_outputs(args.output_dir),
        reference_bed=(args.reference_dir / "union_ref.bed").read_bytes(),
        input_counts=_count_input_rows(args.input_dir),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
