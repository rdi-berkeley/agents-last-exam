"""Local scorer for health_medicine/sa_aki_phenotyping."""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

EXPECTED_HEADER = ["subject_id"]


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    precision: float
    recall: float
    f1: float
    exact_match: bool
    true_positives: int
    false_positives: int
    false_negatives: int
    predicted_count: int
    reference_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _empty_result(reason: str, hard_gate: str | None) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        passed=False,
        reason=reason,
        hard_gate=hard_gate,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        exact_match=False,
        true_positives=0,
        false_positives=0,
        false_negatives=0,
        predicted_count=0,
        reference_count=0,
    )


def _read_subject_id_set(csv_text: str, *, strict_single_column: bool) -> tuple[set[int], str | None]:
    reader = csv.reader(io.StringIO(csv_text))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return set(), "empty_csv"

    header = [cell.strip() for cell in rows[0]]
    if strict_single_column:
        if header != EXPECTED_HEADER:
            return set(), "invalid_header"
    elif not header or header[0] != "subject_id":
        return set(), "invalid_header"

    subject_ids: set[int] = set()
    for row in rows[1:]:
        if not row:
            continue
        if strict_single_column and len(row) != 1:
            return set(), "unexpected_extra_columns"
        raw_value = row[0].strip()
        if not raw_value:
            continue
        try:
            subject_ids.add(int(raw_value))
        except ValueError:
            return set(), "subject_id_not_integer"
    return subject_ids, None


def score_csv_texts(*, candidate_csv_text: str, reference_csv_text: str) -> ScoreResult:
    predicted, candidate_error = _read_subject_id_set(candidate_csv_text, strict_single_column=True)
    if candidate_error:
        return _empty_result("hard_gate_failure", candidate_error)

    reference, reference_error = _read_subject_id_set(reference_csv_text, strict_single_column=False)
    if reference_error:
        return _empty_result("reference_error", reference_error)

    tp = len(predicted & reference)
    fp = len(predicted - reference)
    fn = len(reference - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact_match = predicted == reference
    score = 1.0 if exact_match else 0.0

    return ScoreResult(
        score=score,
        passed=exact_match,
        reason="ok" if exact_match else "set_mismatch",
        hard_gate=None,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=exact_match,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        predicted_count=len(predicted),
        reference_count=len(reference),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Candidate sa_aki_patients.csv path")
    parser.add_argument("--reference", required=True, help="Reference gold_labels.csv path")
    args = parser.parse_args()

    result = score_csv_texts(
        candidate_csv_text=Path(args.candidate).read_text(encoding="utf-8"),
        reference_csv_text=Path(args.reference).read_text(encoding="utf-8"),
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
