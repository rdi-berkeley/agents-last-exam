"""Scoring helpers for ptbxl_diagnostic_superclass_prediction."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


CLASS_ORDER = ["NORM", "CD", "MI", "HYP", "STTC"]
EXPECTED_COLUMNS = ["ecg_id", *CLASS_ORDER]


@dataclass(frozen=True)
class ScoreResult:
    score: float
    macro_auc: float
    passed: bool
    reason: str
    per_class_auc: dict[str, float | None]
    row_count: int
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_csv(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    return reader.fieldnames or [], list(reader)


def _average_ranks(scores: list[float]) -> list[float]:
    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _roc_auc_score_binary(y_true: list[int], y_score: list[float]) -> float | None:
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(y_score)
    rank_sum_pos = sum(rank for rank, y in zip(ranks, y_true) if y == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def score_prediction_tables(
    *,
    agent_csv: str,
    template_csv: str,
    reference_csv: str,
    threshold: float,
) -> ScoreResult:
    agent_fieldnames, agent_rows = _parse_csv(agent_csv)
    template_fieldnames, template_rows = _parse_csv(template_csv)
    reference_fieldnames, reference_rows = _parse_csv(reference_csv)

    if template_fieldnames != EXPECTED_COLUMNS:
        return ScoreResult(0.0, 0.0, False, "template_schema_mismatch", {}, len(agent_rows), threshold)
    if reference_fieldnames != EXPECTED_COLUMNS:
        return ScoreResult(0.0, 0.0, False, "reference_schema_mismatch", {}, len(agent_rows), threshold)
    if agent_fieldnames != EXPECTED_COLUMNS:
        return ScoreResult(0.0, 0.0, False, "agent_schema_mismatch", {}, len(agent_rows), threshold)
    if len(agent_rows) != len(template_rows):
        return ScoreResult(0.0, 0.0, False, "row_count_mismatch", {}, len(agent_rows), threshold)

    template_ids = [row["ecg_id"] for row in template_rows]
    agent_ids = [row["ecg_id"] for row in agent_rows]
    if len(set(agent_ids)) != len(agent_ids):
        return ScoreResult(0.0, 0.0, False, "duplicate_agent_ids", {}, len(agent_rows), threshold)
    if agent_ids != template_ids:
        return ScoreResult(0.0, 0.0, False, "ecg_id_order_mismatch", {}, len(agent_rows), threshold)

    for row in agent_rows:
        for class_name in CLASS_ORDER:
            try:
                value = float(row[class_name])
            except Exception:
                return ScoreResult(0.0, 0.0, False, f"non_numeric_value:{class_name}", {}, len(agent_rows), threshold)
            if not math.isfinite(value):
                return ScoreResult(0.0, 0.0, False, f"non_finite_value:{class_name}", {}, len(agent_rows), threshold)
            if value < 0.0 or value > 1.0:
                return ScoreResult(0.0, 0.0, False, f"out_of_range_value:{class_name}", {}, len(agent_rows), threshold)

    reference_by_id = {row["ecg_id"]: row for row in reference_rows}
    per_class_auc: dict[str, float | None] = {}
    valid_aucs: list[float] = []
    for class_name in CLASS_ORDER:
        y_true = [int(reference_by_id[row["ecg_id"]][class_name]) for row in agent_rows]
        y_score = [float(row[class_name]) for row in agent_rows]
        auc = _roc_auc_score_binary(y_true, y_score)
        per_class_auc[class_name] = auc
        if auc is not None:
            valid_aucs.append(auc)

    if not valid_aucs:
        return ScoreResult(0.0, 0.0, False, "auc_undefined", per_class_auc, len(agent_rows), threshold)

    macro_auc = sum(valid_aucs) / len(valid_aucs)
    passed = macro_auc >= threshold
    score = 1.0 if passed else max(0.0, macro_auc / threshold)
    return ScoreResult(
        score=score,
        macro_auc=macro_auc,
        passed=passed,
        reason="ok" if passed else "below_threshold",
        per_class_auc=per_class_auc,
        row_count=len(agent_rows),
        threshold=threshold,
    )


def evaluate_files(*, output_file: Path, template_file: Path, reference_file: Path, threshold: float) -> dict:
    if not output_file.exists():
        return ScoreResult(0.0, 0.0, False, "missing_output", {}, 0, threshold).to_dict()
    result = score_prediction_tables(
        agent_csv=output_file.read_text(encoding="utf-8", errors="replace"),
        template_csv=template_file.read_text(encoding="utf-8"),
        reference_csv=reference_file.read_text(encoding="utf-8"),
        threshold=threshold,
    )
    return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--template-file", required=True, type=Path)
    parser.add_argument("--reference-file", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    args = parser.parse_args()

    result = evaluate_files(
        output_file=args.output_file,
        template_file=args.template_file,
        reference_file=args.reference_file,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
