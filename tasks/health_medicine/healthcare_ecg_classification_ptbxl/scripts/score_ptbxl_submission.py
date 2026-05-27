from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
PROB_COLS = ["norm_prob", "mi_prob", "sttc_prob", "cd_prob", "hyp_prob"]
PREDICTIONS_NAME = "ptbxl_fold10_predictions.csv"
METRICS_NAME = "metrics.json"
PASS_THRESHOLD = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    return parser.parse_args()


def load_gold(path: Path) -> dict[int, dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            try:
                ecg_id = int(row["ecg_id"])
                rows[ecg_id] = {label: int(row[label]) for label in CLASSES}
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid gold row in {path}: {row!r}") from exc
    if not rows:
        raise ValueError(f"gold labels are empty: {path}")
    return rows


def load_predictions(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = ["ecg_id", *PROB_COLS]
        if reader.fieldnames != expected:
            raise ValueError(f"{PREDICTIONS_NAME} header must be exactly {expected}")
        rows: dict[int, dict[str, float]] = {}
        for row_number, row in enumerate(reader, start=2):
            if row.get(None):
                raise ValueError(f"row {row_number}: unexpected extra CSV columns")
            try:
                ecg_id = int(row["ecg_id"])
            except ValueError as exc:
                raise ValueError(f"row {row_number}: ecg_id must be an integer") from exc
            if ecg_id in rows:
                raise ValueError(f"duplicate ecg_id in predictions: {ecg_id}")
            probs = {}
            for label, column in zip(CLASSES, PROB_COLS, strict=True):
                try:
                    value = float(row[column])
                except ValueError as exc:
                    raise ValueError(f"row {row_number}: {column} is not numeric") from exc
                if not math.isfinite(value) or value < 0.0 or value > 1.0:
                    raise ValueError(f"row {row_number}: {column} must be finite and in [0, 1]")
                probs[label] = value
            rows[ecg_id] = probs
    return rows


def auroc_rank(y_true: list[int], y_score: list[float]) -> float | None:
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    sorted_pairs = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[index][0]:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        positives_in_tie = sum(label for _, label in sorted_pairs[index:end])
        rank_sum_pos += positives_in_tie * average_rank
        rank += end - index
        index = end

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def require_metrics_file(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"{METRICS_NAME} not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{METRICS_NAME} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{METRICS_NAME} must contain a JSON object")
    return payload


def score_submission(submission_dir: Path, reference_dir: Path) -> dict[str, object]:
    predictions_path = submission_dir / PREDICTIONS_NAME
    if not predictions_path.exists():
        raise ValueError(f"{PREDICTIONS_NAME} not found")
    require_metrics_file(submission_dir / METRICS_NAME)

    gold = load_gold(reference_dir / "fold10_labels.csv")
    predictions = load_predictions(predictions_path)
    expected_ids = set(gold)
    actual_ids = set(predictions)
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        details = []
        if missing:
            details.append(f"missing {len(missing)} expected ecg_id values")
        if extra:
            details.append(f"unexpected {len(extra)} ecg_id values")
        raise ValueError("; ".join(details))

    aurocs: dict[str, float | None] = {}
    ordered_ids = sorted(gold)
    for label in CLASSES:
        y_true = [gold[ecg_id][label] for ecg_id in ordered_ids]
        y_score = [predictions[ecg_id][label] for ecg_id in ordered_ids]
        aurocs[label] = auroc_rank(y_true, y_score)

    valid = [value for value in aurocs.values() if value is not None]
    macro_auroc = sum(valid) / len(valid) if valid else 0.0
    passed = macro_auroc >= PASS_THRESHOLD
    return {
        "score": 1.0 if passed else 0.0,
        "raw_macro_auroc": macro_auroc,
        "passed": passed,
        "threshold": PASS_THRESHOLD,
        "row_count": len(predictions),
        "auroc_per_class": aurocs,
    }


def main() -> int:
    args = parse_args()
    try:
        payload = score_submission(Path(args.submission_dir).resolve(), Path(args.reference_dir).resolve())
    except Exception as exc:
        payload = {
            "score": 0.0,
            "raw_macro_auroc": 0.0,
            "passed": False,
            "error": str(exc),
        }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
