"""Scoring helpers for legal/authorship_attribution."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

ATTRIBUTION_COLUMNS = ["query_id", "predicted_candidate_id"]
VERIFICATION_COLUMNS = ["pair_id", "predicted_label_same"]


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "hard_gate": self.hard_gate,
            "details": self.details,
        }


def _read_csv(text: str, expected_columns: list[str], label: str) -> tuple[list[dict[str, str]], str | None]:
    try:
        reader = csv.DictReader(io.StringIO(text), restkey="__extra__")
        if reader.fieldnames != expected_columns:
            return [], (
                f"{label} columns must be exactly {expected_columns}, "
                f"got {reader.fieldnames}"
            )
        rows = list(reader)
    except csv.Error as exc:
        return [], f"{label} is not valid CSV: {exc}"

    if any("__extra__" in row for row in rows):
        return [], f"{label} contains rows with extra fields"
    return rows, None


def _index_rows(
    rows: list[dict[str, str]],
    key: str,
    expected_keys: set[str],
    label: str,
) -> tuple[dict[str, dict[str, str]], str | None]:
    if len(rows) != len(expected_keys):
        return {}, f"{label} row count must be {len(expected_keys)}, got {len(rows)}"

    seen: dict[str, dict[str, str]] = {}
    duplicates: set[str] = set()
    for row in rows:
        row_key = row.get(key, "")
        if row_key in seen:
            duplicates.add(row_key)
        seen[row_key] = row

    if duplicates:
        return {}, f"{label} has duplicated {key}: {sorted(duplicates)[:5]}"

    actual_keys = set(seen)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        return {}, (
            f"{label} key coverage mismatch; "
            f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )

    return seen, None


def _accuracy(correct: int, total: int) -> float:
    return float(correct / total) if total else 0.0


def _binary_macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    scores: list[float] = []
    for label in (0, 1):
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(scores)


def _macro_f1_multiclass(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(scores) if scores else 0.0


def score_output_bundle(
    *,
    attribution_predictions_csv: str,
    verification_predictions_csv: str,
    attribution_labels_csv: str,
    verification_labels_csv: str,
    attribution_queries_csv: str,
    attribution_candidates_csv: str,
    verification_pairs_csv: str,
    pass_threshold: float = 0.70,
) -> ScoreResult:
    verification_pairs, error = _read_csv(
        verification_pairs_csv, ["pair_id", "dataset", "text1", "text2"], "verification_pairs"
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    verification_labels, error = _read_csv(
        verification_labels_csv, ["pair_id", "label_same"], "verification_labels"
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    verification_predictions, error = _read_csv(
        verification_predictions_csv, VERIFICATION_COLUMNS, "verification_predictions"
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    attribution_queries, error = _read_csv(
        attribution_queries_csv,
        ["query_id", "dataset", "n_candidates", "repetition", "query_text", "candidate_pool_id"],
        "attribution_queries",
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    attribution_candidates, error = _read_csv(
        attribution_candidates_csv,
        ["candidate_pool_id", "candidate_id", "candidate_text"],
        "attribution_candidates",
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    attribution_labels, error = _read_csv(
        attribution_labels_csv, ["query_id", "true_candidate_id"], "attribution_labels"
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    attribution_predictions, error = _read_csv(
        attribution_predictions_csv, ATTRIBUTION_COLUMNS, "attribution_predictions"
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    pair_ids = {row["pair_id"] for row in verification_pairs}
    query_ids = {row["query_id"] for row in attribution_queries}
    pred_pairs, error = _index_rows(
        verification_predictions, "pair_id", pair_ids, "verification_predictions"
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    pred_queries, error = _index_rows(
        attribution_predictions, "query_id", query_ids, "attribution_predictions"
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    label_pairs, error = _index_rows(
        verification_labels, "pair_id", pair_ids, "verification_labels"
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    label_queries, error = _index_rows(
        attribution_labels, "query_id", query_ids, "attribution_labels"
    )
    if error:
        return ScoreResult(0.0, False, error, hard_gate=True)

    pool_candidates: dict[str, list[str]] = {}
    for row in attribution_candidates:
        pool_candidates.setdefault(row["candidate_pool_id"], []).append(row["candidate_id"])

    verification_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in ("blog", "email"):
        rows = [row for row in verification_pairs if row["dataset"] == dataset]
        true_values: list[int] = []
        pred_values: list[int] = []
        correct = 0
        for row in rows:
            pair_id = row["pair_id"]
            true = int(label_pairs[pair_id]["label_same"])
            raw_pred = pred_pairs[pair_id]["predicted_label_same"].strip()
            pred = int(raw_pred) if raw_pred in {"0", "1"} else -1
            true_values.append(true)
            pred_values.append(pred)
            correct += int(pred == true)
        verification_by_dataset[dataset] = {
            "accuracy": _accuracy(correct, len(rows)),
            "macro_f1": _binary_macro_f1(true_values, pred_values),
            "correct": correct,
            "total": len(rows),
        }

    attribution_by_config: dict[str, dict[str, Any]] = {}
    for dataset in ("blog", "email"):
        for n_candidates in ("10", "20"):
            rows = [
                row
                for row in attribution_queries
                if row["dataset"] == dataset and row["n_candidates"] == n_candidates
            ]
            true_values: list[str] = []
            pred_values: list[str] = []
            labels: set[str] = set()
            correct = 0
            for row in rows:
                query_id = row["query_id"]
                pool_id = row["candidate_pool_id"]
                valid_candidates = set(pool_candidates.get(pool_id, []))
                labels.update(valid_candidates)
                true = label_queries[query_id]["true_candidate_id"]
                raw_pred = pred_queries[query_id]["predicted_candidate_id"].strip()
                pred = raw_pred if raw_pred in valid_candidates else "__invalid__"
                true_values.append(true)
                pred_values.append(pred)
                correct += int(pred == true)
            attribution_by_config[f"{dataset}_{n_candidates}"] = {
                "accuracy": _accuracy(correct, len(rows)),
                "macro_f1": _macro_f1_multiclass(true_values, pred_values, sorted(labels)),
                "correct": correct,
                "total": len(rows),
            }

    component_scores = [
        verification_by_dataset["blog"]["accuracy"],
        verification_by_dataset["email"]["accuracy"],
        attribution_by_config["blog_10"]["accuracy"],
        attribution_by_config["blog_20"]["accuracy"],
        attribution_by_config["email_10"]["accuracy"],
        attribution_by_config["email_20"]["accuracy"],
    ]
    aggregate = sum(component_scores) / len(component_scores)
    passed = aggregate >= pass_threshold
    return ScoreResult(
        score=aggregate,
        passed=passed,
        reason="passed threshold" if passed else "below threshold",
        hard_gate=False,
        details={
            "pass_threshold": pass_threshold,
            "component_scores": component_scores,
            "verification": verification_by_dataset,
            "attribution": attribution_by_config,
        },
    )
