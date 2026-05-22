"""Local evaluation helpers for the marrow cell-type annotation task."""

import csv
import io
from typing import Any, Optional


PREDICTION_HEADER = ["cell_id", "predicted_cell_type"]
GROUND_TRUTH_HEADER = ["cell_id", "cell_type"]
CANONICAL_ALLOWED_LABELS = [
    "BM CD4 T cell",
    "BM CD8 + CD4 T cell",
    "CD4+ macrophage",
    "Klrb1a/b/c(-) NK cell",
    "MPP Fraction A + HSC",
    "MPP Fraction B",
    "MPP Fraction B + C",
    "NK cell",
    "Unknown Progenitor",
    "basophil",
    "common lymphoid progenitor cell",
    "early pro B cell",
    "granulocyte",
    "granulocyte monocyte progenitor cell",
    "granulocytopoietic cell",
    "immature B cell",
    "late pro B cell",
    "megakaryocyte-erythroid progenitor cell",
    "monocyte + promonocyte",
    "naïve B cell",
    "pre B cell",
]


def _decode_utf8(raw: bytes) -> str:
    return raw.decode("utf-8")


def _read_csv_rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


def parse_allowed_labels(raw: bytes) -> list[str]:
    labels = [line.strip() for line in _decode_utf8(raw).splitlines() if line.strip()]
    if not labels:
        raise ValueError("allowed_labels.txt is empty")
    if len(labels) != len(set(labels)):
        raise ValueError("allowed_labels.txt contains duplicate labels")
    return labels


def parse_ground_truth(raw: bytes) -> list[dict[str, str]]:
    rows = _read_csv_rows(_decode_utf8(raw))
    if not rows:
        raise ValueError("ground_truth.csv is empty")
    if rows[0] != GROUND_TRUTH_HEADER:
        raise ValueError("ground_truth.csv header mismatch")
    records = []
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            raise ValueError(f"ground_truth.csv row {line_no} must have exactly 2 columns")
        records.append({"cell_id": row[0], "cell_type": row[1]})
    if not records:
        raise ValueError("ground_truth.csv has no data rows")
    ids = [row["cell_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("ground_truth.csv contains duplicate cell ids")
    return records


def parse_predictions(raw: bytes) -> list[dict[str, str]]:
    rows = _read_csv_rows(_decode_utf8(raw))
    if not rows:
        raise ValueError("predictions.csv is empty")
    if rows[0] != PREDICTION_HEADER:
        raise ValueError("predictions.csv header must be exactly cell_id,predicted_cell_type")
    records = []
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            raise ValueError(f"predictions.csv row {line_no} must have exactly 2 columns")
        records.append({"cell_id": row[0], "predicted_cell_type": row[1]})
    ids = [row["cell_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("predictions.csv contains duplicate cell ids")
    return records


def compute_macro_f1(truth: list[str], pred: list[str], labels: list[str]) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    total = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(truth, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, pred) if t == label and p != label)
        denom = 2 * tp + fp + fn
        score = 0.0 if denom == 0 else (2 * tp) / denom
        per_class[label] = score
        total += score
    return total / len(labels), per_class


def evaluate_prediction_submission(
    prediction_bytes: bytes,
    ground_truth_bytes: bytes,
    allowed_labels_bytes: Optional[bytes] = None,
    *,
    allowed_labels: Optional[list[str]] = None,
    pass_macro_f1: float,
    pass_accuracy: float,
) -> dict[str, Any]:
    if allowed_labels is None:
        if allowed_labels_bytes is None:
            raise ValueError("allowed labels must be provided")
        allowed_labels = parse_allowed_labels(allowed_labels_bytes)
    ground_truth_rows = parse_ground_truth(ground_truth_bytes)
    prediction_rows = parse_predictions(prediction_bytes)

    truth_ids = [row["cell_id"] for row in ground_truth_rows]
    prediction_ids = [row["cell_id"] for row in prediction_rows]

    if len(prediction_rows) != len(ground_truth_rows):
        return {
            "valid": False,
            "error": f"row_count={len(prediction_rows)} expected={len(ground_truth_rows)}",
        }
    if set(prediction_ids) != set(truth_ids):
        missing = sorted(set(truth_ids) - set(prediction_ids))[:3]
        extra = sorted(set(prediction_ids) - set(truth_ids))[:3]
        return {
            "valid": False,
            "error": f"cell_id mismatch missing={missing} extra={extra}",
        }

    prediction_by_id = {row["cell_id"]: row["predicted_cell_type"] for row in prediction_rows}
    unsupported = sorted(
        {label for label in prediction_by_id.values() if label not in set(allowed_labels)}
    )
    if unsupported:
        return {
            "valid": False,
            "error": f"unsupported labels: {unsupported[:3]}",
        }

    truth = [row["cell_type"] for row in ground_truth_rows]
    pred = [prediction_by_id[row["cell_id"]] for row in ground_truth_rows]
    accuracy = sum(t == p for t, p in zip(truth, pred)) / len(truth)
    macro_f1, per_class_f1 = compute_macro_f1(truth, pred, allowed_labels)
    passes = macro_f1 >= pass_macro_f1 and accuracy >= pass_accuracy
    return {
        "valid": True,
        "error": None,
        "row_count": len(prediction_rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "passes": passes,
    }
