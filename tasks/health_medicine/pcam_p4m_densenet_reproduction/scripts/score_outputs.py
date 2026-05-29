"""Scoring for health_medicine/pcam_p4m_densenet_reproduction.

Called from main.py.evaluate(). Reads the agent's output.csv and the hidden test
labels, enforces the hard-gate schema, runs the calibration sanity check, then
computes AUC and maps it to the rubric in TASK_INTAKE.md §5:

- 1.0 FULL   : hard-gate OK, calibration OK, AUC >= 0.80
- 0.5 PARTIAL: hard-gate OK, calibration OK, 0.65 <= AUC < 0.80
- 0.0 FAIL   : anything else

The scorer is pure NumPy (no sklearn, no pandas). AUC is computed via the
rank-statistic formula (Mann-Whitney U).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

N_TEST = 32768
HEADER = "id,prob"

AUC_FULL_THRESHOLD = 0.80
AUC_PARTIAL_THRESHOLD = 0.65
CALIB_MEAN_LOW = 0.30
CALIB_MEAN_HIGH = 0.70
CALIB_STD_MIN = 0.05


@dataclass
class ScoreResult:
    score: float
    tier: str
    reason: str
    auc: float | None = None
    mean_prob: float | None = None
    std_prob: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "tier": self.tier,
            "reason": self.reason,
            "auc": self.auc,
            "mean_prob": self.mean_prob,
            "std_prob": self.std_prob,
        }


def _fail(reason: str) -> ScoreResult:
    return ScoreResult(score=0.0, tier="FAIL", reason=reason)


def parse_output_csv(csv_bytes: bytes) -> tuple[np.ndarray, np.ndarray] | ScoreResult:
    """Validate and parse the agent's CSV. Returns (ids, probs) or a FAIL result."""
    try:
        text = csv_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _fail(f"output.csv not valid UTF-8: {exc}")

    # Allow a trailing newline, but otherwise require EXACTLY 32769 non-empty lines
    # (header + 32768 rows).
    lines = text.splitlines()
    # Trim trailing empty lines only.
    while lines and lines[-1] == "":
        lines.pop()

    if not lines:
        return _fail("output.csv is empty")

    header = lines[0].strip()
    if header != HEADER:
        return _fail(f"header != '{HEADER}' (got {header!r})")

    body = lines[1:]
    if len(body) != N_TEST:
        return _fail(f"expected {N_TEST} data rows, got {len(body)}")

    ids = np.empty(N_TEST, dtype=np.int64)
    probs = np.empty(N_TEST, dtype=np.float64)
    for i, raw in enumerate(body):
        parts = raw.split(",")
        if len(parts) != 2:
            return _fail(f"row {i}: expected 2 comma-separated fields, got {len(parts)}")
        id_str, prob_str = parts[0].strip(), parts[1].strip()
        try:
            row_id = int(id_str)
        except ValueError:
            return _fail(f"row {i}: id {id_str!r} is not an integer")
        try:
            prob = float(prob_str)
        except ValueError:
            return _fail(f"row {i}: prob {prob_str!r} is not a float")
        if not math.isfinite(prob):
            return _fail(f"row {i}: prob is NaN/Inf")
        if prob < 0.0 or prob > 1.0:
            return _fail(f"row {i}: prob={prob} outside [0, 1]")
        ids[i] = row_id
        probs[i] = prob

    expected_ids = np.arange(N_TEST, dtype=np.int64)
    if not np.array_equal(ids, expected_ids):
        # Identify the first offending row for a helpful message.
        diff_idx = int(np.argmax(ids != expected_ids))
        return _fail(
            f"ids must be 0..{N_TEST - 1} in order; first mismatch at row {diff_idx} "
            f"(got {int(ids[diff_idx])}, expected {diff_idx})"
        )

    return ids, probs


def rank_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via the Mann-Whitney U / rank-sum formula, tie-safe.

    y_true in {0, 1}, y_score real-valued. Returns float in [0, 1].
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_score = np.asarray(y_score).reshape(-1).astype(np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError("rank_auc shape mismatch")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks handle ties correctly.
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    n = y_score.size
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)  # 1-indexed average rank over tied block
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(ranks[y_true == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def score(
    output_csv_bytes: bytes | None,
    test_labels: np.ndarray,
) -> ScoreResult:
    """Main scoring entry point.

    Parameters
    ----------
    output_csv_bytes : bytes or None
        Raw contents of the agent's output/output.csv, or None if the file was
        missing.
    test_labels : np.ndarray
        Ground-truth labels (N_TEST,), dtype integer in {0, 1}.
    """
    if output_csv_bytes is None:
        return _fail("output/output.csv is missing")

    parsed = parse_output_csv(output_csv_bytes)
    if isinstance(parsed, ScoreResult):
        return parsed
    _, probs = parsed

    y = np.asarray(test_labels).reshape(-1).astype(np.int64)
    if y.size != N_TEST:
        # This is an evaluator-side error, not an agent error. Fail loudly.
        return _fail(f"hidden test labels have unexpected length {y.size} (expected {N_TEST})")

    mean_prob = float(probs.mean())
    std_prob = float(probs.std())

    if not (CALIB_MEAN_LOW <= mean_prob <= CALIB_MEAN_HIGH):
        return ScoreResult(
            score=0.0,
            tier="FAIL",
            reason=(
                f"calibration sanity: mean(prob)={mean_prob:.4f} outside "
                f"[{CALIB_MEAN_LOW}, {CALIB_MEAN_HIGH}]"
            ),
            mean_prob=mean_prob,
            std_prob=std_prob,
        )
    if std_prob < CALIB_STD_MIN:
        return ScoreResult(
            score=0.0,
            tier="FAIL",
            reason=f"calibration sanity: std(prob)={std_prob:.4f} < {CALIB_STD_MIN}",
            mean_prob=mean_prob,
            std_prob=std_prob,
        )

    auc = rank_auc(y, probs)

    if auc >= AUC_FULL_THRESHOLD:
        tier = "FULL"
        sc = 1.0
    elif auc >= AUC_PARTIAL_THRESHOLD:
        tier = "PARTIAL"
        sc = 0.5
    else:
        tier = "FAIL"
        sc = 0.0

    return ScoreResult(
        score=sc,
        tier=tier,
        reason=f"AUC={auc:.4f} tier={tier}",
        auc=auc,
        mean_prob=mean_prob,
        std_prob=std_prob,
    )


def load_labels_from_h5_bytes(h5_bytes: bytes) -> np.ndarray:
    """Decode an HDF5 labels file (key 'y', shape (N, 1, 1, 1)) into (N,) int."""
    import h5py

    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        y = f["y"][:]
    return np.asarray(y).reshape(-1).astype(np.int64)
