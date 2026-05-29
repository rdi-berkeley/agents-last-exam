"""Local evaluator for the quadcopter feedback-estimation task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REL_TOLERANCE = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_k_csv(path: Path) -> float:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("empty k.csv")
    rows = [
        [cell.strip() for cell in row]
        for row in csv.reader(text.splitlines())
        if any(cell.strip() for cell in row)
    ]
    if not rows:
        raise ValueError("empty k.csv")

    if len(rows[0]) != 1 or rows[0][0].lower() != "k":
        raise ValueError("k.csv must have a single 'k' header column")
    if len(rows) != 2:
        raise ValueError("k.csv must contain exactly one data row")
    if len(rows[1]) != 1 or not rows[1][0]:
        raise ValueError("k.csv data row must contain exactly one numeric value")

    return float(rows[1][0])


def _plot_metrics(path: Path) -> dict[str, float]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("plot.png is not a readable image")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nonwhite = float(np.mean(gray < 245))
    variance = float(np.var(gray))
    return {
        "height": float(height),
        "width": float(width),
        "nonwhite_fraction": nonwhite,
        "gray_variance": variance,
    }


def _plot_score(path: Path) -> tuple[float, dict[str, float]]:
    metrics = _plot_metrics(path)
    checks = [
        metrics["width"] >= 300,
        metrics["height"] >= 200,
        metrics["nonwhite_fraction"] >= 0.01,
        metrics["gray_variance"] >= 5.0,
    ]
    return sum(1.0 for item in checks if item) / len(checks), metrics


def evaluate_files(
    *,
    output_k_csv: Path,
    output_plot: Path,
    reference_k_csv: Path,
    reference_plot: Path,
) -> dict[str, Any]:
    for path, label in [
        (output_k_csv, "output_k_csv"),
        (output_plot, "output_plot"),
        (reference_k_csv, "reference_k_csv"),
        (reference_plot, "reference_plot"),
    ]:
        if not path.exists():
            return {"score": 0.0, "reason": f"missing_{label}", "details": {}}
        if path.stat().st_size <= 0:
            return {"score": 0.0, "reason": f"empty_{label}", "details": {}}

    if output_k_csv.stat().st_size > 10_000:
        return {"score": 0.0, "reason": "k_csv_too_large", "details": {}}
    if output_plot.stat().st_size < 1_000:
        return {"score": 0.0, "reason": "plot_too_small", "details": {}}

    try:
        output_k = _parse_k_csv(output_k_csv)
        reference_k = _parse_k_csv(reference_k_csv)
    except Exception as exc:
        return {"score": 0.0, "reason": f"k_parse_failed: {exc}", "details": {}}

    if not np.isfinite(output_k):
        return {"score": 0.0, "reason": "k_not_finite", "details": {}}
    if reference_k == 0:
        return {"score": 0.0, "reason": "invalid_reference_k", "details": {}}

    rel_error = abs(output_k - reference_k) / abs(reference_k)
    if rel_error > REL_TOLERANCE:
        return {
            "score": 0.0,
            "reason": "k_outside_tolerance",
            "details": {
                "output_k": output_k,
                "relative_error": rel_error,
                "tolerance": REL_TOLERANCE,
            },
        }

    try:
        plot_score, plot_metrics = _plot_score(output_plot)
        reference_plot_score, reference_plot_metrics = _plot_score(reference_plot)
    except Exception as exc:
        return {"score": 0.0, "reason": f"plot_check_failed: {exc}", "details": {}}

    if reference_plot_score < 1.0:
        return {"score": 0.0, "reason": "invalid_reference_plot", "details": {}}
    if plot_score < 0.75:
        return {
            "score": 0.0,
            "reason": "plot_not_plausible",
            "details": {"plot_metrics": plot_metrics},
        }

    k_score = 1.0
    score = 0.85 * k_score + 0.15 * plot_score

    if _sha256(output_k_csv) == _sha256(reference_k_csv) and _sha256(output_plot) == _sha256(
        reference_plot
    ):
        score = 1.0

    return {
        "score": float(max(0.0, min(1.0, score))),
        "reason": "scored",
        "details": {
            "output_k": output_k,
            "relative_error": rel_error,
            "k_score": k_score,
            "plot_score": plot_score,
            "plot_metrics": plot_metrics,
            "reference_plot_metrics": reference_plot_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-k-csv", required=True, type=Path)
    parser.add_argument("--output-plot", required=True, type=Path)
    parser.add_argument("--reference-k-csv", required=True, type=Path)
    parser.add_argument("--reference-plot", required=True, type=Path)
    args = parser.parse_args()

    result = evaluate_files(
        output_k_csv=args.output_k_csv,
        output_plot=args.output_plot,
        reference_k_csv=args.reference_k_csv,
        reference_plot=args.reference_plot,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
