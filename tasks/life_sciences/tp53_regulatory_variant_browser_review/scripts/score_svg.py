"""Scoring helper for genome-browser SVG outputs."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SVG_NS = "http://www.w3.org/2000/svg"
GRAPHICAL_TAGS = {"rect", "path", "polygon", "polyline", "circle", "ellipse", "line"}


@dataclass
class SvgScoreResult:
    score: float
    checks: dict[str, bool]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "checks": self.checks,
            "notes": self.notes,
        }


def _strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _element_text(root: ET.Element) -> str:
    chunks: list[str] = []
    for elem in root.iter():
        if elem.text:
            chunks.append(elem.text)
        if elem.tail:
            chunks.append(elem.tail)
        for value in elem.attrib.values():
            chunks.append(str(value))
    return " ".join(chunks)


def _numeric_tokens(text: str) -> list[int]:
    values: list[int] = []
    for token in re.findall(r"(?<![A-Za-z])\d{4,}(?![A-Za-z])", text.replace(",", "")):
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values


def _has_coordinate_evidence(text: str, chrom: str, start: int, end: int) -> tuple[bool, str]:
    lowered = text.lower()
    if chrom.lower() not in lowered:
        return False, f"missing chromosome label {chrom}"

    values = _numeric_tokens(text)
    in_range = sorted({value for value in values if start <= value <= end})
    has_endpoint = any(abs(value - start) <= 5 for value in values) or any(
        abs(value - end) <= 5 for value in values
    )
    has_tick_diversity = len(in_range) >= 3
    if has_endpoint and has_tick_diversity:
        return True, "coordinate labels include target chromosome and multiple in-window ticks"
    if in_range:
        return False, f"only {len(in_range)} in-window numeric coordinate label(s) found"
    return False, f"no numeric coordinate labels within {start}-{end}"


def _has_track_evidence(text: str) -> tuple[bool, str]:
    lowered = text.lower()
    variant_terms = ("vcf", "variant", "snv", "encff960ssf", "structural")
    signal_terms = ("h3k27ac", "k27ac", "k562", "chip", "bigwig", "histone")
    has_variant = any(term in lowered for term in variant_terms)
    has_signal = any(term in lowered for term in signal_terms)
    if has_variant and has_signal:
        return True, "both variant and H3K27ac signal track labels are present"
    missing = []
    if not has_variant:
        missing.append("variant/VCF")
    if not has_signal:
        missing.append("H3K27ac/BigWig")
    return False, "missing " + " and ".join(missing) + " track evidence"


def _float_values(value: str) -> list[float]:
    values: list[float] = []
    for raw in re.findall(r"-?\d+(?:\.\d+)?", value):
        try:
            num = float(raw)
        except ValueError:
            continue
        if math.isfinite(num):
            values.append(num)
    return values


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _linear_detrend(values: list[float]) -> list[float]:
    if len(values) < 3:
        return []
    mean_x = (len(values) - 1) / 2
    mean_y = sum(values) / len(values)
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    if denominator == 0:
        return []
    slope = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values)) / denominator
    intercept = mean_y - slope * mean_x
    return [value - (intercept + slope * index) for index, value in enumerate(values)]


def _first_differences(values: list[float]) -> list[float]:
    if len(values) < 3:
        return []
    return [right - left for left, right in zip(values, values[1:])]


def _best_oriented_correlation(left: list[float], right: list[float]) -> float:
    return max(_pearson(left, right), _pearson(left, [-value for value in right]))


def _resample_y_profile(points: list[tuple[float, float]], bins: int) -> list[float]:
    if not points or bins <= 1:
        return []
    ordered = sorted(points)
    min_x = ordered[0][0]
    max_x = ordered[-1][0]
    if max_x <= min_x:
        return []
    sampled: list[float] = []
    cursor = 0
    for index in range(bins):
        target_x = min_x + ((max_x - min_x) * index / (bins - 1))
        while cursor < len(ordered) - 2 and ordered[cursor + 1][0] < target_x:
            cursor += 1
        x1, y1 = ordered[cursor]
        x2, y2 = ordered[min(cursor + 1, len(ordered) - 1)]
        if x2 == x1:
            sampled.append(y1)
            continue
        ratio = (target_x - x1) / (x2 - x1)
        sampled.append(y1 + ((y2 - y1) * ratio))
    return sampled


def _path_points(elem: ET.Element) -> list[tuple[float, float]]:
    tag = _strip_namespace(elem.tag)
    raw = elem.attrib.get("points" if tag == "polyline" else "d", "")
    values = _float_values(raw)
    if len(values) < 2:
        return []
    return list(zip(values[0::2], values[1::2]))


def _has_graphical_evidence(root: ET.Element, reference: dict[str, Any]) -> tuple[bool, str]:
    graphical_count = 0
    best_correlation = 0.0
    best_detrended_correlation = 0.0
    best_delta_correlation = 0.0
    signal_candidate_count = 0
    passing_candidate: tuple[float, float, float] | None = None
    signal_profile = [float(value) for value in reference.get("signal_profile", [])]
    signal_detrended = _linear_detrend(signal_profile)
    signal_deltas = _first_differences(signal_profile)
    for elem in root.iter():
        tag = _strip_namespace(elem.tag)
        if tag not in GRAPHICAL_TAGS:
            continue
        graphical_count += 1
        if tag not in {"path", "polyline"}:
            continue
        points = _path_points(elem)
        if len(points) < 12:
            continue
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        if not x_values or not y_values:
            continue
        x_span = max(x_values) - min(x_values)
        y_span = max(y_values) - min(y_values)
        distinct_y = len({round(value, 1) for value in y_values})
        if x_span < 400 or y_span < 30 or distinct_y < 8:
            continue
        signal_candidate_count += 1
        sampled_y = _resample_y_profile(points, len(signal_profile))
        if not sampled_y or not signal_profile:
            continue
        sampled_detrended = _linear_detrend(sampled_y)
        sampled_deltas = _first_differences(sampled_y)
        raw_correlation = _best_oriented_correlation(signal_profile, sampled_y)
        detrended_correlation = _best_oriented_correlation(signal_detrended, sampled_detrended)
        delta_correlation = _best_oriented_correlation(signal_deltas, sampled_deltas)
        best_correlation = max(best_correlation, raw_correlation)
        best_detrended_correlation = max(best_detrended_correlation, detrended_correlation)
        best_delta_correlation = max(best_delta_correlation, delta_correlation)
        shape_match = detrended_correlation >= 0.70 or delta_correlation >= 0.60
        if raw_correlation >= 0.65 and shape_match:
            passing_candidate = (raw_correlation, detrended_correlation, delta_correlation)

    high_confidence = passing_candidate is not None and passing_candidate[0] >= 0.90
    if passing_candidate is not None and (signal_candidate_count <= 20 or high_confidence):
        raw_correlation, detrended_correlation, delta_correlation = passing_candidate
        return True, (
            "signal path matches hidden BigWig profile "
            f"(raw={raw_correlation:.3f}, detrended={detrended_correlation:.3f}, "
            f"delta={delta_correlation:.3f})"
        )

    return False, (
        "no signal-like path/polyline matched hidden BigWig profile "
        f"(raw={best_correlation:.3f}, detrended={best_detrended_correlation:.3f}, "
        f"delta={best_delta_correlation:.3f}, graphical elements={graphical_count}, "
        f"signal candidates={signal_candidate_count})"
    )


def _has_browser_provenance(text: str) -> tuple[bool, str]:
    lowered = text.lower()
    has_ucsc = "ucsc" in lowered or "genome.ucsc.edu" in lowered
    has_build = "hg19" in lowered or "grch37" in lowered
    if has_ucsc and has_build:
        return True, "UCSC and hg19/GRCh37 provenance are present"
    if has_ucsc:
        return False, "missing hg19/GRCh37 build provenance"
    if has_build:
        return False, "missing UCSC provenance"
    return False, "missing UCSC and hg19/GRCh37 provenance"


def score_svg_bytes(svg_bytes: bytes, reference: dict[str, Any]) -> SvgScoreResult:
    checks = {
        "non_empty": False,
        "valid_svg": False,
        "coordinate_evidence": False,
        "track_evidence": False,
        "graphical_evidence": False,
        "browser_provenance": False,
    }
    notes: list[str] = []

    if not svg_bytes or not svg_bytes.strip():
        return SvgScoreResult(0.0, checks, ["output.svg is missing or empty"])
    checks["non_empty"] = True

    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError as exc:
        return SvgScoreResult(0.0, checks, [f"output.svg is not parseable XML: {exc}"])

    root_tag = _strip_namespace(root.tag).lower()
    if root_tag != "svg":
        return SvgScoreResult(0.0, checks, [f"root element is {root_tag!r}, not svg"])
    checks["valid_svg"] = True

    text = _element_text(root)
    chrom = str(reference["chrom"])
    start = int(reference["start"])
    end = int(reference["end"])

    checks["coordinate_evidence"], coordinate_note = _has_coordinate_evidence(text, chrom, start, end)
    checks["track_evidence"], track_note = _has_track_evidence(text)
    checks["graphical_evidence"], graphical_note = _has_graphical_evidence(root, reference)
    checks["browser_provenance"], provenance_note = _has_browser_provenance(text)
    notes.extend([coordinate_note, track_note, graphical_note, provenance_note])

    core_checks = (
        checks["coordinate_evidence"],
        checks["track_evidence"],
        checks["graphical_evidence"],
        checks["browser_provenance"],
    )
    if not any(core_checks):
        return SvgScoreResult(0.0, checks, notes)

    score = 0.20
    score += 0.25 if checks["coordinate_evidence"] else 0.0
    score += 0.20 if checks["track_evidence"] else 0.0
    score += 0.20 if checks["graphical_evidence"] else 0.0
    score += 0.15 if checks["browser_provenance"] else 0.0

    if not checks["graphical_evidence"]:
        score = min(score, 0.35)
    if not checks["browser_provenance"]:
        score = min(score, 0.60)
    if not all(checks.values()):
        score = min(score, 0.80)
    return SvgScoreResult(round(score, 4), checks, notes)


def score_svg_file(svg_path: str | Path, reference_path: str | Path) -> SvgScoreResult:
    svg_bytes = Path(svg_path).read_bytes()
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    return score_svg_bytes(svg_bytes, reference)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score a genome-browser SVG output.")
    parser.add_argument("svg_path")
    parser.add_argument("reference_path")
    args = parser.parse_args()

    result = score_svg_file(args.svg_path, args.reference_path)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
