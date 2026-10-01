"""Scoring helper for genome-browser SVG outputs."""

from __future__ import annotations

import heapq
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SVG_NS = "http://www.w3.org/2000/svg"
GRAPHICAL_TAGS = {"rect", "path", "polygon", "polyline", "circle", "ellipse", "line"}
NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


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
    if re.sub(NUMBER, "", value).strip(" \t\r\n,") or re.search(r",\s*,|^\s*,|,\s*$", value):
        raise ValueError("invalid SVG number list")
    values = [float(raw) for raw in re.findall(NUMBER, value)]
    if not all(math.isfinite(number) and abs(number) <= 1e12 for number in values):
        raise ValueError("nonfinite or excessive SVG coordinate")
    return values


def _compose_transform(parent: tuple, local: tuple) -> tuple:
    pa, pb, pc, pd, pe, pf = parent
    la, lb, lc, ld, le, lf = local
    result = (
        pa * la + pc * lb,
        pb * la + pd * lb,
        pa * lc + pc * ld,
        pb * lc + pd * ld,
        pa * le + pc * lf + pe,
        pb * le + pd * lf + pf,
    )
    if not all(math.isfinite(value) and abs(value) <= 1e12 for value in result):
        raise ValueError("invalid composed SVG transform")
    return result


def _parse_transform(raw: str) -> tuple:
    matrix = IDENTITY
    cursor = 0
    for match in re.finditer(r"([A-Za-z]+)\s*\(([^()]*)\)", raw):
        if raw[cursor : match.start()].strip(" \t\r\n,"):
            raise ValueError("invalid SVG transform")
        name, arguments = match.groups()
        values = _float_values(arguments)
        if name == "matrix" and len(values) == 6:
            local = tuple(values)
        elif name == "translate" and len(values) in {1, 2}:
            local = (1, 0, 0, 1, values[0], values[1] if len(values) == 2 else 0)
        elif name == "scale" and len(values) in {1, 2}:
            local = (values[0], 0, 0, values[-1], 0, 0)
        elif name == "rotate" and len(values) in {1, 3}:
            angle = math.radians(values[0])
            cosine, sine = math.cos(angle), math.sin(angle)
            local = (cosine, sine, -sine, cosine, 0, 0)
            if len(values) == 3:
                center_x, center_y = values[1:]
                local = _compose_transform(
                    (1, 0, 0, 1, center_x, center_y),
                    _compose_transform(local, (1, 0, 0, 1, -center_x, -center_y)),
                )
        elif name in {"skewX", "skewY"} and len(values) == 1:
            tangent = math.tan(math.radians(values[0]))
            local = (1, 0, tangent, 1, 0, 0) if name == "skewX" else (1, tangent, 0, 1, 0, 0)
        else:
            raise ValueError("unsupported SVG transform")
        matrix = _compose_transform(matrix, local)
        cursor = match.end()
    if raw[cursor:].strip(" \t\r\n,"):
        raise ValueError("invalid SVG transform")
    return matrix


def _transform_points(
    points: list[tuple[float, float]], matrix: tuple
) -> list[tuple[float, float]]:
    scale_x, shear_y, shear_x, scale_y, offset_x, offset_y = matrix
    result = [
        (scale_x * xpos + shear_x * ypos + offset_x, shear_y * xpos + scale_y * ypos + offset_y)
        for xpos, ypos in points
    ]
    if not all(math.isfinite(value) and abs(value) <= 1e12 for point in result for value in point):
        raise ValueError("invalid transformed SVG coordinate")
    return result


def _path_contours(raw: str) -> list[list[tuple[float, float]]]:
    """Parse linear SVG subpaths without connecting separate moveto commands."""
    token_pattern = rf"{NUMBER}|[MmLlHhVvZz]"
    if re.sub(token_pattern, "", raw).strip(" \t\r\n,") or re.search(
        r",\s*,|^\s*,|,\s*$|[MmLlHhVvZz]\s*,|,\s*[MmLlHhVvZz]", raw
    ):
        raise ValueError("unsupported or malformed SVG path")
    tokens = re.findall(token_pattern, raw)
    contours = []
    points = []
    position = (0.0, 0.0)
    command = ""
    cursor = 0
    while cursor < len(tokens):
        if tokens[cursor].isalpha():
            command = tokens[cursor]
            cursor += 1
            if command.upper() == "Z":
                if not points:
                    raise ValueError("closepath without moveto")
                position = points[0]
                contours.append(points + [position])
                points = []
                command = ""
                continue
        if not command or (not points and command.upper() != "M"):
            raise ValueError("path must start with moveto")
        count = 1 if command.upper() in {"H", "V"} else 2
        arguments = tokens[cursor : cursor + count]
        if len(arguments) != count or any(token.isalpha() for token in arguments):
            raise ValueError("incomplete SVG path command")
        values = _float_values(" ".join(arguments))
        cursor += count
        if command.upper() == "M" and points:
            contours.append(points)
            points = []
        if command.upper() == "H":
            position = (values[0] + (position[0] if command.islower() else 0), position[1])
        elif command.upper() == "V":
            position = (position[0], values[0] + (position[1] if command.islower() else 0))
        else:
            position = tuple(
                value + (previous if command.islower() else 0)
                for value, previous in zip(values, position)
            )
        points.append(position)
        if command.upper() == "M":
            command = "l" if command.islower() else "L"
    if points:
        contours.append(points)
    return contours


def _signal_contour(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a monotone signal boundary, excluding an area plot's baseline closure."""
    if len(points) < 2:
        return []
    if points[0] == points[-1]:
        points = points[:-1]
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    for baseline in (min_y, max_y):
        off_baseline = next(
            (
                index
                for index, point in enumerate(points)
                if not math.isclose(point[1], baseline, abs_tol=1e-7)
            ),
            None,
        )
        if off_baseline is None:
            continue
        rotated = points[off_baseline:] + points[:off_baseline]
        start = 0
        for end in range(len(rotated) + 1):
            if end < len(rotated) and math.isclose(rotated[end][1], baseline, abs_tol=1e-7):
                continue
            baseline_x = {point[0] for point in rotated[start:end]}
            if min_x in baseline_x and max_x in baseline_x:
                points = rotated[end - 1 :] + rotated[: start + 1]
                break
            start = end + 1
    if points[0][0] > points[-1][0]:
        points = points[::-1]
    while len(points) > 1 and points[0][0] == points[1][0]:
        points = points[1:]
    while len(points) > 1 and points[-1][0] == points[-2][0]:
        points = points[:-1]
    if any(left[0] > right[0] for left, right in zip(points, points[1:])):
        return []
    return points


def _bar_contour(
    aligned: list[tuple[float, float, float]], baseline: float, direction: str
) -> list[tuple[float, float]]:
    """Extract the visible silhouette of co-baselined bars, including zero-signal gaps."""
    if len(aligned) < 12:
        return []
    ordered = sorted(aligned)
    edges = sorted({edge for left, right, _ in ordered for edge in (left, right)})
    active = []
    cursor = 0
    points = []
    orientation = 1 if direction == "up" else -1
    for left, right in zip(edges, edges[1:]):
        while cursor < len(ordered) and ordered[cursor][0] <= left:
            _, endpoint, tip = ordered[cursor]
            heapq.heappush(active, (orientation * tip, endpoint))
            cursor += 1
        while active and active[0][1] <= left:
            heapq.heappop(active)
        tip = orientation * active[0][0] if active else baseline
        points.extend([(left, tip), (right, tip)])
    return points


def _signal_candidates(root: ET.Element) -> list[tuple[str, list[tuple[float, float]]]]:
    candidates = []
    bars: dict[tuple, list[tuple[float, float, float]]] = {}
    pending = [(root, IDENTITY, {"fill": "black", "stroke": "none"})]
    while pending:
        elem, parent_matrix, inherited_style = pending.pop()
        tag = _strip_namespace(elem.tag)
        if tag in {"defs", "clipPath", "mask", "marker", "pattern", "symbol", "script", "metadata"}:
            continue
        style = dict(inherited_style)
        style.update(
            {
                key: value
                for key, value in elem.attrib.items()
                if key
                in {
                    "fill",
                    "stroke",
                    "display",
                    "visibility",
                    "opacity",
                    "fill-opacity",
                    "stroke-opacity",
                }
            }
        )
        for declaration in elem.get("style", "").split(";"):
            key, separator, value = declaration.partition(":")
            if separator:
                style[key.strip()] = value.strip()
        try:
            opacity = float(style.get("opacity", "1"))
            if style.get("display") == "none" or not math.isfinite(opacity) or opacity <= 0:
                continue
            matrix = _compose_transform(parent_matrix, _parse_transform(elem.get("transform", "")))
            if tag in {"svg", "g", "a"}:
                pending.extend((child, matrix, style) for child in elem)
            if style.get("visibility") in {"hidden", "collapse"}:
                continue
            visible_paint = False
            for paint in ("fill", "stroke"):
                paint_opacity = float(style.get(f"{paint}-opacity", "1"))
                if not math.isfinite(paint_opacity):
                    raise ValueError("nonfinite SVG paint opacity")
                visible_paint |= style.get(paint) != "none" and paint_opacity > 0
            if not visible_paint:
                continue
            if tag == "rect":
                dimensions = []
                for attribute in ("x", "y", "width", "height"):
                    values = _float_values(elem.get(attribute, "0"))
                    if len(values) != 1:
                        raise ValueError("invalid rect dimension")
                    dimensions.append(values[0])
                xpos, ypos, width, height = dimensions
                if width <= 0 or height < 0:
                    continue
                corners = _transform_points(
                    [
                        (xpos, ypos),
                        (xpos + width, ypos),
                        (xpos + width, ypos + height),
                        (xpos, ypos + height),
                    ],
                    matrix,
                )
                if not (
                    math.isclose(corners[0][1], corners[1][1], abs_tol=1e-7)
                    and math.isclose(corners[0][0], corners[3][0], abs_tol=1e-7)
                ):
                    continue
                left, right = sorted((corners[0][0], corners[1][0]))
                top, bottom = sorted((corners[0][1], corners[3][1]))
                for baseline, tip, direction in ((bottom, top, "up"), (top, bottom, "down")):
                    key = (round(baseline, 6), direction, style["stroke"])
                    bars.setdefault(key, []).append((left, right, tip))
            elif tag in {"polyline", "polygon", "path"}:
                if tag == "path":
                    contours = _path_contours(elem.get("d", ""))
                else:
                    values = _float_values(elem.get("points", ""))
                    if len(values) % 2:
                        raise ValueError("unpaired SVG point")
                    contours = [list(zip(values[0::2], values[1::2]))]
                for points in contours:
                    candidates.append((tag, _signal_contour(_transform_points(points, matrix))))
        except (ValueError, OverflowError):
            continue
    for (baseline, direction, _), aligned in bars.items():
        candidates.append(("rect-group", _bar_contour(aligned, baseline, direction)))
    return candidates


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
    slope = (
        sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values)) / denominator
    )
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
    ordered = points
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


def _has_graphical_evidence(root: ET.Element, reference: dict[str, Any]) -> tuple[bool, str]:
    graphical_count = sum(_strip_namespace(elem.tag) in GRAPHICAL_TAGS for elem in root.iter())
    best_correlation = 0.0
    best_detrended_correlation = 0.0
    best_delta_correlation = 0.0
    signal_candidate_count = 0
    candidate_sources: set[str] = set()
    passing_candidate: tuple[str, float, float, float] | None = None
    signal_profile = [float(value) for value in reference.get("signal_profile", [])]
    signal_detrended = _linear_detrend(signal_profile)
    signal_deltas = _first_differences(signal_profile)
    for source, points in _signal_candidates(root):
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
        candidate_sources.add(source)
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
            passing_candidate = (source, raw_correlation, detrended_correlation, delta_correlation)

    if passing_candidate is not None and signal_candidate_count <= 20:
        source, raw_correlation, detrended_correlation, delta_correlation = passing_candidate
        return True, (
            "signal geometry matches hidden BigWig profile "
            f"(raw={raw_correlation:.3f}, detrended={detrended_correlation:.3f}, "
            f"delta={delta_correlation:.3f}, source={source})"
        )

    return False, (
        "no signal geometry matched hidden BigWig profile "
        f"(raw={best_correlation:.3f}, detrended={best_detrended_correlation:.3f}, "
        f"delta={best_delta_correlation:.3f}, graphical elements={graphical_count}, "
        f"signal candidates={signal_candidate_count}, "
        f"sources={','.join(sorted(candidate_sources)) or 'none'})"
    )


def _has_browser_provenance(text: str) -> tuple[bool, str]:
    lowered = text.lower()
    terms = ("ucsc", "genome browser", "igv", "washu", "hg19", "grch37")
    if any(term in lowered for term in terms):
        return True, "browser/build provenance is present"
    return False, "missing genome-browser or hg19 provenance text"


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
        return SvgScoreResult(0.10, checks, [f"output.svg is not parseable XML: {exc}"])

    root_tag = _strip_namespace(root.tag).lower()
    if root_tag != "svg":
        return SvgScoreResult(0.10, checks, [f"root element is {root_tag!r}, not svg"])
    checks["valid_svg"] = True

    text = _element_text(root)
    chrom = str(reference["chrom"])
    start = int(reference["start"])
    end = int(reference["end"])

    checks["coordinate_evidence"], coordinate_note = _has_coordinate_evidence(
        text, chrom, start, end
    )
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
