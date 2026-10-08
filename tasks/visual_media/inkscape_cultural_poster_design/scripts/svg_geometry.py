"""SVG image viewport corners in physical root-canvas coordinates."""

from __future__ import annotations

import math
import re
from xml.etree import ElementTree as ET

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _multiply(left, right):
    a, b, c, d, e, f = left
    g, h, i, j, k, l = right
    return (
        a * g + c * h,
        b * g + d * h,
        a * i + c * j,
        b * i + d * j,
        a * k + c * l + e,
        b * k + d * l + f,
    )


def _numbers(text: str) -> list[float]:
    pattern = rf"\s*{NUMBER}(?:(?:\s*,\s*|\s+){NUMBER})*\s*"
    if not re.fullmatch(pattern, text):
        raise ValueError("Invalid SVG number list")
    values = [float(v) for v in re.findall(NUMBER, text)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite SVG number")
    return values


def _transform(text: str) -> tuple[float, ...]:
    result = IDENTITY
    offset = 0
    for match in re.finditer(r"([A-Za-z]+)\s*\(([^()]*)\)", text):
        gap = text[offset : match.start()]
        if gap.strip(" \t\r\n,") or gap.count(",") > 1 or (offset == 0 and "," in gap):
            raise ValueError("Invalid SVG transform list")
        offset = match.end()
        name = match.group(1)
        values = _numbers(match.group(2))
        if name == "matrix" and len(values) == 6:
            transform = tuple(values)
        elif name == "translate" and len(values) in (1, 2):
            transform = (1, 0, 0, 1, values[0], values[1] if len(values) == 2 else 0)
        elif name == "scale" and len(values) in (1, 2):
            transform = (values[0], 0, 0, values[-1], 0, 0)
        elif name == "rotate" and len(values) in (1, 3):
            angle = math.radians(values[0])
            cosine, sine = math.cos(angle), math.sin(angle)
            transform = (cosine, sine, -sine, cosine, 0, 0)
            if len(values) == 3:
                x, y = values[1:]
                transform = _multiply(
                    _multiply((1, 0, 0, 1, x, y), transform),
                    (1, 0, 0, 1, -x, -y),
                )
        elif name in ("skewX", "skewY") and len(values) == 1:
            angle = math.radians(values[0])
            if abs(math.cos(angle)) < 1e-12:
                raise ValueError("Singular SVG skew")
            tangent = math.tan(angle)
            transform = (1, 0, tangent, 1, 0, 0) if name == "skewX" else (1, tangent, 0, 1, 0, 0)
        else:
            raise ValueError("Unsupported or malformed SVG transform")
        result = _multiply(result, transform)
    if text[offset:].strip() or not all(math.isfinite(v) for v in result):
        raise ValueError("Invalid SVG transform")
    return result


def _length(value: str | None, percentage_base: float) -> float:
    match = re.fullmatch(rf"\s*({NUMBER})\s*(px|mm|cm|in|pt|pc|%)?\s*", value or "0")
    if match is None:
        raise ValueError("Unsupported SVG length")
    factors = {
        None: 1,
        "px": 1,
        "mm": 96 / 25.4,
        "cm": 96 / 2.54,
        "in": 96,
        "pt": 96 / 72,
        "pc": 16,
        "%": percentage_base / 100,
    }
    result = float(match.group(1)) * factors[match.group(2)]
    if not math.isfinite(result):
        raise ValueError("Non-finite SVG length")
    return result


def image_corners_mm(root: ET.Element, image: ET.Element, width_mm: float, height_mm: float):
    """Resolve affine ancestry and the root viewBox; reject unsupported viewports."""
    if not all(math.isfinite(v) and v > 0 for v in (width_mm, height_mm)):
        raise ValueError("Invalid canvas dimensions")
    parents = {child: parent for parent in root.iter() for child in parent}
    chain = [image]
    while chain[-1] is not root:
        chain.append(parents[chain[-1]])
    if any(e.tag.split("}")[-1] == "svg" for e in chain[:-1]):
        raise ValueError("Nested SVG viewports require renderer geometry")
    if root.get("transform"):
        raise ValueError("Root SVG transforms require renderer geometry")
    for element in root.iter():
        if element.tag.split("}")[-1] == "style" and re.search(
            r"\btransform\s*:", element.text or "", re.IGNORECASE
        ):
            raise ValueError("CSS transforms require renderer geometry")
    if any(re.search(r"(?:^|;)\s*transform\s*:", e.get("style", ""), re.IGNORECASE) for e in chain):
        raise ValueError("CSS transforms require renderer geometry")

    sx = sy = 25.4 / 96
    ox = oy = 0.0
    viewport_width, viewport_height = width_mm / sx, height_mm / sy
    if root.get("viewBox"):
        viewbox = _numbers(root.get("viewBox"))
        if len(viewbox) != 4 or viewbox[2] <= 0 or viewbox[3] <= 0:
            raise ValueError("Invalid SVG viewBox")
        min_x, min_y, viewport_width, viewport_height = viewbox
        sx, sy = width_mm / viewport_width, height_mm / viewport_height
        preserve = root.get("preserveAspectRatio", "xMidYMid meet").split()
        if preserve and preserve[0] == "defer":
            preserve = preserve[1:]
        if preserve == ["none"]:
            ox, oy = -min_x * sx, -min_y * sy
        else:
            if not preserve or not re.fullmatch(r"x(Min|Mid|Max)Y(Min|Mid|Max)", preserve[0]):
                raise ValueError("Invalid preserveAspectRatio")
            mode = preserve[1] if len(preserve) == 2 else "meet"
            if len(preserve) > 2 or mode not in ("meet", "slice"):
                raise ValueError("Invalid preserveAspectRatio")
            sx = sy = min(sx, sy) if mode == "meet" else max(sx, sy)
            factors = {"Min": 0, "Mid": 0.5, "Max": 1}
            ox = -min_x * sx + (width_mm - viewport_width * sx) * factors[preserve[0][1:4]]
            oy = -min_y * sy + (height_mm - viewport_height * sy) * factors[preserve[0][5:8]]

    transform = (sx, 0, 0, sy, ox, oy)
    for element in reversed(chain[:-1]):
        transform = _multiply(transform, _transform(element.get("transform", "")))
    a, b, c, d, e, f = transform
    if not all(math.isfinite(v) for v in transform) or a * d - b * c == 0:
        raise ValueError("Degenerate image transform")
    x, y = _length(image.get("x"), viewport_width), _length(image.get("y"), viewport_height)
    width = _length(image.get("width"), viewport_width)
    height = _length(image.get("height"), viewport_height)
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    corners = [
        (a * xx + c * yy + e, b * xx + d * yy + f)
        for xx, yy in ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
    ]
    if not all(math.isfinite(v) for point in corners for v in point):
        raise ValueError("Non-finite image corners")
    return corners
