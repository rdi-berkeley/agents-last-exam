"""Signal geometry regressions using synthetic profiles, not hidden task answers."""

from __future__ import annotations

import json
import math
import random
import socket
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tasks.life_sciences.tp53_locus_variant_histone_browser_svg.scripts import score_svg


PROFILE = [
    12,
    33,
    18,
    62,
    21,
    46,
    87,
    25,
    14,
    72,
    41,
    23,
    52,
    95,
    38,
    16,
    65,
    29,
    81,
    48,
    19,
    57,
    34,
    76,
]
REFERENCE = {"chrom": "chr17", "start": 7571651, "end": 7590910, "signal_profile": PROFILE}
LABELS = "hg19 chr17 7571651 7580000 7590910 K562 H3K27ac BigWig structural variants VCF"


def svg(geometry: str, labels: str = LABELS) -> bytes:
    return (
        f'<svg xmlns="{score_svg.SVG_NS}" width="800" height="400">'
        f"<text>{labels}</text>{geometry}</svg>"
    ).encode()


def points(profile=PROFILE):
    return [
        (50 + 600 * index / (len(profile) - 1), 250 - value) for index, value in enumerate(profile)
    ]


def point_text(vertices):
    return " ".join(f"{xpos:.12g},{ypos:.12g}" for xpos, ypos in vertices)


def signal(kind: str, profile=PROFILE) -> str:
    vertices = points(profile)
    if kind == "polyline":
        return f'<polyline points="{point_text(vertices)}" fill="none" stroke="green"/>'
    if kind == "polygon":
        return f'<polygon points="{point_text(vertices)}"/>'
    if kind == "area":
        vertices = [(50, 250)] + vertices + [(650, 250)]
        return f'<polygon points="{point_text(vertices)}"/>'
    if kind == "path":
        return f'<path d="M{point_text(vertices[:1])} L{point_text(vertices[1:])}"/>'
    if kind == "relative":
        previous = vertices[0]
        commands = [f"m{point_text([previous])}"]
        for vertex in vertices[1:]:
            commands.append(f"l{vertex[0] - previous[0]:.12g},{vertex[1] - previous[1]:.12g}")
            previous = vertex
        return f'<path d="{" ".join(commands)}"/>'
    if kind == "bars":
        return "".join(
            f'<rect x="{50 + index * 25}" y="{250 - value}" width="25" height="{value}"/>'
            for index, value in enumerate(profile)
        )
    if kind in {"step-path", "step-polygon"}:
        vertices = [(50, 250)]
        for index, value in enumerate(profile):
            vertices.extend([(50 + index * 25, 250 - value), (75 + index * 25, 250 - value)])
        vertices.append((650, 250))
        if kind == "step-polygon":
            return f'<polygon points="{point_text(vertices)}"/>'
        commands = [f"M{point_text(vertices[:1])}"]
        for previous, vertex in zip(vertices, vertices[1:]):
            commands.append(f"H{vertex[0]}" if previous[1] == vertex[1] else f"V{vertex[1]}")
        return f'<path d="{" ".join(commands)} Z"/>'
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind", ["polyline", "polygon", "area", "path", "relative", "bars", "step-path", "step-polygon"]
)
def test_equivalent_signal_encodings_receive_identical_full_scores(kind):
    result = score_svg.score_svg_bytes(svg(signal(kind)), REFERENCE)
    assert result.score == 1.0, result.notes
    assert all(result.checks.values())
    assert "raw=1.000, detrended=1.000, delta=1.000" in result.notes[2]
    expected_source = {
        "area": "polygon",
        "relative": "path",
        "bars": "rect-group",
        "step-path": "path",
        "step-polygon": "polygon",
    }.get(kind, kind)
    assert f"source={expected_source}" in result.notes[2]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("offset", [0, 1, 8, 25])
def test_baseline_polygon_is_independent_of_start_vertex_and_winding(reverse, offset):
    vertices = [(50, 250)] + points() + [(650, 250)]
    if reverse:
        vertices.reverse()
    vertices = vertices[offset:] + vertices[:offset]
    vertices.append(vertices[0])
    result = score_svg.score_svg_bytes(
        svg(f'<polygon points="{point_text(vertices)}"/>'), REFERENCE
    )
    assert result.score == 1.0, result.notes


@pytest.mark.parametrize("kind", ["area", "path", "relative", "bars"])
@pytest.mark.parametrize(
    "transforms",
    [
        ("translate(125 -40)", "scale(1.5 0.75)"),
        ("matrix(2 0 0 -1 20 500)", "translate(-10,25)"),
        ("rotate(30 100 200)", "rotate(-30 100 200)"),
        ("skewX(20)", "skewX(-20)"),
        ("skewY(15)", "skewY(-15)"),
    ],
)
def test_nested_transform_equivalence(kind, transforms):
    outer, inner = transforms
    geometry = f'<g transform="{outer}"><g transform="{inner}">{signal(kind)}</g></g>'
    result = score_svg.score_svg_bytes(svg(geometry), REFERENCE)
    assert result.score == 1.0, result.notes


def test_transform_order_and_local_coordinate_scale_are_applied():
    geometry = '<polyline transform="translate(50 250) scale(600 -1)" points="'
    geometry += " ".join(f"{index / 23:.12g},{value}" for index, value in enumerate(PROFILE))
    geometry += '"/>'
    result = score_svg.score_svg_bytes(svg(geometry), REFERENCE)
    assert result.score == 1.0, result.notes
    assert score_svg._transform_points(
        [(2, 3)], score_svg._parse_transform("translate(10 20) scale(2 3)")
    ) == [(14, 29)]


def test_bars_with_nested_groups_local_transforms_and_unrelated_rectangles():
    bars = [
        f'<g transform="translate({50 + index * 25},250)"><g>'
        f'<rect x="0" y="{-value}" width="25" height="{value}"/>'
        "</g></g>"
        for index, value in enumerate(PROFILE)
    ]
    random.Random(17).shuffle(bars)
    geometry = '<rect width="800" height="400" fill="white"/>'
    geometry += '<g fill="green">' + "".join(bars) + "</g>"
    geometry += '<rect x="20" y="20" width="15" height="15" fill="green"/>'
    result = score_svg.score_svg_bytes(svg(geometry), REFERENCE)
    assert result.score == 1.0, result.notes


def test_scientific_notation_leading_decimals_and_implicit_lineto():
    vertices = points()
    geometry = (
        '<path d="M' + " ".join(f"{xpos:+.12e} {ypos:.12e}" for xpos, ypos in vertices) + '"/>'
    )
    assert score_svg.score_svg_bytes(svg(geometry), REFERENCE).score == 1.0
    assert score_svg._float_values(".5,-.25 +2E+2 3e-2") == [0.5, -0.25, 200, 0.03]
    assert score_svg._path_contours("m.5-.25 2,3 h2 v-1 H10 V2") == [
        [(0.5, -0.25), (2.5, 2.75), (4.5, 2.75), (4.5, 1.75), (10, 1.75), (10, 2)]
    ]


def test_subdivided_baseline_closure_does_not_distort_signal():
    vertices = [(50, 250)] + points() + [(650, 250), (450, 250), (250, 250)]
    for offset in range(len(vertices)):
        rotated = vertices[offset:] + vertices[:offset]
        geometry = f'<polygon points="{point_text(rotated)}"/>'
        result = score_svg.score_svg_bytes(svg(geometry), REFERENCE)
        assert result.score == 1.0, result.notes


def test_colored_overlapping_bars_match_their_actual_silhouette():
    root = ET.fromstring(svg(signal("bars")))
    rectangles = root.findall(f"{{{score_svg.SVG_NS}}}rect")
    for index, rect in enumerate(rectangles):
        rect.set("fill", "red" if index % 2 else "green")
        rect.set("width", "25.5")
    result = score_svg.score_svg_bytes(ET.tostring(root), REFERENCE)
    assert result.score == 1.0, result.notes
    contour = score_svg._bar_contour([(0, 2, 8), (1, 3, 4)] * 6, 10, "up")
    assert contour == [(0, 8), (1, 8), (1, 4), (2, 4), (2, 4), (3, 4)]


def test_omitted_zero_height_bars_match_zero_signal_gaps():
    profile = [0 if index % 4 == 1 else value for index, value in enumerate(PROFILE)]
    root = ET.fromstring(svg(signal("bars", profile)))
    for rect in list(root):
        if rect.get("height") == "0":
            root.remove(rect)
    reference = {**REFERENCE, "signal_profile": profile}
    result = score_svg.score_svg_bytes(ET.tostring(root), reference)
    assert result.score == 1.0, result.notes
    assert "raw=1.000, detrended=1.000, delta=1.000" in result.notes[2]


@pytest.mark.parametrize("kind", ["path", "polyline", "polygon", "area", "bars", "step-path"])
@pytest.mark.parametrize(
    "wrong",
    [
        [30] * 24,
        list(range(10, 106, 4)),
        PROFILE[::-1],
        PROFILE[7:] + PROFILE[:7],
        [50 + 35 * math.sin(index * 0.8) for index in range(24)],
    ],
)
def test_wrong_signal_profiles_do_not_pass(kind, wrong):
    result = score_svg.score_svg_bytes(svg(signal(kind, wrong)), REFERENCE)
    assert result.score == 0.8, result.notes
    assert not result.checks["graphical_evidence"]


@pytest.mark.parametrize("mutation", ["baseline", "overlap", "gaps", "varying-widths"])
def test_unrelated_rectangles_cannot_supply_a_signal(mutation):
    root = ET.fromstring(svg(signal("bars")))
    for index, rect in enumerate(root.findall(f"{{{score_svg.SVG_NS}}}rect")):
        if mutation == "baseline":
            rect.set("height", "10")
        elif mutation == "overlap":
            rect.set("x", "50")
        elif mutation == "gaps":
            rect.set("x", str(50 + 25 * index**1.5))
        else:
            rect.set("width", "1000" if index == 0 else "0.01")
    result = score_svg.score_svg_bytes(ET.tostring(root), REFERENCE)
    assert not result.checks["graphical_evidence"], result.notes


def test_decorative_rectangles_and_polygons_do_not_combine_into_a_signal():
    geometry = '<rect width="800" height="400" fill="white"/>'
    geometry += '<rect x="50" y="100" width="600" height="200" fill="none" stroke="gray"/>'
    for xpos, ypos in points():
        geometry += f'<polygon points="{xpos},{ypos} {xpos + 4},{ypos - 4} {xpos + 8},{ypos}"/>'
        geometry += f'<rect x="{xpos}" y="{ypos}" width="8" height="8"/>'
    result = score_svg.score_svg_bytes(svg(geometry), REFERENCE)
    assert result.score == 0.8
    assert not result.checks["graphical_evidence"]


def test_nonmonotone_polygon_does_not_get_sorted_into_the_expected_signal():
    vertices = points()
    random.Random(7).shuffle(vertices)
    geometry = f'<polygon points="{point_text(vertices)}"/>'
    result = score_svg.score_svg_bytes(svg(geometry), REFERENCE)
    assert not result.checks["graphical_evidence"]


@pytest.mark.parametrize(
    "malformed",
    [
        '<polyline points="{points} NaN,12"/>',
        '<polygon points="{points} 1e999,12"/>',
        '<polyline points="{points} 1"/>',
        '<polygon points="{points} garbage"/>',
        '<polyline points="{points},,12,13"/>',
        '<path d="M {points} L"/>',
        '<path d="M {points} L NaN 12"/>',
        '<path d="M {points} X 10 20"/>',
        '<path d="M {points} Z 1 2"/>',
        '<path d="M,{points}"/>',
        '<path d="L {points}"/>',
        '<polyline transform="translate(NaN)" points="{points}"/>',
        '<polyline transform="translate(10)garbage" points="{points}"/>',
        '<g transform="scale(1e999)"><polyline points="{points}"/></g>',
        '<polyline transform="matrix(1 0 0 1 0)" points="{points}"/>',
        '<polyline transform="scale(1e12) scale(1e12)" points="{points}"/>',
        '<polyline transform="scale(0)" points="{points}"/>',
    ],
)
def test_malformed_and_nonfinite_geometry_fails_safely(malformed):
    result = score_svg.score_svg_bytes(
        svg(malformed.format(points=point_text(points()))), REFERENCE
    )
    assert result.score == 0.8
    assert not result.checks["graphical_evidence"]


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("height", "NaN"),
        ("width", "inf"),
        ("x", "1e999"),
        ("y", "12pxjunk"),
        ("height", "-20"),
        ("width", "0"),
    ],
)
def test_malformed_bars_fail_safely(attribute, value):
    root = ET.fromstring(svg(signal("bars")))
    for rect in root.findall(f"{{{score_svg.SVG_NS}}}rect"):
        rect.set(attribute, value)
    result = score_svg.score_svg_bytes(ET.tostring(root), REFERENCE)
    assert not result.checks["graphical_evidence"]


@pytest.mark.parametrize(
    "container",
    [
        "<defs>{}</defs>",
        "<clipPath>{}</clipPath>",
        "<symbol>{}</symbol>",
        '<g display="none">{}</g>',
        '<g style="opacity:0">{}</g>',
        '<g visibility="hidden">{}</g>',
        '<g fill="none" stroke="none">{}</g>',
        '<g fill-opacity="0">{}</g>',
        '<g style="fill-opacity:NaN">{}</g>',
        "<script>{}</script>",
        "<metadata>{}</metadata>",
    ],
)
def test_nonrendered_geometry_is_not_a_candidate(container):
    result = score_svg.score_svg_bytes(svg(container.format(signal("area"))), REFERENCE)
    assert not result.checks["graphical_evidence"]


def test_disconnected_path_subpaths_are_not_stitched_into_a_signal():
    geometry = '<path d="' + " ".join(f"M{xpos},{ypos}h1" for xpos, ypos in points()) + '"/>'
    result = score_svg.score_svg_bytes(svg(geometry), REFERENCE)
    assert not result.checks["graphical_evidence"]


@pytest.mark.parametrize("scale", ["scale(.1 1)", "scale(1 .1)"])
def test_original_signal_extent_thresholds_remain(scale):
    result = score_svg.score_svg_bytes(
        svg(f'<g transform="{scale}">{signal("area")}</g>'), REFERENCE
    )
    assert not result.checks["graphical_evidence"]


def test_original_candidate_limit_remains():
    result = score_svg.score_svg_bytes(svg(signal("area") * 21), REFERENCE)
    assert not result.checks["graphical_evidence"]
    assert "signal candidates=21" in result.notes[2]


@pytest.mark.parametrize(
    "labels,missing",
    [
        (LABELS.replace("chr17", "chr18"), "coordinate_evidence"),
        (LABELS.replace("7571651 7580000 7590910", "10000 11000 12000"), "coordinate_evidence"),
        (LABELS.replace("structural variants VCF", ""), "track_evidence"),
        (LABELS.replace("K562 H3K27ac BigWig", ""), "track_evidence"),
        (LABELS.replace("hg19", ""), "browser_provenance"),
    ],
)
def test_correct_geometry_does_not_bypass_content_checks(labels, missing):
    result = score_svg.score_svg_bytes(svg(signal("area"), labels), REFERENCE)
    assert result.checks["graphical_evidence"]
    assert not result.checks[missing]
    assert result.score <= 0.8


def test_svg_content_is_never_executed_or_fetched(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("SVG scoring must not execute processes or access the network")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    marker = tmp_path / "executed"
    script = f'<script>require("fs").writeFileSync("{marker}", "bad")</script>'
    geometry = '<image href="https://invalid.example/signal.svg"/>' + script + signal("area")
    assert score_svg.score_svg_bytes(svg(geometry), REFERENCE).score == 1.0
    assert not marker.exists()
    external = (
        f'<!DOCTYPE svg [<!ENTITY external SYSTEM "file://{marker}">]><svg>&external;</svg>'
    ).encode()
    assert not score_svg.score_svg_bytes(external, REFERENCE).checks["valid_svg"]


@pytest.mark.parametrize("payload", [b"", b"<svg>", b"<html/>", b"not XML"])
def test_invalid_svg_never_receives_full_credit(payload):
    result = score_svg.score_svg_bytes(payload, REFERENCE)
    assert result.score <= 0.1
    assert not result.checks["valid_svg"]


@pytest.mark.parametrize("variant_index", range(5))
def test_signal_serialization_contract_is_visible_and_aligned_for_every_variant(variant_index):
    from tasks.life_sciences.tp53_locus_variant_histone_browser_svg import main

    config = main._cfg_for_variant(main.VARIANTS[variant_index])
    description = config.task_description
    card = json.loads(Path(main.__file__).with_name("task_card.json").read_text())
    marker = "\nSignal track serialization:\n"
    assert marker in description
    assert description.split(marker)[1] == card["taskPrompt"].split(marker)[1]
    contract = description.split(marker)[1]
    for required in (
        "x-monotone",
        "<polyline>",
        "<path>",
        "M/L/H/V",
        "absolute or relative",
        "final Z",
        "<polygon>",
        "horizontal baseline",
        "<rect>",
        "same explicit `stroke`",
        "12 contour samples or 12 bars",
        "400 units horizontally and 30 vertically",
        "8 signal heights",
        "0.1-unit",
        "finite, unitless",
        "transform",
        "axis-aligned",
        "viewport",
        "root `viewBox`",
        "self-contained",
        "inline `style`",
        "Bezier/arc",
        "C/S/Q/T/A",
        "<use>",
        "<symbol>",
        "<defs>",
        "raster images",
        "CSS stylesheets/classes/variables",
        "scripts",
        "filters",
        "masks",
        "clipping",
        "occlusion",
        "not to unrelated text, fonts, labels",
        "GUI workflow remain unrestricted",
        "post-process an export",
        "without changing its scientific content",
    ):
        assert required in contract
    assert f"{config.CHROM}:{config.START}-{config.END}" in description
    assert config.GENE in description
    assert config.vcf_path in description and config.bigwig_path in description
    assert config.output_svg_path in description
    assert "UCSC Genome Browser, IGV, WashU Epigenome Browser" in description
    assert "filled polygon, or aligned rectangle bars" in card["evaluation"]


@pytest.mark.parametrize("kind", ["path", "polyline", "area", "bars"])
def test_inline_signal_and_unrelated_svg_features_remain_compatible(kind):
    decorative = (
        "<style>.label {font:14px serif}</style>"
        '<defs><path id="glyph" d="M0,0 C1,2 3,4 5,6 Q7,8 9,10 A2,2 0 0,0 12,12"/>'
        '<clipPath id="label-clip"><rect width="40" height="20"/></clipPath></defs>'
        '<text class="label" clip-path="url(#label-clip)"><tspan>TP53</tspan></text>'
        '<use href="#glyph" x="20" y="20"/>'
    )
    geometry = (
        '<g transform="translate(20 10)" style="fill:green;stroke:none;opacity:1">'
        f"{signal(kind)}</g>"
    )
    result = score_svg.score_svg_bytes(svg(decorative + geometry), REFERENCE)
    assert result.score == 1.0, result.notes
    assert all(result.checks.values())
