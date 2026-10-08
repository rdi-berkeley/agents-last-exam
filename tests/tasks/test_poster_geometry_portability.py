from xml.etree import ElementTree as ET

import pytest

from tasks.visual_media.inkscape_cultural_poster_design.main import evaluate_svg_bytes
from tasks.visual_media.inkscape_cultural_poster_design.scripts.svg_geometry import image_corners_mm

SPEC = {
    "canvas_width_mm": 297,
    "canvas_height_mm": 420,
    "orientation": "portrait",
    "required_title": "Title",
    "required_subtitle": "Subtitle",
    "required_phrases": ["Phrase"],
    "min_required_phrase_count": 1,
}


def poster(image, *, viewbox="0 0 297 420", extra=""):
    return (
        f'<svg width="297mm" height="420mm" viewBox="{viewbox}" {extra}>'
        f"<text>Title Subtitle Phrase</text>{image}</svg>"
    ).encode()


@pytest.mark.parametrize(
    "transform",
    [
        "scale(0.2)",
        "matrix(.2,0,0,.2,0,0)",
        "translate(10,10) scale(.2)",
        "rotate(10,100,100) scale(.2)",
    ],
)
def test_parent_transform_is_applied(transform):
    svg = poster(
        f'<g transform="{transform}"><image x="100" y="100" width="600" height="700" '
        'preserveAspectRatio="xMidYMid meet"/></g>'
    )
    score, details = evaluate_svg_bytes(svg, SPEC)
    assert score == 1, details


@pytest.mark.parametrize(
    "transform",
    [
        "scale(0)",
        "translate(1000,0)",
        "scale(NaN)",
        "matrix(1,2,3)",
        "skewX(90)",
        "scale(1e999)",
        "scale(.2) junk",
        ",scale(.2)",
    ],
)
def test_bad_or_offcanvas_geometry_is_rejected(transform):
    svg = poster(
        f'<g transform="{transform}"><image width="600" height="700" '
        'preserveAspectRatio="xMidYMid meet"/></g>'
    )
    assert evaluate_svg_bytes(svg, SPEC)[0] == 0


def test_nested_affine_transforms_match_hand_computed_corners():
    root = ET.fromstring(
        poster(
            '<g transform="translate(10,20)"><g transform="scale(2,3)">'
            '<image x="1" y="2" width="4" height="5"/></g></g>'
        )
    )
    corners = image_corners_mm(root, root.find(".//image"), 297, 420)
    assert corners == [(12, 26), (20, 26), (20, 41), (12, 41)]


def test_viewbox_offset_and_units():
    root = ET.fromstring(
        poster('<image x="11" y="22" width="4" height="5"/>', viewbox="10 20 297 420")
    )
    assert image_corners_mm(root, root.find("image"), 297, 420) == [(1, 2), (5, 2), (5, 7), (1, 7)]


@pytest.mark.parametrize("defect", ["title", "size", "aspect", "css", "nested"])
def test_non_geometry_requirements_and_unsupported_layouts_remain_gated(defect):
    svg = poster('<image width="10" height="20" preserveAspectRatio="xMidYMid meet"/>').decode()
    if defect == "title":
        svg = svg.replace("Title", "Absent")
    elif defect == "size":
        svg = svg.replace("297mm", "500mm")
    elif defect == "aspect":
        svg = svg.replace("xMidYMid meet", "none")
    elif defect == "css":
        svg = svg.replace("<image", '<image style="transform:scale(.5)"')
    else:
        svg = svg.replace("<image", "<svg><image").replace("/></svg>", "/></svg></svg>")
    assert evaluate_svg_bytes(svg.encode(), SPEC)[0] == 0
