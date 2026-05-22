"""Local scorer for business_finance/saas_onepager_brand_refresh_instance_1."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def grayscale_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def resize_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    return a[:height, :width], b[:height, :width]


def simple_ssim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = resize_to_match(a, b)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    denom = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if denom == 0:
        return 0.0
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / denom)


def region_score(sub_png: Path, ref_png: Path, regions_json: Path) -> float:
    submission = Image.open(sub_png).convert("L")
    reference = Image.open(ref_png).convert("L")
    regions = load_json(regions_json)["regions"]
    scores: list[float] = []
    for region in regions:
        box = (
            region["x"],
            region["y"],
            region["x"] + region["w"],
            region["y"] + region["h"],
        )
        sub_arr = np.asarray(submission.crop(box), dtype=np.float32) / 255.0
        ref_arr = np.asarray(reference.crop(box), dtype=np.float32) / 255.0
        scores.append(max(0.0, min(1.0, simple_ssim(sub_arr, ref_arr))))
    return float(np.mean(scores)) if scores else 0.0


def normalize_text(text: str) -> str:
    lowered = text.lower().replace("\n", " ")
    lowered = re.sub(r"-\s+", "-", lowered)
    squashed = re.sub(r"\s+", " ", lowered)
    sanitized = re.sub(r"[^a-z0-9%$./&+\- ]+", "", squashed)
    return sanitized.strip()


def phrase_list_from_expected_text(data: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    for value in data.values():
        if isinstance(value, str):
            phrases.append(value)
        elif isinstance(value, list):
            phrases.extend(item for item in value if isinstance(item, str))
    return phrases


def text_presence_score(all_text: list[str], expected_phrases: list[str]) -> float:
    haystack = normalize_text(" ".join(all_text))
    if not expected_phrases:
        return 1.0
    hits = sum(1 for phrase in expected_phrases if normalize_text(phrase) in haystack)
    return hits / len(expected_phrases)


def numeric_presence_score(all_text: list[str], data: dict[str, Any]) -> float:
    haystack = normalize_text(" ".join(all_text))
    expected: list[str] = []
    for kpi in data.get("kpis", []):
        expected.extend([kpi.get("label", ""), kpi.get("value", "")])
    for row in data.get("pricing", []):
        expected.extend([row.get("plan", ""), row.get("price", "")])
    expected = [item for item in expected if item]
    if not expected:
        return 1.0
    hits = sum(1 for phrase in expected if normalize_text(phrase) in haystack)
    return hits / len(expected)


def _resolve_slide_target(target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join("ppt/slides", target))


def _joined_texts(node: ET.Element) -> list[str]:
    texts: list[str] = []
    for paragraph in node.findall(".//a:p", NS):
        paragraph_text = "".join(run.text or "" for run in paragraph.findall(".//a:t", NS)).strip()
        if paragraph_text:
            texts.append(paragraph_text)
    return texts


def _read_transform(node: ET.Element, xfrm_path: str) -> tuple[int, int, int, int] | None:
    xfrm = node.find(xfrm_path, NS)
    if xfrm is None:
        return None
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    if off is None or ext is None:
        return None
    try:
        x = int(off.attrib["x"])
        y = int(off.attrib["y"])
        w = int(ext.attrib["cx"])
        h = int(ext.attrib["cy"])
    except Exception:
        return None
    return x, y, w, h


def _visible_area(box: tuple[int, int, int, int], slide_width: int, slide_height: int) -> int:
    x, y, w, h = box
    left = max(0, x)
    top = max(0, y)
    right = min(slide_width, x + w)
    bottom = min(slide_height, y + h)
    visible_width = max(0, right - left)
    visible_height = max(0, bottom - top)
    return visible_width * visible_height


def _clip_box_to_slide(
    box: tuple[int, int, int, int] | None,
    slide_width: int,
    slide_height: int,
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x, y, w, h = box
    left = max(0, x)
    top = max(0, y)
    right = min(slide_width, x + w)
    bottom = min(slide_height, y + h)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = rect
    return max(0, right - left) * max(0, bottom - top)


def _intersect_rect(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if a is None or b is None:
        return None
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _rect_union_area(rects: list[tuple[int, int, int, int]]) -> int:
    if not rects:
        return 0
    xs = sorted({rect[0] for rect in rects} | {rect[2] for rect in rects})
    area = 0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals: list[tuple[int, int]] = []
        for rect_left, rect_top, rect_right, rect_bottom in rects:
            if rect_left < right and rect_right > left:
                intervals.append((rect_top, rect_bottom))
        if not intervals:
            continue
        intervals.sort()
        merged_top, merged_bottom = intervals[0]
        covered_height = 0
        for top, bottom in intervals[1:]:
            if top > merged_bottom:
                covered_height += merged_bottom - merged_top
                merged_top, merged_bottom = top, bottom
            else:
                merged_bottom = max(merged_bottom, bottom)
        covered_height += merged_bottom - merged_top
        area += (right - left) * covered_height
    return area


def _visible_area_ratio(
    box: tuple[int, int, int, int] | None,
    slide_width: int,
    slide_height: int,
    slide_area: int,
) -> float | None:
    if box is None:
        return None
    return _visible_area(box, slide_width, slide_height) / slide_area


def _is_on_canvas(box: tuple[int, int, int, int] | None, slide_width: int, slide_height: int) -> bool:
    if box is None:
        return True
    return _visible_area(box, slide_width, slide_height) > 0


def _counts_as_visible(
    box: tuple[int, int, int, int] | None,
    slide_width: int,
    slide_height: int,
    slide_area: int,
    min_area_ratio: float,
) -> bool:
    if box is None:
        return True
    ratio = _visible_area_ratio(box, slide_width, slide_height, slide_area)
    return ratio is not None and ratio >= min_area_ratio


def _uncovered_area_metrics(
    box: tuple[int, int, int, int] | None,
    cover_boxes: list[tuple[int, int, int, int]],
    slide_width: int,
    slide_height: int,
    slide_area: int,
) -> tuple[float, float] | None:
    clipped_box = _clip_box_to_slide(box, slide_width, slide_height)
    if clipped_box is None:
        return None
    target_area = _rect_area(clipped_box)
    if target_area == 0:
        return 0.0, 0.0
    clipped_covers = [
        overlap
        for overlap in (
            _intersect_rect(clipped_box, _clip_box_to_slide(cover_box, slide_width, slide_height))
            for cover_box in cover_boxes
        )
        if overlap is not None
    ]
    covered_area = min(target_area, _rect_union_area(clipped_covers))
    uncovered_area = max(0, target_area - covered_area)
    return uncovered_area / slide_area, uncovered_area / target_area


def _counts_as_uncovered_visible(
    box: tuple[int, int, int, int] | None,
    cover_boxes: list[tuple[int, int, int, int]],
    slide_width: int,
    slide_height: int,
    slide_area: int,
    min_area_ratio: float,
    min_unobscured_fraction: float,
) -> bool:
    if box is None:
        return True
    metrics = _uncovered_area_metrics(box, cover_boxes, slide_width, slide_height, slide_area)
    if metrics is None:
        return False
    uncovered_slide_ratio, uncovered_fraction = metrics
    return uncovered_slide_ratio >= min_area_ratio and uncovered_fraction >= min_unobscured_fraction


def _shape_has_opaque_fill(element: ET.Element) -> bool:
    sp_pr = element.find("p:spPr", NS)
    if sp_pr is None:
        return False
    if sp_pr.find("a:noFill", NS) is not None:
        return False
    return any(
        sp_pr.find(path, NS) is not None
        for path in ("a:solidFill", "a:gradFill", "a:blipFill", "a:pattFill")
    )


def _object_can_occlude(kind: str, element: ET.Element) -> bool:
    if kind == "pic":
        return True
    if kind == "sp":
        return _shape_has_opaque_fill(element)
    return False


@dataclass(frozen=True)
class GroupTransform:
    off_x: int
    off_y: int
    ext_x: int
    ext_y: int
    child_off_x: int
    child_off_y: int
    child_ext_x: int
    child_ext_y: int


def _read_group_transform(node: ET.Element) -> GroupTransform | None:
    xfrm = node.find("p:grpSpPr/a:xfrm", NS)
    if xfrm is None:
        return None
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    child_off = xfrm.find("a:chOff", NS)
    child_ext = xfrm.find("a:chExt", NS)
    if off is None or ext is None or child_off is None or child_ext is None:
        return None
    try:
        return GroupTransform(
            off_x=int(off.attrib["x"]),
            off_y=int(off.attrib["y"]),
            ext_x=int(ext.attrib["cx"]),
            ext_y=int(ext.attrib["cy"]),
            child_off_x=int(child_off.attrib["x"]),
            child_off_y=int(child_off.attrib["y"]),
            child_ext_x=int(child_ext.attrib["cx"]),
            child_ext_y=int(child_ext.attrib["cy"]),
        )
    except Exception:
        return None


def _apply_group_transform(
    box: tuple[int, int, int, int] | None,
    group_transform: GroupTransform,
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    if group_transform.child_ext_x == 0 or group_transform.child_ext_y == 0:
        return None
    x, y, w, h = box
    scale_x = group_transform.ext_x / group_transform.child_ext_x
    scale_y = group_transform.ext_y / group_transform.child_ext_y
    return (
        int(round(group_transform.off_x + (x - group_transform.child_off_x) * scale_x)),
        int(round(group_transform.off_y + (y - group_transform.child_off_y) * scale_y)),
        int(round(w * scale_x)),
        int(round(h * scale_y)),
    )


def _compose_box_with_groups(
    box: tuple[int, int, int, int] | None,
    group_transforms: list[GroupTransform],
) -> tuple[int, int, int, int] | None:
    world_box = box
    for group_transform in reversed(group_transforms):
        world_box = _apply_group_transform(world_box, group_transform)
    return world_box


def _iter_slide_objects(
    node: ET.Element,
    group_transforms: list[GroupTransform],
):
    for child in list(node):
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "grpSp":
            group_transform = _read_group_transform(child)
            next_groups = group_transforms + ([group_transform] if group_transform is not None else [])
            yield from _iter_slide_objects(child, next_groups)
        elif tag == "sp":
            yield (
                "sp",
                child,
                _compose_box_with_groups(_read_transform(child, "p:spPr/a:xfrm"), group_transforms),
            )
        elif tag == "pic":
            yield (
                "pic",
                child,
                _compose_box_with_groups(_read_transform(child, "p:spPr/a:xfrm"), group_transforms),
            )
        elif tag == "graphicFrame":
            yield (
                "graphicFrame",
                child,
                _compose_box_with_groups(_read_transform(child, "p:xfrm"), group_transforms),
            )


def inspect_pptx(pptx_path: Path, constraints: dict[str, Any]) -> dict[str, Any]:
    with zipfile.ZipFile(pptx_path, "r") as archive:
        slide_paths = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )
        result: dict[str, Any] = {
            "slides": len(slide_paths),
            "text": 0,
            "on_canvas_text": 0,
            "images": 0,
            "on_canvas_images": 0,
            "tables": 0,
            "on_canvas_tables": 0,
            "charts": 0,
            "on_canvas_charts": 0,
            "single_image_fullslide": False,
            "max_image_coverage_ratio": 0.0,
            "total_image_coverage_ratio": 0.0,
            "all_text": [],
            "chart_data": [],
        }
        if not slide_paths:
            return result

        presentation_root = ET.fromstring(archive.read("ppt/presentation.xml"))
        slide_size = presentation_root.find("p:sldSz", NS)
        slide_width = int(slide_size.attrib["cx"]) if slide_size is not None else 1
        slide_height = int(slide_size.attrib["cy"]) if slide_size is not None else 1
        slide_area = slide_width * slide_height

        slide_path = slide_paths[0]
        rels_path = slide_path.replace("slides/", "slides/_rels/") + ".rels"
        rels_root = ET.fromstring(archive.read(rels_path))
        rel_targets = {
            rel.attrib["Id"]: _resolve_slide_target(rel.attrib["Target"])
            for rel in rels_root.findall("rel:Relationship", NS)
        }

        slide_root = ET.fromstring(archive.read(slide_path))
        shape_tree = slide_root.find(".//p:spTree", NS)
        max_area_ratio = 0.0
        min_text_box_area_ratio = constraints.get("min_text_box_area_ratio", 0.0)
        min_image_box_area_ratio = constraints.get("min_image_box_area_ratio", 0.0)
        min_table_box_area_ratio = constraints.get("min_table_box_area_ratio", 0.0)
        min_chart_box_area_ratio = constraints.get("min_chart_box_area_ratio", 0.0)
        min_text_unobscured_fraction = constraints.get("min_text_unobscured_fraction", 0.0)
        min_image_unobscured_fraction = constraints.get("min_image_unobscured_fraction", 0.0)
        min_table_unobscured_fraction = constraints.get("min_table_unobscured_fraction", 0.0)
        min_chart_unobscured_fraction = constraints.get("min_chart_unobscured_fraction", 0.0)
        if shape_tree is None:
            return result

        objects = list(_iter_slide_objects(shape_tree, []))
        image_cover_rects = [
            clipped_box
            for kind, _, world_box in objects
            if kind == "pic"
            for clipped_box in [_clip_box_to_slide(world_box, slide_width, slide_height)]
            if clipped_box is not None
        ]

        for index, (kind, element, world_box) in enumerate(objects):
            later_cover_boxes = [
                later_box
                for later_kind, later_element, later_box in objects[index + 1 :]
                if _object_can_occlude(later_kind, later_element)
                and _is_on_canvas(later_box, slide_width, slide_height)
            ]
            if kind == "sp":
                tx_body = element.find("p:txBody", NS)
                if tx_body is None:
                    continue
                shape_texts = _joined_texts(tx_body)
                if not shape_texts:
                    continue
                result["text"] += 1
                if _counts_as_uncovered_visible(
                    world_box,
                    later_cover_boxes,
                    slide_width,
                    slide_height,
                    slide_area,
                    min_text_box_area_ratio,
                    min_text_unobscured_fraction,
                ):
                    result["on_canvas_text"] += 1
                    result["all_text"].extend(shape_texts)
            elif kind == "pic":
                result["images"] += 1
                if not _counts_as_uncovered_visible(
                    world_box,
                    later_cover_boxes,
                    slide_width,
                    slide_height,
                    slide_area,
                    min_image_box_area_ratio,
                    min_image_unobscured_fraction,
                ):
                    continue
                result["on_canvas_images"] += 1
                if world_box is None:
                    continue
                area_ratio = _visible_area_ratio(world_box, slide_width, slide_height, slide_area)
                if area_ratio is None:
                    continue
                max_area_ratio = max(max_area_ratio, area_ratio)
            elif kind == "graphicFrame":
                graphic_data = element.find(".//a:graphicData", NS)
                if graphic_data is None:
                    continue
                uri = graphic_data.attrib.get("uri", "")
                if uri.endswith("/table"):
                    result["tables"] += 1
                    if _counts_as_uncovered_visible(
                        world_box,
                        later_cover_boxes,
                        slide_width,
                        slide_height,
                        slide_area,
                        min_table_box_area_ratio,
                        min_table_unobscured_fraction,
                    ):
                        result["on_canvas_tables"] += 1
                        for cell in element.findall(".//a:tc", NS):
                            cell_text = "".join(text.text or "" for text in cell.findall(".//a:t", NS)).strip()
                            if cell_text:
                                result["all_text"].append(cell_text)
                elif uri.endswith("/chart"):
                    result["charts"] += 1
                    if not _counts_as_uncovered_visible(
                        world_box,
                        later_cover_boxes,
                        slide_width,
                        slide_height,
                        slide_area,
                        min_chart_box_area_ratio,
                        min_chart_unobscured_fraction,
                    ):
                        continue
                    result["on_canvas_charts"] += 1
                    chart_ref = element.find(".//c:chart", NS)
                    if chart_ref is None:
                        continue
                    rel_id = chart_ref.attrib.get(f"{{{NS['r']}}}id", "")
                    target = rel_targets.get(rel_id)
                    if not target:
                        continue
                    chart_root = ET.fromstring(archive.read(target))
                    series_name = ""
                    categories: list[str] = []
                    values: list[float] = []
                    series = chart_root.find(".//c:ser", NS)
                    if series is not None:
                        name_nodes = series.findall(".//c:tx//c:v", NS)
                        if name_nodes:
                            series_name = name_nodes[0].text or ""
                        categories = [node.text or "" for node in series.findall(".//c:cat//c:v", NS)]
                        values = [
                            float(node.text or "0")
                            for node in series.findall(".//c:val//c:v", NS)
                        ]
                    result["chart_data"].append(
                        {
                            "categories": categories,
                            "series": [{"name": series_name, "values": values}],
                        }
                    )

        result["single_image_fullslide"] = (
            max_area_ratio > 0.92
            and result["on_canvas_images"] == 1
            and result["on_canvas_text"] == 0
            and result["on_canvas_tables"] == 0
            and result["on_canvas_charts"] == 0
        )
        result["max_image_coverage_ratio"] = max_area_ratio
        result["total_image_coverage_ratio"] = _rect_union_area(image_cover_rects) / slide_area
        return result


def chart_score(ppt_info: dict[str, Any], expected_chart: dict[str, Any]) -> float:
    if not ppt_info["chart_data"]:
        return 0.0
    chart = ppt_info["chart_data"][0]
    categories_match = chart["categories"] == expected_chart.get("x", [])
    if not chart["series"]:
        return 0.0
    values_match = [float(value) for value in chart["series"][0]["values"]] == [
        float(value) for value in expected_chart.get("y", [])
    ]
    expected_name = expected_chart.get("series_name", "")
    if expected_name:
        name_match = normalize_text(chart["series"][0].get("name", "")) == normalize_text(expected_name)
    else:
        name_match = True
    return 1.0 if (categories_match and values_match and name_match) else 0.0


def structure_score(ppt_info: dict[str, Any], constraints: dict[str, Any]) -> float:
    max_image_coverage_ratio = constraints.get("max_single_image_coverage_ratio", 0.92)
    max_total_image_coverage_ratio = constraints.get("max_total_image_coverage_ratio", 1.0)
    max_slide_count = constraints.get("max_slide_count")
    ok = (
        ppt_info["slides"] >= constraints.get("min_slide_count", 1)
        and (max_slide_count is None or ppt_info["slides"] <= max_slide_count)
        and ppt_info["on_canvas_text"] >= constraints.get("min_text_shapes", 0)
        and ppt_info["on_canvas_images"] >= constraints.get("min_image_shapes", 0)
        and ppt_info["on_canvas_tables"] >= constraints.get("min_table_shapes", 0)
        and ppt_info["on_canvas_charts"] >= constraints.get("min_chart_shapes", 0)
        and ppt_info["max_image_coverage_ratio"] <= max_image_coverage_ratio
        and ppt_info["total_image_coverage_ratio"] <= max_total_image_coverage_ratio
        and not (
            constraints.get("forbid_single_fullslide_raster_only", True)
            and ppt_info["single_image_fullslide"]
        )
    )
    return 1.0 if ok else 0.0


def score_output(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    submission_png = output_dir / "edited_onepager.png"
    submission_pptx = output_dir / "edited_onepager.pptx"
    reference_png = reference_dir / "edited_onepager.png"

    if not submission_png.exists() or not submission_pptx.exists():
        return {"pass": False, "reason": "missing required outputs"}

    thresholds = load_json(reference_dir / "evaluation_thresholds.json")
    constraints = load_json(reference_dir / "structural_constraints.json")

    try:
        sub_img = Image.open(submission_png)
        ref_img = Image.open(reference_png)
    except Exception as exc:
        return {"pass": False, "reason": f"png unreadable: {exc}"}

    resolution_ratio = thresholds["hard_gates"]["min_resolution_ratio_per_dimension"]
    if sub_img.width < ref_img.width * resolution_ratio or sub_img.height < ref_img.height * resolution_ratio:
        return {"pass": False, "reason": "resolution below hard gate"}

    submission_gray = grayscale_array(submission_png)
    if float(submission_gray.std()) < thresholds["hard_gates"]["min_png_stddev"]:
        return {"pass": False, "reason": "png appears blank or near-solid-color"}

    try:
        ppt_info = inspect_pptx(submission_pptx, constraints)
    except Exception as exc:
        return {"pass": False, "reason": f"pptx unreadable: {exc}"}

    if ppt_info["slides"] == 0:
        return {"pass": False, "reason": "pptx has no slides"}
    max_slide_count = constraints.get("max_slide_count")
    if max_slide_count is not None and ppt_info["slides"] > max_slide_count:
        return {"pass": False, "reason": f"pptx exceeds max slide count ({ppt_info['slides']} > {max_slide_count})"}
    if ppt_info["single_image_fullslide"]:
        return {"pass": False, "reason": "pptx is a single full-slide raster only"}

    ssim_score = max(0.0, min(1.0, simple_ssim(submission_gray, grayscale_array(reference_png))))
    edited_region_score = max(
        0.0,
        min(1.0, region_score(submission_png, reference_png, reference_dir / "edited_regions.json")),
    )
    expected_text = load_json(reference_dir / "expected_text_fields.json")
    expected_numeric = load_json(reference_dir / "expected_numeric_fields.json")
    expected_chart = load_json(reference_dir / "expected_chart_data.json")

    text_score = text_presence_score(ppt_info["all_text"], phrase_list_from_expected_text(expected_text))
    numeric_score = numeric_presence_score(ppt_info["all_text"], expected_numeric)
    chart_score_value = chart_score(ppt_info, expected_chart)
    structure_score_value = structure_score(ppt_info, constraints)

    weights = thresholds["weights"]
    final_score = (
        weights["ssim"] * ssim_score
        + weights["edited_region_score"] * edited_region_score
        + weights["text_score"] * text_score
        + weights["numeric_score"] * numeric_score
        + weights["chart_score"] * chart_score_value
        + weights["structure_score"] * structure_score_value
    )

    mins = thresholds["component_thresholds"]
    component_thresholds_pass = (
        ssim_score >= mins["ssim_min"]
        and edited_region_score >= mins["edited_region_score_min"]
        and text_score >= mins["text_score_min"]
        and numeric_score >= mins["numeric_score_min"]
        and chart_score_value >= mins["chart_score_min"]
        and structure_score_value >= mins["structure_score_min"]
    )
    passed = component_thresholds_pass and final_score >= thresholds["final_pass_threshold"]

    return {
        "pass": bool(passed),
        "scores": {
            "ssim_score": round(ssim_score, 4),
            "edited_region_score": round(edited_region_score, 4),
            "text_score": round(text_score, 4),
            "numeric_score": round(numeric_score, 4),
            "chart_score": round(chart_score_value, 4),
            "structure_score": round(structure_score_value, 4),
            "final_score": round(final_score, 4),
        },
        "thresholds": thresholds,
        "ppt_structure": {
            "slides": ppt_info["slides"],
            "text": ppt_info["text"],
            "on_canvas_text": ppt_info["on_canvas_text"],
            "images": ppt_info["images"],
            "on_canvas_images": ppt_info["on_canvas_images"],
            "tables": ppt_info["tables"],
            "on_canvas_tables": ppt_info["on_canvas_tables"],
            "charts": ppt_info["charts"],
            "on_canvas_charts": ppt_info["on_canvas_charts"],
            "max_image_coverage_ratio": round(float(ppt_info["max_image_coverage_ratio"]), 4),
            "total_image_coverage_ratio": round(float(ppt_info["total_image_coverage_ratio"]), 4),
            "single_image_fullslide": ppt_info["single_image_fullslide"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    result = score_output(Path(args.output_dir), Path(args.reference_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
