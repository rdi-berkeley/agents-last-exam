"""Stage 2 implementation for health_medicine/wsi_tumor_localization_1."""

import json
import math
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig



_setup = BaseTaskSetup()

DOMAIN_NAME = "health_medicine"
TASK_NAME = "wsi_tumor_localization_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"


VARIANTS: tuple[dict[str, str], ...] = (
    {
        "name": "center_point",
        "kind": "center_point",
        "title": "Tumor Center Coordinate",
    },
    {
        "name": "bounding_box",
        "kind": "bounding_box",
        "title": "Tumor Bounding Box",
    },
    {
        "name": "grid_cells",
        "kind": "grid_cells",
        "title": "Tumor Grid Cells",
    },
)


def _variant_spec(variant_name: str) -> dict[str, str]:
    for spec in VARIANTS:
        if spec["name"] == variant_name:
            return spec
    raise KeyError(f"Unknown variant: {variant_name}")


@dataclass
class WsiTumorLocalizationConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "center_point"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/prediction.json"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/ground_truth.json"

    @property
    def annotation_file(self) -> str:
        return f"{self.reference_dir}/tumor_001.xml"

    @property
    def wsi_file(self) -> str:
        return f"{self.input_dir}/tumor_001.tif"

    @property
    def wsi_tools_file(self) -> str:
        return f"{self.input_dir}/wsi_tools.py"

    @property
    def runtime_manifest(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def task_kind(self) -> str:
        return _variant_spec(self.VARIANT_NAME)["kind"]

    @property
    def task_description(self) -> str:
        if self.task_kind == "center_point":
            variant_goal = (
                "Find a representative level-0 coordinate near the center of the tumor metastasis. "
                'Write `{"x": <float>, "y": <float>}`.'
            )
        elif self.task_kind == "bounding_box":
            variant_goal = (
                "Find a tight level-0 bounding box around the full visible tumor metastasis. "
                'Write `{"min_x": <float>, "min_y": <float>, "max_x": <float>, "max_y": <float>}`.'
            )
        else:
            variant_goal = (
                "Classify the 16384 by 16384 level-0 grid cells that contain tumor tissue. "
                'Write `{"tumor_cells": ["<x>_<y>", ...]}` using each positive cell\'s top-left coordinate.'
            )

        return f"""\
You are working on a Linux VM with a CAMELYON16 whole-slide pathology image.

Visible files:
- Whole-slide image: `{self.wsi_file}`
- Python helper module: `{self.wsi_tools_file}`
- Optional runtime dependency manifest: `{self.runtime_manifest}`

The helper module provides a `WSINavigationEnvironment` class with methods for slide metadata, thumbnails, region reads, and tissue masks. Use those tools, plus visual inspection of H&E tissue patches, to navigate the slide and localize tumor metastasis.

Your task for this variant:
{variant_goal}

Output requirements:
- Save exactly one JSON file at `{self.output_file}`.
- Use level-0 slide coordinates.
- Do not modify files under `input/`.
- Do not write outside your assigned output directory.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_kind": self.task_kind,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "annotation_file": self.annotation_file,
                "wsi_file": self.wsi_file,
                "wsi_tools_file": self.wsi_tools_file,
                "runtime_manifest": self.runtime_manifest,
            }
        )
        return metadata


config = WsiTumorLocalizationConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=WsiTumorLocalizationConfig(VARIANT_NAME=spec["name"]).task_description,
            metadata=WsiTumorLocalizationConfig(VARIANT_NAME=spec["name"]).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
        for spec in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not await session.exists(meta["output_file"]):
        return [0.0]

    try:
        prediction = json.loads(await session.read_file(meta["output_file"]))
        reference = json.loads(await session.read_file(meta["reference_file"]))
    except Exception:
        return [0.0]

    try:
        if meta["task_kind"] == "center_point":
            annotation_text = await session.read_file(meta["annotation_file"])
            score = _score_center_point(prediction, reference, annotation_text)
        elif meta["task_kind"] == "bounding_box":
            score = _score_bounding_box(prediction, reference)
        elif meta["task_kind"] == "grid_cells":
            score = _score_grid_cells(prediction, reference)
        else:
            score = 0.0
    except Exception:
        score = 0.0

    return [float(max(0.0, min(1.0, score)))]


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload[key]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _score_center_point(
    prediction: dict[str, Any], reference: dict[str, Any], annotation_text: str
) -> float:
    point = (_number(prediction, "x"), _number(prediction, "y"))
    polygon = _parse_annotation_polygon(annotation_text)
    if _point_in_polygon(point, polygon):
        return 1.0
    tolerance = float(reference.get("tolerance_pixels", 10000.0))
    if tolerance <= 0:
        return 0.0
    distance = _distance_to_polygon(point, polygon)
    return max(0.0, 1.0 - distance / tolerance)


def _parse_annotation_polygon(annotation_text: str) -> list[tuple[float, float]]:
    root = ElementTree.fromstring(annotation_text)
    points: list[tuple[float, float]] = []
    for elem in root.iter():
        if elem.tag.endswith("Coordinate") and "X" in elem.attrib and "Y" in elem.attrib:
            points.append((float(elem.attrib["X"]), float(elem.attrib["Y"])))
    if len(points) < 3:
        raise ValueError("annotation polygon has fewer than three points")
    return points


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def _distance_to_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> float:
    distances = [
        _distance_to_segment(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    ]
    return min(distances)


def _distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def _score_bounding_box(prediction: dict[str, Any], reference: dict[str, Any]) -> float:
    pred = {
        "min_x": _number(prediction, "min_x"),
        "min_y": _number(prediction, "min_y"),
        "max_x": _number(prediction, "max_x"),
        "max_y": _number(prediction, "max_y"),
    }
    ref = reference["bbox"]
    gt = {
        "min_x": float(ref["min_x"]),
        "min_y": float(ref["min_y"]),
        "max_x": float(ref["max_x"]),
        "max_y": float(ref["max_y"]),
    }
    if pred["max_x"] <= pred["min_x"] or pred["max_y"] <= pred["min_y"]:
        return 0.0
    inter_w = max(0.0, min(pred["max_x"], gt["max_x"]) - max(pred["min_x"], gt["min_x"]))
    inter_h = max(0.0, min(pred["max_y"], gt["max_y"]) - max(pred["min_y"], gt["min_y"]))
    inter_area = inter_w * inter_h
    pred_area = (pred["max_x"] - pred["min_x"]) * (pred["max_y"] - pred["min_y"])
    gt_area = (gt["max_x"] - gt["min_x"]) * (gt["max_y"] - gt["min_y"])
    union = pred_area + gt_area - inter_area
    return inter_area / union if union > 0 else 0.0


def _score_grid_cells(prediction: dict[str, Any], reference: dict[str, Any]) -> float:
    labels = reference["grid_labels"]
    predicted_cells = prediction.get("tumor_cells")
    if not isinstance(predicted_cells, list):
        raise ValueError("tumor_cells must be a list")
    predicted = {str(cell) for cell in predicted_cells}
    positives = {cell for cell, label in labels.items() if int(label) == 1}
    if not positives and not predicted:
        return 1.0
    true_positive = len(predicted & positives)
    false_positive = len(predicted - positives)
    false_negative = len(positives - predicted)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
