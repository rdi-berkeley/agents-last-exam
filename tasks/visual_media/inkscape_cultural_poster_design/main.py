"""Minimal Stage 2 implementation for the Inkscape cultural poster design task."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree as ET

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local fallback only

    class _FallbackTask:
        def __init__(self, description, metadata, computer):
            self.description = description
            self.metadata = metadata
            self.computer = computer

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    cb = SimpleNamespace(
        Task=_FallbackTask,
        DesktopSession=object,
        tasks_config=_identity_decorator,
        setup_task=_identity_decorator,
        evaluate_task=_identity_decorator,
    )

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_NAME = "inkscape_cultural_poster_design"
TASK_ID = f"design/{TASK_NAME}"
VARIANT_NAME = "base"
INKSCAPE_EXE = r"C:\Program Files\Inkscape\bin\inkscape.exe"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _remote_join(base: str, *parts: str) -> str:
    current = Path(base)
    for part in parts:
        current = current / part
    return str(current).replace("/", "\\")


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "instance_spec", data, 0, len(data), "unable to decode as utf-8/gb18030/gbk"
    )


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y"}


def _parse_instance_spec_bytes(data: bytes) -> dict[str, Any]:
    text = _decode_text_bytes(data)
    spec: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            if current_list_key is not None:
                spec.setdefault(current_list_key, []).append(line[2:].strip())
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            spec[key] = []
            current_list_key = key
            continue
        if key.startswith("canvas_"):
            spec[key] = float(value)
        elif key == "min_required_phrase_count":
            spec[key] = int(value)
        elif (
            key.endswith("_must_be_included")
            or key.endswith("_preserve_aspect_ratio")
            or key.endswith("_inside_canvas")
        ):
            spec[key] = _parse_bool(value)
        else:
            spec[key] = value
    required_keys = {
        "canvas_width_mm",
        "canvas_height_mm",
        "orientation",
        "required_title",
        "required_subtitle",
        "required_phrases",
        "min_required_phrase_count",
    }
    missing = sorted(key for key in required_keys if key not in spec)
    if missing:
        raise ValueError(f"instance spec missing required keys: {missing}")
    return spec


def _parse_length_mm(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    try:
        if lower.endswith("mm"):
            return float(lower[:-2].strip())
        if lower.endswith("cm"):
            return float(lower[:-2].strip()) * 10.0
        if lower.endswith("px"):
            return float(lower[:-2].strip()) * 25.4 / 96.0
        return float(lower)
    except ValueError:
        return None


def _extract_svg_text(root: ET.Element) -> str:
    parts: list[str] = []
    for elem in root.iter():
        text = (elem.text or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _find_image_elements(root: ET.Element) -> list[ET.Element]:
    images: list[ET.Element] = []
    for elem in root.iter():
        if _strip_ns(elem.tag) == "image":
            images.append(elem)
    return images


def _element_attr(elem: ET.Element, name: str) -> str | None:
    return elem.attrib.get(name) or elem.attrib.get(f"{{{XLINK_NS}}}{name}")


def evaluate_svg_bytes(svg_bytes: bytes, spec: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    details: dict[str, Any] = {
        "svg_exists": True,
        "svg_parsable": False,
        "root_is_svg": False,
        "canvas_matches": False,
        "title_present": False,
        "subtitle_present": False,
        "phrase_count_ok": False,
        "image_present": False,
        "preserve_aspect_ratio_ok": False,
        "image_inside_canvas": False,
    }

    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError as exc:
        details["parse_error"] = str(exc)
        return 0.0, details

    details["svg_parsable"] = True
    details["root_is_svg"] = _strip_ns(root.tag).lower() == "svg"
    if not details["root_is_svg"]:
        return 0.0, details

    width_mm = _parse_length_mm(root.attrib.get("width"))
    height_mm = _parse_length_mm(root.attrib.get("height"))
    spec_width = float(spec["canvas_width_mm"])
    spec_height = float(spec["canvas_height_mm"])
    orientation = str(spec["orientation"]).strip().lower()
    orientation_ok = (orientation == "portrait" and spec_height > spec_width) or (
        orientation == "landscape" and spec_width > spec_height
    )
    dimensions_ok = (
        width_mm is not None
        and height_mm is not None
        and abs(width_mm - spec_width) <= 1.0
        and abs(height_mm - spec_height) <= 1.0
    )
    svg_orientation_ok = (orientation == "portrait" and height_mm > width_mm) or (
        orientation == "landscape" and width_mm > height_mm
    )
    details["canvas_matches"] = bool(dimensions_ok and orientation_ok and svg_orientation_ok)

    full_text = _extract_svg_text(root)
    details["title_present"] = str(spec["required_title"]) in full_text
    details["subtitle_present"] = str(spec["required_subtitle"]) in full_text

    required_phrases = [str(item) for item in spec.get("required_phrases", [])]
    phrase_hits = sum(1 for phrase in required_phrases if phrase in full_text)
    details["phrase_hit_count"] = phrase_hits
    details["phrase_count_ok"] = phrase_hits >= int(spec["min_required_phrase_count"])

    image_elements = _find_image_elements(root)
    details["image_present"] = bool(image_elements)
    if image_elements:
        image = image_elements[0]
        preserve = (image.attrib.get("preserveAspectRatio") or "").strip().lower()
        details["preserve_aspect_ratio_ok"] = preserve not in {"", "none"}
        x = _parse_length_mm(image.attrib.get("x"))
        y = _parse_length_mm(image.attrib.get("y"))
        w = _parse_length_mm(image.attrib.get("width"))
        h = _parse_length_mm(image.attrib.get("height"))
        href = _element_attr(image, "href")
        details["image_href"] = href
        if None not in {x, y, w, h} and w and h and width_mm and height_mm:
            details["image_inside_canvas"] = (
                x >= 0.0
                and y >= 0.0
                and w > 0.0
                and h > 0.0
                and x + w <= width_mm + 1.0
                and y + h <= height_mm + 1.0
            )

    passed = all(
        [
            details["svg_parsable"],
            details["root_is_svg"],
            details["canvas_matches"],
            details["title_present"],
            details["subtitle_present"],
            details["phrase_count_ok"],
            details["image_present"],
            details["preserve_aspect_ratio_ok"],
            details["image_inside_canvas"],
        ]
    )
    return (1.0 if passed else 0.0), details


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    check: bool = False,
    timeout: float | None = None,
) -> dict:
    try:
        if timeout is not None:
            return await session.run_command(command, check=check, timeout=timeout)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


def _ps_single_quote(text: str) -> str:
    return text.replace("'", "''")


async def _read_remote_bytes(session: cb.DesktopSession, path: str) -> bytes:
    quoted_path = _ps_single_quote(path)
    result = await _run_command(
        session,
        "powershell -NoProfile -Command "
        f"\"[Convert]::ToBase64String([System.IO.File]::ReadAllBytes('{quoted_path}'))\"",
        check=False,
        timeout=60.0,
    )
    if result.get("return_code") != 0:
        raise RuntimeError(
            f"Failed to read raw bytes from {path}: "
            + (result.get("stderr") or result.get("stdout") or "").strip()
        )
    payload = (result.get("stdout") or "").strip()
    if not payload:
        return b""
    import base64

    return base64.b64decode(payload)


@dataclass
class InkscapeCulturalPosterConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"
    TASK_NAME: str = "inkscape_cultural_poster_design"
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "windows"

    @property
    def input_dir(self) -> str:
        return _remote_join(self.task_dir, "input")

    @property
    def output_poster(self) -> str:
        return _remote_join(self.remote_output_dir, "poster.svg")

    @property
    def spec_path(self) -> str:
        return _remote_join(self.input_dir, "instance_spec.txt")

    @property
    def brief_path(self) -> str:
        return _remote_join(self.input_dir, "design_brief.txt")

    @property
    def installation_photo(self) -> str:
        return _remote_join(self.input_dir, "installation_photo_01.jpg")

    @property
    def output_test_pos_svg(self) -> str:
        return _remote_join(self.task_dir, "output_test_pos", "poster.svg")

    @property
    def output_test_neg_svg(self) -> str:
        return _remote_join(self.task_dir, "output_test_neg", "poster.svg")

    @property
    def task_description(self) -> str:
        return f"""\
You are using Inkscape on Windows to design a cultural exhibition poster.

Read:
- `{self.brief_path}`
- `{self.spec_path}` (GBK/GB18030 encoded Chinese)
- `{self.installation_photo}`

Create the poster in Inkscape and save the final SVG exactly to:
- `{self.output_poster}`

Notes:
- Use Inkscape rather than generic image-generation tools.
- Preserve the source photo's aspect ratio when you place it.
- The current evaluator only checks a stable automated subset of the full rubric.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_dir": self.task_dir,
                "inkscape_exe": INKSCAPE_EXE,
                "input_dir": self.input_dir,
                "brief_path": self.brief_path,
                "spec_path": self.spec_path,
                "installation_photo": self.installation_photo,
                "output_poster": self.output_poster,
                "output_test_pos_svg": self.output_test_pos_svg,
                "output_test_neg_svg": self.output_test_neg_svg,
            }
        )
        return metadata


config = InkscapeCulturalPosterConfig()


@cb.tasks_config(split="train")
def load():
    cfg = InkscapeCulturalPosterConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        spec_bytes = await _read_remote_bytes(session, meta["spec_path"])
        spec = _parse_instance_spec_bytes(spec_bytes)
    except Exception as exc:
        logger.error("Failed to decode or parse instance spec at %s: %s", meta["spec_path"], exc)
        return [0.0]

    output_path = meta["output_poster"]
    if not (await session.file_exists(output_path) or await session.directory_exists(output_path)):
        logger.info("Missing poster output: %s", output_path)
        return [0.0]

    try:
        svg_bytes = await session.read_bytes(output_path)
    except Exception as exc:
        logger.error("Failed to read poster.svg bytes at %s: %s", output_path, exc)
        return [0.0]

    score, details = evaluate_svg_bytes(svg_bytes, spec)
    logger.info("Poster evaluation details: %s", json.dumps(details, ensure_ascii=False))
    return [float(score)]
