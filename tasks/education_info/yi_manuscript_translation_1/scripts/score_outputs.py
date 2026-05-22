"""Local scoring logic for education_info/yi_manuscript_translation_1."""

from __future__ import annotations

import json
import re
from typing import Any


REQUIRED_SECTIONS = [
    "EXAMINATION",
    "SOURCE IMAGE",
    "TECHNIQUE",
    "IDENTIFICATION OF TARGET CHARACTER POSITION",
    "REFERENCE MATERIALS CONSULTED",
    "TRANSLATION",
    "IMPRESSION",
]
OPTIONAL_SECTIONS = ["FINDINGS"]
HEADER_PATTERN = (
    r"EXAMINATION|SOURCE IMAGE|TECHNIQUE|IDENTIFICATION OF TARGET CHARACTER POSITION|"
    r"REFERENCE MATERIALS CONSULTED|TRANSLATION|IMPRESSION|FINDINGS"
)
SECTION_RE = re.compile(
    rf"^(?P<header>{HEADER_PATTERN})\s*:?\s*(?P<body>.*?)(?=^(?:{HEADER_PATTERN})\s*:?\s*|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _parse_bbox(raw: bytes) -> tuple[dict[str, int] | None, str | None]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return None, f"bbox json decode failed: {exc}"

    required = ["x1", "y1", "x2", "y2"]
    if any(key not in payload for key in required):
        return None, "bbox missing required keys"
    bbox: dict[str, int] = {}
    for key in required:
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int):
            return None, f"bbox value for {key} is not an integer"
        bbox[key] = value
    if bbox["x1"] >= bbox["x2"] or bbox["y1"] >= bbox["y2"]:
        return None, "bbox rectangle invalid"
    return bbox, None


def _parse_sections(report_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for match in SECTION_RE.finditer(report_text):
        sections[match.group("header")] = match.group("body").strip()
    return sections


def _bbox_contained(agent: dict[str, int], reference: dict[str, int]) -> bool:
    return (
        agent["x1"] >= reference["x1"]
        and agent["y1"] >= reference["y1"]
        and agent["x2"] <= reference["x2"]
        and agent["y2"] <= reference["y2"]
    )


def evaluate_submission(
    *,
    bbox_bytes: bytes,
    report_bytes: bytes,
    reference_bbox_bytes: bytes,
    ground_truth_bytes: bytes,
    reference_materials_snapshot_bytes: bytes,
) -> dict[str, Any]:
    bbox, bbox_error = _parse_bbox(bbox_bytes)
    reference_bbox, reference_bbox_error = _parse_bbox(reference_bbox_bytes)
    if reference_bbox_error:
        raise ValueError(reference_bbox_error)
    ground_truth = json.loads(ground_truth_bytes.decode("utf-8"))
    materials_snapshot = json.loads(reference_materials_snapshot_bytes.decode("utf-8"))

    try:
        report_text = report_bytes.decode("utf-8")
    except Exception as exc:
        report_text = ""
        report_error = f"report decode failed: {exc}"
    else:
        report_error = None

    hard_gate_failed = bbox_error is not None or report_error is not None

    bbox_score = 0.0
    if bbox is not None:
        bbox_score = 1.0 if _bbox_contained(bbox, reference_bbox) else 0.0

    sections = _parse_sections(report_text) if report_text else {}
    missing_sections = [name for name in REQUIRED_SECTIONS if not sections.get(name)]
    if missing_sections:
        hard_gate_failed = True

    report_score = 0.0
    citation_ok = False
    phonetic_ok = False
    classification_ok = False
    supported_translation_ok = False
    position_ok = False
    translation_ok = False

    if not hard_gate_failed:
        position_text = sections["IDENTIFICATION OF TARGET CHARACTER POSITION"]
        translation_text = sections["TRANSLATION"]
        consulted_text = sections["REFERENCE MATERIALS CONSULTED"]
        combined_findings = "\n".join(
            value
            for key, value in sections.items()
            if key in {"EXAMINATION", "TECHNIQUE", "FINDINGS"}
        )
        whole_report = "\n".join(sections.values())
        materials_blob = "\n".join(str(value) for value in materials_snapshot.values()).lower()
        accepted_translation_terms = [str(item) for item in ground_truth.get("accepted_translation_terms", [])]
        supported_material_terms = [str(item) for item in ground_truth.get("supported_material_terms", accepted_translation_terms)]

        position_ok = (
            _contains_any(position_text, ["first", "1st", "line 1", "upper line", "第一行", "上行"])
            and _contains_any(position_text, ["third", "3rd", "position 3", "第三", "第三个"])
        )
        translation_ok = _contains_any(translation_text, accepted_translation_terms)
        citation_ok = _contains_any(
            consulted_text,
            ["01_", "02_", "03_", "阿余铁日", "Ayu Tieri", "Ta Kung Pao", "大公报", "Xinhua", "新华社"],
        )
        phonetic_ok = _contains_any(
            whole_report,
            ["ṇɔi³³", "ɳɔi33", "noi33", "/ȵi³³/", "ȵi³³"],
        )
        classification_ok = _contains_any(
            combined_findings,
            ["logosyllabic", "syllabic", "ideographic", "表意", "音节", "Nuosu", "诺苏", "Bimo", "毕摩"],
        )
        supported_translation_ok = (
            translation_ok
            and any(token.lower() in materials_blob for token in supported_material_terms)
        )

        if position_ok and translation_ok:
            report_score = (
                float(citation_ok)
                + float(phonetic_ok)
                + float(classification_ok)
                + float(supported_translation_ok)
            ) / 4.0

    if hard_gate_failed:
        score = 0.0
    else:
        score = 0.5 * bbox_score + 0.5 * report_score
    return {
        "score": score,
        "bbox_score": bbox_score,
        "report_score": report_score,
        "bbox_error": bbox_error,
        "report_error": report_error,
        "hard_gate_failed": hard_gate_failed,
        "missing_sections": missing_sections,
        "position_ok": position_ok,
        "translation_ok": translation_ok,
        "citation_ok": citation_ok,
        "phonetic_ok": phonetic_ok,
        "classification_ok": classification_ok,
        "supported_translation_ok": supported_translation_ok,
    }
