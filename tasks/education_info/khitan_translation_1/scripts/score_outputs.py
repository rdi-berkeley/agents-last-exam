"""Scoring logic for education_info/khitan_translation_1."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

IMAGE_NAME = "inscription.png"
IMAGE_WIDTH = 664
IMAGE_HEIGHT = 94
EXPECTED_FILES = {"report.txt", "bounding_boxes.json"}
REFERENCE_FILENAMES_RE = re.compile(
    r"(?:0[1-8]_[A-Za-z0-9_.-]+\.txt|00_README_index\.txt|"
    r"wang_jingru_1933_primary_source\.pdf)"
)

EXPECTED_CHARS = ["宣", "懿", "皇", "后", "哀", "册", "文"]
EXPECTED_CHARS_ALT = ["宣", "懿", "皇", "後", "哀", "冊", "文"]


def _as_text(raw: bytes) -> str:
    return raw.decode("utf-8")


def _section(text: str, label: str) -> str:
    pattern = re.compile(
        rf"^\s*{re.escape(label)}\s*:\s*(?P<body>.*?)(?=^\s*[A-Za-z ]+\s*:|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return "" if match is None else match.group("body").strip()


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _contains_cjk_or_ascii(text: str, needles: list[str]) -> bool:
    return any(needle in text or needle.lower() in text.lower() for needle in needles)


def _parse_boxes(raw: bytes) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(_as_text(raw))
    except Exception as exc:
        return None, f"bounding_boxes.json is not valid UTF-8 JSON: {exc}"

    required_top = {"image", "image_width", "image_height", "boxes"}
    missing = sorted(required_top - set(payload))
    if missing:
        return None, "bounding_boxes.json missing top-level keys: " + ", ".join(missing)
    if payload["image"] != IMAGE_NAME:
        return None, f"image must be {IMAGE_NAME!r}"
    if payload["image_width"] != IMAGE_WIDTH or payload["image_height"] != IMAGE_HEIGHT:
        return None, "image_width/image_height must be 664/94"
    if not isinstance(payload["boxes"], list) or len(payload["boxes"]) != 7:
        return None, "boxes must be a list of exactly 7 objects"

    seen: set[int] = set()
    parsed: list[dict[str, int]] = []
    for item in payload["boxes"]:
        if not isinstance(item, dict):
            return None, "each box must be an object"
        box: dict[str, int] = {}
        for key in ["index", "x", "y", "w", "h"]:
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                return None, f"box key {key!r} must be an integer"
            box[key] = value
        if box["index"] in seen:
            return None, "duplicate box index"
        seen.add(box["index"])
        if not (0 <= box["x"] and 0 <= box["y"]):
            return None, "box coordinates must be non-negative"
        if not (box["x"] + box["w"] <= IMAGE_WIDTH and box["y"] + box["h"] <= IMAGE_HEIGHT):
            return None, "box extends outside image bounds"
        if box["w"] < 15 or box["h"] < 15:
            return None, "box width and height must each be at least 15"
        parsed.append(box)

    if seen != set(range(1, 8)):
        return None, "box indexes must be exactly 1 through 7"
    by_x = sorted(parsed, key=lambda box: box["x"])
    if [box["index"] for box in by_x] != list(range(1, 8)):
        return None, "boxes sorted by x must match index order"
    for left, right in zip(by_x, by_x[1:]):
        overlap = left["x"] + left["w"] - right["x"]
        if overlap > 10:
            return None, "adjacent boxes overlap by more than 10 pixels"

    payload["boxes"] = sorted(parsed, key=lambda box: box["index"])
    return payload, None


def _intersection_over_union(agent: dict[str, int], ref: dict[str, int]) -> float:
    ax1, ay1 = agent["x"], agent["y"]
    ax2, ay2 = agent["x"] + agent["w"], agent["y"] + agent["h"]
    rx1, ry1 = ref["x"], ref["y"]
    rx2, ry2 = ref["x"] + ref["w"], ref["y"] + ref["h"]
    ix1, iy1 = max(ax1, rx1), max(ay1, ry1)
    ix2, iy2 = min(ax2, rx2), min(ay2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_agent = agent["w"] * agent["h"]
    area_ref = ref["w"] * ref["h"]
    union = area_agent + area_ref - inter
    return inter / union if union else 0.0


def _contained_with_slack(agent: dict[str, int], ref: dict[str, int], slack: int = 5) -> bool:
    return (
        agent["x"] >= ref["x"] - slack
        and agent["y"] >= ref["y"] - slack
        and agent["x"] + agent["w"] <= ref["x"] + ref["w"] + slack
        and agent["y"] + agent["h"] <= ref["y"] + ref["h"] + slack
    )


def _distinct_cited_files(citations: str) -> list[str]:
    return sorted(set(REFERENCE_FILENAMES_RE.findall(citations)))


def _source_supports_report_claim(filename: str, report_text: str, materials: dict[str, str]) -> bool:
    if filename == "wang_jingru_1933_primary_source.pdf":
        proxy = materials.get("04_wang_jingru_1933_primary_source.pdf.txt", "")
    else:
        proxy = materials.get(filename, "")
    if not proxy:
        return False
    tokens = re.findall(r"[\w一-鿿]{4,}", proxy, flags=re.UNICODE)
    report_lower = report_text.lower()
    for token in tokens:
        if token.lower() in report_lower:
            return True
    return False


def _has_large_without_negation(text: str) -> bool:
    """Check if classification mentions 'large script' without negating it."""
    large_patterns = [
        r'\blarge\s+script\b',
        r'大字',
        r'契丹大字',
    ]
    negation_patterns = [
        r'\bnot\b.{0,20}\blarge\b',
        r'\brather\s+than\b.{0,20}\blarge\b',
        r'\bnot\s+large\b',
        r'\bisn.t\s+large\b',
        r'并非.{0,10}大字',
        r'不是.{0,10}大字',
        r'而非.{0,10}大字',
        r'\blarge\s+script\b.{0,30}\bhowever\b',
        r'\bresembl(?:es?|ing)\b.{0,20}\blarge\b',
        r'\blooks?\s+like\b.{0,20}\blarge\b',
        r'\bsuperficially\b.{0,30}\blarge\b',
        r'\billusion\b',
        r'\bappears?\b.{0,20}\blarge\b.{0,30}\bactually\b',
    ]
    lowered = text.lower()
    has_large = any(re.search(p, lowered) for p in large_patterns)
    if not has_large:
        return False
    has_negation = any(re.search(p, lowered) for p in negation_patterns)
    return not has_negation


def _per_char_score(report_text: str) -> float:
    """Score per-character mapping: each of the 7 Chinese chars explicitly present."""
    found = 0
    for char, alt in zip(EXPECTED_CHARS, EXPECTED_CHARS_ALT):
        if char in report_text or alt in report_text:
            found += 1
    return found / 7.0


def _score_report_keyword(report_text: str, materials: dict[str, str]) -> dict[str, Any]:
    """Keyword-based report scoring (used as fallback if LLM judge unavailable)."""
    classification = _section(report_text, "Script classification")
    chinese = _section(report_text, "Chinese translation")
    english = _section(report_text, "English translation")
    historical = _section(report_text, "Historical context")
    citations = _section(report_text, "Citations")
    translation_blob = f"{chinese}\n{english}"

    has_small = _contains_cjk_or_ascii(classification, ["small", "小字"])
    has_large_neg = _has_large_without_negation(classification)

    if has_large_neg:
        b1 = 0.0
    elif has_small:
        has_seal = _contains_cjk_or_ascii(
            classification,
            ["seal", "篆书", "篆書", "seal-style", "memorial cover", "册盖", "冊蓋"],
        )
        b1 = 10.0 + (2.0 if has_seal else 0.0)
    else:
        b1 = 0.0

    if has_large_neg:
        b2 = 0.0
    else:
        b2 = 0.0
        b2 += 6.0 if _contains_cjk_or_ascii(translation_blob, ["Xuanyi", "Xuān", "宣懿"]) else 0.0
        b2 += 4.0 if _contains_cjk_or_ascii(translation_blob, ["Empress", "皇后"]) else 0.0
        b2 += 8.0 if _contains_any(
            translation_blob,
            ["funerary", "memorial", "eulogy", "mourning inscription", "哀册", "哀冊"],
        ) else 0.0

    b3 = 0.0
    b3 += 3.0 if _contains_cjk_or_ascii(historical, ["Liao", "辽", "遼"]) else 0.0
    b3 += 3.0 if _contains_cjk_or_ascii(historical, ["Qingling", "Yongfu", "庆陵", "慶陵", "永福"]) else 0.0
    b3 += 4.0 if _contains_cjk_or_ascii(historical, ["Daozong", "道宗", "1075", "1101", "11th-century"]) else 0.0

    cited_files = _distinct_cited_files(citations)
    b4 = min(6.0, 2.0 * len(cited_files))
    b5 = 4.0 if cited_files and all(
        _source_supports_report_claim(filename, report_text, materials) for filename in cited_files
    ) else 0.0

    per_char = _per_char_score(report_text)
    b6 = 10.0 * per_char

    raw = b1 + b2 + b3 + b4 + b5 + b6
    _KW_MAX = 12.0 + 18.0 + 10.0 + 6.0 + 4.0 + 10.0  # 60
    normalized = raw / _KW_MAX * 50.0

    return {
        "b1_classification": b1,
        "b2_translation": b2,
        "b3_context": b3,
        "b4_citations": b4,
        "b5_anti_fabrication": b5,
        "b6_per_char_mapping": b6,
        "cited_files": cited_files,
        "report_total": normalized,
        "llm_judge_used": False,
    }


_LLM_JUDGE_RUBRIC = """\
You are evaluating a student report about a Khitan inscription. Compare the student's report against the reference report below.

## Reference report (ground truth)
{reference_report}

## Student report (submission to evaluate)
{agent_report}

## Scoring rubric — return a JSON object with these exact keys:

1. "d1_classification" (0-12): Script classification quality.
   - 0: says "large script" without qualifying it as a visual illusion
   - 6: correctly says "small script" but no explanation of the seal-style visual ambiguity
   - 10: correctly identifies small script AND mentions seal/篆 style on the cover
   - 12: full explanation of WHY it looks like large script but IS small script (cover vs body distinction)

2. "d2_translation_accuracy" (0-10): Per-character translation accuracy.
   - Award ~1.4 points per correctly identified Chinese character (宣/懿/皇/后/哀/册/文).
   - The complete reading should be "宣懿皇后哀册文". Accept traditional variants (後/冊).
   - 0 if the translation is entirely wrong or missing.

3. "d3_translation_completeness" (0-8): Does the report provide both Chinese and English translations?
   - 0: neither present
   - 4: only one of Chinese or English
   - 8: both present and reasonable

4. "d4_historical_context" (0-10): Accuracy of historical context.
   - Award points for each correct key fact: Liao dynasty (2), Empress Xuanyi/Xiao Guanyin (2), death in 1075 (1), rehabilitation in 1101 (1), Qingling/Yongfu tomb (2), Liaoning Provincial Museum (1), Wang Jingru 1933 decipherment (1).
   - Deduct 2 points for any clearly wrong factual claim (wrong dynasty, wrong person, wrong date).

5. "d5_citation_quality" (0-6): Quality of citations.
   - 0: no citations at all
   - 2: cites file names only without specifics
   - 4: cites file names with some section references
   - 6: detailed citations mapping specific claims to specific sections of specific files

6. "d6_factual_errors" (0 to -6): Penalty for factual errors.
   - 0: no errors detected
   - -2 per significant factual error (wrong dynasty, wrong person, wrong script system, wrong translation)
   - Cap at -6.

7. "reasoning": A brief (2-3 sentence) explanation of the scores.

Return ONLY a JSON object with keys: d1_classification, d2_translation_accuracy, d3_translation_completeness, d4_historical_context, d5_citation_quality, d6_factual_errors, reasoning.
"""


def _score_report_llm(
    report_text: str,
    reference_report: str,
) -> dict[str, Any] | None:
    """Use GPT-4o-mini as a judge to score the report semantically."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.info("OPENAI_API_KEY not set, falling back to keyword scoring")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.info("openai package not installed, falling back to keyword scoring")
        return None

    prompt = _LLM_JUDGE_RUBRIC.format(
        reference_report=reference_report,
        agent_report=report_text,
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        scores = json.loads(raw)
    except Exception as exc:
        logger.warning("LLM judge call failed: %s", exc)
        return None

    d1 = max(0.0, min(12.0, float(scores.get("d1_classification", 0))))
    d2 = max(0.0, min(10.0, float(scores.get("d2_translation_accuracy", 0))))
    d3 = max(0.0, min(8.0, float(scores.get("d3_translation_completeness", 0))))
    d4 = max(0.0, min(10.0, float(scores.get("d4_historical_context", 0))))
    d5 = max(0.0, min(6.0, float(scores.get("d5_citation_quality", 0))))
    d6 = max(-6.0, min(0.0, float(scores.get("d6_factual_errors", 0))))
    reasoning = scores.get("reasoning", "")

    raw = d1 + d2 + d3 + d4 + d5 + d6
    per_char = _per_char_score(report_text)
    b6 = 10.0 * per_char
    raw_with_char = raw + b6
    _LLM_MAX = 12.0 + 10.0 + 8.0 + 10.0 + 6.0 + 10.0  # 56 (d6 can only subtract)
    normalized = max(0.0, raw_with_char / _LLM_MAX * 50.0)

    return {
        "d1_classification": d1,
        "d2_translation_accuracy": d2,
        "d3_translation_completeness": d3,
        "d4_historical_context": d4,
        "d5_citation_quality": d5,
        "d6_factual_errors": d6,
        "b6_per_char_mapping": b6,
        "llm_reasoning": reasoning,
        "report_total": normalized,
        "llm_judge_used": True,
    }


def evaluate_submission(
    *,
    output_files: dict[str, bytes],
    reference_boxes_bytes: bytes,
    reference_materials_snapshot_bytes: bytes,
    reference_report_text: str | None = None,
) -> dict[str, Any]:
    observed_files = set(output_files)
    if observed_files != EXPECTED_FILES:
        return {
            "score": 0.0,
            "passed": False,
            "hard_gate_failed": True,
            "error": f"output directory must contain exactly {sorted(EXPECTED_FILES)}, observed {sorted(observed_files)}",
        }
    empty_files = sorted(name for name, content in output_files.items() if not content)
    if empty_files:
        return {
            "score": 0.0,
            "passed": False,
            "hard_gate_failed": True,
            "error": "required output files must be non-empty: " + ", ".join(empty_files),
        }

    boxes, box_error = _parse_boxes(output_files["bounding_boxes.json"])
    if box_error:
        return {"score": 0.0, "passed": False, "hard_gate_failed": True, "error": box_error}
    try:
        report_text = _as_text(output_files["report.txt"])
    except Exception as exc:
        return {
            "score": 0.0,
            "passed": False,
            "hard_gate_failed": True,
            "error": f"report.txt is not valid UTF-8: {exc}",
        }

    reference_boxes, ref_error = _parse_boxes(reference_boxes_bytes)
    if ref_error:
        raise ValueError("invalid hidden reference boxes: " + ref_error)
    materials = json.loads(_as_text(reference_materials_snapshot_bytes))

    containment_count = 0
    iou_sum = 0.0
    per_box = []
    for agent_box, ref_box in zip(boxes["boxes"], reference_boxes["boxes"]):
        contained = _contained_with_slack(agent_box, ref_box)
        iou = _intersection_over_union(agent_box, ref_box)
        containment_count += int(contained)
        iou_sum += iou
        per_box.append({"index": agent_box["index"], "contained": contained, "iou": iou})

    # Reweighted: containment 10pts, IoU 40pts (was 25/25)
    a_containment = containment_count * (10.0 / 7.0)
    a_iou = iou_sum * (40.0 / 7.0)
    a_total = a_containment + a_iou

    # Report scoring: try LLM judge, fall back to keyword
    report_scores: dict[str, Any] | None = None
    if reference_report_text:
        report_scores = _score_report_llm(report_text, reference_report_text)
    if report_scores is None:
        report_scores = _score_report_keyword(report_text, materials)

    # boxes 0-50 + report 0-50 = 0-100
    total = max(0.0, min(100.0, a_total + report_scores["report_total"]))

    classification_score = report_scores.get("b1_classification", report_scores.get("d1_classification", 0))
    passed = (
        total >= 70.0
        and a_containment >= (10.0 / 7.0) * 6
        and classification_score >= 6.0
    )

    return {
        "score": total / 100.0,
        "score_100": total,
        "passed": passed,
        "hard_gate_failed": False,
        "a_containment": a_containment,
        "a_iou": a_iou,
        "a_total": a_total,
        "containment_count": containment_count,
        "per_box": per_box,
        **report_scores,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score a khitan_translation_1 output directory.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    output_files = {
        path.name: path.read_bytes()
        for path in args.output_dir.iterdir()
        if path.is_file()
    }
    ref_report_path = args.reference_dir / "reference_report.txt"
    ref_report_text = ref_report_path.read_text(encoding="utf-8") if ref_report_path.exists() else None

    result = evaluate_submission(
        output_files=output_files,
        reference_boxes_bytes=(args.reference_dir / "reference_boxes.json").read_bytes(),
        reference_materials_snapshot_bytes=(
            args.reference_dir / "reference_materials_snapshot.json"
        ).read_bytes(),
        reference_report_text=ref_report_text,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not math.isclose(result["score"], 1.0) and args.output_dir.name == "output_test_pos":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
