"""Scoring helpers for legal_ipo_due_diligence_01."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Mapping

from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MIN_SCREENSHOT_WIDTH = 320
MIN_SCREENSHOT_HEIGHT = 180
MIN_THUMB_UNIQUE_COLORS = 8
MIN_LUMA_RANGE = 16
HARD_FAIL_MIN_VALID_SCREENSHOTS = 138


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    expected_count: int
    valid_screenshot_count: int
    missing_count: int
    invalid_count: int
    conclusion_ok: bool
    reason: str


def normalize_expected_filenames(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def inspect_png(data: bytes) -> tuple[bool, str]:
    if not data.startswith(PNG_SIGNATURE):
        return False, "missing PNG signature"
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG":
                return False, "not a PNG image"
            width, height = image.size
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
            thumb = image.convert("RGB").resize((64, 64))
    except Exception as exc:
        return False, f"invalid PNG image: {exc}"
    if width < MIN_SCREENSHOT_WIDTH or height < MIN_SCREENSHOT_HEIGHT:
        return False, f"image too small: {width}x{height}"
    unique_colors = len(set(thumb.getdata()))
    luma_min, luma_max = thumb.convert("L").getextrema()
    if unique_colors < MIN_THUMB_UNIQUE_COLORS or (luma_max - luma_min) < MIN_LUMA_RANGE:
        return False, "low-information or blank screenshot"
    return True, "valid"


def _clean_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("\u3000", " ")
    return re.sub(r"\s+", " ", lowered).strip()


def conclusion_indicates_no_penalty(conclusion: str) -> bool:
    text = _clean_text(conclusion)
    if not text:
        return False

    no_patterns = [
        r"(未发现|未查询到|未检索到|未见|无|没有|不存在).{0,30}?(行政处罚记录|处罚记录|行政处罚|处罚)",
        r"(行政处罚记录|处罚记录|行政处罚|处罚).{0,30}?(未发现|未查询到|未检索到|未见|无|没有|不存在)",
        r"\bno\b.{0,80}?\b(administrative penalty|administrative penalties|penalty|penalties)\b",
        r"\b(administrative penalty|administrative penalties|penalty|penalties)\b.{0,80}?\bnot\b.{0,20}?\bfound\b",
        r"\b(did not|didn't|could not|couldn't)\b.{0,80}?\b(find|identify|locate)\b.{0,80}?\b(administrative penalty|administrative penalties|penalty|penalties)\b",
        r"\bwithout\b.{0,80}?\b(administrative penalty|administrative penalties|penalty|penalties)\b",
        r"\b(zero|none|0)\b.{0,80}?\b(administrative penalty|administrative penalties|penalty|penalties)\b",
        r"\b(administrative penalty|administrative penalties|penalty|penalties)\b.{0,80}?\b(zero|none|0)\b",
    ]

    found_patterns = [
        r"(发现|查询到|检索到|存在|有).{0,30}?(行政处罚记录|处罚记录|行政处罚|处罚)",
        r"(行政处罚记录|处罚记录|行政处罚|处罚).{0,30}?(发现|查询到|检索到|存在|有)",
        r"\bfound\b.{0,80}?\b(administrative penalty|administrative penalties|penalty|penalties)\b",
        r"\b(administrative penalty|administrative penalties|penalty|penalties)\b.{0,80}?\bfound\b",
        r"\b(penalties|penalty)\b.{0,40}?\b(exist|exists)\b",
        r"\b(identified|discovered|located|detected|listed|reported|confirmed)\b.{0,80}?\b(administrative penalty|administrative penalties|penalty|penalties)\b",
        r"\b(administrative penalty|administrative penalties|penalty|penalties)\b.{0,80}?\b(identified|discovered|located|detected|listed|reported|confirmed)\b",
    ]

    no_matches = [match for pattern in no_patterns for match in re.finditer(pattern, text)]
    found_matches = [match for pattern in found_patterns for match in re.finditer(pattern, text)]

    def overlaps(first: tuple[int, int], second: tuple[int, int]) -> bool:
        return max(first[0], second[0]) < min(first[1], second[1])

    has_no_penalty = bool(no_matches)
    has_unnegated_found = any(
        not any(overlaps(found.span(), no_match.span()) for no_match in no_matches)
        for found in found_matches
    )
    if has_unnegated_found:
        return False

    return has_no_penalty


def score_submission(
    *,
    expected_filenames: list[str],
    conclusion_text: str,
    screenshot_bytes_by_name: Mapping[str, bytes],
) -> ScoreBreakdown:
    expected_unique = list(dict.fromkeys(expected_filenames))
    expected_count = len(expected_unique)
    conclusion_ok = conclusion_indicates_no_penalty(conclusion_text)

    valid_count = 0
    invalid_count = 0
    missing_count = 0
    for filename in expected_unique:
        data = screenshot_bytes_by_name.get(filename)
        if data is None:
            missing_count += 1
            continue
        is_valid, _ = inspect_png(data)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

    if not conclusion_ok:
        return ScoreBreakdown(
            score=0.0,
            expected_count=expected_count,
            valid_screenshot_count=valid_count,
            missing_count=missing_count,
            invalid_count=invalid_count,
            conclusion_ok=False,
            reason="conclusion does not state that no administrative penalty records were found",
        )

    if expected_count == 0:
        return ScoreBreakdown(
            score=0.0,
            expected_count=0,
            valid_screenshot_count=valid_count,
            missing_count=missing_count,
            invalid_count=invalid_count,
            conclusion_ok=True,
            reason="empty expected filename list",
        )

    if valid_count < HARD_FAIL_MIN_VALID_SCREENSHOTS:
        return ScoreBreakdown(
            score=0.0,
            expected_count=expected_count,
            valid_screenshot_count=valid_count,
            missing_count=missing_count,
            invalid_count=invalid_count,
            conclusion_ok=True,
            reason="fewer than 138 valid screenshots",
        )

    return ScoreBreakdown(
        score=max(0.0, min(1.0, valid_count / expected_count)),
        expected_count=expected_count,
        valid_screenshot_count=valid_count,
        missing_count=missing_count,
        invalid_count=invalid_count,
        conclusion_ok=True,
        reason="scored by valid expected screenshot completeness",
    )
