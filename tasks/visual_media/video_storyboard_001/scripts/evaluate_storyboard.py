"""Evaluator helpers for visual_media/video_storyboard_001."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.evaluation import llm_multimodal_json, resolve_llm_judge_model

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NAMESPACES = {"w": WORD_NS}
QUESTION_IDS = [f"Q{i}" for i in range(1, 11)]

ANSWER_KEY = {
    "Q1": "B",
    "Q2": "A",
    "Q3": "C",
    "Q4": "B",
    "Q5": "A",
    "Q6": "C",
    "Q7": "C",
    "Q8": "C",
    "Q9": "B",
    "Q10": "D",
}


@dataclass(frozen=True)
class DocxParseResult:
    ok: bool
    text: str
    error: str | None = None


def extract_docx_text(docx_bytes: bytes) -> DocxParseResult:
    """Extract visible text from a DOCX using only the standard library."""
    try:
        with ZipFile(BytesIO(docx_bytes), "r") as docx:
            document_xml = docx.read("word/document.xml")
    except (BadZipFile, KeyError) as exc:
        return DocxParseResult(ok=False, text="", error=f"invalid_docx: {exc}")

    try:
        root = ET.fromstring(document_xml)
    except ET.ParseError as exc:
        return DocxParseResult(ok=False, text="", error=f"invalid_word_xml: {exc}")

    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", NAMESPACES):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{{{WORD_NS}}}t":
                parts.append(node.text or "")
            elif node.tag == f"{{{WORD_NS}}}tab":
                parts.append("\t")
            elif node.tag == f"{{{WORD_NS}}}br":
                parts.append("\n")
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)

    text = "\n".join(paragraphs).strip()
    if not text:
        return DocxParseResult(ok=False, text="", error="empty_docx_text")
    return DocxParseResult(ok=True, text=text)


def _clip_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[TRUNCATED]"


def _normalize_answer(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    if text in {"UNKNOWN", "NOT ANSWERED", "UNANSWERED", "N/A", ""}:
        return "UNKNOWN"
    return "UNKNOWN"


def _normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_answers = payload.get("answers", payload)
    if not isinstance(raw_answers, dict):
        raw_answers = {}

    answers = {qid: _normalize_answer(raw_answers.get(qid)) for qid in QUESTION_IDS}
    correct = {
        qid: answers[qid] == ANSWER_KEY[qid]
        for qid in QUESTION_IDS
    }
    correct_count = sum(1 for value in correct.values() if value)
    score = correct_count / len(QUESTION_IDS)

    return {
        "score": score,
        "correct_count": correct_count,
        "total_questions": len(QUESTION_IDS),
        "answers": answers,
        "correct": correct,
        "raw_judge_payload": payload,
    }


async def answer_questions_from_storyboard(
    *,
    storyboard_text: str,
    question_text: str,
    model: str | None = None,
) -> dict[str, Any]:
    prompt = f"""\
You are grading a video-storyboard task.

You will receive:
1. A candidate storyboard text.
2. A multiple-choice question set.

Answer each question using ONLY the candidate storyboard text. Do not use prior
knowledge of the film, the title, common sense guesses, or the answer choices as
evidence. If the storyboard text does not support an answer, use "UNKNOWN".

Return ONLY valid JSON in this exact shape:
{{
  "answers": {{
    "Q1": "A|B|C|D|UNKNOWN",
    "Q2": "A|B|C|D|UNKNOWN",
    "Q3": "A|B|C|D|UNKNOWN",
    "Q4": "A|B|C|D|UNKNOWN",
    "Q5": "A|B|C|D|UNKNOWN",
    "Q6": "A|B|C|D|UNKNOWN",
    "Q7": "A|B|C|D|UNKNOWN",
    "Q8": "A|B|C|D|UNKNOWN",
    "Q9": "A|B|C|D|UNKNOWN",
    "Q10": "A|B|C|D|UNKNOWN"
  }},
  "evidence": {{
    "Q1": "short supporting quote or UNKNOWN",
    "Q2": "short supporting quote or UNKNOWN",
    "Q3": "short supporting quote or UNKNOWN",
    "Q4": "short supporting quote or UNKNOWN",
    "Q5": "short supporting quote or UNKNOWN",
    "Q6": "short supporting quote or UNKNOWN",
    "Q7": "short supporting quote or UNKNOWN",
    "Q8": "short supporting quote or UNKNOWN",
    "Q9": "short supporting quote or UNKNOWN",
    "Q10": "short supporting quote or UNKNOWN"
  }}
}}

Candidate storyboard text:
---
{_clip_text(storyboard_text, limit=50000)}
---

Question set:
---
{_clip_text(question_text, limit=20000)}
---
"""
    return await llm_multimodal_json(
        content=[{"type": "text", "text": prompt}],
        model=resolve_llm_judge_model(env_var="VIDEO_STORYBOARD_JUDGE_MODEL", default=model),
        max_tokens=2200,
        temperature=0,
    )


async def score_storyboard_docx(
    *,
    storyboard_docx: bytes,
    question_docx: bytes,
    model: str | None = None,
) -> dict[str, Any]:
    storyboard = extract_docx_text(storyboard_docx)
    if not storyboard.ok:
        return {"score": 0.0, "error": storyboard.error}

    questions = extract_docx_text(question_docx)
    if not questions.ok:
        return {"score": 0.0, "error": questions.error}

    judge_payload = await answer_questions_from_storyboard(
        storyboard_text=storyboard.text,
        question_text=questions.text,
        model=model,
    )
    result = _normalize_judge_payload(judge_payload)
    result["storyboard_character_count"] = len(storyboard.text)
    result["question_character_count"] = len(questions.text)
    return result


async def _main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--model")
    args = parser.parse_args()

    with open(args.storyboard, "rb") as handle:
        storyboard_docx = handle.read()
    with open(args.questions, "rb") as handle:
        question_docx = handle.read()

    result = await score_storyboard_docx(
        storyboard_docx=storyboard_docx,
        question_docx=question_docx,
        model=args.model,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if float(result.get("score", 0.0)) >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main_async()))
