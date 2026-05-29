"""Verify the Word template section-swap task output."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SECTION_A = "选题依据与研究内容"
SECTION_B = "已有研究和工作基础"


def _fail(reason: str) -> dict[str, Any]:
    return {"score": 0.0, "passed": False, "reason": reason}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _attr_val(node: ET.Element | None, name: str = "val") -> str:
    if node is None:
        return ""
    return node.attrib.get(f"{{{NS['w']}}}{name}", "")


def _read_docx(path: Path) -> dict[str, Any]:
    try:
        with ZipFile(path) as zf:
            document_xml = zf.read("word/document.xml")
            members = sorted(zf.namelist())
    except (BadZipFile, KeyError) as exc:
        raise ValueError(f"{path.name} is not a valid docx with word/document.xml: {exc}") from exc

    root = ET.fromstring(document_xml)
    text_nodes = [node.text or "" for node in root.findall(".//w:t", NS)]
    full_text = _norm("".join(text_nodes))

    paragraphs: list[dict[str, Any]] = []
    for para in root.findall(".//w:p", NS):
        text = _norm("".join(node.text or "" for node in para.findall(".//w:t", NS)))
        if not text:
            continue
        ppr = para.find("w:pPr", NS)
        pstyle = _attr_val(ppr.find("w:pStyle", NS) if ppr is not None else None)
        jc = _attr_val(ppr.find("w:jc", NS) if ppr is not None else None)
        spacing = ppr.find("w:spacing", NS) if ppr is not None else None
        spacing_sig = tuple(sorted((k.split("}")[-1], v) for k, v in (spacing.attrib if spacing is not None else {}).items()))
        run_sigs = []
        for run in para.findall("w:r", NS):
            rpr = run.find("w:rPr", NS)
            tags = []
            if rpr is not None:
                for child in rpr:
                    tags.append((child.tag.split("}")[-1], tuple(sorted((k.split("}")[-1], v) for k, v in child.attrib.items()))))
            run_sigs.append(tuple(tags))
        paragraphs.append(
            {
                "text": text,
                "pstyle": pstyle,
                "jc": jc,
                "spacing": spacing_sig,
                "runs": tuple(run_sigs),
            }
        )

    return {
        "members": members,
        "text_nodes": text_nodes,
        "full_text": full_text,
        "paragraphs": paragraphs,
    }


def _order_ok(full_text: str) -> bool:
    first = full_text.find(SECTION_B)
    second = full_text.find(SECTION_A)
    return first >= 0 and second >= 0 and first < second


def _paragraph_format_score(output_paras: list[dict[str, Any]], reference_paras: list[dict[str, Any]]) -> float:
    if not reference_paras:
        return 0.0
    n = min(len(output_paras), len(reference_paras))
    if n == 0:
        return 0.0
    hits = 0.0
    for out, ref in zip(output_paras[:n], reference_paras[:n]):
        if out["text"] != ref["text"]:
            continue
        local = 0.40
        local += 0.20 if out["pstyle"] == ref["pstyle"] else 0.0
        local += 0.15 if out["jc"] == ref["jc"] else 0.0
        local += 0.15 if out["spacing"] == ref["spacing"] else 0.0
        local += 0.10 if out["runs"] == ref["runs"] else 0.0
        hits += local
    length_penalty = max(0.0, 1.0 - abs(len(output_paras) - len(reference_paras)) / max(1, len(reference_paras)))
    return (hits / len(reference_paras)) * length_penalty


def verify(output_docx: Path, reference_docx: Path, input_docx: Path) -> dict[str, Any]:
    if not output_docx.exists():
        return _fail("missing swapped_template.docx")
    try:
        out = _read_docx(output_docx)
        ref = _read_docx(reference_docx)
        src = _read_docx(input_docx)
    except Exception as exc:
        return _fail(f"failed to parse docx: {exc}")

    if out["full_text"] == src["full_text"]:
        return _fail("output text is unchanged from the input template")
    if not _order_ok(out["full_text"]):
        return _fail("section order is not swapped: 已有研究和工作基础 must precede 选题依据与研究内容")

    exact_text = out["full_text"] == ref["full_text"]
    content_counter_ok = Counter(out["text_nodes"]) == Counter(ref["text_nodes"])
    format_score = _paragraph_format_score(out["paragraphs"], ref["paragraphs"])

    score = 0.0
    score += 0.55 if exact_text else 0.0
    score += 0.20 if content_counter_ok else 0.0
    score += 0.10
    score += 0.15 * format_score

    if not exact_text:
        score = min(score, 0.60)
    passed = score >= 0.85
    return {
        "score": round(max(0.0, min(1.0, score)), 6),
        "passed": passed,
        "exact_text_match": exact_text,
        "content_multiset_match": content_counter_ok,
        "section_order_ok": True,
        "format_score": round(format_score, 6),
        "output_paragraphs": len(out["paragraphs"]),
        "reference_paragraphs": len(ref["paragraphs"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-docx", required=True)
    parser.add_argument("--reference-docx", required=True)
    parser.add_argument("--input-docx", required=True)
    args = parser.parse_args()
    result = verify(Path(args.output_docx), Path(args.reference_docx), Path(args.input_docx))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("score", 0.0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
