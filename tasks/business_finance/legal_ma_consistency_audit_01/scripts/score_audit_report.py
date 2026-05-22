"""Scoring helpers for legal_ma_consistency_audit_01."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPORT_SIZE_LIMIT_BYTES = 200_000

SECTION_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
PAGE_TOKEN_RE = re.compile(
    r"(?:\b(?:page|pages|p\.|pp\.)\s*(\d+)(?:\s*[-–]\s*(\d+))?\b|第\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*页)",
    re.IGNORECASE,
)
BULLET_LINE_RE = re.compile(r"^\s*(?:[-*+]|(?:\d+[\.\)]))\s+(?P<content>.+?)\s*$")

DOC_ALIASES = {
    "announcement": (
        "announcement",
        "announcement.pdf",
        "document 1",
        "document1",
        "doc 1",
        "doc1",
        "公告",
    ),
    "buyer1_report": (
        "buyer 1",
        "buyer1",
        "buyer 1 report",
        "buyer1 report",
        "buyer1_report",
        "buyer1_report.pdf",
        "document 2",
        "document2",
        "doc 2",
        "doc2",
        "受让方1",
    ),
    "buyer2_report": (
        "buyer 2",
        "buyer2",
        "buyer 2 report",
        "buyer2 report",
        "buyer2_report",
        "buyer2_report.pdf",
        "document 3",
        "document3",
        "doc 3",
        "doc3",
        "受让方2",
    ),
}

CLAIM_TITLE_HINTS = (
    "finding",
    "issue",
    "error",
    "mismatch",
    "contradiction",
    "discrep",
    "clerical",
    "numerical",
    "structural",
    "logical",
)

NUM_146660323_RE = re.compile(r"146\s*,?\s*660\s*,?\s*323(?:\.0+)?")
NUM_146660332_RE = re.compile(r"146\s*,?\s*660\s*,?\s*332(?:\.0+)?")
PERCENT_45_RE = re.compile(r"45\s*%")
PERCENT_40_RE = re.compile(r"40\s*%")


@dataclass(frozen=True)
class Section:
    level: int
    title: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}".strip()


def _normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("\u3000", " ")
    normalized = normalized.replace("：", ":")
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_search_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("\u3000", " ")
    normalized = normalized.replace("：", ":")
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("’", "'").replace("‘", "'")
    return normalized


def _pages_for_match(match: re.Match[str]) -> set[int]:
    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
    else:
        start = int(match.group(3))
        end = int(match.group(4)) if match.group(4) else start
    if end < start:
        start, end = end, start
    return set(range(start, end + 1))


def _pages_present(text: str) -> set[int]:
    pages: set[int] = set()
    for match in PAGE_TOKEN_RE.finditer(text):
        pages.update(_pages_for_match(match))
    return pages


def _doc_mentioned(normalized_text: str, doc_key: str) -> bool:
    return any(alias in normalized_text for alias in DOC_ALIASES[doc_key])


def _location_chunks(text: str) -> list[str]:
    return [_normalize_search_text(line.strip()) for line in text.splitlines() if line.strip()]


def _find_doc_alias_occurrences(text: str) -> list[tuple[int, int, str]]:
    occurrences: list[tuple[int, int, str]] = []
    position = 0
    while position < len(text):
        best_match: tuple[int, str] | None = None
        for doc_key, aliases in DOC_ALIASES.items():
            for alias in aliases:
                if text.startswith(alias, position):
                    candidate = (len(alias), doc_key)
                    if best_match is None or candidate[0] > best_match[0]:
                        best_match = candidate
        if best_match is None:
            position += 1
            continue
        alias_length, doc_key = best_match
        occurrences.append((position, position + alias_length, doc_key))
        position += alias_length
    return occurrences


def _doc_has_page_pair(text: str, doc_key: str, expected_pages: set[int]) -> bool:
    separators = (";", "；", "|")
    for chunk in _location_chunks(text):
        occurrences = _find_doc_alias_occurrences(chunk)
        for index, (start, end, found_doc_key) in enumerate(occurrences):
            if found_doc_key != doc_key:
                continue
            prev_alias_end = occurrences[index - 1][1] if index > 0 else 0
            next_alias_start = occurrences[index + 1][0] if index + 1 < len(occurrences) else len(chunk)

            prev_separator = max((chunk.rfind(separator, 0, start) for separator in separators), default=-1)
            next_separator_candidates = [
                chunk.find(separator, end) for separator in separators if chunk.find(separator, end) != -1
            ]
            next_separator = min(next_separator_candidates) if next_separator_candidates else len(chunk)

            left_bound = max(prev_alias_end, prev_separator + 1)
            right_bound = min(next_alias_start, next_separator)

            local_chunks = (chunk[left_bound:start], chunk[end:right_bound])
            for local_chunk in local_chunks:
                for match in PAGE_TOKEN_RE.finditer(local_chunk):
                    if _pages_for_match(match) & expected_pages:
                        return True
    return False


def _split_sections(text: str) -> list[Section]:
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return [Section(level=1, title="Full Report", body=text.strip())]

    sections: list[Section] = []
    for index, match in enumerate(matches):
        title = match.group("title").strip()
        level = len(match.group("hashes"))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append(Section(level=level, title=title, body=body))
    return sections


def _looks_like_claim_title(title: str) -> bool:
    title_norm = _normalize_text(title)
    return (
        title_norm.startswith("finding ")
        or any(hint in title_norm for hint in CLAIM_TITLE_HINTS)
    )


def _is_supporting_line(content: str) -> bool:
    normalized = _normalize_text(content)
    if not normalized:
        return True
    if normalized.startswith(
        (
            "documents:",
            "document:",
            "category:",
            "why it is inconsistent:",
            "evidence:",
            "location:",
            "page ",
        )
    ):
        return True
    if normalized.startswith("`document") or normalized.startswith("document ") and "page" in normalized:
        return True
    return False


def _claim_start_content(line: str) -> str | None:
    heading_match = SECTION_HEADING_RE.match(line)
    if heading_match:
        title = heading_match.group("title").strip()
        return title if _looks_like_claim_title(title) else None

    bullet_match = BULLET_LINE_RE.match(line)
    if bullet_match:
        content = bullet_match.group("content").strip()
        if _is_supporting_line(content):
            return None
        return content if _looks_like_claim_title(content) else None

    stripped = line.strip()
    if not stripped or _is_supporting_line(stripped):
        return None
    return stripped if _looks_like_claim_title(stripped) else None


def _split_claim_units(text: str) -> list[Section]:
    units: list[Section] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        title = _claim_start_content(raw_line)
        if title is not None:
            if current_title is not None:
                units.append(
                    Section(
                        level=0,
                        title=current_title,
                        body="\n".join(current_lines).strip(),
                    )
                )
            current_title = title
            current_lines = [raw_line.strip()]
            continue

        if current_title is not None:
            current_lines.append(raw_line.rstrip())

    if current_title is not None:
        units.append(
            Section(
                level=0,
                title=current_title,
                body="\n".join(current_lines).strip(),
            )
        )

    return units


def _is_claim_section(section: Section, normalized_text: str) -> bool:
    if _looks_like_claim_title(section.title):
        return True
    return (
        "evidence" in normalized_text
        and bool(_pages_present(normalized_text))
        and any(_doc_mentioned(normalized_text, doc_key) for doc_key in DOC_ALIASES)
    )


def _match_target_a(text: str, normalized_text: str) -> bool:
    return (
        _doc_mentioned(normalized_text, "announcement")
        and "空股股东" in text
        and (
            "控股股东" in text
            or "typo" in normalized_text
            or "clerical" in normalized_text
            or "should read" in normalized_text
            or "instead of" in normalized_text
            or "error" in normalized_text
        )
    )


def _evidence_target_a(text: str) -> bool:
    return "空股股东" in text


def _location_target_a(text: str) -> bool:
    return _doc_has_page_pair(text, "announcement", {1})


def _match_target_b(text: str, normalized_text: str) -> bool:
    return (
        _doc_mentioned(normalized_text, "announcement")
        and _doc_mentioned(normalized_text, "buyer2_report")
        and bool(NUM_146660323_RE.search(text))
        and bool(NUM_146660332_RE.search(text))
    )


def _evidence_target_b(text: str) -> bool:
    has_low_value = bool(NUM_146660323_RE.search(text))
    has_high_value = bool(NUM_146660332_RE.search(text))
    has_chinese_low_side = "协议转让" in text or "对价" in text
    has_chinese_high_side = "元" in text or "受让方2" in text
    return has_low_value and has_high_value and has_chinese_low_side and has_chinese_high_side


def _location_target_b(text: str) -> bool:
    return (
        _doc_has_page_pair(text, "announcement", {2})
        and _doc_has_page_pair(text, "announcement", {11})
        and _doc_has_page_pair(text, "buyer2_report", {7})
    )


def _match_target_c(text: str, normalized_text: str) -> bool:
    return (
        _doc_mentioned(normalized_text, "announcement")
        and _doc_mentioned(normalized_text, "buyer1_report")
        and bool(PERCENT_45_RE.search(text))
        and bool(PERCENT_40_RE.search(text))
        and (
            "聚源盛业" in text
            or "合伙份额" in text
            or "ownership" in normalized_text
            or "shareholder" in normalized_text
        )
    )


def _evidence_target_c(text: str) -> bool:
    return (
        bool(PERCENT_45_RE.search(text))
        and bool(PERCENT_40_RE.search(text))
        and ("聚源盛业" in text or "合伙份额" in text)
    )


def _location_target_c(text: str) -> bool:
    return _doc_has_page_pair(text, "announcement", {4}) and _doc_has_page_pair(
        text, "buyer1_report", {5}
    )


def _match_target_d(text: str, normalized_text: str) -> bool:
    has_loan_language = (
        "借款" in text
        or "银行" in text
        or "bank loan" in normalized_text
        or "loan" in normalized_text
    )
    return (
        _doc_mentioned(normalized_text, "announcement")
        and _doc_mentioned(normalized_text, "buyer1_report")
        and "自有资金" in text
        and has_loan_language
    )


def _evidence_target_d(text: str) -> bool:
    return "自有资金" in text and ("借款" in text or "银行" in text)


def _location_target_d(text: str) -> bool:
    return _doc_has_page_pair(text, "announcement", {2}) and _doc_has_page_pair(
        text, "buyer1_report", {12}
    )


TARGET_RULES = {
    "target_a": {
        "match": _match_target_a,
        "evidence": _evidence_target_a,
        "location": _location_target_a,
    },
    "target_b": {
        "match": _match_target_b,
        "evidence": _evidence_target_b,
        "location": _location_target_b,
    },
    "target_c": {
        "match": _match_target_c,
        "evidence": _evidence_target_c,
        "location": _location_target_c,
    },
    "target_d": {
        "match": _match_target_d,
        "evidence": _evidence_target_d,
        "location": _location_target_d,
    },
}


def _evaluate_target(
    *,
    target_id: str,
    full_text: str,
    normalized_full_text: str,
    sections: list[Section],
) -> dict[str, Any]:
    rule = TARGET_RULES[target_id]
    matched_sections: list[dict[str, Any]] = []
    for section in sections:
        section_text = section.text
        section_norm = _normalize_text(section_text)
        if rule["match"](section_text, section_norm):
            matched_sections.append(
                {
                    "title": section.title,
                    "evidence_ok": bool(rule["evidence"](section_text)),
                    "location_ok": bool(rule["location"](section_text)),
                }
            )

    matched = bool(matched_sections) or bool(rule["match"](full_text, normalized_full_text))
    evidence_ok = False
    location_ok = False
    if matched_sections:
        evidence_ok = any(item["evidence_ok"] for item in matched_sections)
        location_ok = any(item["location_ok"] for item in matched_sections)
    elif matched:
        evidence_ok = bool(rule["evidence"](full_text))
        location_ok = bool(rule["location"](full_text))

    return {
        "matched": matched,
        "evidence_ok": evidence_ok,
        "location_ok": location_ok,
        "matched_sections": matched_sections,
    }


def score_report_text(*, report_text: str, reference_payload: dict[str, Any]) -> dict[str, Any]:
    if not report_text.strip():
        return {"score": 0.0, "reason": "empty_output"}
    if len(report_text.encode("utf-8")) > REPORT_SIZE_LIMIT_BYTES:
        return {"score": 0.0, "reason": "output_too_large"}

    required_findings = reference_payload.get("required_findings")
    if not isinstance(required_findings, list) or not required_findings:
        return {"score": 0.0, "reason": "invalid_reference_payload"}

    sections = _split_sections(report_text)
    normalized_full_text = _normalize_text(report_text)

    per_target: dict[str, Any] = {}
    matched_target_ids: set[str] = set()
    evidence_penalty_targets: list[str] = []
    location_penalty_targets: list[str] = []

    for finding in required_findings:
        target_id = finding["id"]
        target_eval = _evaluate_target(
            target_id=target_id,
            full_text=report_text,
            normalized_full_text=normalized_full_text,
            sections=sections,
        )
        per_target[target_id] = target_eval
        if target_eval["matched"]:
            matched_target_ids.add(target_id)
            if not target_eval["evidence_ok"]:
                evidence_penalty_targets.append(target_id)
            if not target_eval["location_ok"]:
                location_penalty_targets.append(target_id)

    missing_targets = [finding["id"] for finding in required_findings if finding["id"] not in matched_target_ids]
    if missing_targets:
        return {
            "score": 0.0,
            "reason": "missing_required_targets",
            "matched_targets": sorted(matched_target_ids),
            "missing_targets": missing_targets,
            "false_positive_count": 0,
            "evidence_penalty_targets": evidence_penalty_targets,
            "location_penalty_targets": location_penalty_targets,
            "per_target": per_target,
        }

    false_positive_sections: list[dict[str, Any]] = []
    claim_units = _split_claim_units(report_text)
    if not claim_units:
        claim_units = [
            section
            for section in sections
            if _is_claim_section(section, _normalize_text(section.text))
        ]

    for section in claim_units:
        section_text = section.text
        normalized_section = _normalize_text(section_text)
        if not _is_claim_section(section, normalized_section):
            continue
        matched_in_section = [
            target_id
            for target_id, rule in TARGET_RULES.items()
            if rule["match"](section_text, normalized_section)
        ]
        if matched_in_section:
            continue
        false_positive_sections.append(
            {
                "title": section.title,
                "preview": section.body[:240],
            }
        )

    false_positive_count = len(false_positive_sections)
    score = 1.0
    score -= 0.2 * false_positive_count
    score -= 0.2 * len(evidence_penalty_targets)
    score -= 0.1 * len(location_penalty_targets)
    score = max(0.0, min(1.0, round(score, 6)))

    return {
        "score": score,
        "reason": "scored_with_penalties" if score < 1.0 else "perfect_match",
        "matched_targets": sorted(matched_target_ids),
        "missing_targets": [],
        "false_positive_count": false_positive_count,
        "false_positive_sections": false_positive_sections,
        "evidence_penalty_targets": evidence_penalty_targets,
        "location_penalty_targets": location_penalty_targets,
        "per_target": per_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-file", required=True, type=Path)
    parser.add_argument("--reference-file", required=True, type=Path)
    args = parser.parse_args()

    result = score_report_text(
        report_text=args.report_file.read_text(encoding="utf-8", errors="replace"),
        reference_payload=json.loads(args.reference_file.read_text(encoding="utf-8")),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
