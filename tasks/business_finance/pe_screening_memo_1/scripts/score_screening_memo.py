"""Local scorer for business_finance/pe_screening_memo_1 — LLM-judge edition.

Replaces keyword pattern-matching with per-question LLM yes/no judgments
for semantic coverage evaluation.  Hard gates (heading structure, word count,
explicit recommendation) remain rule-based.

Requires: openai, python-dotenv (optional)
Set OPENAI_API_KEY in environment or .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    _script = Path(__file__).resolve()
    for _ancestor in _script.parents:
        _candidate = _ancestor / ".env"
        if _candidate.is_file():
            load_dotenv(_candidate)
            break
except ImportError:
    pass

from openai import OpenAI

TITLE_HEADING = "# Zscaler Screening Memo"
DEFAULT_PASS_THRESHOLD = 0.7
MIN_WORD_COUNT = 250
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are evaluating a private-equity screening memo about Zscaler. "
    "Given a section of the memo and a question, determine whether the section "
    "substantively covers the topic asked about. "
    "Answer with exactly one word: 'yes' or 'no'."
)

RECOMMENDATION_TERMS = [
    "go",
    "no-go",
    "no go",
    "hold",
    "needs more diligence",
    "hold / needs more diligence",
]

# Each list entry is one yes/no question that replaces a keyword pattern group.
SECTION_QUESTIONS: dict[str, list[str]] = {
    "recommendation": [
        "Does this section explain the reasoning or rationale behind the investment recommendation?",
        "Does this section describe specific conditions or criteria that would change the recommendation (e.g., what would need to be true to move from Hold to Go)?",
        "Does this section acknowledge key concerns, risks, or uncertainties that factor into the recommendation?",
    ],
    "investment_thesis": [
        "Does this section discuss Zscaler's core zero-trust security architecture and how it compares to or replaces legacy approaches like VPNs and firewalls?",
        "Does this section discuss the scale of Zscaler's enterprise customer base, such as total customer count or penetration among large enterprises (Fortune 500, Forbes Global 2000)?",
        "Does this section discuss Zscaler's emerging product pillars or strategic growth areas beyond the core platform and their revenue trajectory?",
        "Does this section discuss land-and-expand dynamics, including new customer acquisition, upsell within existing customers, or net retention metrics?",
        "Does this section discuss Zscaler's cloud infrastructure scale or network advantages, such as exchange points, geographic coverage, or telemetry data assets?",
        "Does this section discuss the competitive landscape, including named competitors or competitive positioning?",
    ],
    "financial_summary": [
        "Does this section present Zscaler's revenue figures across multiple periods showing a growth trajectory?",
        "Does this section discuss margin metrics, distinguishing between GAAP and non-GAAP profitability?",
        "Does this section report cash flow metrics such as operating cash flow or free cash flow?",
        "Does this section discuss forward-looking financial indicators such as deferred revenue, remaining performance obligations (RPO), or billings?",
        "Does this section report customer or volume metrics such as ARR, customer counts by spending tier, or net dollar retention rate?",
        "Does this section discuss balance sheet items or identify important financial metrics that are missing from the available data?",
    ],
    "risks": [
        "Does this section discuss competition-related risks or customer renewal/churn risk?",
        "Does this section discuss risks related to channel partner dependency or revenue concentration?",
        "Does this section discuss execution risks for newer or emerging products?",
        "Does this section discuss stock-based compensation, share dilution, or GAAP vs non-GAAP quality-of-earnings concerns?",
        "Does this section discuss risks related to the Red Canary acquisition or M&A integration?",
        "Does this section discuss gaps in the available diligence data or limitations of the source materials?",
    ],
    "appendix": [
        "Does this section describe Zscaler's business model and core value proposition?",
        "Does this section describe specific Zscaler products or solution categories?",
        "Does this section describe Zscaler's customer segments, including enterprise and government customers?",
        "Does this section describe Zscaler's go-to-market strategy, sales channels, or partner ecosystem?",
        "Does this section discuss Zscaler's geographic presence or cloud infrastructure scale?",
        "Does this section mention key financial metrics or recent corporate transactions such as acquisitions?",
    ],
}

# Checked against the full memo text (not per-section).
ANCHOR_QUESTIONS: list[str] = [
    "Does the memo mention Zscaler's zero-trust exchange platform or zero-trust architecture?",
    "Does the memo quantify Zscaler's enterprise customer base, such as total customer count or penetration of Fortune 500 / Forbes Global 2000 companies?",
    "Does the memo discuss Zscaler's newer strategic pillars such as expanded security categories, AI security, or agentic security?",
    "Does the memo discuss the revenue mix between new and existing customers, or specific upsell / land-and-expand metrics?",
    "Does the memo mention Zscaler's cloud exchange network scale, such as the number of exchanges or countries covered?",
    "Does the memo cite specific multi-year revenue figures showing Zscaler's growth trajectory?",
    "Does the memo cite specific gross margin or operating margin percentages, or operating income figures?",
    "Does the memo cite specific free cash flow or operating cash flow dollar amounts?",
    "Does the memo cite specific figures for deferred revenue, RPO, ARR, or net dollar retention rate?",
    "Does the memo mention the Red Canary acquisition or Zscaler's convertible notes?",
]

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9$%.,/-]*")


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None = None
    word_count: int = 0
    matched_heading_count: int = 0
    section_group_hits: dict[str, int] = field(default_factory=dict)
    section_group_totals: dict[str, int] = field(default_factory=dict)
    specific_anchor_hits: int = 0
    specific_anchor_total: int = 0
    missing_required_headings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Text utilities (unchanged from keyword version)
# ---------------------------------------------------------------------------

def _as_text(payload: str | bytes) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8-sig")
    return payload


def _normalize_heading(text: str) -> str:
    normalized = text.replace("—", "-").replace("–", "-").strip()
    normalized = normalized.replace("**", "")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("[Company]", "Zscaler")
    return normalized


def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _heading_level(line: str) -> int:
    return len(line) - len(line.lstrip("#"))


def _extract_sections(markdown_text: str) -> tuple[list[str], dict[str, str]]:
    headings: list[str] = []
    sections: dict[str, list[str]] = {}
    current_heading: str | None = None
    parent_heading: str | None = None

    for raw_line in _as_text(markdown_text).splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("#"):
            normalized = _normalize_heading(stripped)
            level = _heading_level(stripped)
            headings.append(normalized)
            sections.setdefault(normalized, [])
            if level <= 2:
                current_heading = normalized
                parent_heading = normalized
            else:
                current_heading = normalized
            continue
        if current_heading is not None:
            sections[current_heading].append(line)
            if parent_heading is not None and current_heading != parent_heading:
                sections[parent_heading].append(line)

    return headings, {key: "\n".join(value).strip() for key, value in sections.items()}


def _recommendation_is_explicit(text: str) -> bool:
    lowered = text.lower().replace("—", "-").replace("–", "-")
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return any(
        re.search(rf"\b{re.escape(term)}\b", lowered)
        for term in RECOMMENDATION_TERMS
    )


def _missing_headings(
    found_headings: list[str], required_headings: list[str]
) -> list[str]:
    cursor = 0
    missing: list[str] = []
    for required in required_headings:
        while cursor < len(found_headings) and found_headings[cursor] != required:
            cursor += 1
        if cursor >= len(found_headings):
            missing.append(required)
            continue
        cursor += 1
    return missing


def _load_contract(contract_text: str | bytes | None) -> dict:
    if contract_text is None:
        return {}
    return json.loads(_as_text(contract_text))


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Place it in .env or export it."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def _llm_judge(text: str, question: str) -> bool:
    resp = _get_client().chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=5,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Text:\n\"\"\"\n{text}\n\"\"\"\n\nQuestion: {question}",
            },
        ],
    )
    answer = resp.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


def _batch_judge(items: list[tuple[str, str]]) -> list[bool]:
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=10) as pool:
        return list(pool.map(lambda args: _llm_judge(*args), items))


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def score_screening_memo(
    candidate_text: str | bytes,
    reference_text: str | bytes,
    *,
    contract_text: str | bytes | None = None,
) -> ScoreResult:
    del reference_text

    contract = _load_contract(contract_text)
    required_headings = [
        _normalize_heading(heading)
        for heading in contract.get("structural_requirements", {}).get(
            "canonical_headings",
            [
                TITLE_HEADING,
                "## I. Recommendation",
                "## II. Investment Thesis",
                "## III. Financial Summary",
                "## IV. Risks",
                "## V. Appendix - Business Overview",
            ],
        )
    ]
    pass_threshold = float(contract.get("pass_threshold", DEFAULT_PASS_THRESHOLD))

    text = _as_text(candidate_text)
    headings, sections = _extract_sections(text)

    # --- hard gate 1: required headings ---
    missing_required = _missing_headings(headings, required_headings)
    if missing_required:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="memo is missing one or more required top-level headings",
            hard_gate="missing required top-level sections",
            word_count=_count_words(text),
            matched_heading_count=len(required_headings) - len(missing_required),
            missing_required_headings=missing_required,
        )

    # --- hard gate 2: minimum word count ---
    total_words = _count_words(text)
    if total_words < MIN_WORD_COUNT:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"memo is too short ({total_words} words)",
            hard_gate="memo too short",
            word_count=total_words,
            matched_heading_count=len(required_headings),
        )

    # --- hard gate 3: explicit recommendation ---
    recommendation_heading = _normalize_heading("## I. Recommendation")
    recommendation_text = sections.get(recommendation_heading, "")
    if not _recommendation_is_explicit(recommendation_text):
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="recommendation section does not make an explicit Go / No-Go / Hold call",
            hard_gate="explicit recommendation missing",
            word_count=total_words,
            matched_heading_count=len(required_headings),
        )

    # --- build LLM judge batch ---
    judge_items: list[tuple[str, str]] = []
    judge_keys: list[tuple[str, int]] = []

    heading_to_key = {
        _normalize_heading("## I. Recommendation"): "recommendation",
        _normalize_heading("## II. Investment Thesis"): "investment_thesis",
        _normalize_heading("## III. Financial Summary"): "financial_summary",
        _normalize_heading("## IV. Risks"): "risks",
        _normalize_heading("## V. Appendix - Business Overview"): "appendix",
    }
    for heading, key in heading_to_key.items():
        section_text = sections.get(heading, "")
        for qi, question in enumerate(SECTION_QUESTIONS[key]):
            judge_items.append((section_text, question))
            judge_keys.append((key, qi))

    anchor_start = len(judge_items)
    for question in ANCHOR_QUESTIONS:
        judge_items.append((text, question))

    results = _batch_judge(judge_items)

    # --- tally section hits ---
    section_hits: dict[str, int] = {}
    section_totals: dict[str, int] = {}
    for key in SECTION_QUESTIONS:
        section_hits[key] = 0
        section_totals[key] = len(SECTION_QUESTIONS[key])

    # +1 credit for explicit recommendation (hard gate already passed)
    section_hits["recommendation"] += 1
    section_totals["recommendation"] += 1

    for idx, (key, _qi) in enumerate(judge_keys):
        if results[idx]:
            section_hits[key] += 1

    # --- tally anchor hits ---
    anchor_hits = sum(1 for r in results[anchor_start:] if r)
    anchor_total = len(ANCHOR_QUESTIONS)

    # --- composite score (same weights as keyword version) ---
    weighted_coverage = (
        0.15 * (section_hits["recommendation"] / section_totals["recommendation"])
        + 0.25 * (section_hits["investment_thesis"] / section_totals["investment_thesis"])
        + 0.25 * (section_hits["financial_summary"] / section_totals["financial_summary"])
        + 0.20 * (section_hits["risks"] / section_totals["risks"])
        + 0.15 * (section_hits["appendix"] / section_totals["appendix"])
    )

    anchor_score = anchor_hits / anchor_total if anchor_total else 0.0
    score = round(0.75 * weighted_coverage + 0.25 * anchor_score, 6)
    passed = score >= pass_threshold
    reason = (
        "memo satisfies structural, coverage, and evidence thresholds"
        if passed
        else "memo lacks enough coverage or evidence density"
    )
    return ScoreResult(
        score=score,
        passed=passed,
        reason=reason,
        word_count=total_words,
        matched_heading_count=len(required_headings),
        section_group_hits=section_hits,
        section_group_totals=section_totals,
        specific_anchor_hits=anchor_hits,
        specific_anchor_total=anchor_total,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()

    result = score_screening_memo(
        args.candidate.read_text(encoding="utf-8"),
        args.reference.read_text(encoding="utf-8"),
        contract_text=args.contract.read_text(encoding="utf-8") if args.contract else None,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
