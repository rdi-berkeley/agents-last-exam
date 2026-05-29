from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile


THRESHOLD = 84.0
LIVE_SEARCH_TERMS = (
    "威科先行",
    "北大法宝",
    "pkulaw",
    "裁判文书网",
    "中国裁判文书网",
    "法信",
    "无讼",
)
NEGATED_LIVE_SEARCH_PREFIXES = (
    "未使用",
    "没有使用",
    "并未使用",
    "未检索",
    "未通过",
    "不依赖",
    "无需使用",
    "未借助",
)


def _docx_text(path: Path) -> str:
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    text = re.sub(r"</w:p>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _norm(text: str) -> str:
    text = text.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", text)


def _contains(text: str, needle: str) -> bool:
    return needle in _norm(text)


def _case_count(text: str) -> int:
    numbered = len(re.findall(r"案例\d+", text))
    cited = len(re.findall(r"\(\d{4}\)[^\n]{0,50}(?:民终|行终|裁定书|判决书)", text))
    return max(numbered, cited)


def _claims_live_platform_usage(text: str) -> bool:
    term_pattern = "|".join(re.escape(_norm(term).lower()) for term in sorted(LIVE_SEARCH_TERMS, key=len, reverse=True))
    prefix_pattern = "|".join(re.escape(_norm(prefix).lower()) for prefix in NEGATED_LIVE_SEARCH_PREFIXES)
    negated_segment = re.compile(
        rf"(?:{prefix_pattern})(?:{term_pattern})(?:[或和与及、,，/]*(?:{term_pattern}))*"
    )
    for raw_line in text.splitlines():
        line = _norm(raw_line).lower()
        if not line:
            continue
        scrubbed = negated_segment.sub("", line)
        if any(_norm(term).lower() in scrubbed for term in LIVE_SEARCH_TERMS):
            return True
    return False


def _complaint_score(text: str) -> tuple[float, list[str]]:
    score = 0.0
    issues: list[str] = []
    norm = _norm(text)

    if "民事起诉状" in text:
        score += 2
    else:
        issues.append("complaint title missing")

    if re.search(r"[Xx＊*]{2,}", text) or "135-XXXX-XXXX" in text:
        score += 3
    else:
        issues.append("complaint appears insufficiently masked")

    for heading in ["原告", "被告", "诉讼请求", "事实与理由", "证据提示"]:
        if heading in text:
            score += 2
        else:
            issues.append(f"complaint missing section or label: {heading}")

    for keyword, points in [
        ("劳动关系", 3),
        ("工伤认定", 3),
        ("九级", 3),
        ("21天", 2),
        ("仲裁", 2),
        ("2025年5月14日", 2),
        ("停工留薪期", 3),
        ("此致", 2),
    ]:
        if keyword in norm:
            score += points
        else:
            issues.append(f"complaint missing key topic: {keyword}")

    claim_items = [
        "一次性伤残补助金",
        "一次性工伤医疗补助金",
        "一次性伤残就业补助金",
        "停工留薪期工资",
        "医疗费差额",
        "住院伙食补助费",
    ]
    claim_hits = sum(1 for item in claim_items if item in norm)
    score += min(12, claim_hits * 2)
    if claim_hits < 4:
        issues.append("complaint claim coverage is thin")

    if "抵扣" in text or "差额" in text or "垫付" in text:
        score += 3
    else:
        issues.append("complaint does not clearly address already-paid amounts")

    deduction_patterns = [
        ("四米", 10, "complaint states unsupported fall height"),
        ("两三个月", 8, "complaint states unsupported hospitalization duration"),
        ("精神损害", 8, "complaint requests unsupported mental-damages item"),
        ("必胜", 10, "complaint uses outcome-guarantee language"),
    ]
    for pattern, penalty, issue in deduction_patterns:
        if pattern in norm:
            score -= penalty
            issues.append(issue)

    if "叉车" in norm and ("撞" in norm or "碰" in norm) and "不确定" not in norm and "证据不足" not in norm:
        score -= 10
        issues.append("complaint treats forklift collision as proven")

    return max(score, 0.0), issues


def _report_score(text: str) -> tuple[float, list[str]]:
    score = 0.0
    issues: list[str] = []
    norm = _norm(text)

    if "类案检索报告" in text:
        score += 2
    else:
        issues.append("report title missing")

    section_checks = [
        ("检索目的", ("检索目的", "摘要")),
        ("争议焦点", ("争议焦点",)),
        ("类案", ("类案", "案例1")),
        ("裁判规则", ("裁判规则",)),
        ("适用", ("适用", "本案")),
    ]
    for label, options in section_checks:
        if any(option in text for option in options):
            score += 3
        else:
            issues.append(f"report missing section or topic: {label}")

    for keyword, points in [
        ("劳动关系", 4),
        ("工资", 2),
        ("付款主体", 2),
        ("工伤认定", 4),
        ("九级", 3),
        ("垫付", 2),
        ("抵扣", 1),
        ("责任主体", 2),
        ("青岚机电", 1),
        ("本案", 2),
    ]:
        if keyword in norm:
            score += points
        else:
            issues.append(f"report missing key topic: {keyword}")

    cases = _case_count(text)
    if cases >= 5:
        score += 10
    elif cases >= 3:
        score += 6
        issues.append("report cites fewer than five cases")
    else:
        issues.append("report cites too few cases")

    if "必胜" in norm or "一定会支持" in norm:
        score -= 10
        issues.append("report uses outcome-guarantee language")
    if "网上随便搜索" in norm:
        score -= 10
        issues.append("report admits non-rigorous search method")
    if _claims_live_platform_usage(text):
        score -= 20
        issues.append("report claims live legal-platform research instead of the staged static case pack")

    return max(score, 0.0), issues


def evaluate_submission(submission_dir: Path) -> dict:
    complaint_path = submission_dir / "civil_complaint.docx"
    report_path = submission_dir / "similar_case_search_report.docx"
    issues: list[str] = []

    if not complaint_path.exists():
        issues.append("civil_complaint.docx missing")
    if not report_path.exists():
        issues.append("similar_case_search_report.docx missing")
    if issues:
        return {
            "score": 0.0,
            "threshold": THRESHOLD,
            "pass": False,
            "issues": issues,
            "artifacts": {},
        }

    complaint_text = _docx_text(complaint_path)
    report_text = _docx_text(report_path)

    complaint_score, complaint_issues = _complaint_score(complaint_text)
    report_score, report_issues = _report_score(report_text)

    score = complaint_score + report_score
    issues.extend([f"complaint: {msg}" for msg in complaint_issues])
    issues.extend([f"report: {msg}" for msg in report_issues])
    return {
        "score": round(score, 2),
        "threshold": THRESHOLD,
        "pass": score >= THRESHOLD,
        "issues": issues,
        "artifacts": {
            "complaint_score": round(complaint_score, 2),
            "report_score": round(report_score, 2),
            "cited_case_count": _case_count(report_text),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    args = parser.parse_args()
    report = evaluate_submission(Path(args.submission_dir))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
