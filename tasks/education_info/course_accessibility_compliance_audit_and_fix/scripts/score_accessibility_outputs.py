"""Local scorer for course_accessibility_compliance_audit_and_fix."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from typing import Any

import fitz
from bs4 import BeautifulSoup

PUBLIC_DEFECT_FIELDS = (
    "file",
    "element",
    "wcag_criterion",
    "severity",
    "description",
    "remediation_action",
)
DEFECT_IDENTITY_FIELDS = ("file", "element", "wcag_criterion")
CAPTION_WER_THRESHOLD = 0.10
WEIGHTS = {
    "defect_recall": 0.35,
    "html_fix_validity": 0.35,
    "pdf_remediation_validity": 0.15,
    "video_bundle_validity": 0.15,
}
GENERIC_LINK_TEXT = {"click here", "here", "learn more", "more", "link"}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _normalize_wcag(value: str) -> str:
    """Extract the numeric WCAG criterion prefix, e.g. '3.1.1 Language of Page' -> '3.1.1'."""
    m = re.match(r"(\d+\.\d+\.\d+)", value.strip())
    return m.group(1) if m else _normalize_text(value)


def _normalize_element(value: str) -> str:
    """Normalize element identifiers so '<html>', 'html', '<img class=hero-image>' and 'img.hero-image' converge.

    Converts HTML-style '<tag class=cls id=id ...>' into CSS-style 'tag.cls#id',
    then lowercases. Attributes other than class/id are dropped.
    """
    s = (value or "").strip()
    s = re.sub(r"^<|>$", "", s)

    m = re.match(r"(\w+)(.*)", s)
    if not m:
        cls_only = re.match(r"\.([\w-]+)", s)
        if cls_only:
            return _normalize_text(f".{cls_only.group(1)}")
        return _normalize_text(s)
    tag = m.group(1)
    rest = m.group(2)

    cls_match = re.search(r"""(?:class=["\']?|\.)([\w-]+)""", rest, flags=re.IGNORECASE)
    id_match = re.search(r"""(?:id=["\']?|#)([\w-]+)""", rest, flags=re.IGNORECASE)

    result = tag
    if cls_match:
        result += f".{cls_match.group(1)}"
    if id_match:
        result += f"#{id_match.group(1)}"
    return _normalize_text(result)


def _parse_csv_rows(text: str) -> list[dict[str, str]]:
    handle = io.StringIO(text)
    reader = csv.DictReader(handle)
    if tuple(reader.fieldnames or ()) != PUBLIC_DEFECT_FIELDS:
        raise ValueError(f"unexpected defect_report.csv columns: {reader.fieldnames}")
    return [{key: (row.get(key) or "").strip() for key in PUBLIC_DEFECT_FIELDS} for row in reader]


def _manifest_public_rows(manifest_json: str) -> list[dict[str, str]]:
    rows = json.loads(manifest_json)
    return [{field: str(row[field]).strip() for field in PUBLIC_DEFECT_FIELDS} for row in rows]


def _element_matches(a: str, b: str) -> bool:
    """Check if two normalized element identifiers refer to the same element."""
    if a == b:
        return True
    for x, y in [(a, b), (b, a)]:
        if "." in y and y.lstrip(".") in x:
            return True
        if "#" in y and y.lstrip("#") in x:
            return True
    return False


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        _normalize_text(row["file"]),
        _normalize_element(row["element"]),
        _normalize_wcag(row["wcag_criterion"]),
    )


def _score_defect_report(manifest_json: str, report_csv: str) -> tuple[float, list[str]]:
    manifest_rows = _manifest_public_rows(manifest_json)
    report_rows = _parse_csv_rows(report_csv)
    expected_keys = [_row_key(row) for row in manifest_rows]
    observed_keys = [_row_key(row) for row in report_rows]
    matched = 0
    used: set[int] = set()
    for e_file, e_elem, e_wcag in expected_keys:
        for j, (o_file, o_elem, o_wcag) in enumerate(observed_keys):
            if j in used:
                continue
            if e_file == o_file and e_wcag == o_wcag and _element_matches(e_elem, o_elem):
                matched += 1
                used.add(j)
                break
    notes = []
    if matched != len(expected_keys):
        notes.append(f"matched {matched}/{len(expected_keys)} expected defect rows")
    return matched / len(expected_keys), notes


def _score_html(html_expectations_json: str, fixed_html_map: dict[str, str]) -> tuple[float, list[str]]:
    payload = json.loads(html_expectations_json)
    notes = []
    checks = 0
    passed = 0
    soups = {
        filename: BeautifulSoup(fixed_html_map.get(filename, ""), "html.parser")
        for filename in set(payload["required_files"]) | {"styles.css"}
    }
    stylesheet = fixed_html_map.get("styles.css", "")

    def check(condition: bool, note: str) -> None:
        nonlocal checks, passed
        checks += 1
        if condition:
            passed += 1
        else:
            notes.append(note)

    def text_of(node: Any | None) -> str:
        if node is None:
            return ""
        return _normalize_text(" ".join(node.stripped_strings))

    for filename in payload["required_files"]:
        check(filename in fixed_html_map and bool(fixed_html_map.get(filename, "").strip()), f"missing html file {filename}")

    check(bool(stylesheet.strip()), "missing stylesheet styles.css")

    for filename in payload["required_lang"]:
        soup = soups[filename]
        html_tag = soup.find("html")
        lang_value = _normalize_text(html_tag.get("lang") if html_tag else "")
        check(bool(lang_value and lang_value.startswith("en")), f"missing valid English lang tag in {filename}")

    for filename in payload["required_files"]:
        soup = soups[filename]
        stylesheet_link = soup.find("link", attrs={"href": "styles.css"})
        check(
            stylesheet_link is not None,
            f"{filename} is missing the linked stylesheet reference to styles.css",
        )

    index_soup = soups["index.html"]
    launch_control = index_soup.select_one("a.button-link, button.button-link, a.launch-card, button.launch-card")
    hero_image = index_soup.select_one("img.hero-image")
    hero_alt = _normalize_text((hero_image.get("alt") if hero_image else "") or "")
    launch_text = text_of(launch_control)
    check(
        hero_image is not None
        and len(hero_alt.split()) >= 4
        and hero_alt not in {"image", "banner", "hero image"},
        "index.html hero image alt text is still missing or non-descriptive",
    )
    check(
        launch_control is not None
        and launch_control.name in {"a", "button"}
        and bool(launch_text)
        and (
            (launch_control.name == "a" and _normalize_text(launch_control.get("href") or "") == "module1.html")
            or (
                launch_control.name == "button"
                and "module1.html"
                in _normalize_text(
                    " ".join(
                        filter(
                            None,
                            [
                                launch_control.get("onclick"),
                                launch_control.get("formaction"),
                                launch_control.get("data-href"),
                            ],
                        )
                    )
                )
            )
        ),
        "index.html launch control is not a keyboard-reachable link or button",
    )
    check(
        index_soup.select_one("div.launch-card[onclick]") is None,
        "index.html still contains the onclick-only launch card",
    )

    module1_soup = soups["module1.html"]
    module1_headings = [tag.name for tag in module1_soup.find_all(re.compile(r"^h[1-6]$"))]
    summary_heading = module1_soup.select_one(".summary-heading")
    check(
        module1_headings[:2] == ["h1", "h2"],
        f"module1.html heading order is incorrect: {module1_headings[:2]}",
    )
    check(
        summary_heading is not None and summary_heading.name == "h2",
        "module1.html summary heading must be an h2",
    )

    module2_soup = soups["module2.html"]
    lesson_table = module2_soup.select_one("table.lesson-table")
    caption = lesson_table.find("caption") if lesson_table else None
    first_row = lesson_table.find("tr") if lesson_table else None
    resource_link = module2_soup.select_one("a.resource-link")
    resource_link_text = text_of(resource_link)
    check(
        caption is not None and len(text_of(caption).split()) >= 2,
        "module2.html table caption is missing or non-descriptive",
    )
    check(
        lesson_table is not None and len(lesson_table.select("th[scope='col']")) >= 2,
        "module2.html table headers must use th with scope=col",
    )
    check(
        first_row is not None and not first_row.find_all("td") and len(first_row.find_all("th")) >= 1,
        "module2.html first table row still uses td cells for headers",
    )
    check(
        resource_link is not None
        and resource_link_text not in GENERIC_LINK_TEXT
        and len(resource_link_text) >= 8,
        "module2.html resource link text is still generic",
    )

    module3_soup = soups["module3.html"]
    email_label = module3_soup.select_one("label[for='email']")
    accordion = module3_soup.select_one(".accordion-toggle")
    check(
        email_label is not None and text_of(email_label) == "email address",
        "module3.html email input is missing an associated label",
    )
    check(
        accordion is not None
        and accordion.name == "button"
        and accordion.get("aria-expanded") in {"true", "false"}
        and bool(text_of(accordion)),
        "module3.html accordion toggle must be a button with aria-expanded",
    )

    resources_soup = soups["resources.html"]
    icon_link = resources_soup.select_one("a.icon-link")
    icon_link_aria = _normalize_text(
        " ".join(
            filter(
                None,
                [icon_link.get("aria-label"), icon_link.get("title")] if icon_link is not None else [],
            )
        )
    )
    icon_link_name = icon_link_aria if icon_link_aria else text_of(icon_link)
    check(
        icon_link is not None
        and _normalize_text(icon_link.get("href") or "") == "index.html"
        and len(icon_link_name) >= 4,
        "resources.html icon link is missing an accessible name",
    )

    contrast_match = re.search(r"\.contrast-note\s*\{([^}]*)\}", stylesheet, flags=re.IGNORECASE | re.DOTALL)
    contrast_block = contrast_match.group(1).lower() if contrast_match else ""
    focus_match = re.search(r":focus[^{]*\{([^}]*)\}", stylesheet, flags=re.IGNORECASE | re.DOTALL)
    focus_block = focus_match.group(1).lower() if focus_match else ""
    check(
        bool(contrast_block)
        and "#94a3b8" not in contrast_block
        and "color" in contrast_block,
        "styles.css still contains the low-contrast text color",
    )
    check(
        "outline: none" not in _normalize_text(stylesheet)
        and bool(focus_block)
        and "outline" in focus_block
        and "none" not in focus_block,
        "styles.css focus styling is still missing or disabled",
    )

    score = passed / checks if checks else 0.0
    return score, notes


def _pdf_payload(pdf_bytes: bytes) -> tuple[str, str, int, list[str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        lines = []
        for page in doc:
            lines.extend(line.strip() for line in page.get_text("text").splitlines() if line.strip())
        text = "\n".join(lines)
        title = (doc.metadata or {}).get("title") or ""
        return text, title, doc.page_count, lines
    finally:
        doc.close()


def _score_pdfs(
    pdf_expectations_json: str,
    fixed_pdf_map: dict[str, bytes],
    input_pdf_map: dict[str, bytes],
) -> tuple[float, list[str]]:
    expectations = json.loads(pdf_expectations_json)
    notes = []
    checks = 0
    passed = 0
    for filename, expectation in expectations.items():
        pdf_bytes = fixed_pdf_map.get(filename)
        input_bytes = input_pdf_map.get(filename)
        if pdf_bytes is None:
            notes.append(f"missing fixed pdf {filename}")
            continue
        if input_bytes is None:
            notes.append(f"missing staged input pdf {filename}")
            continue
        text, title, page_count, lines = _pdf_payload(pdf_bytes)
        _input_text, _input_title, input_page_count, input_lines = _pdf_payload(input_bytes)
        checks += 1
        if bool(title.strip()):
            passed += 1
        else:
            notes.append(f"missing descriptive pdf title metadata for {filename}")
        checks += 1
        if page_count == input_page_count:
            passed += 1
        else:
            notes.append(f"wrong page count for {filename}: {page_count} != {input_page_count}")
        normalized_text = _normalize_text(text)
        for phrase in expectation["required_phrases"]:
            checks += 1
            if _normalize_text(phrase) in normalized_text:
                passed += 1
            else:
                notes.append(f"missing required pdf phrase in {filename}: {phrase}")
        checks += 1
        fixed_body = _normalize_text("\n".join(lines[1:]))
        input_body = _normalize_text("\n".join(input_lines[1:]))
        if fixed_body == input_body:
            passed += 1
        else:
            notes.append(f"fixed pdf body text does not preserve the staged input content for {filename}")
    score = passed / checks if checks else 0.0
    return score, notes


def _parse_vtt_text(vtt_text: str) -> str:
    words = []
    for line in vtt_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT" or "-->" in stripped:
            continue
        words.append(stripped)
    return " ".join(words)


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = _normalize_text(reference).split()
    hyp_tokens = _normalize_text(hypothesis).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    rows = len(ref_tokens) + 1
    cols = len(hyp_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / len(ref_tokens)


def _sha256_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _score_video_bundle(
    transcript_map: dict[str, str],
    input_video_map: dict[str, bytes],
    output_video_map: dict[str, bytes],
    vtt_map: dict[str, str],
) -> tuple[float, list[str]]:
    notes = []
    checks = 0
    passed = 0
    for slug, transcript in transcript_map.items():
        input_video = input_video_map.get(slug)
        output_video = output_video_map.get(slug)
        vtt_text = vtt_map.get(slug)
        checks += 1
        if input_video is not None and output_video is not None and _sha256_digest(input_video) == _sha256_digest(output_video):
            passed += 1
        else:
            notes.append(f"{slug}.mp4 does not preserve the staged source video bytes")
        checks += 1
        if not vtt_text:
            notes.append(f"missing caption file for {slug}")
            continue
        wer = _word_error_rate(transcript, _parse_vtt_text(vtt_text))
        if wer <= CAPTION_WER_THRESHOLD:
            passed += 1
        else:
            notes.append(f"{slug} WER {wer:.3f} exceeds {CAPTION_WER_THRESHOLD:.2f}")
    return (passed / checks if checks else 0.0), notes


def score_outputs(
    *,
    manifest_json: str,
    html_expectations_json: str,
    pdf_expectations_json: str,
    transcript_map: dict[str, str],
    defect_report_csv: str,
    audit_summary_md: str,
    fixed_html_map: dict[str, str],
    fixed_pdf_map: dict[str, bytes],
    input_pdf_map: dict[str, bytes],
    input_video_map: dict[str, bytes],
    output_video_map: dict[str, bytes],
    vtt_map: dict[str, str],
) -> dict[str, Any]:
    if not audit_summary_md.strip():
        return {"score": 0.0, "reason": "empty audit summary"}

    defect_score, defect_notes = _score_defect_report(manifest_json, defect_report_csv)
    html_score, html_notes = _score_html(html_expectations_json, fixed_html_map)
    pdf_score, pdf_notes = _score_pdfs(pdf_expectations_json, fixed_pdf_map, input_pdf_map)
    video_score, video_notes = _score_video_bundle(transcript_map, input_video_map, output_video_map, vtt_map)

    weighted = (
        WEIGHTS["defect_recall"] * defect_score
        + WEIGHTS["html_fix_validity"] * html_score
        + WEIGHTS["pdf_remediation_validity"] * pdf_score
        + WEIGHTS["video_bundle_validity"] * video_score
    )
    return {
        "score": round(weighted, 6),
        "component_scores": {
            "defect_recall": defect_score,
            "html_fix_validity": html_score,
            "pdf_remediation_validity": pdf_score,
            "video_bundle_validity": video_score,
        },
        "notes": defect_notes + html_notes + pdf_notes + video_notes,
        "input_pdf_sizes": {name: len(data) for name, data in input_pdf_map.items()},
    }
