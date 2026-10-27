"""Workbook scoring contracts from the staged equity research instructions."""

from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Color, Font, PatternFill

from tasks.business_finance.equity_research_summary.scripts.score_workbook import (
    score_workbook_bytes,
)

HEADERS = (
    "Market Data",
    "Valuation Metrics",
    "Income Statement",
    "Balance Sheet",
    "Cash Flow",
    "Key Metrics Dashboard",
)
HEADER_ADDRESSES = dict(zip(HEADERS, ("A2", "A10", "A18", "A26", "A34", "A42")))
FIELD_ADDRESSES = {
    "Current Price ($)": "A3",
    "Market Cap ($B)": "A4",
    "Trailing P/E": "A11",
    "Total Revenue": "A19",
    "Operating Cash Flow": "A35",
    "Capital Expenditures": "A36",
    "Free Cash Flow": "A37",
    "FCF Margin %": "A38",
    "Market Cap / Revenue (x)": "A43",
    "Trailing P/E (x)": "A44",
}


@pytest.fixture
def workbook_case():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "TSLA Financial Summary"
    for title, address in HEADER_ADDRESSES.items():
        sheet[address] = title
        sheet[address].font = Font(bold=True)
        sheet[address].fill = PatternFill("solid", fgColor="1F4E78")
    values = {
        "Current Price ($)": 250.0,
        "Market Cap ($B)": 800.0,
        "Trailing P/E": 60.0,
        "Total Revenue": 96773,
        "Operating Cash Flow": 13256,
        "Capital Expenditures": 8969,
        "Free Cash Flow": "=B35-B36",
        "FCF Margin %": "=B37/B19",
        "Market Cap / Revenue (x)": "=B4*1000/B19",
        "Trailing P/E (x)": "=B11",
    }
    for label, address in FIELD_ADDRESSES.items():
        sheet[address] = label
        sheet.cell(sheet[address].row, sheet[address].column + 1, values[label])
    manifest = {
        "sheet_name_contains": "TSLA",
        "section_headers": [{"text": title, "bold": True, "fill": True} for title in HEADERS],
        "fixed_values": [
            {"label": "Total Revenue", "value": 96773, "tolerance": 1.0},
            {"label": "Operating Cash Flow", "value": 13256, "tolerance": 1.0},
            {"label": "Capital Expenditures", "value": 8969, "tolerance": 1.0},
        ],
        "formula_cells": [
            {
                "label": "Free Cash Flow",
                "must_reference": ["Operating Cash Flow", "Capital Expenditures"],
            },
            {"label": "FCF Margin %", "must_reference": ["Free Cash Flow", "Total Revenue"]},
        ],
        "dashboard_formulas": [
            {
                "label": "Market Cap / Revenue",
                "must_reference": ["Market Cap ($B)", "Total Revenue"],
            },
            {"label": "Trailing P/E (x)", "must_reference": ["Trailing P/E"]},
        ],
        "live_labels": ["Current Price ($)", "Market Cap ($B)", "Trailing P/E"],
    }
    return workbook, manifest


def score_case(workbook_case):
    workbook, manifest = workbook_case
    buffer = BytesIO()
    workbook.save(buffer)
    return score_workbook_bytes(buffer.getvalue(), manifest)


def assert_perfect(workbook_case):
    result = score_case(workbook_case)
    assert result.reasons == []
    assert result.passed
    assert result.score == 1.0


def test_exact_public_labels_and_documented_dashboard_units_pass(workbook_case):
    assert_perfect(workbook_case)


@pytest.mark.parametrize("title", HEADERS)
@pytest.mark.parametrize(
    "template",
    [
        "{}",
        "{} | Yahoo Finance",
        "Yahoo Finance | {}",
        "FY2023 | {} | $mm",
        "Section: {}",
        "***{}***",
        "[{}]",
        "FY2023{}$mm",
        "Sources / {} / Notes",
        "{} (FY2023, $mm)",
    ],
)
def test_header_text_may_have_arbitrary_prefixes_suffixes_and_symbols(
    workbook_case, title, template
):
    workbook_case[0].active[HEADER_ADDRESSES[title]] = template.format(title)
    assert_perfect(workbook_case)


@pytest.mark.parametrize("title", HEADERS)
@pytest.mark.parametrize("separator", ["   ", "\t", "\n", "\u00a0"])
def test_header_matching_normalizes_case_and_whitespace(workbook_case, title, separator):
    workbook_case[0].active[HEADER_ADDRESSES[title]] = (
        " \t" + separator.join(title.swapcase().split()) + "\n "
    )
    assert_perfect(workbook_case)


@pytest.mark.parametrize("title", HEADERS)
@pytest.mark.parametrize("bold,filled", [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize("candidate_address", ["A1", "D1", "Z50"])
def test_any_fully_styled_header_candidate_satisfies_the_section(
    workbook_case, title, bold, filled, candidate_address
):
    sheet = workbook_case[0].active
    candidate = sheet[candidate_address]
    candidate.value = title
    candidate.font = Font(bold=bold)
    candidate.fill = PatternFill("solid", fgColor="1F4E78") if filled else PatternFill()
    assert_perfect(workbook_case)


@pytest.mark.parametrize("bold,filled", [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize("template", ["{}", "Source | {} | FY2023"])
def test_matching_header_still_requires_both_bold_and_colored_fill(
    workbook_case, bold, filled, template
):
    cell = workbook_case[0].active["A2"]
    cell.value = template.format("Market Data")
    cell.font = Font(bold=bold)
    cell.fill = PatternFill("solid", fgColor="1F4E78") if filled else PatternFill()
    result = score_case(workbook_case)
    assert not result.passed
    assert result.score < 1.0
    assert result.reasons


def test_different_candidates_cannot_supply_bold_and_fill_separately(workbook_case):
    sheet = workbook_case[0].active
    sheet["A2"].fill = PatternFill()
    sheet["D2"] = "Market Data"
    sheet["D2"].font = Font(bold=False)
    sheet["D2"].fill = PatternFill("solid", fgColor="1F4E78")
    result = score_case(workbook_case)
    assert not result.passed
    assert result.score < 1.0


def test_one_cell_cannot_stand_in_for_all_section_headers(workbook_case):
    workbook, _ = workbook_case
    sheet = workbook.active
    for address in HEADER_ADDRESSES.values():
        sheet[address] = None
    sheet["D1"] = " | ".join(HEADERS)
    sheet["D1"].font = Font(bold=True)
    sheet["D1"].fill = PatternFill("solid", fgColor="1F4E78")
    result = score_case(workbook_case)
    assert not result.passed
    assert sum(reason.startswith("header_not_distinct:") for reason in result.reasons) == 5


def test_header_matching_uses_alternative_cells_when_needed(workbook_case):
    workbook, _ = workbook_case
    sheet = workbook.active
    sheet["A2"] = "Market Data | Valuation Metrics"
    sheet["A10"] = "Market Data extra context extra context"
    assert score_case(workbook_case).passed


@pytest.mark.parametrize(
    "color",
    [
        pytest.param(Color(rgb="1F4E78"), id="rgb-six-digits"),
        pytest.param(Color(rgb="FF1F4E78"), id="rgb-eight-digits"),
        pytest.param(Color(rgb="000000"), id="rgb-black"),
        pytest.param(Color(rgb="FF000000"), id="argb-black"),
        pytest.param(Color(theme=4), id="theme-accent"),
        pytest.param(Color(theme=4, tint=0.4), id="theme-tinted-accent"),
        pytest.param(Color(indexed=10), id="indexed-red"),
        pytest.param(Color(indexed=12), id="indexed-blue"),
    ],
)
def test_real_rgb_theme_and_indexed_fills_survive_xlsx_roundtrip(workbook_case, color):
    workbook, _ = workbook_case
    workbook.active["A2"].fill = PatternFill("solid", fgColor=color)
    buffer = BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    restored = load_workbook(buffer).active["A2"].fill
    assert restored.patternType == "solid"
    assert restored.fgColor.type == color.type
    assert restored.fgColor.value == color.value
    assert_perfect(workbook_case)


@pytest.mark.parametrize(
    "fill",
    [
        pytest.param(PatternFill(), id="no-fill"),
        pytest.param(PatternFill(fgColor="1F4E78"), id="color-without-pattern"),
        pytest.param(PatternFill("solid", fgColor="FFFFFF"), id="rgb-white"),
        pytest.param(PatternFill("solid", fgColor="FFFFFFFF"), id="argb-white"),
        pytest.param(PatternFill("solid", fgColor=Color(theme=0)), id="theme-white"),
        pytest.param(PatternFill("solid", fgColor=Color(indexed=9)), id="indexed-white"),
        pytest.param(PatternFill("solid", fgColor=Color(auto=True)), id="automatic-color"),
    ],
)
def test_empty_white_or_automatic_fill_is_not_a_colored_background(workbook_case, fill):
    workbook_case[0].active["A2"].fill = fill
    result = score_case(workbook_case)
    assert not result.passed
    assert result.reasons == ["header_no_fill:Market Data"]


@pytest.mark.parametrize("label", FIELD_ADDRESSES)
def test_field_matching_normalizes_only_case_and_whitespace(workbook_case, label):
    workbook_case[0].active[FIELD_ADDRESSES[label]] = (
        " \t" + " \t ".join(label.swapcase().split()) + "\n "
    )
    assert_perfect(workbook_case)


@pytest.mark.parametrize(
    "label,reason_prefix",
    [
        ("Total Revenue", "fixed_label_missing:Total Revenue"),
        ("Free Cash Flow", "formula_label_missing:Free Cash Flow"),
        ("Trailing P/E (x)", "formula_label_missing:Trailing P/E (x)"),
        ("Current Price ($)", "live_label_missing:Current Price ($)"),
        ("Market Cap / Revenue (x)", "formula_label_missing:Market Cap / Revenue"),
    ],
)
@pytest.mark.parametrize("template", ["Not {}", "{} Forecast", "Prior {}"])
def test_nonexact_field_labels_cannot_replace_required_labels(
    workbook_case, label, reason_prefix, template
):
    workbook_case[0].active[FIELD_ADDRESSES[label]] = template.format(label)
    result = score_case(workbook_case)
    assert not result.passed
    assert result.score < 1.0
    assert reason_prefix in result.reasons


@pytest.mark.parametrize("label", ["Operating Cash Flow", "Capital Expenditures"])
def test_formula_references_require_the_exact_source_label(workbook_case, label):
    workbook_case[0].active[FIELD_ADDRESSES[label]] = f"Not {label}"
    result = score_case(workbook_case)
    assert not result.passed
    assert f"formula_ref_label_missing:Free Cash Flow->{label}" in result.reasons


@pytest.mark.parametrize("label", ["Total Revenue", "Trailing P/E", "Free Cash Flow"])
def test_exact_field_wins_over_earlier_misleading_substring_label(workbook_case, label):
    sheet = workbook_case[0].active
    sheet["D1"] = f"Not {label}"
    sheet["E1"] = -999
    assert_perfect(workbook_case)


@pytest.mark.parametrize("separator", ["—", "–", "-", "|"])
def test_field_label_can_include_an_explanatory_suffix(workbook_case, separator):
    sheet = workbook_case[0].active
    sheet[FIELD_ADDRESSES["Free Cash Flow"]] = (
        f"Free Cash Flow {separator} operating cash less capex"
    )
    assert_perfect(workbook_case)


def test_forecast_label_cannot_replace_current_field(workbook_case):
    sheet = workbook_case[0].active
    sheet[FIELD_ADDRESSES["Total Revenue"]] = "Total Revenue — Forecast"
    result = score_case(workbook_case)
    assert "fixed_label_missing:Total Revenue" in result.reasons


def test_dashboard_trailing_pe_cannot_replace_missing_live_trailing_pe(workbook_case):
    workbook_case[0].active["A11"] = None
    result = score_case(workbook_case)
    assert not result.passed
    assert "live_label_missing:Trailing P/E" in result.reasons
    assert "formula_ref_label_missing:Trailing P/E (x)->Trailing P/E" in result.reasons


def test_missing_header_is_reported_without_relaxing_other_checks(workbook_case):
    workbook_case[0].active["A2"] = None
    result = score_case(workbook_case)
    assert not result.passed
    assert result.reasons == ["header_missing:Market Data"]


@pytest.mark.parametrize("value", [True, False, "NaN", "Infinity", "-Infinity"])
def test_nonfinite_and_boolean_live_values_are_not_numbers(workbook_case, value):
    workbook_case[0].active["B3"] = value
    assert "live_value_not_numeric:Current Price ($)" in score_case(workbook_case).reasons


def test_custom_indexed_color_uses_workbook_palette(workbook_case):
    workbook, _ = workbook_case
    workbook._colors = list(workbook._colors)
    workbook._colors[9] = "FF1F4E78"
    workbook.active["A2"].fill = PatternFill("solid", fgColor=Color(indexed=9))
    assert_perfect(workbook_case)
