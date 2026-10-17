from copy import copy
from io import BytesIO

import pytest
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

from tasks.business_finance.equity_research_summary.scripts.score_workbook import (
    score_workbook_bytes,
)


@pytest.fixture
def workbook_case():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "TSLA Financial Summary"
    sheet["A2"] = "Live market data: Yahoo Finance | Historicals: FY2023 SEC 10-K ($mm)"
    sheet["A2"].font = Font(italic=True, color="808080")
    titles = [
        ("Market Data", "Market Data (Yahoo Finance)"),
        ("Valuation Metrics", "Valuation Metrics (Yahoo Finance)"),
        ("Income Statement", "Income Statement (FY2023, $mm)"),
        ("Balance Sheet", "Balance Sheet (FY2023, $mm)"),
        ("Cash Flow", "Cash Flow (FY2023, $mm)"),
        ("Key Metrics Dashboard", "Key Metrics Dashboard"),
    ]
    for index, (_, title) in enumerate(titles):
        cell = sheet.cell(row=4 + index * 6, column=1, value=title)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    sheet["A5"], sheet["B5"] = "Current Price ($)", 250.0
    sheet["A17"], sheet["B17"] = "Total Revenue", 96773
    sheet["A29"], sheet["B29"] = "Operating Cash Flow", 13256
    sheet["A30"], sheet["B30"] = "Capital Expenditures", 8969
    sheet["A31"], sheet["B31"] = "Free Cash Flow", "=B29-B30"
    manifest = {
        "sheet_name_contains": "TSLA",
        "section_headers": [{"text": title, "bold": True, "fill": True} for title, _ in titles],
        "fixed_values": [{"label": "Total Revenue", "value": 96773, "tolerance": 1}],
        "formula_cells": [
            {
                "label": "Free Cash Flow",
                "must_reference": ["Operating Cash Flow", "Capital Expenditures"],
            }
        ],
        "live_labels": ["Current Price ($)"],
    }
    return workbook, manifest


def score_case(workbook_case):
    workbook, manifest = workbook_case
    buffer = BytesIO()
    workbook.save(buffer)
    return score_workbook_bytes(buffer.getvalue(), manifest)


def test_actual_kimi_source_note_and_qualified_headers_pass(workbook_case):
    result = score_case(workbook_case)
    assert result.score == 1.0
    assert result.passed
    assert result.reasons == []


@pytest.mark.parametrize(
    "title",
    [
        "Market Data",
        "  MARKET   DATA  ",
        "Market Data [Yahoo Finance]",
        "1. Market Data:",
        "Market Data: Yahoo Finance",
        "Market Data - Yahoo Finance",
        "Market Data – Yahoo Finance",
        "Market Data — Yahoo Finance",
        "2) MARKET   DATA (USD) — Yahoo Finance",
    ],
)
def test_title_matching_ignores_case_spacing_and_heading_annotations(workbook_case, title):
    workbook_case[0].active["A4"] = title
    assert score_case(workbook_case).passed


@pytest.mark.parametrize("bold,filled", [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize("styled_note", [False, True])
def test_real_header_style_errors_fail_even_with_a_styled_note(
    workbook_case, bold, filled, styled_note
):
    sheet = workbook_case[0].active
    sheet["A4"].font = Font(bold=bold)
    sheet["A4"].fill = PatternFill("solid", fgColor="1F4E78") if filled else PatternFill()
    if styled_note:
        sheet["A2"].font = Font(bold=True)
        sheet["A2"].fill = PatternFill("solid", fgColor="1F4E78")
    expected = []
    if not bold:
        expected.append("header_not_bold:Market Data")
    if not filled:
        expected.append("header_no_fill:Market Data")
    result = score_case(workbook_case)
    assert not result.passed
    assert result.reasons == expected


@pytest.mark.parametrize(
    "first_bold,first_filled,second_bold,second_filled",
    [
        (True, True, True, True),
        (True, True, False, False),
        (False, True, True, True),
        (True, False, True, True),
        (False, False, True, True),
    ],
)
@pytest.mark.parametrize("first_address,second_address", [("D3", "A4"), ("A4", "D4"), ("A4", "D5")])
def test_duplicate_headers_preserve_row_then_column_order_without_style_selection(
    workbook_case,
    first_bold,
    first_filled,
    second_bold,
    second_filled,
    first_address,
    second_address,
):
    sheet = workbook_case[0].active
    for address, title, bold, filled in [
        (first_address, "Market Data: Yahoo Finance", first_bold, first_filled),
        (second_address, "Market Data (another source)", second_bold, second_filled),
    ]:
        sheet[address] = title
        sheet[address].font = Font(bold=bold)
        sheet[address].fill = PatternFill("solid", fgColor="1F4E78") if filled else PatternFill()
    expected = []
    if not first_bold:
        expected.append("header_not_bold:Market Data")
    if not first_filled:
        expected.append("header_no_fill:Market Data")
    result = score_case(workbook_case)
    assert result.passed == (first_bold and first_filled)
    assert result.reasons == expected


@pytest.mark.parametrize("separator", [":", "-", "–", "—"])
@pytest.mark.parametrize("bold,filled", [(False, True), (True, False)])
def test_subtitled_header_still_requires_bold_and_fill(workbook_case, separator, bold, filled):
    sheet = workbook_case[0].active
    sheet["A4"] = f"Market Data {separator} Yahoo Finance"
    sheet["A4"].font = Font(bold=bold)
    sheet["A4"].fill = PatternFill("solid", fgColor="1F4E78") if filled else PatternFill()
    result = score_case(workbook_case)
    assert not result.passed
    assert result.reasons == [
        "header_not_bold:Market Data" if not bold else "header_no_fill:Market Data"
    ]


@pytest.mark.parametrize(
    "note",
    [
        "Live market data: Yahoo Finance | Historicals: FY2023 SEC 10-K ($mm)",
        "Source: Market Data from Yahoo Finance",
        "Market Data sources are Yahoo Finance and SEC filings",
        '=CONCAT("Market Data", " (Yahoo Finance)")',
    ],
)
def test_note_or_formula_cannot_replace_missing_header(workbook_case, note):
    sheet = workbook_case[0].active
    sheet["A4"] = None
    sheet["A2"] = note
    sheet["A2"].font = Font(bold=True)
    sheet["A2"].fill = PatternFill("solid", fgColor="1F4E78")
    assert score_case(workbook_case).reasons == ["header_missing:Market Data"]


def test_header_can_move_to_a_merged_region(workbook_case):
    sheet = workbook_case[0].active
    sheet["D8"] = sheet["A4"].value
    sheet["D8"].font = copy(sheet["A4"].font)
    sheet["D8"].fill = copy(sheet["A4"].fill)
    sheet["A4"] = None
    sheet.merge_cells("D8:F8")
    assert score_case(workbook_case).passed


def test_financial_and_formula_errors_still_fail(workbook_case):
    sheet = workbook_case[0].active
    sheet["B17"] = 1
    sheet["B31"] = 4287
    sheet["B5"] = "unavailable"
    result = score_case(workbook_case)
    assert not result.passed
    assert result.reasons == [
        "fixed_value_mismatch:Total Revenue:1.0!=96773",
        "not_a_formula:Free Cash Flow",
        "live_value_not_numeric:Current Price ($)",
    ]
