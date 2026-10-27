import pytest

from tasks.business_finance.equity_research_summary.scripts.score_workbook import (
    _expected_formula_value,
)


@pytest.fixture
def values():
    return {
        "Total Revenue": 100,
        "Cost of Revenue": 60,
        "Gross Profit": 40,
        "Operating Expenses": 15,
        "Operating Income": 25,
        "Net Income": 20,
        "Total Assets": 200,
        "Total Liabilities": 80,
        "Total Equity": 120,
        "Total Debt": 30,
        "Operating Cash Flow": 40,
        "Capital Expenditures": 10,
        "Free Cash Flow": 30,
        "Market Cap ($B)": 2,
        "Trailing P/E": 10,
    }


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Gross Profit", 40),
        ("Gross Margin %", 0.4),
        ("Operating Income", 25),
        ("Operating Margin %", 0.25),
        ("Net Margin %", 0.2),
        ("Total Equity", 120),
        ("Debt / Equity", 0.25),
        ("Free Cash Flow", 30),
        ("FCF Margin %", 0.3),
        ("Market Cap / Revenue", 20),
        ("Trailing P/E (x)", 10),
        ("ROE", 1 / 6),
        ("Free Cash Flow Yield", 0.015),
        ("Debt / Total Assets", 0.15),
    ],
)
def test_financial_formulas_follow_the_task_instructions(label, expected, values):
    assert _expected_formula_value(label, values) == pytest.approx(expected)
