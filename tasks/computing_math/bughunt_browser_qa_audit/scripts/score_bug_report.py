#!/usr/bin/env python
"""Deterministic scorer for LedgerLite browser-only bug reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BUG_PATTERNS: dict[str, list[str]] = {
    'bug_B01_balance_rounding': ['balance.*round', 'rounding.*balance', 'round\\(amount', 'truncat.*cent', 'decimal.*place', 'precision.*balance', 'balance.*wrong', '\\$99\\.9\\b'],
    'bug_B02_discount_as_percentage': ['discount.*flat', 'discount.*percent', 'discount.*wrong', 'discount.*dollar.*amount', 'treated.*flat', 'discount.*calculat.*wrong', 'total.*wrong.*discount'],
    'bug_B03_exchange_rate_cache_stale': ['exchange.*rate.*cache', 'exchange.*rate.*stale', 'rate.*not.*updat', 'stale.*exchange', 'old.*rate.*still.*used', 'rate.*change.*not.*appl'],
    'bug_B06_monthly_report_first_day': ['off.by.one.*report', 'first.*day.*month.*exclud', 'report.*miss.*invoice', 'monthly.*report.*wrong', 'report.*miss.*first', 'report.*count.*wrong'],
    'bug_B07_recurring_leap_year_crash': ['leap.*year', 'february.*31', 'recurring.*crash', 'recurring.*error', '500.*recurring', 'server.*error.*recurring', 'generate.*crash'],
    'bug_B08_case_insensitive_customers': ['case.insensitiv.*customer', 'case.sensitiv.*customer', 'duplicate.*customer', 'same.*name.*different.*case', 'customer.*case', 'create.*both.*acme'],
    'bug_B09_unicode_product_name': ['unicode.*strip', 'character.*remov', 'special.*character.*product', 'accent.*remov', 'product.*name.*chang', 'name.*different.*saved'],
    'bug_B10_refund_restores_inventory': ['refund.*stock', 'refund.*inventor', 'stock.*not.*restor', 'inventory.*refund', 'stock.*same.*after.*refund', 'refund.*not.*return.*stock'],
    'bug_B12_debug_mode': ['debug.*mode', 'debug.*page', 'werkzeug', 'stack.*trace', 'error.*page.*code', 'debugger', 'internal.*error.*detail'],
    'bug_B15_csv_status_filter': ['csv.*filter.*ignor', 'csv.*status.*ignor', 'export.*status.*ignor', 'export.*filter.*not.*work', 'csv.*all.*invoic', 'export.*wrong.*count', 'filter.*not.*work.*export'],
    'bug_F01_stale_invoice_status': ['stale.*invoice', 'invoice.*list.*not.*updat', 'cached.*invoice', 'still.*shows.*pending', 'pay.*not.*reflect.*list', 'list.*outdated', 'invoice.*not.*refresh'],
    'bug_F02_delete_button_broken': ['delete.*button.*nothing', 'delete.*button.*broken', 'delete.*does.*nothing', 'delete.*not.*work', 'delete.*customer.*broken', 'click.*delete.*nothing'],
    'bug_F03_discount_calc_mismatch': ['preview.*total.*differ', 'preview.*wrong', 'discount.*mismatch', 'total.*change.*after.*sav', 'preview.*correct.*saved.*wrong', 'discount.*calculation.*differ'],
    'bug_F04_pagination_off_by_one': ['math\\.floor', 'math\\.ceil', 'last.*page.*unreachable', 'pagination.*off', 'page.*count.*wrong', 'page.*1.*of.*0', 'pagination.*wrong'],
    'bug_F05_double_click_submit': ['double.click.*creat', 'duplicate.*invoice', 'debounce', 'button.*not.*disabled', 'submit.*twice', 'double.*submit', 'two.*identical.*invoice'],
    'bug_F06_date_off_by_one_timezone': ['timezone.*date', 'date.*off.*one.*day', 'date.*wrong.*day', 'date.*shift', 'day.*behind', 'yesterday.*date', 'date.*display.*wrong'],
    'bug_F07_case_sensitive_search': ['search.*case.sensitiv', 'case.sensitiv.*search', 'search.*no.*result', 'search.*not.*find', 'search.*lowercase', 'search.*doesn.*work'],
    'bug_F08_refund_fires_before_confirm': ['refund.*before.*confirm', 'cancel.*still.*refund', 'refund.*anyway', 'confirm.*cancel.*refund', 'refund.*cancel.*still', 'dialog.*cancel.*refund'],
    'bug_F09_raw_float_display': ['float.*precision', 'raw.*float', '36\\.663', '000000000', 'decimal.*display', 'number.*format.*wrong', 'floating.*point.*display', 'long.*decimal'],
    'bug_B16_multicurrency_unconverted': ['unconvert.*amount', 'multi.*currency.*wrong.*total', 'currency.*not.*convert', 'missing.*exchange.*rate.*report', 'report.*wrong.*total.*currency', 'add.*without.*convert', 'silently.*add.*amount', 'no.*rate.*still.*add', 'multi.*currency.*report.*incorrect'],
    'bug_B17_currency_line_item_mismatch': ['line.*item.*currency.*mismatch', 'line.*item.*usd.*total.*eur', 'line.*item.*don.*add.*up', 'item.*wrong.*currency', 'line.*total.*doesn.*match.*subtotal', 'item.*price.*differ.*invoice.*currency', 'invoice.*item.*currency.*wrong'],
    'bug_B18_recurring_no_stock_decrement': ['recurring.*stock.*not', 'recurring.*inventor.*not', 'stock.*not.*decrement.*recurring', 'recurring.*no.*stock', 'stock.*unchanged.*recurring', 'recurring.*doesn.*reduce.*stock', 'generate.*recurring.*stock'],
    'bug_B19_balance_mixed_currencies': ['balance.*mix.*currenc', 'balance.*wrong.*currenc', 'balance.*multiple.*currenc', 'balance.*different.*currenc', 'customer.*balance.*currenc', 'sum.*across.*currenc', 'balance.*doesn.*account.*currenc'],
    'bug_F11_export_count_off_by_one': ['export.*count.*off', 'preview.*count.*wrong', 'row.*count.*include.*header', 'csv.*count.*off.*one', 'preview.*one.*more', 'export.*preview.*count.*mismatch', 'row.*count.*header'],
    'bug_B20_product_name_truncated': ['product.*name.*truncat', 'name.*cut.*off', 'name.*50.*char', 'name.*shorten', 'name.*too.*long.*silent', 'truncat.*product', 'long.*name.*short', 'name.*limit.*50'],
    'bug_B21_double_payment_accepted': ['pay.*already.*paid', 'double.*pay', 'paid.*invoice.*pay.*again', 'pay.*twice', 'duplicate.*payment', 'charged.*twice', 're.pay.*invoice', 'payment.*accepted.*paid', 'pay.*button.*still.*work.*after.*paid'],
    'bug_F12_dashboard_revenue_mismatch': ['dashboard.*revenue.*wrong', 'dashboard.*report.*differ', 'dashboard.*revenue.*mismatch', 'dashboard.*total.*wrong', 'dashboard.*doesn.*match.*report', 'revenue.*dashboard.*report.*different', 'dashboard.*includes.*refund'],
}

TOTAL_BUGS = len(BUG_PATTERNS)


def score_bug_report(report_text: str) -> dict[str, object]:
    report_lower = report_text.lower()
    identified: list[str] = []
    for bug_name, keywords in BUG_PATTERNS.items():
        for keyword in keywords:
            if re.search(keyword, report_lower):
                identified.append(bug_name)
                break

    backend_found = [bug for bug in identified if bug.startswith("bug_B")]
    frontend_found = [bug for bug in identified if bug.startswith("bug_F")]
    score = round(len(identified) / TOTAL_BUGS, 4)
    return {
        "score": score,
        "bugs_found": len(identified),
        "total_bugs": TOTAL_BUGS,
        "backend_bugs_found": len(backend_found),
        "frontend_bugs_found": len(frontend_found),
        "identified": identified,
        "reason": (
            f"Agent identified {len(identified)}/{TOTAL_BUGS} bugs "
            f"({len(backend_found)} backend, {len(frontend_found)} frontend)"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to bug_report.md")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report_text = Path(args.report).read_text(encoding="utf-8")
    print(json.dumps(score_bug_report(report_text), indent=2))


if __name__ == "__main__":
    main()
