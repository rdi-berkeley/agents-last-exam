import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from ale_run.orchestration.judge_preflight import judge_requirements
from tasks.business_finance.odoo import shared
from tasks.utils import evaluation


@pytest.mark.asyncio
@pytest.mark.parametrize("attachment", ["valid", "missing", "corrupt"])
@pytest.mark.parametrize("invoice_totals,score", [([230, 460], 1.0), ([], 0.92)])
async def test_database_score_does_not_depend_on_a_judge(
    monkeypatch, attachment, invoice_totals, score
):
    tables = [
        "stock_landed_cost", "stock_valuation_adjustment_lines", "stock_warehouse",
        "stock_quant", "account_move", "sale_order", "purchase_order",
        "mrp_production", "stock_picking", "stock_move_line",
    ]
    evidence = {
        "lc_split": {"RM-010": 60, "RM-020": 40},
        "lc_bill_paid_cnt": 1,
        "so": {"state": "sale", "fp_qty": 3, "acc_qty": 3},
        "dropship_done_cnt": 1, "mo_sa_done_cnt": 1, "mo_fp_done_cnt": 1,
        "workorders_done_cnt": 2, "paid_invoices": invoice_totals,
        "paid_credit": [200], "paid_vendor_bills_cnt": 6,
        "fp_lots": ["SSK-0001", "SSK-0002", "SSK-0003"],
        "sc_lots": ["CS-0001", "CS-0002", "CS-0003"],
        "subcontract_sc_distinct_lots": 3, "scrap_rm030_cnt": 1,
        "extra_po_rm030_cnt": 1, "analytic_expense_cnt": 1,
        "wh_qty": [{"wh": "My Co", "product": "RM-010", "qty": 4},
                   {"wh": "My Co", "product": "RM-020", "qty": 7}],
        "ssk2_wh": [{"wh": "My Co", "qty": 1}],
    }
    monkeypatch.setattr(shared, "_resolve_pg_paths", AsyncMock())
    monkeypatch.setattr(shared, "_run_psql_json", AsyncMock(side_effect=[
        dict.fromkeys(tables, "table"), {}, evidence,
    ]))
    judge = Mock(side_effect=AssertionError("Odoo must not call a judge"))
    monkeypatch.setattr(evaluation, "llm_judge_completion", judge)
    stream = BytesIO()
    Image.new("RGB", (2, 2)).save(stream, format="PNG")

    async def read_bytes(path):
        if attachment == "corrupt":
            return b"\xffbad"
        return stream.getvalue() if path.endswith(".png") else b"Operator submission"

    session = SimpleNamespace(
        interface=SimpleNamespace(create_dir=AsyncMock()),
        file_exists=AsyncMock(return_value=attachment != "missing"),
        read_bytes=AsyncMock(side_effect=read_bytes),
        write_file=AsyncMock(),
    )
    config = SimpleNamespace(metadata=shared.cfg_for_variant(shared.VARIANTS[0]).to_metadata())
    assert await shared.evaluate_variant_task(config, session) == [score]
    judge.assert_not_called()
    report = json.loads(session.write_file.call_args.args[1])
    assert report["scoring_basis"] == "database_checks"
    assert len(report["score_items"]) == 25
    assert len(report["delivery_checks"]) == 5
    assert all(check["readable"] == (attachment == "valid") for check in report["delivery_checks"].values())


def test_odoo_does_not_require_judge_credentials():
    assert judge_requirements(["business_finance/odoo"]) == {}
