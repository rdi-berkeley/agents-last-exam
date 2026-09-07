from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

SCORER_PATH = (
    Path(__file__).resolve().parents[2]
    / "tasks/business_finance/ashare_pit_factor_ic_01/scripts/score_factor_outputs.py"
)
SPEC = importlib.util.spec_from_file_location("ashare_pit_factor_scorer", SCORER_PATH)
assert SPEC is not None and SPEC.loader is not None
SCORER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCORER
SPEC.loader.exec_module(SCORER)

FACTORS = SCORER.FACTORS
DATES = ["2024-01-31", "2024-02-29", "2024-03-29"]
INDUSTRIES = ["电子", "医药生物", "计算机"]


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield state / 2**31


def _build_panel(n_names: int = 60, seed: int = 7) -> dict:
    rnd = _lcg(seed)
    panel = {}
    for d in DATES:
        rows = {}
        for i in range(n_names):
            ticker = f"{600000 + i:06d}.SH"
            rec = {"industry_sw1": INDUSTRIES[i % 3]}
            for f in FACTORS:
                rec[f] = round((next(rnd) - 0.5) * 4, 6)
            rec["fwd_ret_20"] = round(0.2 * rec["ep_ttm"] + (next(rnd) - 0.5) * 0.3, 6)
            rows[ticker] = rec
        panel[d] = rows
    return panel


def _panel_csv(panel: dict, columns=None) -> str:
    columns = columns or SCORER.PANEL_COLUMNS
    lines = [",".join(columns)]
    for d in sorted(panel):
        for s in sorted(panel[d]):
            rec = panel[d][s]
            cells = []
            for col in columns:
                if col == "date":
                    cells.append(d)
                elif col == "ticker":
                    cells.append(s)
                elif col == "industry_sw1":
                    cells.append(rec["industry_sw1"])
                else:
                    v = rec.get(col, math.nan)
                    cells.append("" if math.isnan(v) else f"{v:.6f}")
            lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def _report_from_panel(panel: dict) -> str:
    rank_ic = {}
    for f in FACTORS:
        by_date = SCORER.rank_ic_by_date(panel, f)
        ics = list(by_date.values())
        mean = sum(ics) / len(ics)
        var = sum((x - mean) ** 2 for x in ics) / (len(ics) - 1)
        std = math.sqrt(var)
        rank_ic[f] = {
            "mean_ic": mean,
            "ic_std": std,
            "icir": mean / std,
            "n_periods": len(ics),
            "by_date": by_date,
        }
    return json.dumps({"rank_ic": rank_ic})


@pytest.fixture(scope="module")
def reference():
    panel = _build_panel()
    return panel, _panel_csv(panel), _report_from_panel(panel)


def test_exact_reference_scores_one(reference):
    _panel, csv_text, report = reference
    result = SCORER.score_submission(csv_text, report, csv_text, report)
    assert result.hard_gate is None
    assert result.score == pytest.approx(1.0)
    assert result.passed


def test_missing_outputs_score_zero(reference):
    _, csv_text, report = reference
    assert SCORER.score_submission(None, report, csv_text, report).hard_gate == "missing_panel"
    assert SCORER.score_submission(csv_text, None, csv_text, report).hard_gate == "missing_report"


def test_wrong_columns_fail_gate(reference):
    panel, csv_text, report = reference
    bad = _panel_csv(panel, columns=["date", "ticker", *FACTORS, "fwd_ret_20"])
    result = SCORER.score_submission(bad, report, csv_text, report)
    assert result.hard_gate == "panel_schema"
    assert result.score == 0.0


def test_missing_rebalance_date_fails_gate(reference):
    panel, csv_text, report = reference
    partial = {d: rows for d, rows in panel.items() if d != DATES[-1]}
    result = SCORER.score_submission(_panel_csv(partial), report, csv_text, report)
    assert result.hard_gate == "date_set"


def test_fabricated_report_fails_provenance(reference):
    _panel, csv_text, report = reference
    fake = json.loads(report)
    fake["rank_ic"]["ep_ttm"]["mean_ic"] += 0.05
    result = SCORER.score_submission(csv_text, json.dumps(fake), csv_text, report)
    assert result.hard_gate == "report_provenance"


def test_universe_gate_on_low_overlap(reference):
    panel, csv_text, report = reference
    shrunk = {
        d: {s: rec for i, (s, rec) in enumerate(sorted(rows.items())) if i % 4 != 0}
        for d, rows in panel.items()
    }
    result = SCORER.score_submission(
        _panel_csv(shrunk), _report_from_panel(shrunk), csv_text, report
    )
    assert result.hard_gate == "universe_coverage"
    assert result.mean_jaccard < SCORER.COVERAGE_GATE


def test_perturbed_factor_loses_credit_but_no_gate(reference):
    panel, csv_text, report = reference
    wrong = {d: {s: dict(rec) for s, rec in rows.items()} for d, rows in panel.items()}
    for rows in wrong.values():
        for rec in rows.values():
            rec["ep_ttm"] = round(rec["ep_ttm"] + 0.01, 6)
    result = SCORER.score_submission(_panel_csv(wrong), _report_from_panel(wrong), csv_text, report)
    assert result.hard_gate is None
    assert result.column_match_rate["ep_ttm"] == 0.0
    assert result.column_match_rate["mom_6_1"] == 1.0
    expected_panel = (sum(SCORER.COLUMN_WEIGHT.values()) - SCORER.COLUMN_WEIGHT["ep_ttm"]) / sum(
        SCORER.COLUMN_WEIGHT.values()
    )
    assert result.panel_fidelity == pytest.approx(expected_panel)
    assert not result.passed
    assert 0.0 < result.score < SCORER.PASS_THRESHOLD


def test_tolerance_accepts_rounding_noise(reference):
    panel, csv_text, report = reference
    noisy = {d: {s: dict(rec) for s, rec in rows.items()} for d, rows in panel.items()}
    for rows in noisy.values():
        for rec in rows.values():
            for f in FACTORS:
                rec[f] = round(rec[f] + 4e-4, 6)
    result = SCORER.score_submission(_panel_csv(noisy), _report_from_panel(noisy), csv_text, report)
    assert result.hard_gate is None
    assert result.panel_fidelity == pytest.approx(1.0)
    assert result.passed


def test_spearman_matches_known_value():
    assert SCORER.spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(0.8207826816681233)
    assert SCORER.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
