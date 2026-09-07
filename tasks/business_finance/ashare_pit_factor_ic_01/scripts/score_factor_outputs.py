"""Host-side scorer for business_finance/ashare_pit_factor_ic_01.

Pure standard library so the grader can run in the host process without pandas.

Gate-and-score:
  gates (any failure -> score 0): files present and parseable, exact panel columns, exact rebalance
  date set, unique (date, ticker) keys, average per-date universe Jaccard >= 0.90, report schema,
  and provenance: the reported mean_ic must match the rank IC recomputed from the submitted panel.
  score = 0.10 * coverage + 0.55 * panel_fidelity + 0.35 * ic_accuracy
  panel_fidelity is the weighted mean over columns of the per-date share of reference cells the
  submission reproduces within tolerance (ep_ttm carries triple weight: it is the point-in-time part).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

FACTORS = ["mom_6_1", "rev_1m", "vol_20", "turnover_20", "ep_ttm", "size"]
PANEL_COLUMNS = ["date", "ticker", "industry_sw1", *FACTORS, "fwd_ret_20"]
SCORED_COLUMNS = [*FACTORS, "fwd_ret_20"]
REPORT_FIELDS = ["mean_ic", "ic_std", "icir", "n_periods", "by_date"]
MIN_IC_NAMES = 30
COVERAGE_GATE = 0.90
PROVENANCE_TOL = 0.005
PASS_THRESHOLD = 0.95
CELL_TOL = {col: 1e-3 for col in FACTORS}
CELL_TOL["fwd_ret_20"] = 1e-5
COLUMN_WEIGHT = {col: 1.0 for col in SCORED_COLUMNS}
COLUMN_WEIGHT["ep_ttm"] = 3.0
IC_TOL = (0.0005, 0.005)
ICIR_TOL = (0.01, 0.10)


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None = None
    coverage_score: float = 0.0
    panel_fidelity: float = 0.0
    ic_accuracy: float = 0.0
    mean_jaccard: float = 0.0
    column_match_rate: dict[str, float] = field(default_factory=dict)
    column_mean_rho: dict[str, float] = field(default_factory=dict)
    ic_abs_error: dict[str, float] = field(default_factory=dict)
    icir_abs_error: dict[str, float] = field(default_factory=dict)
    provenance_gap: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _as_text(payload: str | bytes) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8-sig")
    return payload.lstrip("﻿")


def _to_float(cell: str) -> float:
    cell = (cell or "").strip()
    if cell == "" or cell.lower() in {"nan", "na", "none", "null"}:
        return math.nan
    return float(cell)


def parse_panel(
    csv_text: str | bytes,
) -> tuple[dict[str, dict[str, dict[str, float | str]]] | None, str | None]:
    """Return {date: {ticker: {col: value}}} or (None, reason)."""
    reader = csv.reader(io.StringIO(_as_text(csv_text)))
    try:
        header = next(reader)
    except StopIteration:
        return None, "panel is empty"
    header = [h.strip() for h in header]
    if header != PANEL_COLUMNS:
        return None, f"panel columns must be exactly {PANEL_COLUMNS}, got {header}"
    panel: dict[str, dict[str, dict]] = {}
    for line_no, row in enumerate(reader, start=2):
        if not row or all(not c.strip() for c in row):
            continue
        if len(row) != len(PANEL_COLUMNS):
            return None, f"panel line {line_no} has {len(row)} cells, expected {len(PANEL_COLUMNS)}"
        d, s = row[0].strip(), row[1].strip()
        if len(d) != 10 or d[4] != "-" or d[7] != "-":
            return None, f"panel line {line_no} has malformed date {d!r}"
        rec: dict[str, float | str] = {"industry_sw1": row[2].strip()}
        try:
            for col, cell in zip(SCORED_COLUMNS, row[3:]):
                rec[col] = _to_float(cell)
        except ValueError:
            return None, f"panel line {line_no} has a non-numeric factor cell"
        bucket = panel.setdefault(d, {})
        if s in bucket:
            return None, f"duplicate (date, ticker) key {d} {s}"
        bucket[s] = rec
    if not panel:
        return None, "panel has no data rows"
    return panel, None


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return math.nan
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return math.nan
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return sxy / math.sqrt(sxx * syy)


def rank_ic_by_date(panel: dict, factor: str) -> dict[str, float]:
    out = {}
    for d in sorted(panel):
        xs, ys = [], []
        for rec in panel[d].values():
            a, b = rec[factor], rec["fwd_ret_20"]
            if not (math.isnan(a) or math.isnan(b)):
                xs.append(a)
                ys.append(b)
        if len(xs) < MIN_IC_NAMES:
            continue
        rho = spearman(xs, ys)
        if not math.isnan(rho):
            out[d] = rho
    return out


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else math.nan


def _linear_credit(x: float, full: float, zero: float) -> float:
    """1.0 when x <= full, 0.0 when x >= zero, linear between."""
    if math.isnan(x):
        return 0.0
    if x <= full:
        return 1.0
    if x >= zero:
        return 0.0
    return (zero - x) / (zero - full)


def parse_report(text: str | bytes) -> tuple[dict | None, str | None]:
    try:
        data = json.loads(_as_text(text))
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"ic_report.json is not valid JSON: {exc}"
    if not isinstance(data, dict) or not isinstance(data.get("rank_ic"), dict):
        return None, "ic_report.json must be an object with a 'rank_ic' object"
    rank_ic = data["rank_ic"]
    for f in FACTORS:
        block = rank_ic.get(f)
        if not isinstance(block, dict):
            return None, f"ic_report.json rank_ic is missing factor {f}"
        for key in REPORT_FIELDS:
            if key not in block:
                return None, f"ic_report.json rank_ic[{f}] is missing {key}"
        for key in ("mean_ic", "ic_std", "icir"):
            if (
                not isinstance(block[key], (int, float))
                or isinstance(block[key], bool)
                or math.isnan(float(block[key]))
            ):
                return None, f"ic_report.json rank_ic[{f}].{key} must be a finite number"
        if not isinstance(block["by_date"], dict):
            return None, f"ic_report.json rank_ic[{f}].by_date must be an object"
    return rank_ic, None


def score_submission(
    output_panel: str | bytes | None,
    output_report: str | bytes | None,
    reference_panel: str | bytes,
    reference_report: str | bytes,
) -> ScoreResult:
    if output_panel is None:
        return ScoreResult(0.0, False, "missing output/factor_panel.csv", hard_gate="missing_panel")
    if output_report is None:
        return ScoreResult(0.0, False, "missing output/ic_report.json", hard_gate="missing_report")

    ref_panel, err = parse_panel(reference_panel)
    if err:
        raise RuntimeError(f"reference panel invalid: {err}")
    ref_report, err = parse_report(reference_report)
    if err:
        raise RuntimeError(f"reference report invalid: {err}")

    panel, err = parse_panel(output_panel)
    if err:
        return ScoreResult(0.0, False, err, hard_gate="panel_schema")
    if set(panel) != set(ref_panel):
        missing = sorted(set(ref_panel) - set(panel))[:3]
        extra = sorted(set(panel) - set(ref_panel))[:3]
        return ScoreResult(
            0.0,
            False,
            f"panel date set differs from rebalance dates (missing {missing}, extra {extra})",
            hard_gate="date_set",
        )
    report, err = parse_report(output_report)
    if err:
        return ScoreResult(0.0, False, err, hard_gate="report_schema")

    jaccards = []
    for d in ref_panel:
        a, b = set(panel[d]), set(ref_panel[d])
        jaccards.append(len(a & b) / len(a | b))
    mean_jaccard = _mean(jaccards)
    if mean_jaccard < COVERAGE_GATE:
        return ScoreResult(
            0.0,
            False,
            f"universe overlap too low (mean Jaccard {mean_jaccard:.3f} < {COVERAGE_GATE})",
            hard_gate="universe_coverage",
            mean_jaccard=mean_jaccard,
        )

    provenance_gap = {}
    for f in FACTORS:
        recomputed = rank_ic_by_date(panel, f)
        own_mean = _mean(list(recomputed.values()))
        provenance_gap[f] = (
            abs(float(report[f]["mean_ic"]) - own_mean) if not math.isnan(own_mean) else math.inf
        )
    worst = max(provenance_gap.values())
    if worst > PROVENANCE_TOL:
        return ScoreResult(
            0.0,
            False,
            f"reported mean_ic is inconsistent with the submitted panel (max gap {worst:.4f})",
            hard_gate="report_provenance",
            mean_jaccard=mean_jaccard,
            provenance_gap=provenance_gap,
        )

    coverage_score = _linear_credit(1.0 - mean_jaccard, full=0.01, zero=0.10)

    column_match_rate, column_mean_rho = {}, {}
    for col in SCORED_COLUMNS:
        rates, rhos = [], []
        for d in ref_panel:
            n_ref = matched = 0
            xs, ys = [], []
            for s, rec in ref_panel[d].items():
                b = rec[col]
                if math.isnan(b):
                    continue
                n_ref += 1
                a = panel[d][s][col] if s in panel[d] else math.nan
                if math.isnan(a):
                    continue
                xs.append(a)
                ys.append(b)
                if abs(a - b) <= CELL_TOL[col]:
                    matched += 1
            if n_ref == 0:
                continue
            rates.append(matched / n_ref)
            rho = spearman(xs, ys)
            if not math.isnan(rho):
                rhos.append(rho)
        column_match_rate[col] = _mean(rates)
        column_mean_rho[col] = _mean(rhos)
    panel_fidelity = sum(COLUMN_WEIGHT[c] * column_match_rate[c] for c in SCORED_COLUMNS) / sum(
        COLUMN_WEIGHT.values()
    )

    ic_abs_error, icir_abs_error, ic_credits = {}, {}, []
    for f in FACTORS:
        d_ic = abs(float(report[f]["mean_ic"]) - float(ref_report[f]["mean_ic"]))
        d_icir = abs(float(report[f]["icir"]) - float(ref_report[f]["icir"]))
        ic_abs_error[f] = d_ic
        icir_abs_error[f] = d_icir
        ic_credits.append(
            0.5 * _linear_credit(d_ic, *IC_TOL) + 0.5 * _linear_credit(d_icir, *ICIR_TOL)
        )
    ic_accuracy = _mean(ic_credits)

    score = 0.10 * coverage_score + 0.55 * panel_fidelity + 0.35 * ic_accuracy
    score = max(0.0, min(1.0, score))
    return ScoreResult(
        score=score,
        passed=score >= PASS_THRESHOLD,
        reason="scored",
        coverage_score=coverage_score,
        panel_fidelity=panel_fidelity,
        ic_accuracy=ic_accuracy,
        mean_jaccard=mean_jaccard,
        column_match_rate=column_match_rate,
        column_mean_rho=column_mean_rho,
        ic_abs_error=ic_abs_error,
        icir_abs_error=icir_abs_error,
        provenance_gap=provenance_gap,
    )


def _read_optional(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Score an output/ directory against a reference/ directory."
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reference-dir", required=True)
    a = ap.parse_args()
    out, ref = Path(a.output_dir), Path(a.reference_dir)
    result = score_submission(
        _read_optional(out / "factor_panel.csv"),
        _read_optional(out / "ic_report.json"),
        (ref / "factor_panel.csv").read_bytes(),
        (ref / "ic_report.json").read_bytes(),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
