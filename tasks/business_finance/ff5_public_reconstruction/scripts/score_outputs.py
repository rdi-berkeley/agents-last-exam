"""Local scorer for business_finance/ff5_public_reconstruction."""

from __future__ import annotations

import argparse
import io
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd

REQUIRED_COLUMNS = ["date", "MKT_RF", "SMB", "HML", "RMW", "CMA"]
SCORED_FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
DATE_FLOOR = "2015-01"
DEFAULT_ALLOWED_DOMAINS = [
    "papers.ssrn.com",
    "stooq.com",
    "sec.gov",
    "data.sec.gov",
    "fred.stlouisfed.org",
    "finance.yahoo.com",
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",
    "github.com",
    "pypi.org",
    "files.pythonhosted.org",
]
_DATE_RE = re.compile(r"^\d{4}-\d{2}$")
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None = None
    aligned_rows: int = 0
    factor_correlations: dict[str, float] = field(default_factory=dict)
    factor_slopes: dict[str, float] = field(default_factory=dict)
    factor_intercepts: dict[str, float] = field(default_factory=dict)
    trace_paths_checked: list[str] = field(default_factory=list)
    non_allowlisted_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _as_text(payload: str | bytes) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8-sig")
    return payload


def _load_contract_text(contract_text: str | bytes | None) -> dict:
    if contract_text is None:
        return {}
    return json.loads(_as_text(contract_text))


def _normalize_host(host: str | None) -> str:
    if not host:
        return ""
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _host_is_allowed(host: str, allowed_domains: Iterable[str]) -> bool:
    for allowed in allowed_domains:
        allowed_host = _normalize_host(allowed)
        if not allowed_host:
            continue
        if host == allowed_host or host.endswith(f".{allowed_host}"):
            return True
    return False


def _find_non_allowlisted_urls(
    trace_texts: Iterable[str | bytes],
    allowed_domains: Iterable[str],
) -> list[str]:
    offenders: list[str] = []
    for payload in trace_texts:
        text = _as_text(payload)
        for match in _URL_RE.findall(text):
            host = _normalize_host(urlparse(match).hostname)
            if host and not _host_is_allowed(host, allowed_domains):
                offenders.append(match)
    return sorted(set(offenders))


def _read_csv_frame(csv_text: str | bytes) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(_as_text(csv_text)))


def _normalize_frame(frame: pd.DataFrame, *, label: str) -> tuple[pd.DataFrame | None, str | None]:
    if list(frame.columns) != REQUIRED_COLUMNS:
        return None, f"{label} columns must be exactly {REQUIRED_COLUMNS}, got {list(frame.columns)}"

    normalized = frame.copy()
    normalized["date"] = normalized["date"].astype(str).str.strip()
    if not normalized["date"].map(lambda value: bool(_DATE_RE.match(value))).all():
        return None, f"{label} contains malformed YYYY-MM dates"

    try:
        normalized["_period"] = pd.PeriodIndex(normalized["date"], freq="M")
    except Exception as exc:  # pragma: no cover - defensive against pandas parser drift
        return None, f"{label} date parsing failed: {exc}"

    if normalized["_period"].duplicated().any():
        return None, f"{label} contains duplicate monthly dates"

    for factor in SCORED_FACTORS:
        normalized[factor] = pd.to_numeric(normalized[factor], errors="coerce")

    if normalized[SCORED_FACTORS].isna().any().any():
        return None, f"{label} contains non-numeric or missing factor values"

    floor_period = pd.Period(DATE_FLOOR, freq="M")
    normalized = normalized.loc[normalized["_period"] >= floor_period].copy()
    if normalized.empty:
        return None, f"{label} has no valid rows on or after {DATE_FLOOR}"

    normalized = normalized.sort_values("date").reset_index(drop=True)
    return normalized, None


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) == 0 or len(right) == 0:
        raise ValueError("zero aligned observations")
    if left.nunique(dropna=False) <= 1 or right.nunique(dropna=False) <= 1:
        return 0.0
    corr = left.corr(right)
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _fit_affine(agent: pd.Series, reference: pd.Series) -> tuple[float, float]:
    reference_var = float(reference.var(ddof=0))
    if reference_var == 0.0:
        return 0.0, float(agent.mean())
    slope = float(reference.cov(agent, ddof=0) / reference_var)
    intercept = float(agent.mean() - slope * reference.mean())
    return slope, intercept


def score_factor_csv(
    agent_csv_text: str | bytes,
    reference_csv_text: str | bytes,
    *,
    contract_text: str | bytes | None = None,
    trace_texts: Iterable[str | bytes] | None = None,
    trace_paths: Iterable[str] | None = None,
) -> ScoreResult:
    contract = _load_contract_text(contract_text)
    allowed_domains = contract.get("solve_time_allowed_domains", DEFAULT_ALLOWED_DOMAINS)
    pass_threshold = float(contract.get("pass_threshold", 0.7))
    unit_consistency = contract.get("unit_consistency", {})
    min_slope = float(unit_consistency.get("min_slope", 0.8))
    max_slope = float(unit_consistency.get("max_slope", 1.25))
    max_abs_intercept = float(unit_consistency.get("max_abs_intercept", 0.5))

    trace_payloads = list(trace_texts or [])
    offenders = _find_non_allowlisted_urls(trace_payloads, allowed_domains)
    if offenders:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="observable non-allowlisted URL detected",
            hard_gate="observable solve-time access outside the visible allowlist",
            trace_paths_checked=list(trace_paths or []),
            non_allowlisted_urls=offenders,
        )

    try:
        agent_frame = _read_csv_frame(agent_csv_text)
    except Exception as exc:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"failed to parse agent CSV: {exc}",
            hard_gate="missing or unreadable required output file",
            trace_paths_checked=list(trace_paths or []),
        )

    try:
        reference_frame = _read_csv_frame(reference_csv_text)
    except Exception as exc:  # pragma: no cover - hidden reference should remain valid
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"failed to parse reference CSV: {exc}",
            hard_gate="evaluator reference unreadable",
            trace_paths_checked=list(trace_paths or []),
        )

    normalized_agent, agent_error = _normalize_frame(agent_frame, label="agent output")
    if agent_error:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=agent_error,
            hard_gate=agent_error,
            trace_paths_checked=list(trace_paths or []),
        )

    normalized_reference, reference_error = _normalize_frame(reference_frame, label="reference")
    if reference_error:  # pragma: no cover - hidden reference should remain valid
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=reference_error,
            hard_gate="evaluator reference invalid",
            trace_paths_checked=list(trace_paths or []),
        )

    if normalized_agent["date"].tolist() != normalized_reference["date"].tolist():
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="candidate date coverage does not exactly match the hidden monthly benchmark from 2015-01 onward",
            hard_gate="missing or extra monthly dates after 2015-01",
            trace_paths_checked=list(trace_paths or []),
        )

    merged = normalized_agent.merge(
        normalized_reference[["date", *SCORED_FACTORS]],
        on="date",
        how="inner",
        suffixes=("_agent", "_reference"),
    )

    correlations: dict[str, float] = {}
    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    try:
        for factor in SCORED_FACTORS:
            agent_series = merged[f"{factor}_agent"]
            reference_series = merged[f"{factor}_reference"]
            correlations[factor] = _safe_corr(agent_series, reference_series)
            slope, intercept = _fit_affine(agent_series, reference_series)
            slopes[factor] = slope
            intercepts[factor] = intercept
    except ValueError:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="zero aligned observations for at least one scored factor",
            hard_gate="zero aligned observations for scored factors",
            aligned_rows=int(len(merged)),
            trace_paths_checked=list(trace_paths or []),
        )

    for factor in SCORED_FACTORS:
        slope = slopes[factor]
        intercept = intercepts[factor]
        if slope < min_slope or slope > max_slope or abs(intercept) > max_abs_intercept:
            return ScoreResult(
                score=0.0,
                passed=False,
                reason=(
                    f"{factor} violates unit-consistency checks "
                    f"(slope={slope:.6f}, intercept={intercept:.6f})"
                ),
                hard_gate="factor magnitudes are inconsistent with the required percent-unit benchmark",
                aligned_rows=int(len(merged)),
                factor_correlations=correlations,
                factor_slopes=slopes,
                factor_intercepts=intercepts,
                trace_paths_checked=list(trace_paths or []),
                non_allowlisted_urls=offenders,
            )

    raw_score = sum(correlations.values()) / len(SCORED_FACTORS)
    score = max(0.0, min(1.0, raw_score))
    passed = score >= pass_threshold
    return ScoreResult(
        score=score,
        passed=passed,
        reason=f"mean Pearson correlation={score:.6f}",
        aligned_rows=int(len(merged)),
        factor_correlations=correlations,
        factor_slopes=slopes,
        factor_intercepts=intercepts,
        trace_paths_checked=list(trace_paths or []),
        non_allowlisted_urls=offenders,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="Path to the candidate CSV")
    parser.add_argument("--reference", required=True, help="Path to the hidden reference CSV")
    parser.add_argument("--contract", help="Optional evaluation-contract JSON path")
    parser.add_argument(
        "--trace-path",
        action="append",
        default=[],
        help="Optional observable trace text/JSON file to enforce the browsing allowlist",
    )
    args = parser.parse_args()

    contract_text = None
    if args.contract:
        with open(args.contract, "rb") as handle:
            contract_text = handle.read()

    trace_payloads: list[bytes] = []
    for trace_path in args.trace_path:
        with open(trace_path, "rb") as handle:
            trace_payloads.append(handle.read())

    with open(args.agent, "rb") as handle:
        agent_text = handle.read()
    with open(args.reference, "rb") as handle:
        reference_text = handle.read()

    result = score_factor_csv(
        agent_text,
        reference_text,
        contract_text=contract_text,
        trace_texts=trace_payloads,
        trace_paths=args.trace_path,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=True))


if __name__ == "__main__":
    main()
