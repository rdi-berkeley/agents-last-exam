"""Strict CSV scorer for the SwissADME transporter-risk task."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

REQUIRED_COLUMNS = [
    "Compound Key",
    "SMILES",
    "IC50 (nM)",
    "LogP",
    "TPSA",
    "GI absorption",
    "P-gp substrate",
    "Renal Risk Score",
    "Transporter Risk",
]

LOGP_TOL = 0.01
TPSA_TOL = 0.1


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "details": self.details,
        }


def _read_csv(text: str) -> tuple[list[str], list[dict[str, str]]]:
    stream = io.StringIO(text.lstrip("\ufeff"))
    reader = csv.DictReader(stream)
    columns = reader.fieldnames or []
    rows = []
    for row in reader:
        if not row:
            continue
        if all((value or "").strip() == "" for value in row.values()):
            continue
        rows.append({key: (value or "").strip() for key, value in row.items()})
    return columns, rows


def _as_float(value: str, *, label: str, details: list[str]) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        details.append(f"{label} is not numeric: {value!r}")
        return None


def _numeric_equal(agent: str, ref: str) -> bool:
    a = _as_float(agent, label="agent numeric", details=[])
    r = _as_float(ref, label="reference numeric", details=[])
    return a is not None and r is not None and a == r


def score_output_csv(output_csv: str, reference_csv: str) -> ScoreResult:
    details: list[str] = []
    agent_columns, agent_rows = _read_csv(output_csv)
    ref_columns, ref_rows = _read_csv(reference_csv)

    if agent_columns != REQUIRED_COLUMNS:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="output columns do not match required schema",
            details=[f"expected={REQUIRED_COLUMNS!r}", f"actual={agent_columns!r}"],
        )
    if ref_columns != REQUIRED_COLUMNS:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="reference columns do not match required schema",
            details=[f"reference={ref_columns!r}"],
        )

    agent_by_key = {row["Compound Key"]: row for row in agent_rows}
    ref_by_key = {row["Compound Key"]: row for row in ref_rows}
    if len(agent_by_key) != len(agent_rows):
        return ScoreResult(0.0, False, "duplicate Compound Key in output")
    if len(ref_by_key) != len(ref_rows):
        return ScoreResult(0.0, False, "duplicate Compound Key in reference")

    if set(agent_by_key) != set(ref_by_key):
        missing = sorted(set(ref_by_key) - set(agent_by_key))
        extra = sorted(set(agent_by_key) - set(ref_by_key))
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="compound key set differs from reference",
            details=[f"missing={missing!r}", f"extra={extra!r}"],
        )

    for key in sorted(ref_by_key):
        agent = agent_by_key[key]
        ref = ref_by_key[key]

        for column in ["SMILES", "GI absorption", "P-gp substrate", "Renal Risk Score", "Transporter Risk"]:
            if agent[column] != ref[column]:
                details.append(
                    f"{key}: {column} mismatch; expected {ref[column]!r}, got {agent[column]!r}"
                )

        if not _numeric_equal(agent["IC50 (nM)"], ref["IC50 (nM)"]):
            details.append(
                f"{key}: IC50 (nM) mismatch; expected {ref['IC50 (nM)']!r}, got {agent['IC50 (nM)']!r}"
            )

        logp_agent = _as_float(agent["LogP"], label=f"{key} LogP", details=details)
        logp_ref = _as_float(ref["LogP"], label=f"{key} reference LogP", details=details)
        if logp_agent is not None and logp_ref is not None and abs(logp_agent - logp_ref) > LOGP_TOL:
            details.append(
                f"{key}: LogP outside tolerance; expected {logp_ref}, got {logp_agent}"
            )

        tpsa_agent = _as_float(agent["TPSA"], label=f"{key} TPSA", details=details)
        tpsa_ref = _as_float(ref["TPSA"], label=f"{key} reference TPSA", details=details)
        if tpsa_agent is not None and tpsa_ref is not None and abs(tpsa_agent - tpsa_ref) > TPSA_TOL:
            details.append(
                f"{key}: TPSA outside tolerance; expected {tpsa_ref}, got {tpsa_agent}"
            )

    if details:
        return ScoreResult(0.0, False, "one or more row values differ", details)
    return ScoreResult(1.0, True, "output matches reference")
