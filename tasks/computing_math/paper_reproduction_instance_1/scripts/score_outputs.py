"""Scoring for computing_math/paper_reproduction_instance_1.

Reads the agent's output/results.json and the hidden reference/gold_table2.json,
enforces the hard gate (results.json must exist and parse), then computes a
weighted score per the TASK_INTAKE rubric:

  Rule 1 (20%, pass/fail)  : identified_key_table == "Table 2"
  Rule 2 (20%, percentage) : claimed datasets downloaded AND verified on disk
  Rule 3 (60%, percentage) : fraction of 40 table2_values cells with relative
                             error <= 10% versus gold

Rule 2 takes an on-VM probe for each claimed dataset. On the cpu-free-ubuntu
host the full OOD validation splits do not fit, so an honest cached-artifact
agent will legitimately score 0 on Rule 2 and still pass via Rules 1 + 3.

Final = 0.2 * rule1 + 0.2 * rule2 + 0.6 * rule3. Pass threshold (not enforced
by this script, only reported): 0.70.
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

EXPECTED_DATASET_TOKENS = (
    "imagenet-v2",
    "imagenet-s",
    "imagenet-r",
    "imagenet-a",
    "objectnet",
)

RELATIVE_TOLERANCE = 0.10


@dataclass
class ScoreResult:
    score: float
    rule1: float
    rule2: float
    rule3: float
    reason: str
    cells_matched: int = 0
    cells_total: int = 0
    datasets_present: int = 0
    datasets_claimed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "rule1": self.rule1,
            "rule2": self.rule2,
            "rule3": self.rule3,
            "cells_matched": self.cells_matched,
            "cells_total": self.cells_total,
            "datasets_present": self.datasets_present,
            "datasets_claimed": self.datasets_claimed,
            "reason": self.reason,
        }


def _fail(reason: str) -> ScoreResult:
    return ScoreResult(
        score=0.0, rule1=0.0, rule2=0.0, rule3=0.0, reason=reason
    )


def parse_results(agent_bytes: bytes) -> dict | ScoreResult:
    if not agent_bytes:
        return _fail("results.json missing or empty")
    try:
        text = agent_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _fail(f"results.json not UTF-8: {exc}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return _fail(f"results.json not valid JSON: {exc}")
    if not isinstance(data, dict):
        return _fail("results.json top-level is not an object")
    return data


def score_rule1(agent: dict) -> float:
    return 1.0 if agent.get("identified_key_table") == "Table 2" else 0.0


def score_rule2(agent: dict, datasets_verified_on_vm: Iterable[str]) -> tuple[float, int, int]:
    claimed = agent.get("datasets_downloaded", [])
    if not isinstance(claimed, list):
        return 0.0, 0, 0
    claimed_normalized = [str(x).strip().lower() for x in claimed if isinstance(x, str)]
    claimed_expected = [c for c in claimed_normalized if c in EXPECTED_DATASET_TOKENS]
    verified = set(datasets_verified_on_vm)
    counted = sum(1 for c in claimed_expected if c in verified)
    return counted / 5.0, counted, len(claimed_expected)


def score_rule3(agent: dict, gold_cells: dict) -> tuple[float, int, int]:
    agent_cells = agent.get("table2_values", {})
    if not isinstance(agent_cells, dict):
        return 0.0, 0, len(gold_cells)
    matched = 0
    for key, gold_val in gold_cells.items():
        agent_val = agent_cells.get(key)
        if agent_val is None:
            continue
        try:
            a = float(agent_val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(a):
            continue
        g = float(gold_val)
        if g == 0.0:
            if a == 0.0:
                matched += 1
            continue
        if abs(a - g) / abs(g) <= RELATIVE_TOLERANCE:
            matched += 1
    return matched / len(gold_cells), matched, len(gold_cells)


def score(
    agent_bytes: bytes | None,
    gold_cells: dict,
    datasets_verified_on_vm: Iterable[str],
) -> ScoreResult:
    parsed = parse_results(agent_bytes or b"")
    if isinstance(parsed, ScoreResult):
        return parsed
    agent = parsed

    r1 = score_rule1(agent)
    r2, n_verified, n_claimed = score_rule2(agent, datasets_verified_on_vm)
    r3, cells_matched, cells_total = score_rule3(agent, gold_cells)

    final = 0.2 * r1 + 0.2 * r2 + 0.6 * r3

    return ScoreResult(
        score=final,
        rule1=r1,
        rule2=r2,
        rule3=r3,
        cells_matched=cells_matched,
        cells_total=cells_total,
        datasets_present=n_verified,
        datasets_claimed=n_claimed,
        reason=(
            f"rule1={r1:.2f} rule2={r2:.2f}({n_verified}/5 on-disk, {n_claimed} claimed) "
            f"rule3={r3:.3f}({cells_matched}/{cells_total}) final={final:.3f}"
        ),
    )
