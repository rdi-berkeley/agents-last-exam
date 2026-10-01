"""Repeated-trial reliability helpers for benchmark experiments.

The benchmark score for one agent/task pair is a bounded random variable in
[0, 1]. A single run therefore measures one draw, not the stability of the
agent. This module keeps repeated scientific trials separate from
infrastructure retries and provides time-uniform uncertainty summaries that
remain valid when a run stops after reaching a requested precision.

The confidence sequence uses a simple union-bound construction. At trial n we
allocate delta_n = 6 * delta / (pi^2 * n^2) and apply Hoeffding's inequality.
Because sum_n delta_n <= delta, all reported intervals simultaneously cover
the true mean with probability at least confidence. This is conservative, but
it is dependency-free and safe under optional stopping.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .run_writer import slug_agent, slug_model, slug_task

if TYPE_CHECKING:
    from .experiment_spec import ReliabilitySpec, RunUnit


_TERMINAL_TRIAL_STATUSES = frozenset({"completed", "timeout"})


@dataclass(frozen=True)
class TrialObservation:
    """One independent benchmark trial recovered from or written to disk."""

    trial_index: int
    status: str
    score: float | None
    run_dir: str | None = None


def anytime_hoeffding_interval(
    scores: Iterable[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return a time-uniform Hoeffding interval for a bounded [0, 1] mean.

    Returns (lower, upper, half_width). half_width is the conservative raw
    radius used for stopping (clipped to at most 1 because the parameter
    itself is bounded to [0, 1]).
    """

    values = [float(x) for x in scores]
    if not values:
        raise ValueError("at least one score is required")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in values):
        raise ValueError("scores must be finite values in [0, 1]")

    n = len(values)
    mean = statistics.fmean(values)
    delta = 1.0 - confidence
    delta_n = delta * 6.0 / (math.pi**2 * n**2)
    radius = math.sqrt(math.log(2.0 / delta_n) / (2.0 * n))
    half_width = min(1.0, radius)
    return max(0.0, mean - radius), min(1.0, mean + radius), half_width


def pass_k_estimate(*, successes: int, total: int, k: int) -> float | None:
    """Unbiased estimator of p^k from total Bernoulli trials."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total")
    if k < 1:
        raise ValueError("k must be >= 1")
    if total < k:
        return None
    if successes < k:
        return 0.0
    return math.comb(successes, k) / math.comb(total, k)


def summarize_trials(
    observations: Iterable[TrialObservation],
    *,
    confidence: float,
    pass_threshold: float | None,
    pass_k: tuple[int, ...],
) -> dict[str, Any]:
    """Build the statistical summary for terminal repeated trials."""

    obs = sorted(observations, key=lambda item: item.trial_index)
    terminal = [item for item in obs if item.status in _TERMINAL_TRIAL_STATUSES]
    scores = [
        float(item.score)
        for item in terminal
        if item.status == "completed" and item.score is not None
    ]
    if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in scores):
        raise ValueError("completed trial scores must be finite values in [0, 1]")

    summary: dict[str, Any] = {
        "terminal_trials": len(terminal),
        "scored_trials": len(scores),
        "non_scored_terminal_trials": len(terminal) - len(scores),
        "mean_score": statistics.fmean(scores) if scores else None,
        "sample_sd": (
            statistics.stdev(scores)
            if len(scores) >= 2
            else (0.0 if scores else None)
        ),
        "completion_rate": len(scores) / len(terminal) if terminal else None,
        "confidence": confidence,
        "confidence_method": "time_uniform_hoeffding_union_bound",
        "confidence_interval": None,
        "confidence_half_width": None,
        "trials": [
            {
                "trial_index": item.trial_index,
                "status": item.status,
                "score": item.score,
                "run_dir": item.run_dir,
            }
            for item in terminal
        ],
    }

    if scores:
        lo, hi, half_width = anytime_hoeffding_interval(scores, confidence=confidence)
        summary["confidence_interval"] = [lo, hi]
        summary["confidence_half_width"] = half_width

    if pass_threshold is not None:
        successes = sum(score >= pass_threshold for score in scores)
        total = len(terminal)
        summary["pass_reliability"] = {
            "threshold": pass_threshold,
            "successes": successes,
            "trials": total,
            "pass_rate": successes / total if total else None,
            "pass_k": {
                str(k): pass_k_estimate(successes=successes, total=total, k=k)
                for k in pass_k
            },
        }

    return summary


def precision_reached(summary: dict[str, Any], spec: ReliabilitySpec) -> bool:
    """Whether the anytime-valid interval is narrow enough to stop."""

    if spec.target_half_width is None:
        return False
    if int(summary["scored_trials"]) < spec.min_trials:
        return False
    width = summary.get("confidence_half_width")
    return width is not None and float(width) <= spec.target_half_width


def _variant_dir(output_root: Path, unit: RunUnit) -> Path:
    model = str((unit.agent_spec.config or {}).get("model", ""))
    return (
        output_root
        / slug_agent(unit.agent_id)
        / slug_model(model)
        / slug_task(unit.task_path)
        / f"v{unit.variant_index}"
    )


def load_trial_observations(
    output_root: Path,
    unit: RunUnit,
) -> list[TrialObservation]:
    """Recover the latest attempt for every explicit trial directory."""

    variant_dir = _variant_dir(output_root, unit)
    if not variant_dir.is_dir():
        return []

    observations: list[TrialObservation] = []
    for trial_dir in sorted(variant_dir.glob("trial_[0-9][0-9][0-9]")):
        try:
            trial_index = int(trial_dir.name.removeprefix("trial_")) - 1
        except ValueError:
            continue
        if trial_index < 0:
            continue

        run_files = sorted(trial_dir.glob("*/run.json"))
        if not run_files:
            continue
        latest = run_files[-1]
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        score = payload.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            score = None
        else:
            score = float(score)

        observations.append(
            TrialObservation(
                trial_index=trial_index,
                status=str(payload.get("status") or "unknown"),
                score=score,
                run_dir=str(latest.parent),
            )
        )
    return observations


def write_reliability_summary(
    *,
    output_root: Path,
    unit: RunUnit,
    spec: ReliabilitySpec,
    observations: Iterable[TrialObservation],
    stop_reason: str,
) -> Path:
    """Write reliability.json next to the unit's trial directories."""

    summary = summarize_trials(
        observations,
        confidence=spec.confidence,
        pass_threshold=spec.pass_threshold,
        pass_k=spec.pass_k,
    )
    payload = {
        "schema_version": 1,
        "agent_id": unit.agent_id,
        "model": str((unit.agent_spec.config or {}).get("model", "")),
        "task_path": unit.task_path,
        "variant_index": unit.variant_index,
        "stopping": {
            "min_trials": spec.min_trials,
            "max_trials": spec.max_trials,
            "target_half_width": spec.target_half_width,
            "stop_reason": stop_reason,
        },
        **summary,
    }

    variant_dir = _variant_dir(output_root, unit)
    variant_dir.mkdir(parents=True, exist_ok=True)
    path = variant_dir / "reliability.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path
