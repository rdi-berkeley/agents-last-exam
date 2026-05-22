"""Proxy scalar scoring for SimGlucose hidden summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    formula: str
    episodes: int
    mean_tir_70_180: float
    catastrophic_episode_count: int
    completion_ratio: float
    eligible_for_ranking: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "formula": self.formula,
            "episodes": self.episodes,
            "mean_tir_70_180": self.mean_tir_70_180,
            "catastrophic_episode_count": self.catastrophic_episode_count,
            "completion_ratio": self.completion_ratio,
            "eligible_for_ranking": self.eligible_for_ranking,
        }


def score_hidden_summary(summary: dict[str, Any]) -> ScoreResult:
    episodes = max(int(summary.get("episodes", 0) or 0), 1)
    mean_tir = float(summary.get("mean_tir_70_180", 0.0) or 0.0)
    catastrophic = int(summary.get("catastrophic_episode_count", 0) or 0)
    completion_ratio = max(0.0, 1.0 - (catastrophic / float(episodes)))

    # Proxy scalar for AgentHLE automation:
    # favor time-in-range while penalizing episode-level catastrophic failures.
    score = max(0.0, min(1.0, mean_tir * (completion_ratio ** 1.5)))
    passed = score >= 0.10
    reason = "proxy_threshold_met" if passed else "proxy_threshold_not_met"

    return ScoreResult(
        score=score,
        passed=passed,
        reason=reason,
        formula="mean_tir_70_180 * (1 - catastrophic_episode_count / episodes) ** 1.5",
        episodes=episodes,
        mean_tir_70_180=mean_tir,
        catastrophic_episode_count=catastrophic,
        completion_ratio=completion_ratio,
        eligible_for_ranking=bool(summary.get("eligible_for_ranking", False)),
    )
