"""Trajectory schema: ATIF-inspired Pydantic models, ALE-v1.0.

Strict subset of harbor's ATIF (omitting fields we don't need) plus a
small ``extra`` dict on Step and Trajectory for agent-specific metadata
that doesn't fit the standard shape.

Deployers populate this from their agent's structured logs (stream-json
for claude-code, event jsonl for openclaw, ...). The framework seeds a
leading ``user``-source Step (the instruction); ``BaseAgentDeployer.
parse_artifacts`` appends the rest. Sub-agents attach under
:attr:`Trajectory.subagent_trajectories`.

Storage is the orchestrator's job: ``trajectory.model_dump_json(indent=2)``
to a file. Screenshots are referenced **by path** (see :class:`ImageSource`)
and written separately — never inline base64 in the JSON.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "ALE-v1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# =============================================================================
# Multimodal content
# =============================================================================

class ImageSource(BaseModel):
    """Reference to an image. Prefer ``path`` (relative to the run dir).

    ``data`` (inline base64) is supported but discouraged for long episodes —
    the framework is responsible for moving base64 captures to disk and
    rewriting references to ``path`` form before persistence.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["path", "url", "base64"] = "path"
    path: str | None = None
    url: str | None = None
    data: str | None = None
    media_type: str = "image/png"
    alt_text: str | None = None


class ContentPart(BaseModel):
    """One piece of structured content. Either text or an image."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image"]
    text: str | None = None
    image: ImageSource | None = None


# =============================================================================
# Tool calls + observations
# =============================================================================

class ToolCall(BaseModel):
    """A tool invocation emitted by the agent within one Step."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """The environment's response to one or more tool calls.

    ``results`` aligns with ``tool_calls`` from the **previous** Step
    (matched by ``tool_call_id``). For a step that is purely an env update
    (no preceding tool call), ``results`` may be empty and the message
    carries the content.
    """

    model_config = ConfigDict(extra="forbid")

    results: list["ToolResult"] = Field(default_factory=list)
    error: str | None = None


class ToolResult(BaseModel):
    """One tool's structured result. ``content`` may be text or an image."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    content: list[ContentPart] = Field(default_factory=list)
    is_error: bool = False


# =============================================================================
# Metrics
# =============================================================================

class StepMetrics(BaseModel):
    """Per-step LLM accounting. All fields optional — populate what's available."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cost_usd: float | None = None
    duration_ms: int | None = None


class FinalMetrics(BaseModel):
    """Trajectory-wide totals + outcome."""

    model_config = ConfigDict(extra="forbid")

    total_steps: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    reward: float | None = None
    status: Literal["completed", "timeout", "failed"] = "completed"


# =============================================================================
# Step + Trajectory
# =============================================================================

Source = Literal["system", "user", "agent", "environment"]


class Step(BaseModel):
    """One step in the trajectory.

    Semantic shape varies by ``source``:

    - ``user``        — instruction or human turn. ``message`` set.
    - ``agent``       — model output. Some combination of ``message``,
                        ``reasoning``, ``tool_calls`` set. ``metrics``
                        records the LLM call's token/cost.
    - ``environment`` — env response (tool results or state update).
                        ``observation`` set.
    - ``system``      — system prompt or framework note (cancellations,
                        timeouts, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    step_id: int = Field(ge=1)
    timestamp: str = Field(default_factory=_now_iso)
    source: Source
    message: str | list[ContentPart] | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    observation: Observation | None = None
    metrics: StepMetrics | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AgentInfo(BaseModel):
    """Identifies the agent that produced the trajectory."""

    model_config = ConfigDict(extra="forbid")

    name: str                          # "claude-code", "ale-claw", ...
    version: str | None = None         # CLI version or commit
    model: str | None = None           # the LLM id this agent used
    extra: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """A complete episode. Built incrementally; finalized once at the end.

    Long-running episodes split across multiple files using
    ``continued_trajectory_ref``. Concatenation: walk the chain back via
    ``continued_trajectory_ref`` until ``None``; concat steps in order.

    Sub-agent traces (e.g. claude-code ``Task`` tool, native agents that
    spawn sub-loops) attach under :attr:`subagent_trajectories`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["ALE-v1.0"] = SCHEMA_VERSION
    episode_id: str
    agent: AgentInfo
    task_path: str
    variant_index: int
    instruction: str = ""
    steps: list[Step] = Field(default_factory=list)
    final_metrics: FinalMetrics | None = None
    started_at: str = Field(default_factory=_now_iso)
    ended_at: str | None = None

    # ---- nested / spanning fields ----
    subagent_trajectories: list["Trajectory"] = Field(default_factory=list)
    """Sub-trajectories from spawned subagents."""

    continued_trajectory_ref: str | None = None
    """When the episode is too long to fit one file, the writer flushes
    every N steps and starts a new Trajectory chunk with this field
    pointing at the previous chunk's relative path."""

    extra: dict[str, Any] = Field(default_factory=dict)


# Forward refs.
Observation.model_rebuild()


# =============================================================================
# Builder helper — canonical constructor for the schema above.
# =============================================================================

class TrajectoryBuilder:
    """Mutable helper used during a run. Append steps; finalize once."""

    def __init__(
        self,
        *,
        episode_id: str | None = None,
        agent_name: str,
        agent_version: str | None = None,
        model: str | None = None,
        task_path: str,
        variant_index: int,
        instruction: str = "",
    ):
        self._traj = Trajectory(
            episode_id=episode_id or uuid.uuid4().hex,
            agent=AgentInfo(name=agent_name, version=agent_version, model=model),
            task_path=task_path,
            variant_index=variant_index,
            instruction=instruction,
        )
        self._next_step_id = 1
        self._t0 = time.monotonic()
        self._final_metrics_override: dict[str, float] = {}

    #: FinalMetrics fields a deployer may override in :meth:`finalize`.
    _OVERRIDABLE_METRICS = frozenset({
        "total_input_tokens",
        "total_output_tokens",
        "total_cache_read_tokens",
        "total_cache_creation_tokens",
        "total_cost_usd",
    })

    @property
    def trajectory(self) -> Trajectory:
        return self._traj

    def override_final_metrics(self, **totals: float | None) -> None:
        """Record authoritative trajectory totals to apply in :meth:`finalize`.

        :meth:`finalize` defaults to summing per-step :class:`StepMetrics`, which
        is lossy for some agents — e.g. ale_claw's transcript carries neither the
        prompt-cache read/write split nor the final/helper turns, so the per-step
        sum under-counts tokens and cost. A deployer that can compute exact totals
        from richer artifacts records them here; finalize then prefers them over
        the per-step sum for exactly the keys provided (others still come from the
        sum). Passing ``None`` for a key is a no-op, so callers can offer a metric
        only when they actually have it.
        """
        for key, value in totals.items():
            if key not in self._OVERRIDABLE_METRICS:
                raise ValueError(f"non-overridable FinalMetrics field: {key!r}")
            if value is not None:
                self._final_metrics_override[key] = value

    def add_step(
        self,
        source: Source,
        *,
        message: str | list[ContentPart] | None = None,
        reasoning: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        observation: Observation | None = None,
        metrics: StepMetrics | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Step:
        step = Step(
            step_id=self._next_step_id,
            source=source,
            message=message,
            reasoning=reasoning,
            tool_calls=list(tool_calls or []),
            observation=observation,
            metrics=metrics,
            extra=dict(extra or {}),
        )
        self._next_step_id += 1
        self._traj.steps.append(step)
        return step

    def finalize(
        self,
        *,
        reward: float | None,
        status: Literal["completed", "timeout", "failed"] = "completed",
    ) -> Trajectory:
        m = FinalMetrics(
            total_steps=len(self._traj.steps),
            reward=reward,
            status=status,
            total_duration_ms=int((time.monotonic() - self._t0) * 1000),
        )
        for s in self._traj.steps:
            if s.metrics is None:
                continue
            m.total_input_tokens += s.metrics.input_tokens or 0
            m.total_output_tokens += s.metrics.output_tokens or 0
            m.total_cache_read_tokens += s.metrics.cache_read_tokens or 0
            m.total_cache_creation_tokens += s.metrics.cache_creation_tokens or 0
            if s.metrics.cost_usd is not None:
                m.total_cost_usd += s.metrics.cost_usd
        # Authoritative deployer-supplied totals win over the per-step sum,
        # per provided key (see :meth:`override_final_metrics`).
        for key, value in self._final_metrics_override.items():
            setattr(m, key, value)
        self._traj.final_metrics = m
        self._traj.ended_at = _now_iso()
        return self._traj
