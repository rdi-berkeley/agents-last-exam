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

import base64
import binascii
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


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


# =============================================================================
# Screenshot persistence — base64 captures → on-disk PNGs + relative refs.
#
# The framework promise (see :class:`ImageSource` docstring + LOG_SPEC §"Sub-
# shapes"): inline base64 image captures never survive into the persisted
# ``trajectory.json``. Before the writer serialises, every capture is written
# to ``<run_dir>/screenshots/NNNN.<ext>`` and its reference rewritten to
# ``type="path"`` with a path **relative to the run dir** (e.g.
# ``screenshots/0000.png``).
#
# Ported from agenthle ``orchestration/external/base.py::save_screenshots_from_log``,
# generalised for the ATIF schema:
#   * the agenthle convention — a transient ``_screenshot_b64`` on a step's
#     ``extra`` dict (still emitted by e.g. the openhands deployer) — and
#   * inline ``ImageSource(type="base64", data=...)`` ContentParts anywhere in
#     a step's ``message`` or ``observation`` content.
# Both are handled, sub-agent trajectories are walked recursively, and a single
# run-wide counter keeps filenames unique across the whole episode.
# =============================================================================

# media_type → file extension. Defaults to ``png`` for anything unrecognised.
_MEDIA_EXT: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


def _ext_for(media_type: str | None) -> str:
    return _MEDIA_EXT.get((media_type or "").strip().lower(), "png")


def _strip_data_url(b64: str) -> str:
    """Return the raw base64 payload from an optional ``data:...;base64,`` URL."""
    if b64.startswith("data:"):
        marker = "base64,"
        idx = b64.find(marker)
        if idx != -1:
            return b64[idx + len(marker):]
    return b64


def _decode_b64(b64: str) -> bytes:
    """Tolerant base64 decode: strip any data-URL prefix + fix missing padding."""
    payload = _strip_data_url(b64).strip()
    payload += "=" * (-len(payload) % 4)  # restore stripped padding
    return base64.b64decode(payload)


def _iter_steps(traj: "Trajectory") -> Iterator["Step"]:
    """Yield every step in *traj*, descending into sub-agent trajectories."""
    yield from traj.steps
    for sub in traj.subagent_trajectories:
        yield from _iter_steps(sub)


def persist_screenshots(trajectory: "Trajectory", run_dir: str | Path) -> int:
    """Move inline base64 image captures to disk; rewrite refs to relative paths.

    Mutates *trajectory* in place and writes ``<run_dir>/screenshots/NNNN.<ext>``.
    Returns the number of screenshots written. Pure framework helper — call it
    once, after the trajectory is finalized and before it is serialised.

    Capture forms handled (per step, in walk order):

    1. ``ImageSource`` with inline ``data`` (``type != "path"``) inside any
       ``message`` or ``observation`` ContentPart → the bytes are written out
       and the source is rewritten to ``type="path"``, ``path="screenshots/
       NNNN.<ext>"``, ``data=None`` (relative to *run_dir*).
    2. A transient ``step.extra["_screenshot_b64"]`` → written out, the key
       dropped, and ``extra["screenshot_index"]`` + ``extra["screenshot_path"]``
       set so existing agenthle-style tooling and the relative-path consumers
       both resolve it.

    Decode/write failures for a single capture are logged and skipped — they
    never abort persistence of the rest of the trajectory.
    """
    run_dir = Path(run_dir)
    pending: list[tuple[str, str]] = []  # (relative_path, raw_b64)

    def _take(b64: str, media_type: str | None) -> str:
        rel = f"screenshots/{len(pending):04d}.{_ext_for(media_type)}"
        pending.append((rel, b64))
        return rel

    def _rewrite_content(content: Any) -> None:
        if not isinstance(content, list):
            return
        for part in content:
            if not isinstance(part, ContentPart):
                continue
            img = part.image
            if part.type != "image" or img is None:
                continue
            if img.data and img.type != "path":
                rel = _take(img.data, img.media_type)
                img.type = "path"
                img.path = rel
                img.data = None

    for step in _iter_steps(trajectory):
        # 1. inline ImageSource base64 in message + observation content
        _rewrite_content(step.message)
        if step.observation is not None:
            for result in step.observation.results:
                _rewrite_content(result.content)
        # 2. agenthle ``_screenshot_b64`` convention on step.extra
        b64 = step.extra.pop("_screenshot_b64", None)
        if b64:
            idx = len(pending)
            rel = _take(b64, "image/png")
            step.extra["screenshot_index"] = idx
            step.extra["screenshot_path"] = rel

    if not pending:
        return 0

    screenshots_dir = run_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    written = 0

    def _write_one(item: tuple[str, str]) -> bool:
        rel, b64 = item
        try:
            (run_dir / rel).write_bytes(_decode_b64(b64))
            return True
        except (binascii.Error, ValueError, OSError) as e:
            logger.warning("persist_screenshots: failed to write %s: %s", rel, e)
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        written = sum(pool.map(_write_one, pending))

    return written
