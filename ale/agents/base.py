"""Base agent contracts: BaseAgentConfig + BaseAgent + EpisodeResult.

Two flavors of agent share this base:

- :class:`NativeAgent` — step loop runs orchestrator-side, LLM called in-
  process. Subclass implements ``act(obs, state) -> list[Action]``.
- :class:`InstalledAgent` — agent CLI runs inside the VM. Subclass-injected
  ``InstalledAgentDeployer`` does install / launch / collect.

Both produce a uniform :class:`EpisodeResult` carrying an ALE-v1.0
:class:`Trajectory`. Tools that consume runs downstream don't need to
branch on the agent flavor — they read one schema.

Config rules:
    - ``BaseAgentConfig`` holds the six fields every agent has reason to expose.
    - ``name`` is a ``ClassVar`` on each concrete config subclass — not an
      __init__ field — so config-vs-agent registry stays simple.
    - Subclass-specific knobs live on the subclass (``NativeAgentConfig``,
      ``InstalledAgentConfig``, ``ClaudeCodeConfig``, ...).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import ClassVar

from ale.agents.trajectory import Trajectory
from ale.core.env import AgenthleEnv


# =============================================================================
# Config
# =============================================================================

@dataclass
class BaseAgentConfig:
    """Shared tunables for any agent.

    Subclasses MUST set the class attribute ``name``. They MAY add their own
    fields; they SHOULD NOT redefine the six standard fields below.
    """

    # Identifier — set on concrete config subclasses, NOT __init__ field.
    name: ClassVar[str] = ""

    # ---- standard fields ----
    model: str = ""
    """LLM model id, e.g. ``claude-opus-4-7``, ``gpt-5``. Empty string = use
    the agent's own default (each subclass documents its default)."""

    max_turns: int | None = None
    """Upper bound on agent turns. None = framework default (no cap)."""

    timeout_s: float = 1800.0
    """Wall-clock budget for the whole episode (including evaluate)."""

    save_screenshots: bool = True
    """Whether to keep screenshots in the trajectory's image references."""

    api_keys: dict[str, str] = field(default_factory=dict)
    """Env-var name → value. Caller passes these explicitly. Never auto-read
    from ``os.environ`` (avoids cross-experiment leakage). Empty = the agent
    must look elsewhere (config subclass field, raise, etc.)."""


# =============================================================================
# Episode result
# =============================================================================

@dataclass
class EpisodeResult:
    """One agent.run() outcome — same shape for native and installed agents.

    ``trajectory`` follows the ALE-v1.0 schema. It's optional only because
    a failure can occur before any step is recorded (rare in practice).
    """

    reward: float | None
    """Score from the task's ``evaluate()``. ``None`` if evaluation didn't run."""

    status: str = "completed"
    """``"completed"`` | ``"timeout"`` | ``"failed"``."""

    error: str | None = None
    """Short description when ``status != "completed"``."""

    instruction: str | None = None
    """The rendered task prompt that started this episode."""

    trajectory: Trajectory | None = None
    """The ALE-v1.0 trajectory. None only if the run died before reset_async."""

    duration_s: float | None = None
    """Wall time from reset to submit."""

    task_path: str | None = None
    variant_index: int | None = None # TODO: use task_id to integrate task_path and variant_index

    # ---- evaluator telemetry (pulled from final Submit observation) ----
    eval_status: str = "not_executed"
    """``"success"`` | ``"failed"`` | ``"not_executed"`` (Submit never fired)."""

    eval_duration_s: float | None = None
    """Wall time the ``task.evaluate()`` call took. None if never ran."""

    eval_error: dict[str, Any] | None = None
    """``{"type", "message", "traceback"}`` populated when eval_status == ``"failed"``."""


# =============================================================================
# Base agent
# =============================================================================

class BaseAgent(abc.ABC):
    """One method: ``run(env, *, variant_index) → EpisodeResult``.

    The env is already bound to a task via :func:`ale.make`. The agent
    just picks the variant to run.
    """

    @abc.abstractmethod
    async def run(
        self,
        env: AgenthleEnv,
        *,
        variant_index: int = 0,
    ) -> EpisodeResult:
        """Drive one episode end-to-end against ``env``."""
