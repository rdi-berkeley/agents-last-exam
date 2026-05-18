"""BaseAgentDeployer — the minimal contract every ALE agent implements.

A deployer is **just code**: a few Python methods that get *placed* and
*run* by the framework's runtime layer. Where they run (vm / local /
docker) is a framework concern, expressed in yaml as ``runtime: <kind>``
and validated against the deployer's ``supported_runtimes`` ClassVar.

The deployer never receives an env, a session, or a substrate object.
It receives ONE thing at construction: an :class:`AgentRuntime` (passive
context — work_dir, vm_endpoint, vm_os, config). Methods are plain
async functions that use ``self.runtime`` + Python stdlib (subprocess,
pathlib, json) to do their work. Where ``self`` happens to live (VM
Python, container Python, host Python) is decided by the framework's
:class:`Executor` for the chosen runtime.

Contract:

  class BaseAgentDeployer(abc.ABC):
      supported_runtimes: ClassVar[frozenset[str]]      # subclass must set

      def __init__(self, runtime: AgentRuntime): ...    # framework calls
      async def install(self) -> None: ...              # subclass implements
      async def launch(self, prompt: str) -> AgentRunResult: ...
      @classmethod
      def parse_artifacts(cls, *, work_dir, config, run_result, builder) -> None: ...

That's it. No ``collect``, no ``work_dir`` method, no ``mirror_artifacts``,
no ``run(env)``. The framework owns env lifecycle (task setup + evaluate),
runtime construction (where deployer lives), artifact gathering (work_dir
→ host), and trajectory finalize. The deployer only owns: stage prereqs,
launch the agent, parse its on-disk logs.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ale.agents.trajectory import Trajectory, TrajectoryBuilder

if TYPE_CHECKING:
    from ale.runtime.base import AgentRuntime


# =============================================================================
# Config
# =============================================================================

@dataclass
class BaseAgentConfig:
    """Shared tunables for any agent.

    Subclasses MUST set the class attribute ``name``. They MAY add their
    own fields; they SHOULD NOT redefine the standard fields below.
    """

    # Identifier — set on concrete config subclasses, NOT __init__ field.
    name: ClassVar[str] = ""

    model: str = ""
    """LLM model id, e.g. ``claude-opus-4-7``, ``gpt-5``. Empty string =
    use the deployer's own default (each subclass documents its default)."""

    max_turns: int | None = None
    """Upper bound on agent turns. None = no cap (deployer enforces other
    limits like timeout / max_budget)."""

    timeout_s: float = 1800.0
    """Wall-clock budget for the whole episode (including evaluate)."""

    save_screenshots: bool = True
    """Hint to deployers that capture screenshots."""

    api_keys: dict[str, str] = field(default_factory=dict)
    """Bag of name → value for arbitrary env vars. Caller passes explicitly;
    never auto-read from os.environ."""


# =============================================================================
# Run + episode results
# =============================================================================

@dataclass
class AgentRunResult:
    """Outcome of :meth:`BaseAgentDeployer.launch` — handed to
    :meth:`parse_artifacts` along with the gathered work_dir.

    Pure data; serializable across runtime boundaries (docker exec, VM
    python_exec return value).
    """

    status: str                          # "completed" | "timeout" | "failed"
    transcript_path: str | None = None
    stderr_path: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    duration_s: float | None = None
    error: str | None = None


@dataclass
class EpisodeResult:
    """The framework lifecycle's final assembly. Returned by
    :func:`ale.runner.lifecycle.run_one_unit` (indirectly).
    """

    reward: float | None
    status: str = "completed"
    error: str | None = None
    instruction: str | None = None
    trajectory: Trajectory | None = None
    duration_s: float | None = None
    task_path: str | None = None
    variant_index: int | None = None

    eval_status: str = "not_executed"
    eval_duration_s: float | None = None
    eval_error: dict[str, Any] | None = None


# =============================================================================
# Deployer ABC
# =============================================================================

class BaseAgentDeployer(abc.ABC):
    """Minimal deployer contract.

    Subclasses MUST set :attr:`supported_runtimes` (declares which
    substrates this agent can run on: any subset of ``{"vm","local","docker"}``).
    The framework validates yaml ``runtime`` against this set.
    """

    supported_runtimes: ClassVar[frozenset[str]] = frozenset()
    """Subclass overrides — strings match yaml ``runtime: <kind>`` values.
    Empty set is a programmer error caught at :func:`resolve_agent` time."""

    hot_artifacts: ClassVar[tuple[str, ...]] = ()
    """Relative paths under work_dir that the framework should incrementally
    pull from the VM during agent execution (e.g. ``("transcript.jsonl",
    "stderr.log")``). Only relevant for ``runtime: vm`` deployers — local /
    docker work_dirs are already host-visible. Empty default → no pull.
    Newline-delimited / JSONL files only; binary streams aren't supported."""

    def __init__(self, runtime: "AgentRuntime"):
        self.runtime = runtime
        self.config = runtime.config        # convenience alias

    # ---- abstract methods ----

    @abc.abstractmethod
    async def install(self) -> None:
        """Stage prereqs for this run. Use ``self.runtime.work_dir``,
        ``self.runtime.vm_endpoint``, ``self.runtime.vm_os``, and stdlib
        primitives (``subprocess``, ``pathlib``, ``json``). The substrate
        in which this runs (VM, container, host process) is the
        framework's concern — the agent code is identical anywhere."""

    @abc.abstractmethod
    async def launch(self, prompt: str) -> AgentRunResult:
        """Spawn the agent and wait for it to finish.

        Always return an :class:`AgentRunResult` (errors → ``status="failed"``
        with ``error=...``). Raise only if even *starting* failed (the
        framework will catch and treat as failed-run too)."""

    @classmethod
    @abc.abstractmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: BaseAgentConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Read on-disk artifacts in ``work_dir``, populate ``builder`` with
        :class:`Step` entries (call ``builder.add_step(source=..., ...)``).

        Pure function — always runs on the framework host after the
        framework has gathered the runtime's work_dir locally. Doesn't
        need a runtime instance; static across all runtime kinds for a
        given agent. Partial / missing logs are valid; emit a single
        ``source="system"`` step explaining the gap and return cleanly."""

    # ---- optional metadata ----

    @property
    def version(self) -> str | None:
        """CLI / SDK version string surfaced in run.json + trajectory.agent.version.
        Override if the agent has a meaningful version pin."""
        return None
