"""ExperimentSpec dataclasses — pure data, no IO.

Built by :mod:`ale.runner.loader` from a yaml file, consumed by
:class:`ale.runner.Runner`. One ExperimentSpec describes a whole batch:
which provider, which agents (inline configs), which tasks × variants,
and how to mirror artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskSpec:
    """One task entry. ``variants`` is the list of variant indices to run."""

    path: str
    variants: list[int] = field(default_factory=lambda: [0])


@dataclass
class AgentSpec:
    """One agent entry. Multiple AgentSpecs in an experiment ⇒ matrix.

    Attributes:
        id: short user-chosen label, used as the top folder in the output
            tree (e.g. ``"cc_sonnet"``). Lets a single experiment run
            multiple config-variants of the same class.
        class_: either a shortcut (``"claude_code"``) registered in
            ``ale.runner.factory.AGENT_REGISTRY``, or a fully-qualified
            Deployer class path (e.g. ``"my_pkg.MyDeployer"``).
        config: kwargs passed verbatim into the deployer's Config dataclass.
        runtime: which substrate to run the deployer in: ``"vm"`` (inside
            the eval VM), ``"local"`` (this Python process), or
            ``"docker"`` (host docker container). Must be in
            ``DeployerCls.supported_runtimes`` or ``None`` (factory picks
            the sole value when the deployer supports exactly one; else
            picks the first sorted value as a friendly default — currently
            ``"local"`` wins over ``"docker"`` for native-flavor deployers).
    """

    id: str
    class_: str
    config: dict[str, Any] = field(default_factory=dict)
    runtime: str | None = None


@dataclass
class ProviderSpec:
    """VM provider selection. ``kind`` picks the impl; ``config`` is its kwargs."""

    kind: str                                    # gcs_direct | static | (stub for tests)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactsSpec:
    """ArtifactMirror knobs. Maps directly onto :class:`ArtifactMirrorConfig`."""

    gcs_bucket: str | None = None
    gcs_local_key_file: str | None = None
    gcs_vm_key_file: str | None = None
    fallback_to_cua: bool = True


@dataclass
class OutputSpec:
    """Where the on-disk run dirs go."""

    root: str = ".logs/ale"


@dataclass
class ExperimentSpec:
    name: str
    output: OutputSpec
    provider: ProviderSpec
    agents: list[AgentSpec]
    tasks: list[TaskSpec]
    artifacts: ArtifactsSpec = field(default_factory=ArtifactsSpec)
    concurrency: int = 1
    """Max simultaneous run units. ``1`` = sequential. With ``StaticProvider``
    (one shared VM) keep this at 1 to avoid work_dir collisions; with real
    ephemeral providers, set higher to parallelize across VMs."""

    eval_timeout_s: float = 3600.0
    """Per-task wall budget for ``task.evaluate`` (the scoring step that
    runs on the VM after the agent finishes). Independent from the agent's
    own ``timeout_s`` so a heavy evaluator can't be silently capped by it.
    Default 1h; raise for tasks with multi-stage / network-heavy scoring."""


# =============================================================================
# Derived run units (one per agent × task × variant combination)
# =============================================================================

@dataclass
class RunUnit:
    """One concrete (agent, task, variant) tuple to execute."""

    agent_id: str               # AgentSpec.id (user-chosen label)
    agent_spec: AgentSpec
    task_path: str
    variant_index: int

    @property
    def slug(self) -> str:
        return f"{self.agent_id}/{self.task_path}/v{self.variant_index}"


@dataclass
class UnitResult:
    """Per-unit outcome. No aggregation; the Runner returns ``list[UnitResult]``."""

    unit: RunUnit
    status: str                                  # completed | failed | cancelled | not_executed
    score: float | None = None
    eval_status: str = "not_executed"
    duration_s: float | None = None
    run_dir: Path | None = None
    error: str | None = None
