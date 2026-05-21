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

    kind: str                                    # gcloud | static | (stub for tests)
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
    """Max units running simultaneously. ``1`` = sequential. Each unit
    holds a slot for its full lifetime — VM acquire + agent run + post-
    launch fan-out + eval — so the cap is effectively "max VMs alive at
    once". Size to ``min(GCP quota, LLM rate-limit / N)``. With
    ``StaticProvider`` (one shared VM) must stay at 1 to avoid work_dir
    collisions."""

    cleanup_mode: str = "delete"
    """VM disposition after a unit finishes (simprun parity).

    - ``"delete"`` (default): tear the VM down via ``gcloud instances delete``.
    - ``"stop"``: ``gcloud instances stop``; the disk remains so a later
      run can re-start the VM and inspect agent artifacts.
    - ``"keep"``: leave the VM running (debug / reproducer use).

    Rate-limit-triggered runs always force ``"keep"`` regardless of this
    setting — losing the VM on a rate-limit retryable failure costs the
    next attempt's boot time for nothing."""


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
