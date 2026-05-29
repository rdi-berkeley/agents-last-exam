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
        executor: which substrate to run the deployer in: ``"sandbox"``
            (in the cua-server VM), ``"local"`` (this Python process), or
            ``"docker"`` (host docker container). Must be in
            ``DeployerCls.supported_executors`` or ``None`` (factory uses
            ``DeployerCls.default_executor``).
    """

    id: str
    class_: str
    config: dict[str, Any] = field(default_factory=dict)
    executor: str | None = None


@dataclass
class ProviderSpec:
    """VM provider selection. ``kind`` picks the impl; ``config`` is its kwargs."""

    kind: str                                    # gcloud | static | (stub for tests)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactsSpec:
    """Artifact path config — yaml ``artifacts_path:`` block.

    ``task_data_source`` selects where task data comes from:
    ``"baked_in_sandbox"`` (image already has it — the default), a
    ``"gs://<bucket>"`` prefix (rsync from GCS; public mirror is
    ``gs://ale-data-public``), or ``"hf://<dataset>"``.

    ``output_path`` controls what happens to the env's output dir after the
    agent finishes. Tri-state:

    * ``None`` (yaml ``null``) — skip output gather entirely. The agent's
      output files stay on the VM and are lost on VM teardown. Smallest
      footprint; the only signal that survives is the eval score.
    * ``"local"`` — pull files from the VM straight to
      ``<run_dir>/output/`` via cua HTTP (no GCS round-trip). Right for
      dev / smoke / small outputs.
    * ``"gs://<bucket>[/<prefix>]"`` — push from the VM to that GCS
      bucket via ``gsutil`` (one hop, fast on large dirs). Nothing lands
      on the host run dir in this mode. Right for large-scale batches
      where you'll process outputs later out-of-band. Hard fail if GCS
      push fails — no fallback in V1.
    """

    task_data_source: str = "baked_in_sandbox"
    output_path: str | None = None


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
    """VM disposition after a unit finishes.

    - ``"delete"`` (default): tear the VM down via ``gcloud instances delete``.
    - ``"stop"``: ``gcloud instances stop``; the disk remains so a later
      run can re-start the VM and inspect agent artifacts.
    - ``"keep"``: leave the VM running (debug / reproducer use).
    """

    prompt_suffix: str = ""
    """Text appended to *every* task's prompt before it is handed to the
    agent. Empty (default) ⇒ no change. The suffix is appended after the
    task description with a blank-line separator so it reads as its own
    paragraph; it also lands in the recorded trajectory so the run reflects
    exactly what the agent saw. Set via the top-level ``prompt_suffix:``
    yaml key — a yaml ``|`` block scalar is convenient for multi-line text."""


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
