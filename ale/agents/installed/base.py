"""InstalledAgent + InstalledAgentDeployer + InstalledAgentConfig.

The agent CLI runs *inside* the VM. The deployer (agent-specific) knows how
to install / launch / parse-into-trajectory. The framework drives the
lifecycle:

    env.reset_async(...)                              # acquire VM + task.start
    deployer.install(session)                         # verify prereqs, write configs
    run = deployer.launch(session, prompt, timeout)   # spawn CLI, wait for completion
    deployer.collect(session, run, builder)           # parse CLI logs → trajectory.steps
    env.step_async(Submit())                          # task.evaluate, get reward
    env.close_async()                                 # release VM

The deployer never owns the env / instruction / submit / final reward.
Those belong to the framework so all installed agents produce uniform
:class:`EpisodeResult` shape.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ale.agents.base import BaseAgent, BaseAgentConfig, EpisodeResult
from ale.agents.trajectory import TrajectoryBuilder
from ale.core.env import AgenthleEnv
from ale.core.types import Submit

from .install_paths import InstallPaths

if TYPE_CHECKING:
    import cua_bench as cb


# =============================================================================
# Config
# =============================================================================

@dataclass
class InstalledAgentConfig(BaseAgentConfig):
    """Common config for any in-VM CLI agent.

    Subclasses add their CLI-specific fields (API keys, tool restrictions,
    model-specific flags). The base provides ``install_paths`` so per-image
    layout overrides don't require touching deployer code.
    """

    install_paths: InstallPaths = field(default_factory=InstallPaths)


# =============================================================================
# Deployer contract
# =============================================================================

@dataclass
class AgentRunResult:
    """Outcome of :meth:`InstalledAgentDeployer.launch` — pre-collect.

    Passed to :meth:`InstalledAgentDeployer.collect` so the deployer can
    read whatever it wrote (transcript path, stderr path, PID/exit code).
    """

    status: str                          # "completed" | "timeout" | "failed"
    transcript_path: str | None = None
    stderr_path: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    duration_s: float | None = None
    error: str | None = None


class InstalledAgentDeployer(abc.ABC):
    """Three-phase ABC. Subclasses are CLI-specific (Claude Code, OpenClaw, ...)."""

    @property
    @abc.abstractmethod
    def config(self) -> InstalledAgentConfig:
        """The deployer's bound config. Used by the framework for trajectory metadata."""

    @property
    def version(self) -> str | None:
        """Override to surface a CLI / SDK version string in trajectory metadata."""
        return None

    @abc.abstractmethod
    async def install(self, session: "cb.DesktopSession") -> None:
        """Verify image prereqs + stage per-run config on the VM. Should be fast.

        Raise loudly if the snapshot is missing a required binary — we don't
        do runtime downloads.
        """

    @abc.abstractmethod
    async def launch(
        self,
        session: "cb.DesktopSession",
        *,
        prompt: str,
        timeout_s: float,
    ) -> AgentRunResult:
        """Spawn the CLI and wait for completion or timeout.

        Always return an :class:`AgentRunResult` (errors → ``status="failed"``,
        ``error=...``). Raise only if even *starting* failed.
        """

    @abc.abstractmethod
    async def collect(
        self,
        session: "cb.DesktopSession",
        run: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Parse the CLI's structured log files and append Steps to ``builder``.

        The framework owns the builder (it's already seeded with the
        ``user``-source instruction Step). The deployer:

        - reads ``run.transcript_path`` / ``run.stderr_path``
        - appends ``agent`` Steps (LLM turns: messages, tool calls, metrics)
        - appends ``environment`` Steps (tool results from the in-VM agent)
        - may set ``builder.trajectory.extra`` for CLI-specific metadata

        Partial / empty transcripts are valid — log a warning, do nothing.
        """

    @abc.abstractmethod
    def work_dir(self, session: "cb.DesktopSession") -> str | None:
        """The on-VM directory this deployer writes to (transcript / scripts / logs).

        Used by :meth:`InstalledAgent.mirror_artifacts` to pull raw files
        from the VM. Returning ``None`` skips origin_log mirroring for
        this deployer.
        """


# =============================================================================
# InstalledAgent — wires Env + Deployer through the lifecycle
# =============================================================================

class InstalledAgent(BaseAgent):
    """A :class:`BaseAgent` whose work happens inside the env's VM."""

    def __init__(self, deployer: InstalledAgentDeployer):
        self._deployer = deployer

    @property
    def config(self) -> InstalledAgentConfig:
        return self._deployer.config

    @property
    def deployer(self) -> InstalledAgentDeployer:
        return self._deployer

    async def mirror_artifacts(
        self,
        env: AgenthleEnv,
        mirror: "Any",  # ale.io.artifact_mirror.ArtifactMirror (lazy import to avoid cycle)
    ) -> dict[str, Any]:
        """Pull deployer ``work_dir`` + task ``remote_output_dir`` to local disk.

        Returns ``{"origin_log": {...}, "output": {...}}`` with transport
        + file counts for each. Call after :meth:`run` returns but
        **before** ``env.close_async()`` (we need the session alive).
        """
        if env._session is None:                              # noqa: SLF001
            return {"origin_log": {"error": "no session"},
                    "output": {"error": "no session"}}
        session = env._session                                # noqa: SLF001
        report: dict[str, Any] = {}

        # 1. Deployer work_dir → origin_log/<agent_name>/
        work_dir = self._deployer.work_dir(session)
        if work_dir:
            report["origin_log"] = await mirror.pull_dir(
                session, work_dir, f"origin_log/{self.config.name}",
            )
        else:
            report["origin_log"] = {"transport": "skipped",
                                     "reason": "deployer.work_dir() = None"}

        # 2. Task remote_output_dir → output/
        lt = env._lt                                          # noqa: SLF001
        output_dir: str | None = None
        if lt is not None and lt.cb_task and lt.cb_task.metadata:
            output_dir = lt.cb_task.metadata.get("remote_output_dir")
        if output_dir:
            report["output"] = await mirror.pull_dir(session, output_dir, "output")
        else:
            report["output"] = {"transport": "skipped",
                                 "reason": "task has no remote_output_dir"}
        return report

    async def run(
        self,
        env: AgenthleEnv,
        *,
        variant_index: int = 0,
    ) -> EpisodeResult:
        cfg = self._deployer.config
        task_path = env.task_path
        builder = TrajectoryBuilder(
            agent_name=cfg.name,
            agent_version=self._deployer.version,
            model=cfg.model or None,
            task_path=task_path,
            variant_index=variant_index,
        )
        t0 = time.monotonic()
        instruction: str | None = None
        run_result: AgentRunResult | None = None
        status = "completed"
        error: str | None = None
        reward: float | None = None

        try:
            # 1. Reset env — acquire VM, run task.start, get instruction.
            obs = await env.reset_async(variant_index=variant_index)
            instruction = obs.instruction or ""
            builder.trajectory.instruction = instruction
            builder.add_step(source="user", message=instruction)

            # 2. Install — verify image prereqs + stage config files.
            await self._deployer.install(env.session)

            # 3. Launch — spawn the CLI, wait for completion / timeout.
            run_result = await self._deployer.launch(
                env.session, prompt=instruction, timeout_s=cfg.timeout_s,
            )
            status = run_result.status
            error = run_result.error

            # 4. Collect — parse logs into trajectory steps (even on timeout/failure).
            try:
                await self._deployer.collect(env.session, run_result, builder)
            except Exception as collect_exc:        # noqa: BLE001
                builder.add_step(
                    source="system",
                    message=f"collect failed: {type(collect_exc).__name__}: {collect_exc}",
                    extra={"reason": "collect_error"},
                )

            # 5. Submit — task.evaluate runs against the VM, returns score.
            final_obs = await env.step_async(Submit())
            reward = final_obs.reward

            traj = builder.finalize(reward=reward, status=status)
            return EpisodeResult(
                reward=reward,
                status=status,
                error=error,
                instruction=instruction,
                trajectory=traj,
                duration_s=time.monotonic() - t0,
                task_path=task_path,
                variant_index=variant_index,
                eval_status=final_obs.eval_status or "not_executed",
                eval_duration_s=final_obs.eval_duration_s,
                eval_error=final_obs.eval_error,
            )

        except Exception as exc:
            builder.add_step(
                source="system",
                message=f"run failed: {type(exc).__name__}: {exc}",
                extra={"reason": "exception"},
            )
            traj = builder.finalize(reward=None, status="failed")
            return EpisodeResult(
                reward=None,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                instruction=instruction,
                trajectory=traj,
                duration_s=time.monotonic() - t0,
                task_path=task_path,
                variant_index=variant_index,
            )
