"""Single agent contract: BaseAgentDeployer.

A deployer fully owns one agent's lifecycle on one task: ``install`` (set
up wherever the agent process runs), ``launch`` (kick off, wait for
completion), ``collect`` (translate that agent's logs into the standard
ALE-v1.0 trajectory schema). The framework drives the env, mirrors
artifacts, and finalizes the run dir. Subclasses focus only on the
install / launch / collect glue.

Two flavors share this base, distinguished only by ``work_dir_on_vm``:

- **In-VM** (default ``work_dir_on_vm = True``). The agent CLI runs
  inside the guest; ``install`` stages binaries on the VM via
  ``session``; ``work_dir`` is a VM path; mirror pulls it via cua
  direct or the GCS bridge.
- **Native** (``work_dir_on_vm = False``). The agent process runs on
  the ALE host (local subprocess, docker container, ...). ``install``
  may use ``session`` only to read VM info (``os_type``, endpoint) the
  local process needs; ``work_dir`` is a local path; mirror does a
  ``shutil.copytree`` instead of a VM pull.

Both produce uniform :class:`EpisodeResult` carrying an ALE-v1.0
:class:`Trajectory`. Downstream consumers don't branch on flavor.
"""
from __future__ import annotations

import abc
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ale.agents.install_paths import InstallPaths
from ale.agents.trajectory import Trajectory, TrajectoryBuilder
from ale.core.env import AgenthleEnv
from ale.core.types import Submit

if TYPE_CHECKING:
    import cua_bench as cb


# =============================================================================
# Config
# =============================================================================

@dataclass
class BaseAgentConfig:
    """Shared tunables for any agent.

    Subclasses MUST set the class attribute ``name``. They MAY add their own
    fields; they SHOULD NOT redefine the standard fields below.
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

    install_paths: InstallPaths = field(default_factory=InstallPaths)
    """In-VM path layout. Used by in-VM deployers; native deployers can
    ignore. Defaults match the baked agenthle Linux/Windows images."""


# =============================================================================
# Run + episode results
# =============================================================================

@dataclass
class AgentRunResult:
    """Outcome of :meth:`BaseAgentDeployer.launch` — pre-collect.

    Passed to :meth:`BaseAgentDeployer.collect` so the deployer can read
    whatever it wrote (transcript path, stderr path, exit code).
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
    """One :meth:`BaseAgentDeployer.run` outcome.

    ``trajectory`` is optional only because a failure can occur before any
    step is recorded (rare in practice).
    """

    reward: float | None
    """Score from the task's ``evaluate()``. ``None`` if evaluation didn't run."""

    status: str = "completed"
    """``"completed"`` | ``"timeout"`` | ``"failed"`` | ``"cancelled"``."""

    error: str | None = None
    """Short description when ``status != "completed"``."""

    instruction: str | None = None
    """The rendered task prompt that started this episode."""

    trajectory: Trajectory | None = None
    """The ALE-v1.0 trajectory. None only if the run died before reset_async."""

    duration_s: float | None = None
    """Wall time from reset to submit."""

    task_path: str | None = None
    variant_index: int | None = None

    # ---- evaluator telemetry (pulled from the final Submit observation) ----
    eval_status: str = "not_executed"
    """``"success"`` | ``"failed"`` | ``"not_executed"`` (Submit never fired)."""

    eval_duration_s: float | None = None
    """Wall time the ``task.evaluate()`` call took. None if never ran."""

    eval_error: dict[str, Any] | None = None
    """``{"type", "message", "traceback"}`` populated when eval_status == ``"failed"``."""


# =============================================================================
# Deployer
# =============================================================================

class BaseAgentDeployer(abc.ABC):
    """Single agent contract. CLI- / runtime-specific subclasses implement
    install / launch / collect; the base class drives env + finalizes results.
    """

    work_dir_on_vm: ClassVar[bool] = True
    """Where this deployer's :meth:`work_dir` lives:

    - ``True`` (default) — in-VM. Mirror reads from the VM via the
      :class:`ale.io.artifact_mirror.ArtifactMirror` (GCS bridge or cua
      direct).
    - ``False`` — native. Mirror copies the local directory directly.
    """

    # ---- subclass contract --------------------------------------------------

    @property
    @abc.abstractmethod
    def config(self) -> BaseAgentConfig:
        """The deployer's bound config. Used for trajectory metadata + run.json."""

    @property
    def version(self) -> str | None:
        """Override to surface a CLI / SDK version string."""
        return None

    @abc.abstractmethod
    async def install(self, session: "cb.DesktopSession") -> None:
        """Stage whatever the agent process needs to start.

        In-VM deployers (default) write files / verify binaries on the VM
        via ``session``. Native deployers may ignore ``session`` (or use
        it only to read OS metadata) and instead set up local resources
        (docker pull, venv create, etc.).
        """

    @abc.abstractmethod
    async def launch(
        self,
        session: "cb.DesktopSession",
        *,
        prompt: str,
        timeout_s: float,
    ) -> AgentRunResult:
        """Spawn the agent and wait for it to finish or time out.

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
        """Parse the agent's structured logs into trajectory Steps.

        The framework owns the builder (already seeded with the
        ``user``-source instruction Step). Append ``agent`` / ``environment``
        Steps; set ``builder.trajectory.extra`` for CLI-specific metadata.
        Partial logs are valid — log a warning and do nothing.
        """

    @abc.abstractmethod
    def work_dir(self, session: "cb.DesktopSession") -> str | None:
        """Where this deployer wrote its files this run.

        Path semantics depend on :attr:`work_dir_on_vm` (VM path vs
        local path). Returning ``None`` skips origin_log mirroring.
        """

    # ---- framework lifecycle (concrete) -------------------------------------

    async def run(
        self,
        env: AgenthleEnv,
        *,
        variant_index: int = 0,
    ) -> EpisodeResult:
        """Drive one episode end-to-end against ``env``.

        Steps: reset → install → launch → collect → submit. Always returns
        an :class:`EpisodeResult` (failures captured, not raised).
        """
        cfg = self.config
        task_path = env.task_path
        builder = TrajectoryBuilder(
            agent_name=cfg.name,
            agent_version=self.version,
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

            # 2. Install — stage agent prereqs (VM-side or local).
            await self.install(env.session)

            # 3. Launch — spawn the agent, wait for completion / timeout.
            run_result = await self.launch(
                env.session, prompt=instruction, timeout_s=cfg.timeout_s,
            )
            status = run_result.status
            error = run_result.error

            # 4. Collect — parse logs into trajectory steps (always, even on failure).
            try:
                await self.collect(env.session, run_result, builder)
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

    async def mirror_artifacts(
        self,
        env: AgenthleEnv,
        mirror: Any,    # ale.io.artifact_mirror.ArtifactMirror (lazy to avoid cycle)
    ) -> dict[str, Any]:
        """Pull deployer ``work_dir`` + task ``remote_output_dir`` to local disk.

        Returns ``{"origin_log": {...}, "output": {...}}`` with transport
        + file counts for each. Call after :meth:`run` returns but
        **before** ``env.close_async()`` (the session must still be alive
        for VM-side pulls).

        Branches on :attr:`work_dir_on_vm`:

        - ``True`` → ``mirror.pull_dir(session, work_dir, ...)`` (VM)
        - ``False`` → ``shutil.copytree(work_dir, local_root/origin_log/...)``

        The task's ``remote_output_dir`` is always on the VM by definition,
        so it always goes through the mirror.
        """
        if env._session is None:                              # noqa: SLF001
            return {"origin_log": {"error": "no session"},
                    "output": {"error": "no session"}}
        session = env._session                                # noqa: SLF001
        report: dict[str, Any] = {}

        # 1. Deployer work_dir → origin_log/<agent_name>/
        wd = self.work_dir(session)
        sub = f"origin_log/{self.config.name}"
        if not wd:
            report["origin_log"] = {"transport": "skipped",
                                    "reason": "deployer.work_dir() = None"}
        elif self.work_dir_on_vm:
            report["origin_log"] = await mirror.pull_dir(session, wd, sub)
        else:
            report["origin_log"] = _copy_local(
                Path(wd), Path(mirror._cfg.local_root) / sub,            # noqa: SLF001
            )

        # 2. Task remote_output_dir → output/  (always VM-side by definition)
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


# =============================================================================
# Helpers
# =============================================================================

def _copy_local(src: Path, dst: Path) -> dict[str, Any]:
    """``shutil.copytree`` with a mirror-shaped status dict."""
    if not src.exists():
        return {"transport": "local_copy", "files": 0,
                "error": f"src missing: {src}"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    files = sum(1 for p in dst.rglob("*") if p.is_file())
    return {"transport": "local_copy", "files": files,
            "src": str(src), "dst": str(dst), "error": None}
