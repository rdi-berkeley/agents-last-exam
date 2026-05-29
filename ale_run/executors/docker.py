"""DockerExecutor — deployer runs in a fresh container per unit.

Per-unit ``docker run --rm`` with:

* ``-v <host_work>:/work``           — work_dir is bind-mounted; reads/
                                       writes go straight to host fs
* ``-v <host_ale_root>:/ale_src``    — ship the local checkout's
                                       ``ale_run/`` into the container at
                                       import time (no copy needed)
* ``--network host``                 — so the deployer can reach the
                                       sandbox VM's cua-server endpoint
* ``--env-file <api_keys>``          — keeps api keys off the cmdline +
                                       out of ``docker inspect``. Written
                                       to a private tempdir OUTSIDE the
                                       bind mount so it never lands in the
                                       gathered host log dir; removed after
                                       the run.

Entrypoint is ``python -m ale_run.executors._docker_entry`` —
:mod:`_docker_entry` reads ``/work/_spec.json`` and reconstructs the
deployer + a :class:`LocalExecutor` in-container.

Because work_dir is bind-mounted, :meth:`gather_dir` is a no-op and
:meth:`download_range` is a local seek+read on the host side of the
bind mount.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ..base_interface import (
    BaseExecutor,
    GatherReport,
    RangeResult,
    SandboxHandle,
)
from ._secrets import SECRET_GATHER_EXCLUDES, SECRETS_FILE, write_secrets

if TYPE_CHECKING:
    from ..base_interface import AgentRunResult, BaseAgentDeployer

logger = logging.getLogger(__name__)


# Default image. Operator can override at construction time (see field below).
_DEFAULT_IMAGE_TAG = "ale/native-base:0.1.0"

# How much wall-clock slack we give the container over the agent budget
# (covers ``uv sync`` / dep install + container teardown).
_DOCKER_SLACK_S = 180.0


def _host_repo_root() -> Path:
    """Path to ``agents-last-exam/`` on the host — parent of ``ale_run/``.

    The container needs the source tree mounted so it can import ale_run.
    """
    # This module: .../agents-last-exam/ale_run/executors/docker.py
    return Path(__file__).resolve().parents[2]


_ENV_PASSTHROUGH_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "BRAVE_API_KEY",
)


@dataclass
class DockerExecutor(BaseExecutor):
    """In-container substrate. work_dir is host-bind-mounted at ``/work``."""

    type: ClassVar[str] = "docker"

    image: str = _DEFAULT_IMAGE_TAG
    """Docker image tag to ``docker run``. Operator overrides if a
    deployer needs a custom base (claude CLI baked in, etc.)."""

    extra_run_args: list[str] = field(default_factory=list)
    """Extra args spliced into the ``docker run`` command (memory limits,
    GPU pass-through, etc.). Free-form — operator's responsibility."""

    async def run_deployer(
        self,
        *,
        deployer_cls: type["BaseAgentDeployer"],
        prompt: str,
        timeout_s: float,
    ) -> "AgentRunResult":
        from ..base_interface import AgentRunResult

        if not _docker_available():
            return AgentRunResult(
                status="failed",
                error="DockerExecutor: docker CLI not found on PATH",
            )
        if not _image_present(self.image):
            return AgentRunResult(
                status="failed",
                error=f"DockerExecutor: image {self.image!r} not present "
                      f"on host. Pull or build it first.",
            )

        host_work = Path(self.work_dir)
        host_work.mkdir(parents=True, exist_ok=True)

        # 1. Write spec.json into the bind mount.
        #    Secrets (api keys etc.) are deliberately KEPT OUT of the spec —
        #    work_dir IS the host log dir for docker runs, so _spec.json is a
        #    host log file and must stay keyless. The env goes in a separate
        #    _secrets.json that the entry reads once and deletes.
        spec = {
            "ale_src_root": "/ale_src",  # container view
            "deployer_module": deployer_cls.__module__,
            "deployer_class": deployer_cls.__name__,
            "config_module": self.config.__class__.__module__,
            "config_class": self.config.__class__.__name__,
            "config_kwargs": _config_to_kwargs(self.config),
            "sandbox_kwargs": _sandbox_to_kwargs(self.sandbox),
            "work_dir": "/work",
            "secrets_file": SECRETS_FILE,
            "prompt": prompt,
            "timeout_s": float(timeout_s),
        }
        (host_work / "_spec.json").write_text(json.dumps(spec, indent=2))

        # 1b. Write the read-once secrets sidecar into the bind mount. The
        #     in-container entry reads it then deletes it, so it does not
        #     persist in the host log dir. chmod 600 while it lives.
        write_secrets(host_work, dict(self.env or {}))

        # 2. Write env-file (keeps api keys off cmdline + docker inspect).
        #    Lives in a private tempdir OUTSIDE the bind-mounted work_dir so
        #    it is never part of the host log dir; removed after the run.
        env_lines = []
        for k in _ENV_PASSTHROUGH_KEYS:
            v = os.environ.get(k) or (self.env or {}).get(k)
            if v:
                env_lines.append(f"{k}={v}")
        env_tmp_dir = Path(tempfile.mkdtemp(prefix="ale-env-"))
        env_file = env_tmp_dir / "env"
        env_file.write_text("\n".join(env_lines) + ("\n" if env_lines else ""))
        try:
            env_file.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except OSError:
            pass

        try:
            return await self._run_container(
                deployer_cls=deployer_cls,
                host_work=host_work,
                env_file=env_file,
                timeout_s=timeout_s,
            )
        finally:
            # Remove the env-file tempdir and, defensively, the bind-mounted
            # secrets sidecar in case the container died before the entry
            # could read+delete it (so no key reaches the host log dir).
            shutil.rmtree(env_tmp_dir, ignore_errors=True)
            try:
                (host_work / SECRETS_FILE).unlink()
            except OSError:
                pass

    async def _run_container(
        self,
        *,
        deployer_cls: type["BaseAgentDeployer"],
        host_work: Path,
        env_file: Path,
        timeout_s: float,
    ) -> "AgentRunResult":
        from ..base_interface import AgentRunResult

        # 3. docker run argv
        container_name = f"ale-{deployer_cls.__name__.lower()}-{uuid.uuid4().hex[:8]}"
        host_repo = _host_repo_root()
        docker_argv = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "host",
            "-v", f"{host_work}:/work:rw",
            "-v", f"{host_repo}:/ale_src:ro",
            "-e", "PYTHONPATH=/ale_src",
            "--env-file", str(env_file),
            *self.extra_run_args,
            self.image,
            "python", "-m", "ale_run.executors._docker_entry",
        ]
        logger.info(
            "docker: starting %s (image=%s, work_dir=%s)",
            container_name, self.image, host_work,
        )

        # 4. Run with wall-budget = agent timeout + slack
        t0 = time.monotonic()
        wall = timeout_s + _DOCKER_SLACK_S
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=wall,
                )
            except asyncio.TimeoutError:
                logger.warning("docker: %s wall budget %.0fs exceeded — killing",
                               container_name, wall)
                await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                )
                return AgentRunResult(
                    status="timeout",
                    duration_s=time.monotonic() - t0,
                    error=f"docker wall budget {wall:.0f}s exceeded",
                )
        except Exception as e:                                     # noqa: BLE001
            return AgentRunResult(
                status="failed",
                duration_s=time.monotonic() - t0,
                error=f"docker run transport: {type(e).__name__}: {e}",
            )

        duration_s = time.monotonic() - t0
        rc = proc.returncode
        if rc != 0:
            tail = (stderr or b"").decode("utf-8", errors="replace")[-1500:]
            logger.error("docker: container rc=%s\nstderr tail:\n%s", rc, tail)

        # 5. Read result.json
        result_path = host_work / "_result.json"
        if not result_path.exists():
            return AgentRunResult(
                status="failed",
                duration_s=duration_s,
                error=f"container rc={rc} wrote no _result.json; "
                      f"stderr tail: "
                      f"{(stderr or b'').decode('utf-8', errors='replace')[-800:]}",
            )
        try:
            out = json.loads(result_path.read_text())
        except json.JSONDecodeError as e:
            return AgentRunResult(
                status="failed",
                duration_s=duration_s,
                error=f"_result.json parse failed: {e}",
            )

        if not out.get("ok", False):
            tb = out.get("traceback") or ""
            err = out.get("error") or "container bootstrap failed"
            return AgentRunResult(
                status=out.get("status", "failed"),
                error=f"{err}\n{tb}" if tb else err,
                duration_s=out.get("duration_s") or duration_s,
            )
        return AgentRunResult(
            status=out.get("status", "failed"),
            error=out.get("error"),
            transcript_path=out.get("transcript_path"),
            stderr_path=out.get("stderr_path"),
            pid=out.get("pid"),
            exit_code=out.get("exit_code"),
            duration_s=out.get("duration_s") or duration_s,
        )

    async def gather_dir(
        self, *, src: str, dst: Path,
    ) -> GatherReport:
        # Bind-mounted: src on host == dst (or already a host path).
        # If src and dst differ, do a host-side copy.
        src_path = Path(src)
        if src_path == dst:
            return GatherReport(transport="docker-bindmount", files=0, bytes=0)
        if not src_path.exists():
            return GatherReport(
                transport="docker-bindmount", error=f"src not found: {src}",
            )

        def _copy() -> tuple[int, int, str | None]:
            files = 0
            total = 0
            try:
                dst.mkdir(parents=True, exist_ok=True)
                for entry in src_path.rglob("*"):
                    if entry.is_dir():
                        continue
                    if entry.name in SECRET_GATHER_EXCLUDES:
                        continue
                    rel = entry.relative_to(src_path)
                    target = dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(entry, target)
                    files += 1
                    total += target.stat().st_size
                return files, total, None
            except Exception as e:                                  # noqa: BLE001
                return files, total, str(e)

        files, total, err = await asyncio.to_thread(_copy)
        return GatherReport(
            transport="docker-bindmount", files=files, bytes=total, error=err,
        )

    async def download_range(
        self, *, src: str, start: int, max_bytes: int,
    ) -> RangeResult:
        def _read() -> RangeResult:
            p = Path(src)
            if not p.exists():
                return RangeResult(success=False, error="file not found")
            try:
                size = p.stat().st_size
                if start >= size:
                    return RangeResult(success=True, new_data=b"", new_size=size)
                with open(p, "rb") as f:
                    f.seek(start)
                    data = f.read(max_bytes)
                return RangeResult(success=True, new_data=data, new_size=size)
            except OSError as e:
                return RangeResult(success=False, error=str(e))
        return await asyncio.to_thread(_read)


# ======================================================================
# helpers
# ======================================================================


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _image_present(tag: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _config_to_kwargs(cfg) -> dict:
    import dataclasses

    out = {}
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if isinstance(val, (str, int, float, bool, type(None), list, dict, tuple)):
            out[f.name] = val
    return out


def _sandbox_to_kwargs(sb: SandboxHandle) -> dict:
    return {
        "id": sb.id,
        "endpoint": sb.endpoint,
        "os": sb.os,
        "work_dir_base": sb.work_dir_base,
        "task_data_root": sb.task_data_root,
        "node": sb.node,
        "python": sb.python,
        "mcp_server_dir": sb.mcp_server_dir,
        "metadata": dict(sb.metadata or {}),
    }
