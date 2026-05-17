"""DockerExecutor — runs the deployer in a host docker container.

Stages a spec.json + env file into the run_dir (which gets bind-mounted
as /work in the container), launches the container with
``ale/native-base:0.1.0`` and an entrypoint that:

  1. ``cd /projects/agents-last-exam``
  2. ``uv sync --all-packages --quiet`` (installs cua-* + litellm + agent deps;
     uv cache mounted from host keeps subsequent runs fast)
  3. ``uv run python -m ale.runtime._docker_entry``

The container has ``--network host`` so it reaches the eval VM's
cua-server endpoint. ``--env-file`` keeps API keys out of cmdline +
``docker inspect``.

Gather is no-op: work_dir is bind-mounted, so files written by the
container at /work are already on host at the bind-mount source.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .executor import EXECUTORS, Executor

if TYPE_CHECKING:
    from ale.agents.base import AgentRunResult, BaseAgentDeployer

    from .base import AgentRuntime
    from .docker import DockerRuntime

logger = logging.getLogger(__name__)


# ---- conventions ----

IMAGE_TAG = "ale/native-base:0.1.0"
"""Built locally via `docker build -t ale/native-base:0.1.0 -f ale/runtime/Dockerfile.native_base ale/runtime/`."""

# Bind-mount source for /projects in the container — the parent of both
# ale repo + agenthle submodule. We mount this so uv sync can resolve
# the path-pinned cua-* deps in root pyproject.toml.
_HOST_PROJECTS_DIR = Path("/Users/weichen/projects/agenthle-overall")

# uv cache persisted across runs to skip re-resolution.
_HOST_UV_CACHE = Path.home() / ".cache" / "uv"


# =============================================================================
# Executor
# =============================================================================

class DockerExecutor(Executor):
    kind = "docker"

    async def run_deployer(
        self,
        *,
        deployer_cls: type["BaseAgentDeployer"],
        runtime: "AgentRuntime",
        prompt: str,
        timeout_s: float,
    ) -> "AgentRunResult":
        from ale.agents.base import AgentRunResult

        if runtime.kind != "docker":
            raise TypeError(
                f"DockerExecutor.run_deployer needs DockerRuntime, got {type(runtime).__name__}"
            )
        host_work_dir = runtime.work_dir   # this IS the host bind-mount source
        host_work_dir.mkdir(parents=True, exist_ok=True)

        # ---- spec.json (read by _docker_entry inside container) ----
        spec = {
            "deployer_module": deployer_cls.__module__,
            "deployer_class": deployer_cls.__name__,
            "config_module": runtime.config.__class__.__module__,
            "config_class": runtime.config.__class__.__name__,
            "config_kwargs": _config_to_kwargs(runtime.config),
            "work_dir": "/work",            # container view
            "vm_endpoint": runtime.vm_endpoint,
            "vm_os": runtime.vm_os,
            "prompt": prompt,
        }
        (host_work_dir / "_spec.json").write_text(json.dumps(spec, indent=2))

        # ---- env file (API keys; --env-file keeps them off cmdline) ----
        env_lines = _collect_env_vars(runtime.config)
        env_file = host_work_dir / "_env"
        env_file.write_text("\n".join(f"{k}={v}" for k, v in env_lines.items()) + "\n")

        # ---- ensure uv cache + image exist ----
        _HOST_UV_CACHE.mkdir(parents=True, exist_ok=True)
        if not _image_present(IMAGE_TAG):
            raise RuntimeError(
                f"docker image {IMAGE_TAG!r} not found. "
                f"Build it: docker build -t {IMAGE_TAG} -f ale/runtime/Dockerfile.native_base ale/runtime/"
            )

        # ---- container args ----
        container_name = f"ale-{deployer_cls.__name__.lower()}-{uuid.uuid4().hex[:8]}"
        container_cmd = (
            "cd /projects/agents-last-exam && "
            "uv sync --all-packages --quiet && "
            "uv run python -m ale.runtime._docker_entry"
        )
        docker_argv = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "host",
            "--memory", "4g", "--cpus", "2",
            "-v", f"{_HOST_PROJECTS_DIR}:/projects:rw",
            "-v", f"{host_work_dir}:/work:rw",
            "-v", f"{_HOST_UV_CACHE}:/root/.cache/uv:rw",
            "--env-file", str(env_file),
            IMAGE_TAG,
            "bash", "-c", container_cmd,
        ]

        logger.info("docker: starting %s (image=%s, work_dir=%s)",
                    container_name, IMAGE_TAG, host_work_dir)
        t0 = time.monotonic()
        # NOTE: we let docker run block here. timeout_s is the agent's wall
        # budget; docker run timeout = timeout_s + 120 (allow for uv sync).
        proc = await asyncio.create_subprocess_exec(
            *docker_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_s + 120,
            )
        except asyncio.TimeoutError:
            logger.warning("docker: container %s timed out, killing", container_name)
            subprocess.run(["docker", "rm", "-f", container_name],
                           capture_output=True)
            return AgentRunResult(
                status="timeout",
                duration_s=time.monotonic() - t0,
                error=f"docker wall budget {timeout_s + 120}s exceeded",
            )

        duration = time.monotonic() - t0
        rc = proc.returncode
        if rc != 0:
            tail_err = stderr.decode(errors="replace")[-1500:]
            logger.error("docker: container exited rc=%s\nstderr tail:\n%s", rc, tail_err)

        # ---- read result.json ----
        result_path = host_work_dir / "_result.json"
        if not result_path.exists():
            return AgentRunResult(
                status="failed",
                duration_s=duration,
                error=f"docker container exited rc={rc} but wrote no result.json; "
                      f"stderr tail: {stderr.decode(errors='replace')[-800:]}",
            )
        out = json.loads(result_path.read_text())
        return AgentRunResult(
            status=out.get("status", "failed"),
            error=out.get("error"),
            transcript_path=out.get("transcript_path"),
            stderr_path=out.get("stderr_path"),
            pid=out.get("pid"),
            exit_code=out.get("exit_code"),
            duration_s=out.get("duration_s") or duration,
        )

# =============================================================================
# helpers
# =============================================================================

def _config_to_kwargs(cfg) -> dict:
    """Same as VmExecutor — serialize dataclass fields to JSON-friendly dict."""
    out = {}
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if isinstance(val, (str, int, float, bool, type(None), list, dict, tuple)):
            out[f.name] = val
    return out


def _collect_env_vars(cfg) -> dict[str, str]:
    """Pull API keys out of config into env-file lines.

    Conventions match the in-process deployer's _patched_environ pattern:
    keys named ``<provider>_api_key`` → ``<PROVIDER>_API_KEY`` env var.
    """
    env: dict[str, str] = {}
    for attr, env_name in [
        ("openrouter_api_key", "OPENROUTER_API_KEY"),
        ("anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("openai_api_key", "OPENAI_API_KEY"),
        ("brave_api_key", "BRAVE_API_KEY"),
    ]:
        val = getattr(cfg, attr, None)
        if val:
            env[env_name] = val
    return env


def _image_present(tag: str) -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
    )
    return r.returncode == 0


# Register at import time
EXECUTORS["docker"] = DockerExecutor()
