"""Container-side entry for :class:`DockerExecutor`.

Invoked as ``python -m ale_run.executors._docker_entry`` inside the
container. Reads ``/work/_spec.json`` (mounted from the host's
``work_dir``), reconstructs config + sandbox + a :class:`LocalExecutor`
in-container, runs the deployer, writes ``/work/_result.json``.

Mirrors :mod:`_sandbox_entry` but does NOT go through ``cua.python_exec``
— this is a normal module run, full import system available, so
``from X import Y`` is fine.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("docker_entry")


SPEC_PATH = Path("/work/_spec.json")
RESULT_PATH = Path("/work/_result.json")


async def _run() -> dict:
    spec = json.loads(SPEC_PATH.read_text())

    # Read the read-once secrets sidecar (api keys etc.) from the bind
    # mount and delete it immediately so it never persists in the host
    # log dir (work_dir is bind-mounted = host log dir for docker runs).
    # Fall back to a legacy in-spec env if present (older host writers).
    from ale_run.executors._secrets import read_and_delete_secrets
    env = read_and_delete_secrets(SPEC_PATH.parent) or (spec.get("env") or {})
    for k, v in env.items():
        os.environ[str(k)] = str(v)

    # Make scp'd ale_run importable (DockerExecutor mounts ale_run at /ale_run
    # via -v <host_repo>/ale_run:/ale_run/ale_run — convention below).
    src = spec["ale_src_root"]
    if src not in sys.path:
        sys.path.insert(0, src)

    # Install agent-declared Python deps before importing the deployer
    # so top-level imports (e.g. `import yaml`) don't crash.
    from ale_run.base_interface import BaseExecutor
    BaseExecutor.install_agent_deps(spec["deployer_module"])

    from ale_run.base_interface import SandboxHandle
    from ale_run.executors.local import LocalExecutor

    cfg_mod = importlib.import_module(spec["config_module"])
    dep_mod = importlib.import_module(spec["deployer_module"])
    cfg_cls = getattr(cfg_mod, spec["config_class"])
    dep_cls = getattr(dep_mod, spec["deployer_class"])

    cfg = cfg_cls(**spec["config_kwargs"])
    sandbox = SandboxHandle(**spec["sandbox_kwargs"])
    executor = LocalExecutor(
        config=cfg,
        work_dir=spec["work_dir"],
        sandbox=sandbox,
        env=env,
    )
    deployer = dep_cls(executor)

    logger.info(
        "docker_entry: %s.install (work_dir=%s)",
        dep_cls.__name__, executor.work_dir,
    )
    await deployer.install()
    logger.info("docker_entry: %s.launch", dep_cls.__name__)
    timeout_s = float(spec.get("timeout_s") or 1800.0)
    result = await asyncio.wait_for(
        deployer.launch(spec["prompt"]), timeout=timeout_s,
    )
    return {
        "ok": True,
        "status": result.status,
        "error": result.error,
        "transcript_path": result.transcript_path,
        "stderr_path": result.stderr_path,
        "pid": result.pid,
        "exit_code": result.exit_code,
        "duration_s": result.duration_s,
    }


def main() -> int:
    try:
        out = asyncio.run(_run())
    except Exception as exc:                                       # noqa: BLE001
        logger.exception("docker_entry crashed")
        out = {
            "ok": False,
            "status": (
                "timeout" if isinstance(exc, asyncio.TimeoutError) else "failed"
            ),
            "error": f"docker_entry: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    RESULT_PATH.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
