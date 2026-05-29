"""Sandbox-side entry. Invoked as a normal Python module:

    python -m ale_run.executors._sandbox_entry <spec_path>

inside the sandbox VM after :class:`SandboxExecutor` has scp'd the
``ale_run/`` source tree to the sandbox's ``ale_src_root`` and exported
that directory on ``PYTHONPATH``.

Reads ``<spec_path>`` (a JSON file the host wrote into the sandbox's
work_dir), reconstructs config + sandbox handle + a :class:`LocalExecutor`
in-sandbox, runs the deployer end-to-end, writes ``_result.json`` +
``_done.marker`` next to the spec for the host-side poller to read.

Symmetric with :mod:`_docker_entry`. No cua ``python_exec`` involved —
this is a normal Python process spawned via ``setsid`` (linux) or
``Start-Process`` (windows) by the host-side ``SandboxExecutor``.
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
logger = logging.getLogger("sandbox_entry")


def run(spec: dict, env: dict | None = None) -> dict:
    """Drive ``install() + launch()`` of the deployer. Return result dict.

    Pure data in / out. Caught exceptions become ``ok=False`` with a
    full traceback so the host poller can surface them.

    ``env`` holds the framework-supplied secrets (api keys, base URLs)
    read from the read-once ``_secrets.json`` sidecar; it is never part
    of ``spec`` (which is gathered to host logs and must stay keyless).
    """
    # Inject framework env vars so the deployer's spawned subprocess
    # inherits them. Fall back to a legacy in-spec env for forward/back
    # compatibility, but the writer no longer puts secrets in the spec.
    if env is None:
        env = spec.get("env") or {}
    for k, v in env.items():
        os.environ[str(k)] = str(v)

    try:
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
        # We are running INSIDE the sandbox: cua-server is co-located on
        # loopback at the image's declared port. The handle's ``endpoint`` was
        # the host-side URL (e.g. a socat sidecar port) which is NOT reachable
        # from in here, so point it at loopback. This is what the cua MCP bridge
        # (via executor.cua_bridge_url) and any in-sandbox run_command must use.
        sandbox.endpoint = f"http://127.0.0.1:{sandbox.cua_server_port}"
        executor = LocalExecutor(
            config=cfg,
            work_dir=spec["work_dir"],
            sandbox=sandbox,
            env=env,
        )
        deployer = dep_cls(executor)

        loop = asyncio.new_event_loop()
        try:
            logger.info(
                "sandbox_entry: %s.install (work_dir=%s)",
                dep_cls.__name__, executor.work_dir,
            )
            loop.run_until_complete(deployer.install())
            timeout_s = float(spec.get("timeout_s") or 1800.0)
            logger.info(
                "sandbox_entry: %s.launch (timeout_s=%.0f)",
                dep_cls.__name__, timeout_s,
            )
            result = loop.run_until_complete(
                asyncio.wait_for(
                    deployer.launch(spec["prompt"]),
                    timeout=timeout_s,
                )
            )
        finally:
            loop.close()

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
    except Exception as exc:                                       # noqa: BLE001
        logger.exception("sandbox_entry crashed")
        return {
            "ok": False,
            "status": (
                "timeout" if isinstance(exc, asyncio.TimeoutError) else "failed"
            ),
            "error": f"sandbox_entry: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    """Read ``spec_path`` from argv, run, write ``_result.json`` +
    ``_done.marker`` into the work_dir specified in the spec."""
    if len(sys.argv) < 2:
        print(
            "usage: python -m ale_run.executors._sandbox_entry <spec_path>",
            file=sys.stderr,
        )
        return 2
    spec_path = Path(sys.argv[1])
    try:
        spec = json.loads(spec_path.read_text())
    except Exception as e:                                          # noqa: BLE001
        print(f"sandbox_entry: cannot read spec {spec_path}: {e}", file=sys.stderr)
        return 2

    # Read the read-once secrets sidecar (api keys) and delete it
    # immediately so it never lingers to be gathered into host logs.
    from ale_run.executors._secrets import read_and_delete_secrets
    env = read_and_delete_secrets(spec_path.parent)

    out = run(spec, env)

    work_dir = Path(spec["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "_result.json").write_text(json.dumps(out, indent=2))
    # done.marker last — the host poller treats its presence as "result is
    # ready to read".
    (work_dir / "_done.marker").write_text("0\n" if out.get("ok") else "1\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
