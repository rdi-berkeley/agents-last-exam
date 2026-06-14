"""CarDeployer — Common Agent Runtime (CAR) native deployer.

Runs out-of-sandbox on the host (``local``) or in a docker container
(``docker``). CAR is the system under test: the deployer launches the
``car run-task`` headless runner, hands it stdio MCP bridges that reach the eval
VM's cua-server, and CAR drives its own propose -> validate -> execute loop.

The deployer's surface:

  install():          check the ``car`` binary, an API key env var, and a pinned
                      model; mkdir work_dir.
  launch(prompt):     install the vm (+ optional cua) MCP bridges, write the
                      runner's MCP config, subprocess ``car run-task``, and return
                      an AgentRunResult. The transcript lands in work_dir for
                      parse_artifacts.
  parse_artifacts():  read the CAR transcript JSONL -> ATIF Steps via the in-tree
                      translator.

Why a transcript and not CAR's eventlog: CAR's engine eventlog is metadata-only
(it records action ids + durations, not tool names / parameters / outputs), so
the runner emits a separate content-rich transcript that this deployer converts.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentDeployer,
    TrajectoryBuilder,
)

from ale_run.agents._bootstrap import (
    cua_bridge_env,
    ensure_cua_mcp_server_at,
    ensure_node_npm,
    ensure_vm_mcp_server,
    vm_bridge_env,
)

from .config import CarConfig
from .transcript_to_trajectory import parse_transcript_into

logger = logging.getLogger(__name__)


class CarDeployer(BaseAgentDeployer):
    """Common Agent Runtime deployer. Runs on host or in a docker container."""

    default_executor: ClassVar[str] = "local"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"local", "docker"})

    # At least one of these env vars must be set or ``install`` raises.
    _api_key_alternatives: ClassVar[tuple[str, ...]] = (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
    )

    # =========================================================================
    # install
    # =========================================================================

    async def install(self) -> None:
        cfg: CarConfig = self.config  # type: ignore[assignment]
        if shutil.which(cfg.car_bin) is None and not Path(cfg.car_bin).is_file():
            raise RuntimeError(
                f"{type(self).__name__}: car binary {cfg.car_bin!r} not found on "
                "PATH — install the CAR CLI or set config.car_bin to its path."
            )
        if not cfg.model:
            raise RuntimeError(
                f"{type(self).__name__}: config.model is unset — pin a CAR model "
                "id (e.g. an Anthropic/OpenAI catalog id) for the benchmark run."
            )
        if not any(os.environ.get(k) for k in self._api_key_alternatives):
            raise RuntimeError(
                f"{type(self).__name__}: no LLM API key in env — set one of "
                f"{', '.join(self._api_key_alternatives)}"
            )
        Path(self.executor.work_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "%s: install ok (model=%s, work_dir=%s, executor=%s)",
            type(self).__name__, cfg.model, self.executor.work_dir, self.executor.type,
        )

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: CarConfig = self.config  # type: ignore[assignment]
        work_dir = Path(self.executor.work_dir)
        car_dir = work_dir / "car"
        car_dir.mkdir(parents=True, exist_ok=True)

        goal_path = car_dir / "goal.txt"
        mcp_config_path = car_dir / "mcp.json"
        transcript_path = car_dir / "transcript.jsonl"
        eventlog_path = car_dir / "eventlog.jsonl"
        goal_path.write_text(prompt, encoding="utf-8")

        # ---- 1. Install the stdio MCP bridges that reach the eval cua-server ----
        # Native (host) agents run the bridge on the host; it points at the host
        # endpoint the executor exposes (cua_bridge_url == sb.endpoint here).
        node_path, _ = await ensure_node_npm()
        servers: list[dict[str, Any]] = []

        vm_bridge_dir = await ensure_vm_mcp_server(str(car_dir / "mcp" / "vm"))
        servers.append({
            "name": "vm",
            "command": node_path,
            "args": [os.path.join(vm_bridge_dir, "src", "index.js")],
            "env": vm_bridge_env(self.executor),
        })
        if cfg.gui:
            cua_bridge_dir = await ensure_cua_mcp_server_at(str(car_dir / "mcp" / "cua"))
            servers.append({
                "name": "cua",
                "command": node_path,
                "args": [os.path.join(cua_bridge_dir, "src", "index.js")],
                "env": cua_bridge_env(self.executor),
            })

        import json
        mcp_config_path.write_text(json.dumps({"servers": servers}, indent=2), encoding="utf-8")
        logger.info("car: launch — work_dir=%s servers=%s", work_dir, [s["name"] for s in servers])

        # ---- 2. Subprocess the headless runner ----
        argv = [
            cfg.car_bin, "run-task",
            "--goal-file", str(goal_path),
            "--mcp-config", str(mcp_config_path),
            "--transcript", str(transcript_path),
            "--max-turns", str(cfg.max_turns),
            "--model", str(cfg.model),
        ]
        if cfg.eventlog:
            argv += ["--eventlog", str(eventlog_path)]

        env = {**os.environ, **cfg.extra_env}
        t0 = time.monotonic()
        # The episode wall budget is orchestration-owned: the executor wraps
        # launch() in asyncio.wait_for(timeout=timeout_s). On cancellation we
        # reap the child and re-raise so the framework records a timeout.
        proc = await asyncio.create_subprocess_exec(
            *argv, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await proc.communicate()
        except asyncio.CancelledError:
            logger.info("car: launch cancelled (wall budget) — terminating runner")
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
            raise

        duration_s = time.monotonic() - t0
        if stderr:
            logger.info("car run-task stderr:\n%s", stderr.decode("utf-8", "replace")[-4000:])

        # run-task exits 0 on a done signal, 1 on max-turns / error. A non-zero
        # exit is not necessarily a hard failure (max-turns still produced work),
        # so map by whether a transcript exists; parse_artifacts reads the detail.
        if transcript_path.exists():
            status = "completed" if proc.returncode == 0 else "failed"
            error = None if proc.returncode == 0 else "runner exited non-zero (see transcript run_end)"
        else:
            status = "failed"
            error = f"car run-task produced no transcript (exit={proc.returncode})"

        return AgentRunResult(
            status=status,
            duration_s=duration_s,
            transcript_path=str(transcript_path) if transcript_path.exists() else None,
            exit_code=proc.returncode,
            error=error,
        )

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: CarConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        if not work_dir.exists():
            builder.add_step(
                source="system",
                message=f"car: work_dir missing {work_dir}",
                extra={"reason": "no_work_dir"},
            )
            return
        try:
            parse_transcript_into(work_dir, builder)
        except Exception as exc:  # noqa: BLE001
            logger.exception("car: parse_artifacts failed")
            builder.add_step(
                source="system",
                message=f"transcript parse failed: {type(exc).__name__}: {exc}",
                extra={"reason": "parse_error"},
            )
        builder.trajectory.extra.setdefault("car", {}).update({
            "work_dir": str(work_dir),
            "transcript_path": run_result.transcript_path,
            "run_status": run_result.status,
            "exit_code": run_result.exit_code,
        })
