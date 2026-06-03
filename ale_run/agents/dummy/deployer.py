"""DummyDeployer — a no-LLM smoke agent for the whole pipeline.

Purpose: exercise orchestration + provider/sandbox provisioning + task-data
staging + the output pull + the scorer across all 147 tasks WITHOUT spending
model tokens.

Per unit it:

  1. Scans the rendered task prompt for path tokens (``pathscan.scan``).
  2. Connects to the eval VM over cua-server (same ``RemoteDesktopSession``
     path AleClaw uses, ``executor: local``).
  3. Checks every ``input``-bearing path for existence on the VM — proving
     task data was staged.
  4. For each detected ``output`` directory, derives the task's GCS prefix
     (from ``sandbox.task_data_root``) and pulls ``output_test_pos`` straight
     into that dir with ``gsutil rsync`` ON the VM — simulating an agent that
     produced the correct answer, so the scorer should reward it.

Best-effort by design: a task with no output path, or one whose
``output_test_pos`` isn't on GCS, is recorded in ``dummy_report.json`` and
skipped — it does not fail the run or mask the rest of the sweep.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentDeployer,
    TrajectoryBuilder,
)
# Reuse the exact gsutil invocation the framework's own staging uses (handles
# the requester-pays ``-u <project>`` flag + linux/windows command shapes).
from ale_run.environments.task_data.gsbucket import _gsutil, _rsync_cmd

from . import pathscan
from .config import DummyConfig

logger = logging.getLogger(__name__)

_REPORT_NAME = "dummy_report.json"


class DummyDeployer(BaseAgentDeployer):
    """No-LLM smoke agent. Runs on the host, drives the eval VM."""

    default_executor: ClassVar[str] = "local"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"local", "docker"})
    hot_artifacts: ClassVar[tuple[str, ...]] = (_REPORT_NAME,)

    @property
    def version(self) -> str | None:
        return "dummy-0.2.0"

    # =========================================================================
    # install — just ensure work_dir + import the VM session class
    # =========================================================================

    async def install(self) -> None:
        from cua_bench.computers.remote import RemoteDesktopSession  # noqa: F401

        Path(self.executor.work_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "dummy: install ok (work_dir=%s, executor=%s, os=%s)",
            self.executor.work_dir,
            self.executor.type,
            getattr(self.executor.sandbox, "os", "?"),
        )

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        from cua_bench.computers.remote import RemoteDesktopSession

        cfg: DummyConfig = self.config  # type: ignore[assignment]
        work_dir = Path(self.executor.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        scan = pathscan.scan(prompt)
        sb = self.executor.sandbox
        report: dict[str, Any] = {
            "os": getattr(sb, "os", None),
            "task_data_root": getattr(sb, "task_data_root", None),
            "scanned": {
                "n_all_paths": len(scan.all_paths),
                "n_input_paths": len(scan.input_paths),
                "n_output_paths": len(scan.output_paths),
                "n_output_dirs": len(scan.output_dirs),
            },
            "input_paths": scan.input_paths,
            "output_paths": scan.output_paths,
            "output_dirs": scan.output_dirs,
            "input_checks": [],
            "missing_inputs": [],
            "output_pulls": [],
            "skipped": False,
            "skip_reason": None,
            "errors": [],
        }

        # ---- skip: prompt named no input AND no output paths ----
        if not scan.has_input and not scan.has_output:
            report["skipped"] = True
            report["skip_reason"] = "no input/output path tokens found in prompt"
            self._write_report(work_dir, report)
            logger.warning("dummy: SKIP — no input/output paths in prompt")
            return self._result("completed", t0, work_dir)

        # ---- connect to the eval VM ----
        session = RemoteDesktopSession(
            api_url=sb.endpoint,
            os_type=sb.os,
            ephemeral=False,          # env lifecycle owned by the framework
            headless=True,
        )
        ready = await session.wait_until_ready(timeout=cfg.connect_timeout_s)
        if not ready:
            report["errors"].append("VM cua-server not responsive within timeout")
            self._write_report(work_dir, report)
            return self._result("failed", t0, work_dir, error="VM not reachable")

        status = "completed"
        error: str | None = None

        # ---- 1. input existence checks (data-staging probe) ----
        for p in scan.input_paths:
            try:
                exists = await self._path_exists(session, p)
            except Exception as exc:  # noqa: BLE001
                exists = False
                report["errors"].append(f"exists({p!r}) raised: {type(exc).__name__}: {exc}")
            report["input_checks"].append({"path": p, "exists": bool(exists)})
            if not exists:
                report["missing_inputs"].append(p)
        if report["missing_inputs"]:
            logger.warning("dummy: %d missing input(s)", len(report["missing_inputs"]))
            if cfg.fail_on_missing_input:
                status = "failed"
                error = f"{len(report['missing_inputs'])} input path(s) missing on VM"

        # ---- 2. pull output_test_pos from GCS into each output dir ----
        if not scan.output_dirs:
            report["skip_reason"] = "input present but no output directory in prompt"
            logger.info("dummy: no output dir in prompt — nothing to pull")
        else:
            for out_dir in scan.output_dirs:
                entry = await self._pull_pos(session, sb, cfg, out_dir)
                report["output_pulls"].append(entry)
                if entry.get("error"):
                    report["errors"].append(entry["error"])

        self._write_report(work_dir, report)
        n_pulled = sum(1 for e in report["output_pulls"] if e.get("pulled"))
        logger.info(
            "dummy: done status=%s inputs=%d missing=%d out_dirs=%d pulled=%d",
            status, len(scan.input_paths), len(report["missing_inputs"]),
            len(scan.output_dirs), n_pulled,
        )
        return self._result(status, t0, work_dir, error=error)

    # =========================================================================
    # GCS pull (one output dir)
    # =========================================================================

    async def _pull_pos(
        self, session: Any, sb: Any, cfg: DummyConfig, out_dir: str,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "output_dir": out_dir,
            "candidates": [],
            "source": None,
            "pulled": False,
            "marker_written": False,
            "error": None,
        }
        candidates = pathscan.gcs_pos_candidates(
            out_dir, getattr(sb, "task_data_root", None),
            bucket=cfg.data_bucket, pos_name=cfg.pos_subdir,
        )
        entry["candidates"] = candidates

        # Always ensure the output dir exists on the VM.
        try:
            await self._mkdir(session, sb, out_dir)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"mkdir({out_dir!r}) failed: {type(exc).__name__}: {exc}"
            return entry

        src = None
        for cand in candidates:
            if await self._gcs_exists(session, sb, cand):
                src = cand
                break

        if src is None:
            entry["error"] = (
                f"no {cfg.pos_subdir} on GCS for {out_dir} "
                f"(tried: {candidates or 'none'})"
            )
            logger.warning("dummy: %s", entry["error"])
            if cfg.write_marker_when_no_pos:
                entry["marker_written"] = await self._write_marker(session, cfg, out_dir)
            return entry

        entry["source"] = src
        cmd = _rsync_cmd(sb, src, out_dir)
        try:
            r = await session.run_command(cmd, check=False)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"rsync raised: {type(exc).__name__}: {exc}"
            return entry
        if r.get("return_code", 1) != 0:
            entry["error"] = (
                f"rsync {src} -> {out_dir} rc={r.get('return_code')}: "
                f"{(r.get('stderr') or '')[:300]}"
            )
            logger.error("dummy: %s", entry["error"])
            return entry

        entry["pulled"] = True
        logger.info("dummy: pulled %s -> %s", src, out_dir)
        return entry

    @staticmethod
    async def _path_exists(session: Any, path: str) -> bool:
        """Installed cua_bench has no combined ``exists`` — OR the two probes."""
        if await session.file_exists(path):
            return True
        return await session.directory_exists(path)

    @staticmethod
    async def _mkdir(session: Any, sb: Any, path: str) -> None:
        """Installed cua_bench has no ``makedirs`` — mkdir via run_command."""
        if sb.is_linux:
            cmd = f"mkdir -p '{path}'"
        else:
            cmd = (
                f"powershell -NoProfile -Command "
                f"\"New-Item -ItemType Directory -Force -Path '{path}' | Out-Null\""
            )
        await session.run_command(cmd, check=False)

    async def _gcs_exists(self, session: Any, sb: Any, url: str) -> bool:
        gsutil = _gsutil(sb)
        if sb.is_linux:
            cmd = f"{gsutil} ls '{url}' >/dev/null 2>&1"
        else:
            cmd = (
                f"powershell -NoProfile -Command "
                f"\"{gsutil} ls '{url}' *> $null; exit $LASTEXITCODE\""
            )
        try:
            r = await session.run_command(cmd, check=False)
        except Exception:  # noqa: BLE001
            return False
        return r.get("return_code", 1) == 0

    @staticmethod
    async def _write_marker(session: Any, cfg: DummyConfig, out_dir: str) -> bool:
        path = pathscan.join_path(out_dir, cfg.marker_filename)
        payload = json.dumps(
            {"agent": "dummy", "note": "no output_test_pos on GCS; marker only"},
            indent=2,
        )
        try:
            await session.write_file(path, payload)
            return True
        except Exception:  # noqa: BLE001
            return False

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: DummyConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        report_path = work_dir / _REPORT_NAME
        if not report_path.exists():
            builder.add_step(
                source="system",
                message=f"dummy: report missing at {report_path}",
                extra={"reason": "no_report", "run_status": run_result.status},
            )
            return
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            builder.add_step(
                source="system",
                message=f"dummy: report unreadable: {type(exc).__name__}: {exc}",
                extra={"reason": "bad_report"},
            )
            return

        if report.get("skipped"):
            builder.add_step(
                source="system",
                message=f"dummy: SKIPPED — {report.get('skip_reason')}",
                extra={"reason": "skipped"},
            )
        else:
            n_in = len(report.get("input_checks", []))
            n_missing = len(report.get("missing_inputs", []))
            pulls = report.get("output_pulls", [])
            n_pulled = sum(1 for e in pulls if e.get("pulled"))
            builder.add_step(
                source="system",
                message=(
                    f"dummy: checked {n_in} input path(s) ({n_missing} missing); "
                    f"pulled output_test_pos into {n_pulled}/{len(pulls)} output dir(s)."
                ),
                extra={
                    "missing_inputs": report.get("missing_inputs", []),
                    "output_pulls": pulls,
                    "errors": report.get("errors", []),
                },
            )

        builder.trajectory.extra.setdefault("dummy", {}).update({
            "work_dir": str(work_dir),
            "run_status": run_result.status,
            "report": report,
        })

    # =========================================================================
    # helpers
    # =========================================================================

    def _result(
        self, status: str, t0: float, work_dir: Path, *, error: str | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            status=status,
            duration_s=time.monotonic() - t0,
            transcript_path=str(work_dir / _REPORT_NAME),
            error=error,
        )

    @staticmethod
    def _write_report(work_dir: Path, report: dict[str, Any]) -> None:
        (work_dir / _REPORT_NAME).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
