"""LocalExecutor — deployer runs in a supervised host subprocess.

Substrate IS the host:

* ``run_deployer`` supervises ``install() + launch()`` in a host worker.
* ``gather_dir`` is a no-op — work_dir already lives on host fs.
* ``download_range`` is a local seek+read.

Used by harness-style agents that drive the sandbox VM over the
network from the host (e.g. AleClaw runs the OpenClaw harness loop
in-process, reaching the eval VM via :attr:`SandboxHandle.endpoint`).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ..base_interface import BaseExecutor, GatherReport, RangeResult
from ._secrets import SECRETS_FILE, config_to_kwargs_and_secrets, write_secrets

if TYPE_CHECKING:
    from ..base_interface import AgentRunResult, BaseAgentDeployer

logger = logging.getLogger(__name__)


@dataclass
class LocalExecutor(BaseExecutor):
    """Host substrate with per-run process containment."""

    type: ClassVar[str] = "local"
    _run_token: str = field(default_factory=lambda: uuid.uuid4().hex, init=False)
    _supervisor: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _stop_task: asyncio.Task | None = field(default=None, init=False, repr=False)

    async def run_deployer(
        self,
        *,
        deployer_cls: type["BaseAgentDeployer"],
        prompt: str,
        timeout_s: float,
    ) -> "AgentRunResult":
        from ..base_interface import AgentRunResult

        work_dir = Path(self.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        config_kwargs, secrets = config_to_kwargs_and_secrets(self.config)
        spec = {
            "deployer_module": deployer_cls.__module__,
            "deployer_class": deployer_cls.__name__,
            "config_module": self.config.__class__.__module__,
            "config_class": self.config.__class__.__name__,
            "config_kwargs": config_kwargs,
            "sandbox_kwargs": asdict(self.sandbox),
            "work_dir": str(work_dir),
            "prompt": prompt,
            "timeout_s": timeout_s,
            "run_token": self._run_token,
            "sandbox_local": False,
            "install_agent_deps": False,
        }
        for name in ("_result.json", "_worker_result.json", "_done.marker"):
            (work_dir / name).unlink(missing_ok=True)
        spec_path = work_dir / "_spec.json"
        spec_path.write_text(json.dumps(spec))
        write_secrets(work_dir, {**self.env, **secrets})
        process_env = dict(os.environ)
        process_env["PYTHONPATH"] = os.pathsep.join(str(Path(path).resolve()) for path in sys.path)
        started = time.monotonic()
        try:
            with (work_dir / "_entry.log").open("wb") as log:
                self._supervisor = subprocess.Popen(
                    [sys.executable, "-m", "ale_run.executors._sandbox_entry", str(spec_path)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=process_env,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            while self._supervisor.poll() is None:
                if time.monotonic() - started >= timeout_s:
                    return AgentRunResult(
                        status="timeout", duration_s=time.monotonic() - started,
                        error=f"local deployer wall-clock {timeout_s}s exceeded",
                    )
                await asyncio.sleep(0.05)
            out = json.loads((work_dir / "_result.json").read_text())
            return AgentRunResult(
                status=out.get("status", "failed"), error=out.get("error"),
                transcript_path=out.get("transcript_path"), stderr_path=out.get("stderr_path"),
                pid=out.get("pid"), exit_code=out.get("exit_code"),
                duration_s=out.get("duration_s") or time.monotonic() - started,
            )
        finally:
            try:
                await self.stop_deployer()
            finally:
                (work_dir / SECRETS_FILE).unlink(missing_ok=True)

    async def stop_deployer(self) -> None:
        if self._supervisor is None:
            return
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(
                asyncio.wait_for(self._stop_and_verify(), timeout=60),
            )
        cancelled = False
        while not self._stop_task.done():
            try:
                await asyncio.shield(self._stop_task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        try:
            self._stop_task.result()
        except Exception as exc:
            raise RuntimeError("local solver termination unconfirmed; reference blocked") from exc
        if cancelled:
            raise asyncio.CancelledError

    async def _stop_and_verify(self) -> None:
        work_dir = Path(self.work_dir)
        (work_dir / "_stop.request").write_text(self._run_token)
        while True:
            try:
                report = json.loads((work_dir / "_stopped.json").read_text())
            except (FileNotFoundError, ValueError):
                report = {}
            if report.get("run_token") == self._run_token:
                if report.get("stopped") is not True:
                    raise RuntimeError(f"local process cleanup failed: {report.get('error')}")
                while self._supervisor.poll() is None:
                    await asyncio.sleep(0.02)
                return
            if self._supervisor.poll() is not None:
                raise RuntimeError("local supervisor exited without termination confirmation")
            await asyncio.sleep(0.02)

    async def gather_dir(
        self, *, src: str, dst: Path,
    ) -> GatherReport:
        # work_dir IS on host; deployer wrote files there directly.
        # If src != dst the caller wants an actual copy — handle it.
        src_path = Path(src)
        if src_path == dst:
            return GatherReport(transport="local", files=0, bytes=0)
        if not src_path.exists():
            return GatherReport(
                transport="local", files=0, bytes=0,
                error=f"src not found: {src}",
            )

        def _copy() -> tuple[int, int, str | None]:
            import shutil
            files = 0
            total = 0
            try:
                dst.mkdir(parents=True, exist_ok=True)
                for entry in src_path.rglob("*"):
                    if entry.is_dir():
                        continue
                    rel = entry.relative_to(src_path)
                    target = dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(entry, target)
                    files += 1
                    total += target.stat().st_size
                return files, total, None
            except Exception as e:                  # noqa: BLE001
                return files, total, str(e)

        files, total, err = await asyncio.to_thread(_copy)
        return GatherReport(
            transport="local", files=files, bytes=total, error=err,
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
