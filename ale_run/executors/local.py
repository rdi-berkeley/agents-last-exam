"""LocalExecutor — deployer runs in the framework's own Python process.

Substrate IS the host:

* ``run_deployer`` constructs the deployer with ``self`` and ``await``s
  ``install() + launch()`` directly. No bridging, no IPC.
* ``gather_dir`` is a no-op — work_dir already lives on host fs.
* ``download_range`` is a local seek+read.

Used by harness-style agents that drive the sandbox VM over the
network from the host (e.g. AleClaw runs the OpenClaw harness loop
in-process, reaching the eval VM via :attr:`SandboxHandle.endpoint`).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ..base_interface import BaseExecutor, GatherReport, RangeResult

if TYPE_CHECKING:
    from ..base_interface import AgentRunResult, BaseAgentDeployer

logger = logging.getLogger(__name__)


class LocalExecutor(BaseExecutor):
    """In-process substrate."""

    type: ClassVar[str] = "local"

    async def run_deployer(
        self,
        *,
        deployer_cls: type["BaseAgentDeployer"],
        prompt: str,
        timeout_s: float,
    ) -> "AgentRunResult":
        from ..base_interface import AgentRunResult

        deployer = deployer_cls(self)
        try:
            await deployer.install()
        except Exception as exc:
            import traceback
            return AgentRunResult(
                status="failed",
                error=f"install crashed: {type(exc).__name__}: {exc}\n"
                      f"{traceback.format_exc()}",
            )
        try:
            return await asyncio.wait_for(
                deployer.launch(prompt), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return AgentRunResult(
                status="timeout",
                error=f"local deployer wall-clock {timeout_s}s exceeded",
            )
        except Exception as exc:
            import traceback
            return AgentRunResult(
                status="failed",
                error=f"launch crashed: {type(exc).__name__}: {exc}\n"
                      f"{traceback.format_exc()}",
            )

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
