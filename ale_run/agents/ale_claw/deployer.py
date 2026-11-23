"""ALE integration for the standalone ALE-Claw harness."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from ale_claw import AleClawConfig, run
from ale_run.base_interface import AgentRunResult, BaseAgentDeployer, TrajectoryBuilder

from .transcript_to_trajectory import parse_transcripts_into

logger = logging.getLogger(__name__)


class AleClawDeployer(BaseAgentDeployer):
    default_executor: ClassVar[str] = "local"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"local", "docker"})

    async def install(self) -> None:
        Path(self.executor.work_dir).mkdir(parents=True, exist_ok=True)

    async def launch(self, prompt: str) -> AgentRunResult:
        sandbox = self.executor.sandbox
        result = await run(
            prompt, config=self.config, sandbox_url=sandbox.endpoint,
            os_type=sandbox.os, output_dir=self.executor.work_dir,
        )
        return AgentRunResult(
            status=result.status, duration_s=result.duration_s,
            error=result.error, transcript_path=result.transcript_path,
        )

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: AleClawConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Parse on-disk transcripts → ATIF Steps via the in-tree translator."""
        if not work_dir.exists():
            builder.add_step(
                source="system",
                message=f"ale-claw: work_dir missing {work_dir}",
                extra={"reason": "no_work_dir"},
            )
            return
        try:
            parse_transcripts_into(work_dir, builder)
        except Exception as exc:
            logger.exception("ale-claw: parse_artifacts failed")
            builder.add_step(
                source="system",
                message=f"transcript parse failed: {type(exc).__name__}: {exc}",
                extra={"reason": "parse_error"},
            )
        # Surface useful debug info on trajectory.extra
        builder.trajectory.extra.setdefault("ale_claw", {}).update({
            "work_dir": str(work_dir),
            "transcript_path": run_result.transcript_path,
            "run_status": run_result.status,
        })
