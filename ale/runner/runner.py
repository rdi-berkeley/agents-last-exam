"""Runner: yaml-described experiment → concurrent run units.

Sequential is just ``concurrency=1``. Per-unit isolation is achieved by:
    - One fresh ``ale.make(task_path)`` per unit (env binds task at ctor)
    - One fresh deployer instance per unit (configs are per-run state)
    - asyncio.Semaphore bounds in-flight unit count

Provider is shared across units in the batch — that's the point: real
providers (gcs_direct) acquire a fresh VM per ``acquire()`` call, so
concurrent acquires give concurrent VMs.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable

from .factory import build_provider
from .lifecycle import install_signal_handlers, run_one_unit
from .spec import ExperimentSpec, RunUnit, UnitResult

logger = logging.getLogger(__name__)


class Runner:
    """Owns the provider; produces and executes run units."""

    def __init__(self, spec: ExperimentSpec):
        self._spec = spec
        self._provider = build_provider(spec.provider)
        self._output_root = Path(spec.output.root) / spec.name

    @property
    def spec(self) -> ExperimentSpec:
        return self._spec

    @property
    def output_root(self) -> Path:
        return self._output_root

    # ---- enumeration ----

    def enumerate_units(self) -> list[RunUnit]:
        """Cartesian product of agents × tasks × variants."""
        out: list[RunUnit] = []
        for agent in self._spec.agents:
            for task in self._spec.tasks:
                for vi in task.variants:
                    out.append(RunUnit(
                        agent_id=agent.id,
                        agent_spec=agent,
                        task_path=task.path,
                        variant_index=vi,
                    ))
        return out

    # ---- execution ----

    async def run(
        self,
        units: Iterable[RunUnit] | None = None,
    ) -> list[UnitResult]:
        """Run all units (or a filtered subset). Returns ``list[UnitResult]``.

        No aggregation, no summary — caller does whatever rollup it wants.
        """
        install_signal_handlers()
        unit_list = list(units) if units is not None else self.enumerate_units()
        if not unit_list:
            logger.warning("Runner.run: no units to execute")
            return []

        self._output_root.mkdir(parents=True, exist_ok=True)

        sem = asyncio.Semaphore(self._spec.concurrency)
        async def _bounded(u: RunUnit) -> UnitResult:
            async with sem:
                logger.info("[%s] starting", u.slug)
                result = await run_one_unit(
                    unit=u,
                    provider=self._provider,
                    output_root=self._output_root,
                    artifacts=self._spec.artifacts,
                )
                logger.info("[%s] done: status=%s score=%s duration=%.1fs",
                            u.slug, result.status, result.score, result.duration_s or 0)
                return result

        results = await asyncio.gather(
            *(_bounded(u) for u in unit_list),
            return_exceptions=False,
        )
        return list(results)
