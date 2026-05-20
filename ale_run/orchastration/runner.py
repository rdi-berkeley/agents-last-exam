"""Runner: yaml-described experiment → concurrent run units.

Per-unit isolation:
    - One fresh ``ale.make(task_path)`` per unit (env binds task at ctor)
    - One fresh deployer instance per unit (configs are per-run state)

Concurrency is a single ``asyncio.Semaphore`` sized to ``spec.concurrency``
(matches simprun's one-knob model). Each unit holds the slot for its
full lifetime — VM acquire + agent run + post-launch fan-out + eval —
so the cap is effectively "max VMs alive at once". Size to
``min(GCP quota, LLM rate-limit / N)``.

Provider is shared across units — real providers (gcloud) acquire
a fresh VM per ``acquire()`` call, so concurrent acquires give concurrent
VMs.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable

from .factory import build_provider
from .spec import ExperimentSpec, RunUnit, UnitResult

logger = logging.getLogger(__name__)


class Runner:
    """Owns the provider; produces and executes run units."""

    def __init__(self, spec: ExperimentSpec):
        self._spec = spec
        self._provider = None  # built lazily — keeps --dry-run free of provider deps
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
        from .lifecycle import install_signal_handlers, run_one_unit

        install_signal_handlers()
        unit_list = list(units) if units is not None else self.enumerate_units()
        if not unit_list:
            logger.warning("Runner.run: no units to execute")
            return []

        if self._provider is None:
            self._provider = build_provider(self._spec.provider)
        self._output_root.mkdir(parents=True, exist_ok=True)

        n = self._spec.concurrency
        sem = asyncio.Semaphore(n)
        logger.info("runner: %d units, concurrency=%d", len(unit_list), n)

        async def _drive(u: RunUnit) -> UnitResult:
            logger.info("[%s] queued", u.slug)
            result = await run_one_unit(
                unit=u,
                provider=self._provider,
                output_root=self._output_root,
                artifacts=self._spec.artifacts,
                sem=sem,
                cleanup_mode=self._spec.cleanup_mode,
            )
            logger.info("[%s] done: status=%s score=%s duration=%.1fs",
                        u.slug, result.status, result.score, result.duration_s or 0)
            return result

        results = await asyncio.gather(
            *(_drive(u) for u in unit_list),
            return_exceptions=False,
        )
        return list(results)
