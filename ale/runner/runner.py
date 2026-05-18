"""Runner: yaml-described experiment → concurrent run units.

Per-unit isolation:
    - One fresh ``ale.make(task_path)`` per unit (env binds task at ctor)
    - One fresh deployer instance per unit (configs are per-run state)

Concurrency uses TWO asyncio.Semaphores (a la simprun engine.py:171-176)
to avoid one phase starving the other:

    - ``provision_sem`` (default = run_concurrency) bounds in-flight VM
      acquires (``provider.acquire``). Sized to GCP quota / gcloud client
      throughput.
    - ``run_sem`` (= ``run_concurrency``) bounds in-flight agent runs
      (launch + post-launch fan-out + eval). Sized to API rate-limit.

Each unit holds ``provision_sem`` ONLY during reset_async, releases it,
then waits on ``run_sem`` for the active phase. So slow provisioning
doesn't pin a run-slot, and fully-occupied run-slots don't pin a
provision-slot. A unit may sit with a ready VM idle for a beat while
waiting on ``run_sem`` — the trade-off vs leaving the API pipeline starved
is favorable when provision is the slow side (GCP cold boot ~minutes).

Provider is shared across units — real providers (gcs_direct) acquire
a fresh VM per ``acquire()`` call, so concurrent acquires give concurrent
VMs.
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

        run_n = self._spec.run_concurrency
        prov_n = self._spec.provision_concurrency or run_n
        run_sem = asyncio.Semaphore(run_n)
        provision_sem = asyncio.Semaphore(prov_n)
        logger.info(
            "runner: %d units, provision_concurrency=%d, run_concurrency=%d",
            len(unit_list), prov_n, run_n,
        )

        async def _drive(u: RunUnit) -> UnitResult:
            logger.info("[%s] queued", u.slug)
            result = await run_one_unit(
                unit=u,
                provider=self._provider,
                output_root=self._output_root,
                artifacts=self._spec.artifacts,
                eval_timeout_s=self._spec.eval_timeout_s,
                provision_sem=provision_sem,
                run_sem=run_sem,
            )
            logger.info("[%s] done: status=%s score=%s duration=%.1fs",
                        u.slug, result.status, result.score, result.duration_s or 0)
            return result

        results = await asyncio.gather(
            *(_drive(u) for u in unit_list),
            return_exceptions=False,
        )
        return list(results)
