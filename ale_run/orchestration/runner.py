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
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from .experiment_spec import ExperimentSpec, RunUnit, UnitResult
from .factory import EnvironmentRouter
from .reliability import (
    TrialObservation,
    load_trial_observations,
    precision_reached,
    summarize_trials,
    write_reliability_summary,
)

logger = logging.getLogger(__name__)


class Runner:
    """Owns the provider; produces and executes run units."""

    def __init__(self, spec: ExperimentSpec):
        self._spec = spec
        # Router resolves each unit's snapshot to its provider, building +
        # caching provider instances lazily (keeps --dry-run provider-free).
        self._router = EnvironmentRouter(spec.environment)
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
        *,
        max_attempts: int = 1,
    ) -> list[UnitResult]:
        """Run all units (or a filtered subset).

        With reliability disabled, behavior is unchanged: each unit is measured
        once and infrastructure failures may be retried up to max_attempts.

        With reliability enabled, every base unit is expanded into independent
        scientific trials. Each trial still has its own infrastructure-retry
        loop, so retries never masquerade as extra measurements.
        """
        from .lifecycle import install_signal_handlers, run_one_unit

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        install_signal_handlers()
        unit_list = list(units) if units is not None else self.enumerate_units()
        if not unit_list:
            logger.warning("Runner.run: no units to execute")
            return []

        self._output_root.mkdir(parents=True, exist_ok=True)

        n = self._spec.concurrency
        sem = asyncio.Semaphore(n)
        logger.info("runner: %d units, concurrency=%d", len(unit_list), n)

        async def _run_with_retries(u: RunUnit) -> UnitResult:
            for attempt in range(1, max_attempts + 1):
                logger.info(
                    "[%s] queued (attempt %d/%d)", u.slug, attempt, max_attempts,
                )
                result = await run_one_unit(
                    unit=u,
                    router=self._router,
                    output_root=self._output_root,
                    artifacts=self._spec.artifacts,
                    sem=sem,
                    cleanup_mode=self._spec.cleanup_mode,
                    prompt_suffix=self._spec.prompt_suffix,
                    wall_time_s=self._spec.wall_time_s,
                )
                logger.info(
                    "[%s] done: status=%s score=%s duration=%.1fs (attempt %d/%d)",
                    u.slug,
                    result.status,
                    result.score,
                    result.duration_s or 0,
                    attempt,
                    max_attempts,
                )
                if result.status != "failed" or attempt == max_attempts:
                    return result
                logger.warning(
                    "[%s] failed on attempt %d/%d; requeued",
                    u.slug,
                    attempt,
                    max_attempts,
                )

            raise AssertionError("unreachable")

        reliability = self._spec.reliability
        if not reliability.enabled:
            results = await asyncio.gather(
                *(_run_with_retries(u) for u in unit_list),
                return_exceptions=False,
            )
            return list(results)

        async def _drive_reliable(base_unit: RunUnit) -> list[UnitResult]:
            observations = (
                load_trial_observations(self._output_root, base_unit)
                if self._spec.auto_resume
                else []
            )
            by_index = {item.trial_index: item for item in observations}
            executed: list[UnitResult] = []
            stop_reason = "max_trials"

            for trial_index in range(reliability.max_trials):
                current = sorted(
                    by_index.values(),
                    key=lambda item: item.trial_index,
                )
                terminal_count = sum(
                    item.status in {"completed", "timeout"}
                    for item in current
                )
                if terminal_count >= reliability.min_trials:
                    summary = summarize_trials(
                        current,
                        confidence=reliability.confidence,
                        pass_threshold=reliability.pass_threshold,
                        pass_k=reliability.pass_k,
                    )
                    if precision_reached(summary, reliability):
                        stop_reason = "target_precision"
                        break

                previous = by_index.get(trial_index)
                if previous is not None and previous.status in {"completed", "timeout"}:
                    continue

                trial_unit = replace(base_unit, trial_index=trial_index)
                result = await _run_with_retries(trial_unit)
                executed.append(result)

                if result.status in {"completed", "timeout"}:
                    by_index[trial_index] = TrialObservation(
                        trial_index=trial_index,
                        status=result.status,
                        score=result.score,
                        run_dir=str(result.run_dir) if result.run_dir is not None else None,
                    )
                    continue

                stop_reason = "infrastructure_failure"
                break

            final_observations = sorted(
                by_index.values(),
                key=lambda item: item.trial_index,
            )
            if stop_reason == "max_trials" and final_observations:
                final_summary = summarize_trials(
                    final_observations,
                    confidence=reliability.confidence,
                    pass_threshold=reliability.pass_threshold,
                    pass_k=reliability.pass_k,
                )
                if precision_reached(final_summary, reliability):
                    stop_reason = "target_precision"

            write_reliability_summary(
                output_root=self._output_root,
                unit=base_unit,
                spec=reliability,
                observations=final_observations,
                stop_reason=stop_reason,
            )
            logger.info(
                "[%s] reliability: %d terminal trial(s), stop=%s",
                base_unit.slug,
                len(
                    [
                        item
                        for item in final_observations
                        if item.status in {"completed", "timeout"}
                    ]
                ),
                stop_reason,
            )
            return executed

        grouped_results = await asyncio.gather(
            *(_drive_reliable(u) for u in unit_list),
            return_exceptions=False,
        )
        return [result for group in grouped_results for result in group]
