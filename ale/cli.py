"""CLI entry: ``python -m ale run experiments/foo.yaml``."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import ale
from .runner import Runner
from .runner.loader import load_experiment
from .runner.spec import RunUnit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ale",
        description="agent-last-exam: run benchmark experiments.",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_run = subparsers.add_parser("run", help="Run an experiment yaml.")
    p_run.add_argument("spec_path", type=Path, help="Path to experiment yaml.")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Show the run matrix without executing.")
    p_run.add_argument("--agent", action="append", dest="filter_agents",
                       metavar="ID", help="Filter: only run agents with these ids.")
    p_run.add_argument("--task", action="append", dest="filter_tasks",
                       metavar="PATH", help="Filter: only run these task paths.")
    p_run.add_argument("--verbose", "-v", action="store_true")

    p_list = subparsers.add_parser("list", help="List discoverable tasks.")
    p_list.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    if args.cmd == "run":
        return asyncio.run(_cmd_run(args))
    if args.cmd == "list":
        return _cmd_list(args)
    return 1


# =============================================================================
# Commands
# =============================================================================

async def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_experiment(args.spec_path)
    runner = Runner(spec)
    units = _filter_units(runner.enumerate_units(), args)

    if args.dry_run:
        print(f"experiment: {spec.name}")
        print(f"provider:   {spec.provider.kind}")
        print(f"output:     {runner.output_root}")
        print(f"concurrency: {spec.concurrency}")
        print(f"units ({len(units)}):")
        for u in units:
            print(f"  {u.agent_id:20s}  {u.task_path:40s}  v{u.variant_index}")
        return 0

    results = await runner.run(units)
    _print_results_table(results)
    bad = sum(1 for r in results if r.status not in ("completed",))
    return 0 if bad == 0 else 1


def _cmd_list(args: argparse.Namespace) -> int:
    envs = ale.list_envs()
    print(f"discoverable tasks ({len(envs)}):")
    for e in envs:
        print(f"  {e}")
    return 0


# =============================================================================
# Helpers
# =============================================================================

def _filter_units(units: list[RunUnit], args: argparse.Namespace) -> list[RunUnit]:
    if args.filter_agents:
        units = [u for u in units if u.agent_id in args.filter_agents]
    if args.filter_tasks:
        units = [u for u in units if u.task_path in args.filter_tasks]
    return units


def _print_results_table(results) -> None:
    if not results:
        print("(no results)")
        return
    print()
    print(f"{'agent':20s}  {'task':40s}  {'var':>3s}  {'status':10s}  {'score':>6s}  {'dur':>6s}")
    print("-" * 100)
    for r in results:
        score = f"{r.score:.2f}" if r.score is not None else "  -  "
        dur = f"{r.duration_s:.1f}s" if r.duration_s is not None else "   -  "
        print(
            f"{r.unit.agent_id:20s}  {r.unit.task_path:40s}  "
            f"{r.unit.variant_index:>3d}  {r.status:10s}  {score:>6s}  {dur:>6s}"
        )


if __name__ == "__main__":
    sys.exit(main())
