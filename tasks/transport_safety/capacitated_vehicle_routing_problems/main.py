"""AgentHLE task: transport_safety/capacitated_vehicle_routing_problems."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.transport_safety.capacitated_vehicle_routing_problems.scripts.score_outputs import (
    score_solution_dir,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

DOMAIN_NAME = "transport_safety"
TASK_NAME = "capacitated_vehicle_routing_problems"
VARIANT_NAME = "base"
SELECTED_INSTANCES = ("M-n101-k10", "X-n101-k25", "X-n106-k14")


@dataclass
class CVRPTaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    def __init__(self, *, REMOTE_OUTPUT_DIR: str = ""):
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=REMOTE_OUTPUT_DIR or os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def problem_spec(self) -> str:
        return f"{self.input_dir}/problem_spec.md"

    @property
    def instances_dir(self) -> str:
        return f"{self.input_dir}/instances"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_cvrp_env.sh"

    @property
    def solutions_dir(self) -> str:
        return f"{self.remote_output_dir}/solutions"

    @property
    def task_description(self) -> str:
        required = "\n".join(f"- `{self.solutions_dir}/{stem}.sol`" for stem in SELECTED_INSTANCES)
        return f"""\
You are working on a Linux VM.

## Your Task
Build a reproducible CVRP-solving workflow and generate VRPLIB-format solutions for exactly three selected instances.

## Visible Inputs
- Problem specification: `{self.problem_spec}`
- Instance directory: `{self.instances_dir}/`
- Runtime manifest: `{self.runtime_env_dir}/pyproject.toml`
- Runtime lockfile: `{self.runtime_env_dir}/uv.lock`
- Python wrapper: `{self.python_wrapper}`

## What You Must Produce
Write these required files:
{required}

You may also write helper scripts and logs anywhere under `{self.remote_output_dir}`.

## Rules
- Read `{self.problem_spec}` first.
- Do not modify files under `{self.input_dir}`.
- Do not write outside `{self.remote_output_dir}`.
- Use only the staged local inputs and software already available on this VM.
"""


config = CVRPTaskConfig()


async def _download_remote_tree(
    session: cb.DesktopSession, remote_root: str, local_root: Path
) -> None:
    entries = await session.list_dir(remote_root)
    local_root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            name = entry.get("name")
        else:
            name = getattr(entry, "name", None)
        if not name or name in {".", ".."}:
            continue
        remote_path = f"{remote_root}/{name}"
        local_path = local_root / name
        if await session.exists(remote_path):
            try:
                children = await session.list_dir(remote_path)
            except Exception:
                children = None
            if children is not None:
                await _download_remote_tree(session, remote_path, local_path)
                continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(await session.read_bytes(remote_path))


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    with tempfile.TemporaryDirectory(prefix="agenthle_cvrp_eval_") as temp_dir:
        temp_root = Path(temp_dir)
        local_input_instances = temp_root / "input_instances"
        local_reference = temp_root / "reference"
        local_solutions = temp_root / "solutions"

        await _download_remote_tree(
            session, f'{meta["input_dir"]}/instances', local_input_instances
        )
        await _download_remote_tree(session, meta["reference_dir"], local_reference)
        if await session.exists(f'{meta["remote_output_dir"]}/solutions'):
            await _download_remote_tree(
                session, f'{meta["remote_output_dir"]}/solutions', local_solutions
            )

        result = score_solution_dir(local_solutions, local_input_instances, local_reference)

    for row in result.per_instance:
        logger.info(
            "instance=%s exists=%s feasible=%s gap=%s passed=%s details=%s",
            row.name,
            row.exists,
            row.feasible,
            row.gap,
            row.passed,
            row.details,
        )
    logger.info("score=%s passed_count=%s", result.score, result.passed_count)
    return [float(result.score)]
