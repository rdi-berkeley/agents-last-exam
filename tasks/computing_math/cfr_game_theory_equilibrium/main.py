"""Stage 2 task implementation for computing_math/cfr_game_theory_equilibrium."""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "cfr_game_theory_equilibrium"
VARIANT_NAME = "base"

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import score_outputs  # noqa: E402


@dataclass
class CFRGameTheoryConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    @property
    def problem_spec(self) -> str:
        return f"{self.input_dir}/problem_spec.md"

    @property
    def runtime_manifest(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def runtime_lock(self) -> str:
        return f"{self.input_dir}/runtime_env/uv.lock"

    @property
    def runtime_python(self) -> str:
        return f"{self.software_dir}/python_with_task_deps.sh"

    @property
    def runtime_bootstrap(self) -> str:
        return f"{self.software_dir}/bootstrap_runtime.sh"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/results.json"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/results.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Your Task
Implement equilibrium solvers for three progressively harder two-player zero-sum games and save one combined `results.json`.

## Visible Input
- Authoritative problem specification: `{self.problem_spec}`
- Solve-time runtime manifest: `{self.runtime_manifest}`
- Solve-time runtime lockfile: `{self.runtime_lock}`

## Runtime Entry Points
- Bootstrap the staged NumPy runtime with `{self.runtime_bootstrap}`
- Run Python with staged task dependencies using `{self.runtime_python}`

## What You Must Produce
Write exactly one file at `{self.output_file}`.

The JSON must contain top-level keys `tier1`, `tier2`, and `tier3`:
- `tier1`: matrix-game equilibrium value, row/column mixed strategies, and row/column support indices
- `tier2`: Kuhn poker CFR iterations, final exploitability, average strategy, game value estimate, and information-set count
- `tier3`: 4-rank Leduc MCCFR iterations, final exploitability, full average strategy map, information-set count, and exact game parameters

## Rules
- Read `{self.problem_spec}` first and follow it closely.
- Do not modify any files under `{self.input_dir}`.
- Do not write outside `{self.output_dir}`.
- Use only the staged local inputs and the software already available on this VM.
"""

    def to_metadata(self):
        metadata = super().to_metadata()
        metadata.update(
            {
                "problem_spec": self.problem_spec,
                "runtime_manifest": self.runtime_manifest,
                "runtime_lock": self.runtime_lock,
                "runtime_python": self.runtime_python,
                "runtime_bootstrap": self.runtime_bootstrap,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = CFRGameTheoryConfig()


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
    if not (await session.file_exists(meta["output_file"]) or await session.directory_exists(meta["output_file"])):
        logger.error("missing results.json at %s", meta["output_file"])
        return [0.0]

    try:
        agent_text = (await session.read_bytes(meta["output_file"])).decode("utf-8-sig")
    except Exception as exc:
        logger.error("failed to read evaluation payloads: %s", exc)
        return [0.0]

    result = score_outputs.score_submission_texts(agent_text)
    logger.info(
        "score=%.2f tier1=%s tier2=%s tier3=%s details=%s",
        result.score,
        result.tier1_passed,
        result.tier2_passed,
        result.tier3_passed,
        json.dumps(result.details, ensure_ascii=True),
    )
    return [float(result.score)]
