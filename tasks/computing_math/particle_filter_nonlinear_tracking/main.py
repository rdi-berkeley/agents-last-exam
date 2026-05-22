"""AgentHLE task: computing_math/particle_filter_nonlinear_tracking."""

from __future__ import annotations

import json
import logging
import posixpath
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_particle_filter_outputs import (  # noqa: E402
    REQUIRED_OUTPUT_FILES,
    REQUIRED_REFERENCE_FILES,
    score_submission,
)

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "particle_filter_nonlinear_tracking"
VARIANT_NAME = "base"
CANONICAL_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIRS:
        raise ValueError(f"REMOTE_OUTPUT_DIR must normalize to one of {sorted(CANONICAL_OUTPUT_DIRS)}")
    return normalized


async def _read_bytes_map(session: cb.DesktopSession, path_map: dict[str, str]) -> tuple[dict[str, bytes], list[str]]:
    payloads: dict[str, bytes] = {}
    missing: list[str] = []
    for name, path in path_map.items():
        if not await session.exists(path):
            missing.append(name)
            continue
        payloads[name] = await session.read_bytes(path)
    return payloads, missing


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def output_dir_name(self) -> str:
        return _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def problem_spec_file(self) -> str:
        return f"{self.input_dir}/problem_spec.md"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def runtime_pyproject(self) -> str:
        return f"{self.runtime_env_dir}/pyproject.toml"

    @property
    def runtime_lockfile(self) -> str:
        return f"{self.runtime_env_dir}/uv.lock"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python.sh"

    @property
    def runtime_scratch_dir(self) -> str:
        return f"{self.task_dir}/output/.agent_runtime_env"

    @property
    def output_files(self) -> dict[str, str]:
        return {
            "pf_solver.py": f"{self.remote_output_dir}/pf_solver.py",
            "tier1_results.npz": f"{self.remote_output_dir}/tier1_results.npz",
            "tier2_results.npz": f"{self.remote_output_dir}/tier2_results.npz",
            "tier3_results.npz": f"{self.remote_output_dir}/tier3_results.npz",
            "results.json": f"{self.remote_output_dir}/results.json",
        }

    @property
    def reference_files(self) -> dict[str, str]:
        return {
            "tier2_results.npz": f"{self.reference_dir}/tier2_results.npz",
            "tier3_results.npz": f"{self.reference_dir}/tier3_results.npz",
        }

    @property
    def task_description(self) -> str:
        return f"""You are working on an Ubuntu coding task about particle filtering for nonlinear tracking.

Task directory:
- `{self.task_dir}`

Visible input files:
- Problem specification: `{self.problem_spec_file}`
- Python runtime manifest: `{self.runtime_pyproject}`
- Python lockfile: `{self.runtime_lockfile}`
- Canonical Python wrapper: `{self.python_wrapper}`

Recommended workflow:
1. Change into `{self.task_dir}`
2. Read `{self.problem_spec_file}` in full before coding
3. Use `"{self.python_wrapper}" your_script.py` or `"{self.python_wrapper}" -c "..."` to run Python with the staged NumPy/SciPy environment
4. Implement your solver and save it to `{self.remote_output_dir}/pf_solver.py`
5. Run the solver and write every required artifact under `{self.remote_output_dir}`

Required outputs under `{self.remote_output_dir}`:
- `pf_solver.py`
- `tier1_results.npz`
- `tier2_results.npz`
- `tier3_results.npz`
- `results.json`

Requirements:
- Use only Python + NumPy + SciPy. Do not use FilterPy, particles, pyro, Stan, or any state-estimation package.
- Follow the seed, model, and schema requirements in `{self.problem_spec_file}`.
- Benchmark acceptance thresholds used during evaluation:
  - Tier 1: `max_abs_error_mean < 0.20` and `max_rel_error_variance < 0.35`
  - Tier 2: `overall_rmse_pos < 1.5`, `overall_rmse_vel < 0.3`, and `mean_ess > 1000`
  - Tier 3: `overall_rmse_filter_pos < 3.0`, `overall_rmse_smoother_pos < overall_rmse_filter_pos`, and `mean_ess > 500`
- Tier 1 and Tier 2 are required for any passing score. Tier 3 upgrades the task from partial credit to full credit.
- If you cannot complete Tier 3, still write truthful Tier 1 / Tier 2 outputs and a truthful `results.json` rather than fabricating missing work.
- Do not modify files outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "output_dir_name": self.output_dir_name,
                "problem_spec_file": self.problem_spec_file,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lockfile": self.runtime_lockfile,
                "python_wrapper": self.python_wrapper,
                "runtime_scratch_dir": self.runtime_scratch_dir,
                "output_files": self.output_files,
                "reference_files": self.reference_files,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = TaskConfig(DOMAIN_NAME=DOMAIN_NAME, TASK_NAME=TASK_NAME, VARIANT_NAME=VARIANT_NAME)


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    output_payloads, missing_outputs = await _read_bytes_map(session, meta["output_files"])
    reference_payloads, missing_references = await _read_bytes_map(session, meta["reference_files"])

    try:
        report = score_submission(output_payloads, reference_payloads)
    except Exception as exc:
        logger.exception("Evaluation failed unexpectedly: %s", exc)
        return [0.0]

    if missing_outputs:
        logger.info("Missing output files during evaluation: %s", missing_outputs)
    if missing_references:
        logger.info("Missing reference files during evaluation: %s", missing_references)

    logger.info(
        "particle_filter_eval_report=%s",
        json.dumps(report.to_dict(), sort_keys=True),
    )
    return [report.score]
