"""AgentHLE task: business_finance/american_option_pricing_ls."""

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

from score_american_option_outputs import score_submission  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "american_option_pricing_ls"
VARIANT_NAME = "base"
CANONICAL_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIRS:
        raise ValueError(
            f"OUTPUT_SUBDIR must normalize to one of {sorted(CANONICAL_OUTPUT_DIRS)}"
        )
    return normalized


async def _read_bytes_map(
    session: cb.DesktopSession, path_map: dict[str, str]
) -> tuple[dict[str, bytes], list[str]]:
    payloads: dict[str, bytes] = {}
    missing: list[str] = []
    for name, path in path_map.items():
        if not (await session.file_exists(path) or await session.directory_exists(path)):
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
        return _canonical_output_dir_name(self.OUTPUT_SUBDIR)

    @property
    def output_dir(self) -> str:
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
            "results.json": f"{self.output_dir}/results.json",
            "exercise_boundary_tier2.npy": f"{self.output_dir}/exercise_boundary_tier2.npy",
        }

    @property
    def reference_files(self) -> dict[str, str]:
        return {
            "results.json": f"{self.reference_dir}/results.json",
            "exercise_boundary_tier2.npy": f"{self.reference_dir}/exercise_boundary_tier2.npy",
        }

    @property
    def task_description(self) -> str:
        return f"""You are working on an Ubuntu coding task about pricing American-style options with Monte Carlo and Longstaff-Schwartz regression.

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
3. Use `"{self.python_wrapper}" your_script.py` or `"{self.python_wrapper}" -c "..."` to run Python with the staged NumPy/SciPy runtime
4. Implement your solver under `{self.output_dir}`
5. Write every required artifact under `{self.output_dir}`

Required outputs under `{self.output_dir}`:
- `results.json`
- `exercise_boundary_tier2.npy`

Requirements:
- Use only Python + NumPy + SciPy. Do not use QuantLib, autograd, JAX, PyTorch, TensorFlow, or other financial / autodiff libraries.
- Use exact log-normal GBM simulation, not Euler discretization.
- Follow the fixed seeds, path counts, regression design, and output schema from `{self.problem_spec_file}`.
- Tier 1 and Tier 2 are required for any passing score; Tier 3 is required for full credit.
- If Tier 3 is incomplete, keep Tier 1 / Tier 2 outputs truthful rather than fabricating Tier 3 metrics.
- Do not modify files outside `{self.output_dir}`.
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

    logger.info("american_option_eval_report=%s", json.dumps(report.to_dict(), sort_keys=True))
    return [report.score]
