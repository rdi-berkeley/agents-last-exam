"""AgentHLE task: engineering/aerospace_low_thrust_trajectory."""

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

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tasks.engineering.aerospace_low_thrust_trajectory.scripts.score_outputs import (  # noqa: E402
    EvaluatorReferenceError,
    ScoreReport,
    score_submission,
    validate_reference,
)

logger = logging.getLogger(__name__)

TASK_NAME = "aerospace_low_thrust_trajectory"
DOMAIN_NAME = "engineering"
VARIANT_NAME = "base"
REQUIRED_OUTPUT_FILES = (
    "results.json",
    "tier2_trajectory.npy",
    "tier3_trajectory.npy",
    "tier3_control.npy",
)
CANONICAL_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIR_NAMES:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: output, output_test_pos, output_test_neg"
        )
    return normalized


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
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def runtime_manifest_file(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def python_entry_file(self) -> str:
        return f"{self.software_dir}/python"

    @property
    def uv_entry_file(self) -> str:
        return f"{self.software_dir}/uv"

    @property
    def runtime_env_python_file(self) -> str:
        return f"{self.task_dir}/.runtime_env/bin/python"

    @property
    def solve_output_dir(self) -> str:
        return f"{self.task_dir}/output"

    @property
    def task_description(self) -> str:
        return f"""You are working on a Linux VM.

Task root:
- `{self.task_dir}`

Visible files:
- Problem specification: `{self.problem_spec_file}`
- Task prompt: `{self.task_prompt_file}`
- Output contract: `{self.output_contract_file}`
- Python runtime manifest: `{self.runtime_manifest_file}`
- Task-scoped Python entry point: `{self.python_entry_file}`
- uv entry point: `{self.uv_entry_file}`

Required output directory:
- `{self.solve_output_dir}`

Read the staged problem specification and output contract, then create the required result and trajectory files directly under the output directory. Use `{self.python_entry_file}` for Python/NumPy/SciPy work. If the task-scoped Python runtime is missing, use `{self.uv_entry_file}` and `{self.runtime_manifest_file}` to create or repair it under `{self.task_dir}/.runtime_env`. Do not modify files under `{self.input_dir}`.

Save at least 1,000 finite states in each trajectory array. Tier 2 timestamps must be strictly increasing, starting at zero and including the final state. Saved output spacing is independent of the integrator's internal steps: dynamics are checked by propagating the stated gravity-plus-tangential-thrust equations between saved states, not by finite-differencing sparse output.

Tier 3 uses the full_mee_j2_minimum_fuel_v2 contract in the staged specification. Minimize fuel with throttle/coast over exactly 300 days using full osculating MEE+J2, including the normal-acceleration longitude term and all seven evolving costates. The Tier 3 trajectory now has 15 columns, including lambda_L before lambda_m; its control vectors include throttle. Follow the explicit certification mesh and numerical tolerances in the specification, including independent full-duration Cartesian replay of the saved piecewise-linear controls. Multiple shooting and collocation are permitted. No averaged reference history, fixed thrust fraction, or always-on fuel objective applies.
"""

    def output_file(self, filename: str) -> str:
        return f"{self.remote_output_dir}/{filename}"

    def reference_file(self, filename: str) -> str:
        return f"{self.reference_dir}/{filename}"

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "output_dir_name": self.output_dir_name,
                "problem_spec_file": self.problem_spec_file,
                "task_prompt_file": self.task_prompt_file,
                "output_contract_file": self.output_contract_file,
                "runtime_manifest_file": self.runtime_manifest_file,
                "python_entry_file": self.python_entry_file,
                "uv_entry_file": self.uv_entry_file,
                "runtime_env_python_file": self.runtime_env_python_file,
                "solve_output_dir": self.solve_output_dir,
                "required_output_files": list(REQUIRED_OUTPUT_FILES),
                "tier3_contract_version": "full_mee_j2_minimum_fuel_v2",
                "candidate_files": {
                    filename: self.output_file(filename) for filename in REQUIRED_OUTPUT_FILES
                },
                "reference_files": {
                    filename: self.reference_file(filename) for filename in REQUIRED_OUTPUT_FILES
                },
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
    reference_paths = meta.get("reference_files")
    if not isinstance(reference_paths, dict) or any(
        not isinstance(reference_paths.get(name), str) or not reference_paths[name]
        for name in REQUIRED_OUTPUT_FILES
    ):
        raise EvaluatorReferenceError("Missing or malformed evaluator reference metadata")
    reference_files: dict[str, bytes] = {}
    for filename in REQUIRED_OUTPUT_FILES:
        reference_path = reference_paths[filename]
        if not await session.file_exists(reference_path):
            raise EvaluatorReferenceError(f"Missing evaluator reference file: {reference_path}")
        try:
            reference_files[filename] = await session.read_bytes(reference_path)
        except FileNotFoundError as error:
            raise EvaluatorReferenceError(f"Missing evaluator reference file: {reference_path}") from error
    validate_reference(reference_files)
    candidate_files: dict[str, bytes] = {}
    for filename in REQUIRED_OUTPUT_FILES:
        candidate_path = meta["candidate_files"][filename]
        if not await session.file_exists(candidate_path):
            logger.info("Missing output file: %s", candidate_path)
            return [0.0]
        try:
            candidate_files[filename] = await session.read_bytes(candidate_path)
        except FileNotFoundError:
            return [0.0]
    report: ScoreReport = score_submission(candidate_files, reference_files)
    logger.info("Evaluation report: %s", json.dumps(report.to_dict(), sort_keys=True))
    return [report.score]
