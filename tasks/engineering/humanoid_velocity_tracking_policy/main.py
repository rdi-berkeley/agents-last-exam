"""AgentHLE task: humanoid_velocity_tracking_policy."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

# cua_bench loads task modules via exec_module without always pre-registering
# them in sys.modules; dataclass needs this for string annotation handling.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.engineering.humanoid_velocity_tracking_policy.scripts.score_humanoid_velocity_policy import (
    evaluate_submission,
    validate_submission,
)
from tasks.engineering.humanoid_velocity_tracking_policy.scripts.simulation_process import run_simulation


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [("base", "Humanoid velocity tracking policy")]
EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\humanoid_velocity_tracking_policy"
DEFAULT_ISAAC_PYTHON = r"C:\Softwares\IsaacLab-2.3.2\env_isaaclab\Scripts\python.exe"
SIM_POLL_INTERVAL = 30
SIM_TIMEOUT = 3600


async def _missing(session: cb.DesktopSession, path: str, *, label: str, tag: str) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


@dataclass
class HumanoidVelocityTrackingConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "engineering"
    TASK_NAME: str = "humanoid_velocity_tracking_policy"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def input_task_brief(self) -> str:
        return rf"{self.input_dir}\task_brief.md"

    @property
    def input_success(self) -> str:
        return rf"{self.input_dir}\SUCCESS.md"

    @property
    def input_proprio_spec(self) -> str:
        return rf"{self.input_dir}\PROPRIO_SPEC.md"

    @property
    def input_eval_dir(self) -> str:
        return rf"{self.input_dir}\eval"

    @property
    def input_run_eval(self) -> str:
        return rf"{self.input_eval_dir}\run_eval.py"

    @property
    def input_assets_dir(self) -> str:
        return rf"{self.input_dir}\assets"

    @property
    def input_robot_urdf(self) -> str:
        return rf"{self.input_assets_dir}\robot.urdf"

    @property
    def input_submission_template_dir(self) -> str:
        return rf"{self.input_dir}\submission_template"

    @property
    def software_readme(self) -> str:
        return rf"{self.software_dir}\README.txt"

    @property
    def output_policy(self) -> str:
        return rf"{self.remote_output_dir}\policy.py"

    @property
    def output_checkpoint(self) -> str:
        return rf"{self.remote_output_dir}\checkpoint.pt"

    @property
    def reference_baseline_policy(self) -> str:
        return rf"{self.reference_dir}\baseline\policy.py"

    @property
    def reference_baseline_checkpoint(self) -> str:
        return rf"{self.reference_dir}\baseline\checkpoint.pt"

    @property
    def task_description(self) -> str:
        return f"""\
You are producing a policy submission for an Isaac Lab humanoid velocity-tracking benchmark.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input Files
- Task brief: `{self.input_task_brief}`
- Success criteria and exact evaluation command: `{self.input_success}`
- Observation/action ordering: `{self.input_proprio_spec}`
- Robot asset and articulation config: `{self.input_assets_dir}`
- Evaluation harness: `{self.input_eval_dir}`
- Starter template: `{self.input_submission_template_dir}`

## What You Must Do
1. Read the staged task brief, proprioceptive-state spec, and success criteria.
2. Train, distill, hand-author, or otherwise produce a deterministic policy that
   implements the required `Policy` API.
3. Save exactly two files in `{self.remote_output_dir}`:
   `policy.py` and `checkpoint.pt`.

## Output Requirements
- `policy.py` must define `Policy(checkpoint_path: str, device: str)`.
- `Policy.inference(obs)` must return exactly `{{"action": tensor}}`.
- The action tensor must have shape `(N, 23)`, dtype `torch.float32`, and live
  on the same device as `obs["command"]`.
- Do not write final answers outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_dir": self.input_dir,
                "input_task_brief": self.input_task_brief,
                "input_success": self.input_success,
                "input_proprio_spec": self.input_proprio_spec,
                "input_eval_dir": self.input_eval_dir,
                "input_run_eval": self.input_run_eval,
                "input_assets_dir": self.input_assets_dir,
                "input_robot_urdf": self.input_robot_urdf,
                "input_submission_template_dir": self.input_submission_template_dir,
                "software_readme": self.software_readme,
                "output_policy": self.output_policy,
                "output_checkpoint": self.output_checkpoint,
                "reference_baseline_policy": self.reference_baseline_policy,
                "reference_baseline_checkpoint": self.reference_baseline_checkpoint,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=HumanoidVelocityTrackingConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=HumanoidVelocityTrackingConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
        for variant_name, variant_label in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _run_full_sim_once(task_cfg, session: cb.DesktopSession, remote_results: str) -> bool:
    meta = task_cfg.metadata
    isaac_python = os.environ.get("AGENTHLE_HUMANOID_ISAAC_PYTHON", DEFAULT_ISAAC_PYTHON)
    arguments = [
        "-u", meta["input_run_eval"],
        "--policy", meta["output_policy"], "--checkpoint", meta["output_checkpoint"],
        "--seeds", *map(str, range(10)), "--num_envs", "64", "--rollout_seconds", "20.0",
        "--device", "cuda", "--results_path", remote_results,
    ]
    return await run_simulation(
        session, executable=isaac_python, arguments=arguments,
        working_directory=meta["task_dir"], results_path=remote_results,
        timeout=SIM_TIMEOUT, poll_interval=SIM_POLL_INTERVAL,
    )


async def _maybe_run_full_sim(
    task_cfg,
    session: cb.DesktopSession,
    local_results: Path,
    local_repeat_results: Path,
) -> bool:
    if os.environ.get("AGENTHLE_HUMANOID_SKIP_FULL_SIM") == "1":
        return False

    await session.interface.create_dir(EVAL_TMP_DIR)
    remote_results = rf"{EVAL_TMP_DIR}\results_run_1.json"
    remote_repeat_results = rf"{EVAL_TMP_DIR}\results_run_2.json"
    await _run_full_sim_once(task_cfg, session, remote_results)
    await _run_full_sim_once(task_cfg, session, remote_repeat_results)
    local_results.write_bytes(await session.read_bytes(remote_results))
    local_repeat_results.write_bytes(await session.read_bytes(remote_repeat_results))
    return True


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]

    for key, label in [
        ("output_policy", "output policy.py"),
        ("output_checkpoint", "output checkpoint.pt"),
        ("reference_baseline_policy", "hidden baseline policy.py"),
        ("reference_baseline_checkpoint", "hidden baseline checkpoint.pt"),
    ]:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            if key.startswith("reference_"):
                raise RuntimeError(f"Missing evaluator reference: {meta[key]}")
            logger.error("[%s] Missing %s at %s", tag, label, meta[key])
            return [0.0]

    with tempfile.TemporaryDirectory(prefix="humanoid_velocity_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output = tmp / "output"
        local_reference = tmp / "reference"
        (local_output).mkdir()
        (local_reference / "baseline").mkdir(parents=True)
        local_results = tmp / "results.json"
        local_repeat_results = tmp / "results_repeat.json"

        try:
            (local_output / "policy.py").write_bytes(await session.read_bytes(meta["output_policy"]))
            (local_output / "checkpoint.pt").write_bytes(await session.read_bytes(meta["output_checkpoint"]))
            (local_reference / "baseline" / "policy.py").write_bytes(
                await session.read_bytes(meta["reference_baseline_policy"])
            )
            (local_reference / "baseline" / "checkpoint.pt").write_bytes(
                await session.read_bytes(meta["reference_baseline_checkpoint"])
            )
            validation = validate_submission(local_output)
            if validation["score"] is not None:
                logger.info("[%s] submission validation=%s", tag, json.dumps(validation, sort_keys=True))
                return [float(validation["score"])]
            full_sim_ran = await _maybe_run_full_sim(
                task_cfg,
                session,
                local_results,
                local_repeat_results,
            )
            result = evaluate_submission(
                local_output,
                local_reference,
                results_path=local_results if full_sim_ran else None,
                repeat_results_path=local_repeat_results if full_sim_ran else None,
                allow_fixture_hash_fallback=not full_sim_ran,
            )
        except Exception as exc:
            logger.exception("[%s] Evaluation failed: %s", tag, exc)
            raise

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, sort_keys=True))
    return [float(result.get("score", 0.0))]
