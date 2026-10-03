"""humanoid_loco_manipulation — synthesize a G1 whole-body trajectory."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.engineering.humanoid_loco_manipulation.scripts.score_outputs import (
    evaluate_submission,
)


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [("base", "Turn+walk to the thermos, grasp it, carry it back to the start, hold it")]
GRADER_TIMEOUT = 900


@dataclass
class HumanoidLocoManipulationConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "engineering"
    TASK_NAME: str = "humanoid_loco_manipulation"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_task_brief(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def input_success(self) -> str:
        return f"{self.input_dir}/SUCCESS.md"

    @property
    def input_schema(self) -> str:
        return f"{self.input_dir}/TRAJECTORY_SCHEMA.md"

    @property
    def input_assets_dir(self) -> str:
        return f"{self.input_dir}/assets"

    @property
    def input_policy_dir(self) -> str:
        return f"{self.input_dir}/policy"

    @property
    def output_trajectory(self) -> str:
        return f"{self.remote_output_dir}/trajectory.npz"

    @property
    def reference_trajectory(self) -> str:
        return f"{self.reference_dir}/trajectory.npz"

    @property
    def reference_grader(self) -> str:
        return f"{self.reference_dir}/grader/run_grader.py"

    @property
    def task_description(self) -> str:
        return f"""\
You are synthesizing a fetch-and-return whole-body trajectory for a Unitree G1
humanoid (43 DoF, right-hand Dex3 3-finger gripper) in a fixed MuJoCo scene.
The robot starts facing AWAY from the target thermos.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input Files
- Task brief: `{self.input_task_brief}`
- Success criteria and metrics: `{self.input_success}`
- Trajectory format and deterministic replay/control contract: `{self.input_schema}`
- Frozen self-contained simulation: `{self.input_assets_dir}`
  (scene.mjb, robot_params.json incl. spawn_xy, init_state.npy, amo_init_state.npz)
- RL locomotion policy the robot is driven by: `{self.input_policy_dir}`
  (amo_policy.py + TorchScript weights)

## What You Must Do
1. Read the task brief, the trajectory schema, and the success criteria.
2. Build your own control loop over `{self.input_assets_dir}/scene.mjb` +
   `{self.input_policy_dir}`. There is no motion planner, no grasp database, and
   no inverse-kinematics helper. Command base velocities + heading to the AMO
   locomotion policy to turn and walk to the thermos; command waist/arm/hand
   targets to grasp and lift it; then turn around and carry it back to the start
   position (`spawn_xy` in robot_params.json). Solve navigation, grasp-pose
   selection, and arm kinematics yourself. Note: `amo_policy_target_yaw` is
   relative to the robot's initial facing.
3. Save exactly one file in `{self.remote_output_dir}`: `trajectory.npz`, with
   the four float32 arrays defined in `{self.input_schema}`.

## Output Requirements
- `trajectory.npz` must contain `action` [T,43], `amo_policy_command` [T,9],
  `amo_policy_target_yaw` [T,1], and `amo_policy_turning_flag` [T,1], all finite
  and the same length T.
- When replayed under the contract in `{self.input_schema}`, the robot must end
  upright (pelvis > 0.6 m), within 0.6 m of `spawn_xy`, and still holding the
  thermos in the right hand (object z > 0.6 m AND within 0.15 m of the Dex3
  fingertip center). Scoring is binary: all three or 0.
- Do not write final answers outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_task_brief": self.input_task_brief,
                "input_success": self.input_success,
                "input_schema": self.input_schema,
                "input_assets_dir": self.input_assets_dir,
                "input_policy_dir": self.input_policy_dir,
                "output_trajectory": self.output_trajectory,
                "reference_trajectory": self.reference_trajectory,
                "reference_grader": self.reference_grader,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=HumanoidLocoManipulationConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=HumanoidLocoManipulationConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
        for variant_name, variant_label in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _run_grader(task_cfg, session: cb.DesktopSession, remote_results: str) -> bool:
    """Replay the submitted trajectory on the VM (GPU) and write a results JSON."""
    meta = task_cfg.metadata
    cmd = (
        f'python3 "{meta["reference_grader"]}" '
        f'--trajectory "{meta["output_trajectory"]}" '
        f'--assets "{meta["input_assets_dir"]}" '
        f'--policy "{meta["input_policy_dir"]}" '
        f'--results "{remote_results}"'
    )
    await session.run_command(f'rm -f "{remote_results}"', check=False)
    await session.run_command(cmd, check=False, timeout=GRADER_TIMEOUT)
    return await session.file_exists(remote_results)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]

    for key, label in [
        ("output_trajectory", "output trajectory.npz"),
        ("reference_trajectory", "hidden reference trajectory.npz"),
        ("reference_grader", "hidden grader"),
    ]:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            logger.error("[%s] Missing %s at %s", tag, label, meta[key])
            return [0.0]

    remote_results = f"{meta['remote_output_dir']}/.grader_result.json"

    with tempfile.TemporaryDirectory(prefix="humanoid_loco_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output = tmp / "output"
        local_output.mkdir()
        local_results = tmp / "results.json"

        try:
            (local_output / "trajectory.npz").write_bytes(
                await session.read_bytes(meta["output_trajectory"])
            )
            grader_ran = await _run_grader(task_cfg, session, remote_results)
            if grader_ran:
                local_results.write_bytes(await session.read_bytes(remote_results))
            result = evaluate_submission(
                local_output,
                results_path=local_results if grader_ran else None,
            )
        except Exception as exc:
            logger.exception("[%s] Evaluation failed: %s", tag, exc)
            return [0.0]

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, sort_keys=True))
    return [float(result.get("score", 0.0))]
