"""robot_dexterous_grasping — synthesize G1 upper-body grasp+lift trajectories."""

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
from tasks.engineering.robot_dexterous_grasping.scripts.score_outputs import (
    evaluate_submission,
)


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [("base", "Grasp and lift each of the three target objects")]
GRADER_TIMEOUT = 1800


@dataclass
class RobotDexterousGraspingConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "engineering"
    TASK_NAME: str = "robot_dexterous_grasping"
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
    def input_objects_json(self) -> str:
        return f"{self.input_assets_dir}/objects.json"

    @property
    def input_policy_dir(self) -> str:
        return f"{self.input_dir}/policy"

    @property
    def reference_grader(self) -> str:
        return f"{self.reference_dir}/grader/run_grader.py"

    @property
    def task_description(self) -> str:
        return f"""\
You are synthesizing dexterous grasping trajectories for a Unitree G1 humanoid
(Dex3 3-finger hands) in fixed MuJoCo scenes.

The robot is ALREADY STANDING at the table in front of each object and its
LOWER BODY DOES NOT MOVE: you do not control the legs or the base, and the robot
never walks. A bundled RL balance policy holds it standing in place. You control
ONLY the 31 upper-body joints: waist (3), left arm (7), right arm (7), left Dex3
hand (7), right Dex3 hand (7). The right hand is the grasping hand.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input Files
- Task brief: `{self.input_task_brief}`
- Success criteria and metrics: `{self.input_success}`
- Trajectory format and deterministic replay/control contract: `{self.input_schema}`
- Target object ids: `{self.input_objects_json}`
- Per-object frozen simulation: `{self.input_assets_dir}/obj_<id>/`
  (scene.mjb, init_state.npy, amo_init_state.npz, robot_params.json incl.
  upper_body_joint_names / target_body / init_object_z / stand_command)
- Bundled RL balance policy (holds the robot standing; you never command it):
  `{self.input_policy_dir}`

## What You Must Do
1. Read the task brief, the trajectory schema, and the success criteria.
2. For EACH object id in `{self.input_objects_json}`, build your own control loop
   over that object's `scene.mjb` + `{self.input_policy_dir}` and synthesize an
   upper-body motion that reaches to the object, closes the Dex3 fingers on it,
   and lifts it. There is no grasp database and no inverse-kinematics helper:
   grasp-pose selection, arm IK, finger closure and motion timing are yours.
3. Save one file per object in `{self.remote_output_dir}`:
   `trajectory_<object_id>.npz`, each containing the float32 array `upper_body`
   of shape [T, 31] defined in `{self.input_schema}`.

## Output Requirements
- One `trajectory_<object_id>.npz` per object id, each with `upper_body` [T, 31],
  finite, T >= 1.
- On replay under the contract in `{self.input_schema}`, EACH object must end
  grasped in the right hand (within 0.15 m of the Dex3 fingertip centre), lifted
  >= 0.08 m above its starting height, robot still upright (pelvis > 0.6 m).
- Scoring is all-or-nothing: all three objects, or 0.
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
                "input_objects_json": self.input_objects_json,
                "input_policy_dir": self.input_policy_dir,
                "reference_grader": self.reference_grader,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=RobotDexterousGraspingConfig(
                VARIANT_NAME=variant_name, VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=RobotDexterousGraspingConfig(
                VARIANT_NAME=variant_name, VARIANT_LABEL=variant_label,
            ).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
        for variant_name, variant_label in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _run_grader(task_cfg, session: cb.DesktopSession, remote_results: str) -> bool:
    """Replay every submitted trajectory on the VM (GPU) and write a results JSON."""
    meta = task_cfg.metadata
    cmd = (
        f'python3 "{meta["reference_grader"]}" '
        f'--output "{meta["remote_output_dir"]}" '
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

    if not (await session.file_exists(meta["reference_grader"])
            or await session.directory_exists(meta["reference_grader"])):
        logger.error("[%s] Missing hidden grader: %s", tag, meta["reference_grader"])
        return [0.0]
    if not (await session.file_exists(meta["remote_output_dir"])
            or await session.directory_exists(meta["remote_output_dir"])):
        logger.error("[%s] Missing output directory: %s", tag, meta["remote_output_dir"])
        return [0.0]

    remote_results = f"{meta['remote_output_dir']}/.grader_result.json"

    with tempfile.TemporaryDirectory(prefix="dexterous_grasping_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output = tmp / "output"
        local_output.mkdir()
        local_results = tmp / "results.json"

        try:
            object_ids = json.loads(await session.read_text(meta["input_objects_json"]))["object_ids"]
            for oid in object_ids:
                remote_traj = f"{meta['remote_output_dir']}/trajectory_{oid}.npz"
                if await session.file_exists(remote_traj):
                    (local_output / f"trajectory_{oid}.npz").write_bytes(
                        await session.read_bytes(remote_traj))
            grader_ran = await _run_grader(task_cfg, session, remote_results)
            if grader_ran:
                local_results.write_bytes(await session.read_bytes(remote_results))
            result = evaluate_submission(
                local_output,
                object_ids=object_ids,
                results_path=local_results if grader_ran else None,
            )
        except Exception as exc:
            logger.exception("[%s] Evaluation failed: %s", tag, exc)
            return [0.0]

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, sort_keys=True))
    return [float(result.get("score", 0.0))]
