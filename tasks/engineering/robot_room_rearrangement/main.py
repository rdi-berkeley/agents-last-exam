"""humanoid_object_stowing — synthesize one continuous G1 whole-body tidy-up trajectory."""

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
from tasks.engineering.humanoid_object_stowing.scripts.score_outputs import (
    evaluate_submission,
)


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [("base", "Walk to the table object, pick it up, and drop it into the floor bin")]
GRADER_TIMEOUT = 3600


@dataclass
class HumanoidObjectStowingConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "engineering"
    TASK_NAME: str = "humanoid_object_stowing"
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
    def reference_grader(self) -> str:
        return f"{self.reference_dir}/grader/run_grader.py"

    @property
    def task_description(self) -> str:
        return f"""\
You are synthesizing a single continuous whole-body trajectory for a Unitree G1
humanoid (Dex3 3-finger hands) in a fixed MuJoCo bedroom scene.

An object sits on a table and a bin stands on the floor nearby. The robot starts
standing away from the table and must walk to the object, grasp it, carry it to
the bin and drop it in, ending with the object inside the bin and the robot still
standing.

This is a WHOLE-BODY task. You control the legs and the upper body: one
trajectory drives walking, turning, reaching, grasping and releasing. It is ONE
continuous episode -- not one trajectory per object. A bundled RL balance policy
converts your locomotion command stream into leg motion; you must still decide
where to walk, when to turn, how to reach and how to grasp.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input Files
- Task brief: `{self.input_task_brief}`
- Success criteria and metrics: `{self.input_success}`
- Trajectory format and deterministic replay/control contract: `{self.input_schema}`
- Frozen simulation: `{self.input_assets_dir}/` (scene.mjb, init_state.npy,
  amo_init_state.npz, robot_params.json incl. joint order, PD gains, torque
  limits, the bin's axis-aligned bounds and the target object label)
- Bundled RL balance policy: `{self.input_policy_dir}`

## What You Must Do
1. Read the task brief, the trajectory schema, and the success criteria.
2. Build your own control loop over `scene.mjb` + `{self.input_policy_dir}` and
   synthesize ONE continuous episode that puts the target object in the bin. There
   is no motion planner, no grasp database and no inverse-kinematics helper:
   navigation, grasp-pose selection, arm IK, finger closure and motion timing are
   all yours to solve.
3. Save `{self.remote_output_dir}/trajectory.npz` containing the four float32
   arrays defined in `{self.input_schema}`: `action` [T, 43],
   `amo_policy_command` [T, 9], `amo_policy_target_yaw` [T, 1] and
   `amo_policy_turning_flag` [T, 1]. All four must share the same T.

## Output Requirements
- Exactly one `trajectory.npz` with the four arrays above, finite, T >= 1.
- On replay under the contract in `{self.input_schema}`, the target object must
  end inside the bin (within its XY bounds and below the rim) and the robot must
  still be upright (pelvis > 0.60 m).
- Scoring is binary: 1.0 or 0.0.
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
                "reference_grader": self.reference_grader,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=HumanoidObjectStowingConfig(
                VARIANT_NAME=variant_name, VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=HumanoidObjectStowingConfig(
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
    """Replay the submitted trajectory on the VM (GPU) and write a results JSON."""
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

    with tempfile.TemporaryDirectory(prefix="humanoid_object_stowing_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output = tmp / "output"
        local_output.mkdir()
        local_results = tmp / "results.json"

        try:
            remote_traj = f"{meta['remote_output_dir']}/trajectory.npz"
            if await session.file_exists(remote_traj):
                (local_output / "trajectory.npz").write_bytes(
                    await session.read_bytes(remote_traj))
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
