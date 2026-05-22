"""AgentHLE task: humanoid_velocity_tracking_policy."""

from __future__ import annotations

import asyncio
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
)


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [("base", "Humanoid velocity tracking policy")]
EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\humanoid_velocity_tracking_policy"
DEFAULT_ISAAC_PYTHON = r"C:\Softwares\IsaacLab-2.3.2\env_isaaclab\Scripts\python.exe"
SIM_POLL_INTERVAL = 30
SIM_TIMEOUT = 3600


async def _missing(session: cb.DesktopSession, path: str, *, label: str, tag: str) -> bool:
    if await session.exists(path):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


@dataclass
class HumanoidVelocityTrackingConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = r"E:\agenthle"
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
    """Launch Isaac sim in background on VM and poll for the results file.

    The CUA REST/WebSocket layer times out after ~5min / ~2min respectively,
    but a full 10-seed rollout takes 30-40 minutes.  We work around this by
    writing a small .bat wrapper, launching it detached via ``wmic``, and
    polling for the results file.
    """
    meta = task_cfg.metadata
    isaac_python = os.environ.get("AGENTHLE_HUMANOID_ISAAC_PYTHON", DEFAULT_ISAAC_PYTHON)

    wrapper_bat = remote_results + ".bat"

    bat_content = (
        "@echo off\r\n"
        f"cd /d \"{meta['task_dir']}\"\r\n"
        f"\"{isaac_python}\" input\\eval\\run_eval.py "
        f"--policy \"{meta['output_policy']}\" "
        f"--checkpoint \"{meta['output_checkpoint']}\" "
        "--seeds 0 1 2 3 4 5 6 7 8 9 "
        "--num_envs 64 --rollout_seconds 20.0 "
        "--device cuda "
        f"--results_path \"{remote_results}\"\r\n"
    )

    await session.run_command(f'del /f /q "{remote_results}" 2>nul', check=False)
    await session.write_file(wrapper_bat, bat_content)
    wmic_cmd = f'wmic process call create "cmd /c \\"{wrapper_bat}\\""'
    await session.run_command(wmic_cmd, check=False)

    elapsed = 0
    while elapsed < SIM_TIMEOUT:
        await asyncio.sleep(SIM_POLL_INTERVAL)
        elapsed += SIM_POLL_INTERVAL
        if await session.exists(remote_results):
            break
        if elapsed % 120 == 0:
            logger.info("Isaac sim still running (%ds elapsed)...", elapsed)

    if not await session.exists(remote_results):
        logger.error("full Isaac simulation timed out after %ds", SIM_TIMEOUT)
        return False
    return True


async def _maybe_run_full_sim(
    task_cfg,
    session: cb.DesktopSession,
    local_results: Path,
    local_repeat_results: Path,
) -> bool:
    if os.environ.get("AGENTHLE_HUMANOID_SKIP_FULL_SIM") == "1":
        return False

    await session.makedirs(EVAL_TMP_DIR)
    remote_results = rf"{EVAL_TMP_DIR}\results_run_1.json"
    remote_repeat_results = rf"{EVAL_TMP_DIR}\results_run_2.json"
    if not await _run_full_sim_once(task_cfg, session, remote_results):
        return False
    if not await _run_full_sim_once(task_cfg, session, remote_repeat_results):
        return False
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
        if not await session.exists(meta[key]):
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
            return [0.0]

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, sort_keys=True))
    return [float(result.get("score", 0.0))]
