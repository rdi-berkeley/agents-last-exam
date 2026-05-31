"""AgentHLE task: GLM Lake Mendota calibration."""

import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
REMOTE_EVAL_TMP_DIR = "/tmp/agenthle_eval/glm_lake_calibration"


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: Optional[float] = None,
    check: bool = False,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


def _parse_json_stdout(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("verifier returned empty stdout")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"unable to parse verifier JSON from stdout: {text[:500]}")


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


@dataclass
class GLMLakeCalibrationConfig(LinuxTaskConfig):
    """Task configuration for the GLM lake calibration benchmark."""

    DOMAIN_NAME: str = "physical_sciences"
    TASK_NAME: str = "glm_lake_calibration"
    VARIANT_NAME: str = "base"
    OS_TYPE: str = "linux"

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.task_dir}/output_test_neg"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/output.nc"

    @property
    def glm_config_path(self) -> str:
        return f"{self.input_dir}/glm3.nml"

    @property
    def observation_file(self) -> str:
        return f"{self.input_dir}/field_temp_oxy.csv"

    @property
    def forcing_dir(self) -> str:
        return f"{self.input_dir}/bcs"

    @property
    def glm_run_script(self) -> str:
        return f"{self.software_dir}/run_glm_from_input.sh"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_glm_env.sh"

    @property
    def glm_binary(self) -> str:
        return f"{self.software_dir}/bin/glm.bin"

    @property
    def task_description(self) -> str:
        return f"""\
You are calibrating GLM (General Lake Model) 3 for Lake Mendota, Wisconsin.

## Your Task
Tune the staged GLM namelist so the simulated vertical water-temperature
profiles match the staged observations with RMSE below 1.5 C.

## Visible Inputs
- GLM namelist to edit: `{self.glm_config_path}`
- Observation file: `{self.observation_file}`
- Forcing directory: `{self.forcing_dir}`
- Runtime wrapper: `{self.glm_run_script}`

## Software
- Use the staged GLM runtime wrapper at `{self.glm_run_script}`
- The wrapper uses the benchmark-provided GLM 3 binary and shared libraries in
  `{self.software_dir}`
- Do not assume `/usr/local/bin/glm` exists on this VM

## What You Must Do
1. Modify only `{self.glm_config_path}`.
2. Leave the observation CSV, forcing CSVs, and staged GLM runtime unchanged.
3. Run `{self.glm_run_script}` to generate the simulation output.
4. Save the final simulation NetCDF exactly to `{self.output_file}`.

## Important Constraints
- Only edit `{self.glm_config_path}` and write under `{self.output_dir}`
- Do not replace the staged GLM binary
- The output must cover the required 2009-01-01 through late-2015 simulation window
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_name": self.VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "output_file": self.output_file,
                "glm_config_path": self.glm_config_path,
                "observation_file": self.observation_file,
                "forcing_dir": self.forcing_dir,
                "glm_run_script": self.glm_run_script,
                "python_wrapper": self.python_wrapper,
                "glm_binary": self.glm_binary,
                "canonical_gcs_root": "gs://ale-data-all/physical_sciences/glm_lake_calibration/base/",
            }
        )
        return metadata


config = GLMLakeCalibrationConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score on the VM after post-submit hidden evaluator assets are staged."""

    meta = task_cfg.metadata
    mode = Path(meta["output_dir"]).name
    hidden_paths = [meta["reference_dir"], f"{meta['reference_dir']}/fixture_metrics.json"]
    for path in hidden_paths:
        if not (await session.file_exists(path) or await session.directory_exists(path)):
            logger.error(
                "[%s] hidden evaluator path missing at evaluate() time: %s",
                meta["variant_name"],
                path,
            )
            return [0.0]

    await session.interface.create_dir(REMOTE_EVAL_TMP_DIR)
    verify_script_path = f"{REMOTE_EVAL_TMP_DIR}/verify_outputs.py"
    await session.write_file(verify_script_path, _read_script("verify_outputs.py"))

    command = (
        f"{shlex.quote(meta['python_wrapper'])} "
        f"{shlex.quote(verify_script_path)} "
        f"--mode {shlex.quote(mode)} "
        f"--input-dir {shlex.quote(meta['input_dir'])} "
        f"--software-dir {shlex.quote(meta['software_dir'])} "
        f"--reference-dir {shlex.quote(meta['reference_dir'])} "
        f"--output-dir {shlex.quote(meta['output_dir'])}"
    )
    result = await _run_command(
        session,
        command,
        timeout=1800.0,
        check=False,
    )

    try:
        payload = _parse_json_stdout(result.get("stdout", ""))
    except ValueError as exc:
        logger.error(
            "[%s] failed to parse VM verifier output: %s; stderr=%s",
            meta["variant_name"],
            exc,
            (result.get("stderr", "") or "")[:1000],
        )
        return [0.0]

    score = float(payload.get("score", 0.0))
    logger.info("[%s] verifier=%s", meta["variant_name"], json.dumps(payload))
    if result.get("return_code", 0) != 0:
        logger.error(
            "[%s] VM verifier exited non-zero (%s): %s",
            meta["variant_name"],
            result.get("return_code", 0),
            (result.get("stderr", "") or "")[:1000],
        )
        return [0.0]
    return [score]
