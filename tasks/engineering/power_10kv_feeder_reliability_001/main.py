"""AgentHLE task: power_10kv_feeder_reliability_001."""

from __future__ import annotations

import json
import inspect
import logging
import math
import shlex
from pathlib import Path
from typing import Any, Optional

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "power_10kv_feeder_reliability_001"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
SCIENTIFIC_CONTRACT = (Path(__file__).resolve().parent / "scientific_contract.md").read_text(
    encoding="utf-8"
)
EVAL_TMP_DIR = "/tmp/agenthle_eval/power_10kv_feeder_reliability_001"
OUTPUT_FILENAME = "reliability_indices.json"
REFERENCE_FILENAME = "reliability_indices.normal-affine-v2.1.0.json"
EXPECTED_TOP_LEVEL_KEYS = [
    "feeder",
    "N_T",
    "SAIFI_F",
    "SAIDI_F_h",
    "SAIDI_F_min",
    "SAIFI_D",
    "SAIDI_D_h",
    "SAIDI_D_min",
    "SAIFI_S",
    "SAIDI_S_h",
    "SAIDI_S_min",
    "SAIFI",
    "SAIDI_h",
    "SAIDI_min",
    "CAIDI_h",
    "CAIDI_min",
    "ASAI",
    "fault_rows",
    "device_fault_rows",
    "scheduled_rows",
    "data_quality",
    "missing_exposure_terms",
]


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: Optional[float] = None,
    check: bool = False,
) -> dict[str, Any]:
    kwargs = {"check": check}
    if timeout is not None:
        try:
            inspect.signature(session.run_command).bind(command, timeout=timeout, check=check)
        except TypeError:
            pass
        else:
            kwargs["timeout"] = timeout
    return await session.run_command(command, **kwargs)


def _parse_json_stdout(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("verifier returned empty stdout")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"unable to parse verifier JSON from stdout: {text[:500]}")


class PowerFeederReliabilityConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    def __init__(self) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
        )

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.task_dir}/output_test_neg"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/{OUTPUT_FILENAME}"

    @property
    def input_xml(self) -> str:
        return f"{self.input_dir}/gis.null.xml"

    @property
    def input_svg(self) -> str:
        return f"{self.input_dir}/gis.null.svg"

    @property
    def input_params(self) -> str:
        return f"{self.input_dir}/params.json"

    @property
    def input_pyproject(self) -> str:
        return f"{self.input_dir}/pyproject.toml"

    @property
    def input_uv_lock(self) -> str:
        return f"{self.input_dir}/uv.lock"

    @property
    def task_description(self) -> str:
        return f"""\
You are computing supply reliability indices for a 10kV distribution feeder under the specified planning model.

## Your Task
Read the staged feeder model, the SVG topology drawing, and the reliability parameters.
Compute the feeder reliability indices and the section-level contribution tables.
Complete the whole-feeder model with affine terms for unknown line lengths, without imputing measurements.

## Visible Inputs
- CIM/RDF XML: `{self.input_xml}`
- SVG topology: `{self.input_svg}`
- Reliability parameters: `{self.input_params}`
- Agent-facing dependency manifests: `{self.input_pyproject}` and `{self.input_uv_lock}`

## Software
- Use the staged Python environment manifests under `input/` if you need packages.
- Do not preinstall task-specific Python packages system-wide.
- The benchmark runtime should stay isolated from any global package state.

## Conventions
- For this task, `N_T` is the number of `PowerTransformer` objects assigned to the current feeder; each counts as one equally weighted load point. Do not use the `EnergyConsumer` count for `N_T`.
- Use these exact row fields:
  - `fault_rows`: `section, name, length_km, lambda_i, perm_users, t_iso_h, r_i_h, lambda_N, lambda_N_r`
  - `device_fault_rows`: `type, section, name, device_count, lambda_per_device, affected_users, t_repair_h, lambda_N, lambda_N_r`
  - `scheduled_rows`: `section, name, length_km, users, lambda_N, lambda_N_r`
- `section` may be any locally unique identifier. Its literal value and row ordering are not significant.

{SCIENTIFIC_CONTRACT}

## What You Must Do
1. Produce one JSON file exactly at `{self.output_file}`.
2. Include recorded-exposure scalar subtotals, section tables, mandatory `data_quality`, and `missing_exposure_terms` for the full affine model.
3. Follow the public scientific model and the exact schema given here; validation data is hidden.
4. Write `ASAI` as a fraction between 0 and 1.

## Output Requirements
- Required top-level keys:
  `feeder, N_T, SAIFI_F, SAIDI_F_h, SAIDI_F_min, SAIFI_D, SAIDI_D_h, SAIDI_D_min, SAIFI_S, SAIDI_S_h, SAIDI_S_min, SAIFI, SAIDI_h, SAIDI_min, CAIDI_h, CAIDI_min, ASAI, fault_rows, device_fault_rows, scheduled_rows, data_quality, missing_exposure_terms`
- Keep the section tables in the JSON output.
- Do not write any extra files into `{self.remote_output_dir}`.

## Practical Note
The task is designed for Python-based network analysis and data processing. If you need additional Python packages, install them from the staged manifests under `input/` using `uv`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_name": VARIANT_NAME,
                "scientific_contract_version": "normal-affine-v2.1.0",
                "reference_filename": REFERENCE_FILENAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "remote_output_dir": self.remote_output_dir,
                "output_file": self.output_file,
                "input_xml": self.input_xml,
                "input_svg": self.input_svg,
                "input_params": self.input_params,
                "input_pyproject": self.input_pyproject,
                "input_uv_lock": self.input_uv_lock,
                "expected_top_level_keys": list(EXPECTED_TOP_LEVEL_KEYS),
            }
        )
        return metadata


config = PowerFeederReliabilityConfig()


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
    tag = meta["variant_name"]
    output_file = meta["output_file"]
    reference_file = f"{meta['reference_dir']}/{REFERENCE_FILENAME}"

    if not await session.file_exists(reference_file):
        raise RuntimeError(f"[{tag}] Controlled reference missing or not a file: {reference_file}")

    await session.interface.create_dir(EVAL_TMP_DIR)
    verify_script_path = f"{EVAL_TMP_DIR}/verify_reliability_indices.py"
    await session.write_file(verify_script_path, _read_script("verify_reliability_indices.py"))

    result = await _run_command(
        session,
        f"python {shlex.quote(verify_script_path)} --agent {shlex.quote(output_file)} --ref {shlex.quote(reference_file)}",
        timeout=300.0,
        check=False,
    )

    if not isinstance(result, dict) or result.get("return_code") != 0:
        raise RuntimeError(f"[{tag}] Verifier command failed: {str(result)[:1000]}")

    try:
        payload = _parse_json_stdout(result["stdout"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"[{tag}] Invalid verifier JSON response") from exc
    if not isinstance(payload, dict) or payload.get("status") not in {"ok", "candidate_error"}:
        raise RuntimeError(f"[{tag}] Verifier reported an evaluation failure: {str(payload)[:500]}")
    score = payload.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0 <= score <= 1
        or type(payload.get("passed")) is not bool
        or not isinstance(payload.get("reason"), str)
        or not isinstance(payload.get("issues"), list)
        or (payload["status"] == "candidate_error" and (score != 0 or payload["passed"]))
    ):
        raise RuntimeError(f"[{tag}] Invalid verifier result schema: {str(payload)[:500]}")
    logger.info(
        "[%s] score=%.3f passed=%s reason=%s",
        tag,
        score,
        payload.get("passed"),
        payload.get("reason"),
    )
    return [float(score)]
