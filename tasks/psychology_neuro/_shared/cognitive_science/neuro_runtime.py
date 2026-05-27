"""Shared runtime helpers for cognitive-science neuroimaging GUI tasks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.psychology_neuro._shared.cognitive_science.neuro_common import (
    DATA_ROOT,
    DEFAULT_VARIANT,
    DESKTOP_ROOT,
    DOMAIN_NAME,
    SCENE_SPECS,
    SceneSpec,
)

logger = logging.getLogger(__name__)

EVAL_TMP_ROOT = "/tmp/agenthle_eval/cognitive_science"
START_DEPS = "python3 -m pip install --user --quiet pillow"


def _read_script(task_dir: Path, name: str) -> str:
    return (task_dir / "scripts" / name).read_text(encoding="utf-8")


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: Optional[float] = None,
) -> dict[str, object]:
    try:
        if timeout is not None:
            result = await session.run_command(command, timeout=timeout)
        else:
            result = await session.run_command(command)
    except TypeError:
        result = await session.run_command(command)
    if isinstance(result, dict):
        return {
            "return_code": result.get("return_code", result.get("returncode", 1)),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
    return {
        "return_code": getattr(result, "return_code", getattr(result, "returncode", 1)),
        "stdout": getattr(result, "stdout", ""),
        "stderr": getattr(result, "stderr", ""),
    }


async def ensure_layout(session: cb.DesktopSession, task_name: str, domain_name: str = DOMAIN_NAME) -> None:
    desktop_variant_root = f"{DESKTOP_ROOT}/{domain_name}/{task_name}/{DEFAULT_VARIANT}"
    data_variant_root = f"{DATA_ROOT}/{domain_name}/{task_name}/{DEFAULT_VARIANT}"
    desktop_parent = desktop_variant_root.rsplit("/", 1)[0]
    result = await _run_command(
        session,
        (
            f'mkdir -p "{data_variant_root}/output" "{desktop_parent}" '
            f'&& ln -sfn "{data_variant_root}" "{desktop_variant_root}"'
        ),
        timeout=120.0,
    )
    if result["return_code"] != 0:
        raise RuntimeError(f"failed to ensure task layout: {result['stderr'][:400]}")


async def ensure_runtime_deps(session: cb.DesktopSession) -> None:
    check = await _run_command(
        session,
        "python3 - <<'PY'\nimport numpy\nfrom PIL import Image\nprint(\"ok\")\nPY",
        timeout=60.0,
    )
    if check["return_code"] == 0:
        return
    install = await _run_command(session, START_DEPS, timeout=600.0)
    if install["return_code"] != 0:
        raise RuntimeError(f"failed to install runtime deps: {install['stderr'][:400]}")


@dataclass
class NeuroTaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = ""
    VARIANT_NAME: str = DEFAULT_VARIANT
    TASK_TITLE: str = ""
    OS_TYPE: str = "linux"

    @property
    def spec(self) -> SceneSpec:
        return SCENE_SPECS[self.TASK_NAME]

    @property
    def task_dir(self) -> str:
        return f"{DESKTOP_ROOT}/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}"

    @property
    def data_task_dir(self) -> str:
        return f"{DATA_ROOT}/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        return f"{self.task_dir}/input"

    @property
    def reference_dir(self) -> str:
        return f"{self.task_dir}/reference"

    @property
    def software_dir(self) -> str:
        return f"{self.task_dir}/software"

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.REMOTE_OUTPUT_DIR}"

    @property
    def launch_script(self) -> str:
        return f"{self.software_dir}/launch_gui.sh"

    @property
    def task_description(self) -> str:
        inputs = "\n".join(f"- `{self.input_dir}/{name}`" for name in self.spec.input_files)
        outputs = "\n".join(
            f"- `{self.remote_output_dir}/{name}`" for name in self.spec.required_outputs
        )
        return f"""\
You are a neuroimaging analyst completing a GUI workflow in {self.spec.software_name} {self.spec.software_version}.

## Your Task
{self.spec.instruction_text.strip()}

## Input Files
All task inputs are staged in:
- `{self.input_dir}`

Key staged files:
{inputs}

## Software
Launch the correct GUI workflow with:
- `{self.launch_script}`

The Desktop task folder is a symlink into `/media/user/data/agenthle/...`, so write all outputs under the staged task directory only.

## Required Output Files
Save these outputs exactly under `{self.remote_output_dir}`:
{outputs}

## Evaluation
You only pass if all required output files exist, are readable, and satisfy the task-specific correctness checks against hidden reference data.
Focus on this correctness target:
- {self.spec.evaluation_hint}
Do not read from `reference/`, `output_test_pos/`, or `output_test_neg/`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_title": self.TASK_TITLE,
                "task_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
                "input_dir": self.input_dir,
                "reference_dir": self.reference_dir,
                "software_dir": self.software_dir,
                "remote_output_dir": self.remote_output_dir,
                "launch_script": self.launch_script,
                "required_outputs": list(self.spec.required_outputs),
                "description_pdf": self.spec.description_pdf,
            }
        )
        return metadata


def load_single_task(task_name: str, task_title: str, *, domain_name: str | None = None):
    kwargs = {"TASK_NAME": task_name, "TASK_TITLE": task_title}
    if domain_name is not None:
        kwargs["DOMAIN_NAME"] = domain_name
    cfg = NeuroTaskConfig(**kwargs)
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


async def ensure_neuro_runtime(session: cb.DesktopSession, task_name: str, domain_name: str = DOMAIN_NAME) -> None:
    """Verify the canonical neuro-runtime layout and runtime deps on the VM.

    Output-dir creation is orchestration's job (clean_remote_output_dir wipes
    and recreates remote_output_dir after start()) so this helper only owns
    the truly neuro-specific bits: ensure the data-root symlink layout and
    that the conda env + Python deps are available.
    """
    await ensure_layout(session, task_name, domain_name)
    await ensure_runtime_deps(session)
    logger.info("[%s] neuro runtime verified", task_name)


async def evaluate_single_task(task_cfg, session: cb.DesktopSession, task_dir: Path) -> list[float]:
    task_name = task_cfg.metadata["task_name"]
    domain_name = task_cfg.metadata.get("domain_name", DOMAIN_NAME)
    await ensure_layout(session, task_name, domain_name)
    await ensure_runtime_deps(session)

    eval_tmp_dir = f"{EVAL_TMP_ROOT}/{task_name}"
    verify_script_path = f"{eval_tmp_dir}/verify_outputs.py"
    await _run_command(session, f'mkdir -p "{eval_tmp_dir}"', timeout=60.0)
    await session.write_file(verify_script_path, _read_script(task_dir, "verify_outputs.py"))

    result = await _run_command(
        session,
        (
            f'python3 "{verify_script_path}" '
            f'--task-name "{task_name}" '
            f'--input-dir "{task_cfg.metadata["input_dir"]}" '
            f'--reference-dir "{task_cfg.metadata["reference_dir"]}" '
            f'--output-dir "{task_cfg.metadata["remote_output_dir"]}"'
        ),
        timeout=300.0,
    )
    if result["return_code"] != 0 and not str(result["stdout"]).strip():
        logger.error("[%s] verifier failed: %s", task_name, str(result["stderr"])[:400])
        return [0.0]
    try:
        payload = json.loads(str(result["stdout"]))
    except Exception:
        logger.error("[%s] invalid verifier output: %r", task_name, result)
        return [0.0]
    score = float(payload.get("score", 0.0))
    logger.info("[%s] score=%.3f reasons=%s", task_name, score, payload.get("reasons"))
    return [score]
