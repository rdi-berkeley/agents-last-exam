"""physical_sciences/hst_acs_wfc_visit_reduction -- Linux task."""

from __future__ import annotations

import json
import logging
import os
import shlex
import asyncio
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup  # noqa: E402
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402
from tasks.physical_sciences.hst_acs_wfc_visit_reduction.scripts.score_outputs import (  # noqa: E402
    REQUIRED_VISIT_FILES,
)

_setup = BaseTaskSetup()


logger = logging.getLogger(__name__)

DOMAIN_NAME = "physical_sciences"
TASK_NAME = "hst_acs_wfc_visit_reduction"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).parent / "scripts"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"
STATIC_OUTPUT_DIRS = {"output_test_pos", "output_test_neg"}


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


@dataclass
class HstAcsWfcVisitReductionConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def prompt_file(self) -> str:
        return f"{self.input_dir}/TASK_PROMPT.md"

    @property
    def visible_visit_dir(self) -> str:
        return f"{self.input_dir}/acs_visit_f606w_lockman"

    @property
    def starter_script(self) -> str:
        return f"{self.input_dir}/starter_project/reduce_visit.py"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_with_task_deps.sh"

    @property
    def candidate_script(self) -> str:
        return f"{self.remote_output_dir}/reduce_visit.py"

    @property
    def hidden_input_dir(self) -> str:
        return f"{self.reference_dir}/hidden_input"

    @property
    def reference_outputs_dir(self) -> str:
        return f"{self.reference_dir}/reference_outputs"

    @property
    def task_description(self) -> str:
        return f"""\
You are reducing synthetic HST ACS/WFC visit data on a Linux VM. Read the full
instructions in `{self.prompt_file}` and use the visible visit data under:
`{self.visible_visit_dir}`.

## Goal
Implement a reusable ACS/WFC visit-reduction script at:
`{self.candidate_script}`

Your script must accept:
`python reduce_visit.py --input <visit_root_or_parent_input_dir> --output <output_dir>`

For each visit in the input, write a folder directly under the supplied output
directory, for example `<output_dir>/<visit_id>/`, containing:
- `drizzled_image.csv`
- `source_catalog.csv`
- `alignment_solution.csv`
- `photometry_qc.json`
- `reduction_report.md`

The starter implementation is at `{self.starter_script}`. The runtime manifest is
under `{self.runtime_env_dir}`, and the helper wrapper is:
`{self.python_wrapper}`

Do not modify files under `{self.input_dir}`. Write your final implementation and
any scratch artifacts only under `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{DOMAIN_NAME}/{TASK_NAME}",
                "prompt_file": self.prompt_file,
                "visible_visit_dir": self.visible_visit_dir,
                "starter_script": self.starter_script,
                "runtime_env_dir": self.runtime_env_dir,
                "python_wrapper": self.python_wrapper,
                "candidate_script": self.candidate_script,
                "hidden_input_dir": self.hidden_input_dir,
                "reference_outputs_dir": self.reference_outputs_dir,
                "required_visit_files": sorted(REQUIRED_VISIT_FILES),
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = HstAcsWfcVisitReductionConfig()


@cb.tasks_config(split="train")
def load():
    cfg = HstAcsWfcVisitReductionConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _write_tree_from_mapping(root: Path, files: dict[str, bytes]) -> None:
    for rel_path, payload in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


async def _pull_tree(session: cb.DesktopSession, remote_root: str) -> dict[str, bytes]:
    result = await session.run_command(
        f"find {_shell_quote(remote_root)} -type f -printf '%P\\n'",
        check=False,
    )
    if result.get("return_code") not in (0, 1):
        raise RuntimeError(f"could not list remote tree {remote_root}: {result}")
    files: dict[str, bytes] = {}
    for rel_path in result.get("stdout", "").splitlines():
        clean = rel_path.strip()
        if clean:
            files[clean] = await session.read_bytes(f"{remote_root}/{clean}")
    return files


async def _run_candidate(session: cb.DesktopSession, meta: dict[str, Any]) -> str:
    candidate = meta["candidate_script"]
    if not await session.exists(candidate):
        raise RuntimeError(f"missing required candidate script: {candidate}")

    run_dir = f"{EVAL_TMP_DIR}/candidate_run"
    input_dir = f"{run_dir}/combined_input"
    generated_dir = f"{run_dir}/generated_output"
    command = "\n".join(
        [
            "set -euo pipefail",
            f"rm -rf {_shell_quote(run_dir)}",
            f"mkdir -p {_shell_quote(input_dir)} {_shell_quote(generated_dir)}",
            f"cp -a {_shell_quote(meta['visible_visit_dir'])} {_shell_quote(input_dir)}/",
            (
                f"if [ -d {_shell_quote(meta['hidden_input_dir'])} ]; "
                f"then cp -a {_shell_quote(meta['hidden_input_dir'])}/* {_shell_quote(input_dir)}/; fi"
            ),
            (
                f"{_shell_quote(meta['python_wrapper'])} {_shell_quote(candidate)} "
                f"--input {_shell_quote(input_dir)} --output {_shell_quote(generated_dir)}"
            ),
        ]
    )
    result = await session.run_command(f"bash -lc {_shell_quote(command)}", check=False)
    if result.get("return_code") != 0:
        raise RuntimeError(
            "candidate reduction failed: "
            + str(result.get("stdout", ""))[-1000:]
            + str(result.get("stderr", ""))[-1000:]
        )
    return generated_dir


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_dir_name = Path(meta["remote_output_dir"]).name
    try:
        reference_files = await _pull_tree(session, meta["reference_outputs_dir"])
        if output_dir_name in STATIC_OUTPUT_DIRS:
            output_root = meta["remote_output_dir"]
        else:
            output_root = await _run_candidate(session, meta)
        output_files = await _pull_tree(session, output_root)
    except Exception as exc:
        logger.error("failed to collect evaluation assets: %s", exc)
        return [0.0]

    with tempfile.TemporaryDirectory(prefix="agenthle_hst_acs_eval_") as tmp:
        tmp_path = Path(tmp)
        local_output = tmp_path / "output"
        local_reference = tmp_path / "reference_outputs"
        _write_tree_from_mapping(local_output, output_files)
        _write_tree_from_mapping(local_reference, reference_files)

        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                str(SCRIPTS_DIR / "score_outputs.py"),
                "--output-dir",
                str(local_output),
                "--reference-dir",
                str(local_reference),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    if result.stderr:
        logger.info("local scorer stderr: %s", result.stderr.strip()[:2000])
    payload: dict[str, Any] | None = None
    try:
        candidate = json.loads(result.stdout)
        if isinstance(candidate, dict) and "score" in candidate:
            payload = candidate
    except json.JSONDecodeError:
        for line in reversed(result.stdout.strip().splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "score" in candidate:
                payload = candidate
                break
    if payload is None:
        logger.error(
            "could not parse scorer JSON; rc=%s stdout=%s", result.returncode, result.stdout[:2000]
        )
        return [0.0]

    try:
        score = float(payload["score"])
    except (TypeError, ValueError):
        logger.error("scorer score was not numeric: %r", payload.get("score"))
        return [0.0]
    logger.info("evaluation result: %s", json.dumps(payload, sort_keys=True))
    return [max(0.0, min(1.0, score))]


if __name__ == "__main__":
    for task in load():
        print(task.description)
