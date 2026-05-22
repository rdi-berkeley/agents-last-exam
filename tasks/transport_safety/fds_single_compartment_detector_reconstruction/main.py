"""AgentHLE task: FDS single-compartment detector reconstruction."""

import json
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "transport_safety"
TASK_NAME = "fds_single_compartment_detector_reconstruction"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


async def _missing(session: cb.DesktopSession, path: str, *, label: str) -> bool:
    if await session.exists(path):
        return False
    logger.error("Missing %s: %s", label, path)
    return True


@dataclass
class FDSDetectorReconstructionConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")

    @property
    def project_dir(self) -> str:
        return f"{self.input_dir}/project"

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/TASK_PROMPT.md"

    @property
    def project_prompt_file(self) -> str:
        return f"{self.project_dir}/TASK_PROMPT.md"

    @property
    def visible_input_dir(self) -> str:
        return f"{self.project_dir}/input/visible"

    @property
    def starter_cli(self) -> str:
        return f"{self.project_dir}/reconstruct_fire_case.py"

    @property
    def output_cli(self) -> str:
        return f"{self.remote_output_dir}/reconstruct_fire_case.py"

    @property
    def reference_evaluator_dir(self) -> str:
        return f"{self.reference_dir}/evaluator_reference"

    @property
    def reference_evaluator(self) -> str:
        return f"{self.reference_evaluator_dir}/evaluate.py"

    @property
    def task_description(self) -> str:
        return f"""\
You are acting as a senior fire protection engineer reconstructing a synthetic single-compartment detector-response incident on a Linux VM.

## Visible Starter Project
- Starter project directory: `{self.project_dir}`
- Task prompt: `{self.task_prompt_file}`
- Visible scenario input: `{self.visible_input_dir}`
- Starter CLI: `{self.starter_cli}`

## Your Task
Complete the reconstruction CLI so it can run with this exact contract:

```bash
python reconstruct_fire_case.py --input-dir <scenario_input_dir> --output-dir <output_dir>
```

Use the visible project files and the documented reconstruction contract. Your implementation must generalize to scenario directories with the same schema; do not hard-code the visible case ID or output values.

## Required Final Submission
Write your completed submission package to:

`{self.remote_output_dir}`

At minimum, `{self.output_cli}` must exist. The evaluator will run that script against scenario input directories and check the generated FDS/Smokeview artifacts.

## Required CLI Outputs
For each scenario, your CLI must write these files into the CLI `--output-dir`:
- `completed_case.fds`
- `device_manifest.json`
- `hrr_reconstruction.csv`
- `detector_activation.csv`
- `tenability_summary.csv`
- `grid_sensitivity.csv`
- `engineering_memo.md`

## Important Rules
- Do not modify files under `{self.input_dir}`.
- Use `{self.remote_output_dir}` only for your final submitted files.
- FDS/Smokeview are available at `{self.software_dir}/fds` and `{self.software_dir}/smokeview` for optional professional validation, but the submitted CLI must run offline with Python standard-library code.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "project_dir": self.project_dir,
                "task_prompt_file": self.task_prompt_file,
                "project_prompt_file": self.project_prompt_file,
                "visible_input_dir": self.visible_input_dir,
                "starter_cli": self.starter_cli,
                "output_cli": self.output_cli,
                "reference_evaluator_dir": self.reference_evaluator_dir,
                "reference_evaluator": self.reference_evaluator,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = FDSDetectorReconstructionConfig()


@cb.tasks_config(split="train")
def load():
    cfg = FDSDetectorReconstructionConfig(
        REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output")
    )
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


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    for key, label in [
        ("remote_output_dir", "submission output directory"),
        ("output_cli", "submitted reconstruct_fire_case.py"),
        ("reference_evaluator", "hidden evaluator script"),
        ("reference_evaluator_dir", "hidden evaluator directory"),
    ]:
        if not await session.exists(meta[key]):
            logger.error("Missing %s at %s", label, meta[key])
            return [0.0]

    await session.makedirs(EVAL_TMP_DIR)
    verifier_path = f"{EVAL_TMP_DIR}/evaluate_submission_safe.py"
    await session.write_file(verifier_path, _read_script("evaluate_submission_safe.py"))

    cmd = " ".join(
        [
            "python",
            shlex.quote(verifier_path),
            "--submission-dir",
            shlex.quote(meta["remote_output_dir"]),
            "--reference-dir",
            shlex.quote(meta["reference_evaluator_dir"]),
            "--work-dir",
            shlex.quote(EVAL_TMP_DIR),
        ]
    )
    result = await session.run_command("bash -lc " + json.dumps(cmd), check=False)
    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    if not stdout:
        logger.error("Evaluator produced empty stdout; stderr=%s", stderr)
        return [0.0]

    try:
        report = json.loads(stdout)
    except json.JSONDecodeError:
        logger.exception("Evaluator returned non-JSON stdout=%s stderr=%s", stdout, stderr)
        return [0.0]

    try:
        raw_score = float(report.get("score", 0.0))
    except (TypeError, ValueError):
        raw_score = 0.0

    await session.write_file(
        f"{EVAL_TMP_DIR}/autograde_report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    score = max(0.0, min(1.0, raw_score / 100.0))
    logger.info(
        "[%s] raw_score=%.3f normalized=%.3f pass=%s", TASK_ID, raw_score, score, report.get("pass")
    )
    return [score]


if __name__ == "__main__":  # pragma: no cover
    for task in load():
        print(task.description)
