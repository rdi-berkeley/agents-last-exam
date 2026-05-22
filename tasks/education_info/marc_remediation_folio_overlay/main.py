"""AgentHLE task: MARC remediation and FOLIO overlay decisioning."""

from __future__ import annotations

import json
import logging
import os
import posixpath
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local import fallback only

    class _FallbackTask:
        def __init__(self, description, metadata, computer):
            self.description = description
            self.metadata = metadata
            self.computer = computer

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    cb = SimpleNamespace(
        Task=_FallbackTask,
        DesktopSession=object,
        tasks_config=_identity_decorator,
        setup_task=_identity_decorator,
        evaluate_task=_identity_decorator,
    )

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup  # noqa: E402
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402

_setup = BaseTaskSetup()


logger = logging.getLogger(__name__)

DOMAIN_NAME = "education_info"
TASK_NAME = "marc_remediation_folio_overlay"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}
FIXTURE_OUTPUT_DIRS = {"output_test_pos", "output_test_neg"}
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/")).strip("/")
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: " + ", ".join(sorted(ALLOWED_OUTPUT_DIRS))
        )
    return normalized


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    check: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, check=check, timeout=timeout)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


class MarcRemediationFolioOverlayConfig(LinuxTaskConfig):
    def __init__(self, remote_output_dir: str = "output") -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=remote_output_dir,
        )

    @property
    def output_dir_name(self) -> str:
        return _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def starter_project_dir(self) -> str:
        return f"{self.input_dir}/starter_project"

    @property
    def task_spec_file(self) -> str:
        return f"{self.input_dir}/task_spec.md"

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/TASK_PROMPT.md"

    @property
    def public_case_dir(self) -> str:
        return f"{self.starter_project_dir}/input/public_case"

    @property
    def candidate_submission_dir(self) -> str:
        return f"{self.remote_output_dir}/submission"

    @property
    def candidate_public_outputs_dir(self) -> str:
        return f"{self.candidate_submission_dir}/outputs/public_case"

    @property
    def evaluator_dir(self) -> str:
        return f"{self.reference_dir}/evaluator"

    @property
    def evaluator_script(self) -> str:
        return f"{self.evaluator_dir}/evaluate.py"

    @property
    def software_readme(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def python_entry_point(self) -> str:
        return f"{self.software_dir}/python3.12"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to complete a MARC remediation and FOLIO overlay starter project.

## Task Directory
`{self.task_dir}`

## Visible Inputs
- Full prompt: `{self.task_prompt_file}`
- Harness summary: `{self.task_spec_file}`
- Editable starter project: `{self.starter_project_dir}`
- Visible public case: `{self.public_case_dir}`
- Software note: `{self.software_readme}`
- Python 3.12 entry point: `{self.python_entry_point}`

## Your Task
1. Copy `{self.starter_project_dir}` to `{self.candidate_submission_dir}`.
2. Implement `scripts/remediate_catalog.py` in the copied project while preserving this CLI:
   `python scripts/remediate_catalog.py --case-dir <case_dir> --output-dir <output_dir>`
3. Use the case files and policy rules to remediate MARC records, choose FOLIO overlay actions, and emit all required output artifacts.
4. Run the CLI on the visible public case before finishing:
   `{self.python_entry_point} scripts/remediate_catalog.py --case-dir input/public_case --output-dir outputs/public_case`

## Final Deliverable
Leave the completed project at:

`{self.candidate_submission_dir}`

Do not modify files under `{self.input_dir}`. Write final work only under `{self.remote_output_dir}`.
The benchmark harness exposes only the intended input/software surface and the writable output directory while you solve; evaluator-only reference data is not part of the solve-time task.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "output_dir_name": self.output_dir_name,
                "starter_project_dir": self.starter_project_dir,
                "task_spec_file": self.task_spec_file,
                "task_prompt_file": self.task_prompt_file,
                "public_case_dir": self.public_case_dir,
                "candidate_submission_dir": self.candidate_submission_dir,
                "candidate_public_outputs_dir": self.candidate_public_outputs_dir,
                "evaluator_dir": self.evaluator_dir,
                "evaluator_script": self.evaluator_script,
                "software_readme": self.software_readme,
                "python_entry_point": self.python_entry_point,
                "eval_tmp_dir": EVAL_TMP_DIR,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = MarcRemediationFolioOverlayConfig(
    remote_output_dir=os.environ.get("REMOTE_OUTPUT_DIR", "output")
)


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
    if not await session.exists(meta["candidate_submission_dir"]):
        logger.error(
            "[%s] missing candidate submission: %s", TASK_NAME, meta["candidate_submission_dir"]
        )
        return [0.0]

    if not await session.exists(meta["evaluator_script"]):
        raise RuntimeError(f"missing evaluator script: {meta['evaluator_script']}")

    verifier_path = f'{meta["eval_tmp_dir"]}/evaluate_submission.py'
    await session.makedirs(meta["eval_tmp_dir"])
    await session.write_file(
        verifier_path,
        (SCRIPTS_DIR / "evaluate_submission.py").read_text(encoding="utf-8"),
    )

    command = (
        "PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache "
        f'uv run --isolated --python python3.12 --no-project python "{verifier_path}" '
        f'--candidate-dir "{meta["candidate_submission_dir"]}" '
        f'--evaluator-dir "{meta["evaluator_dir"]}"'
    )
    result = await _run_command(session, command, check=False, timeout=120.0)
    stdout = str(result.get("stdout", "")).strip()
    stderr = str(result.get("stderr", "")).strip()
    if result.get("return_code", 0) != 0:
        logger.error(
            "[%s] evaluator launcher failed: stdout=%s stderr=%s",
            TASK_NAME,
            stdout[-4000:],
            stderr[-4000:],
        )
        return [0.0]

    try:
        report = json.loads(stdout)
    except json.JSONDecodeError:
        logger.error("[%s] evaluator returned non-JSON stdout: %s", TASK_NAME, stdout[-4000:])
        return [0.0]

    logger.info("[%s] evaluation report: %s", TASK_NAME, json.dumps(report, sort_keys=True))
    return [float(report.get("normalized_score", 0.0))]


if __name__ == "__main__":
    for task in load():
        print(task.description)
