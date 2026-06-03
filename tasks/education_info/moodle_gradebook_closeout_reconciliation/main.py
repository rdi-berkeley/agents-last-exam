"""AgentHLE task: education_info/moodle_gradebook_closeout_reconciliation."""

from __future__ import annotations

import json
import logging
import os
import posixpath
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local fallback only

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

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "education_info"
TASK_NAME = "moodle_gradebook_closeout_reconciliation"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
VISIBLE_ROOT = f"{LinuxTaskConfig.REMOTE_ROOT_DIR}/{TASK_ID}/{VARIANT_NAME}"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}

REQUIRED_OUTPUT_FILES = [
    "corrected_course.mbz",
    "final_grade_export.csv",
    "final_grade_export.xml",
    "audit_report.csv",
    "audit_report.json",
    "exception_log.csv",
    "decisions.md",
    "oneroster_package/manifest.csv",
    "oneroster_package/users.csv",
    "oneroster_package/classes.csv",
    "oneroster_package/enrollments.csv",
    "oneroster_package/lineItems.csv",
    "oneroster_package/results.csv",
]


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def _parse_json_text(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty JSON payload")
    return json.loads(text)


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: output, output_test_pos, output_test_neg"
        )
    return normalized


class MoodleGradebookCloseoutConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    def __init__(self, remote_output_dir: str | None = None) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=remote_output_dir or os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def output_dir_name(self) -> str:
        return _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def reference_dir(self) -> str:
        return super().reference_dir

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.task_dir}/output_test_neg"

    @property
    def remote_output_dir(self) -> str:
        if self.output_dir_name == "output_test_pos":
            return self.output_test_pos_dir
        if self.output_dir_name == "output_test_neg":
            return self.output_test_neg_dir
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def input_task_prompt(self) -> str:
        return f"{self.input_dir}/TASK_PROMPT.md"

    @property
    def input_bundle_readme(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def input_bundle_lib(self) -> str:
        return f"{self.input_dir}/bundle_lib.py"

    @property
    def starter_project_dir(self) -> str:
        return f"{self.input_dir}/starter_project"

    @property
    def starter_backup(self) -> str:
        return f"{self.starter_project_dir}/course_backup.mbz"

    @property
    def starter_roster(self) -> str:
        return f"{self.starter_project_dir}/roster.csv"

    @property
    def starter_rebuild_tool(self) -> str:
        return f"{self.starter_project_dir}/tools/rebuild_exports.py"

    @property
    def runtime_manifest(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def runtime_lock(self) -> str:
        return f"{self.input_dir}/runtime_env/uv.lock"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_with_task_deps.sh"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on Linux on an offline Moodle gradebook closeout task.

## Task Root
- `{self.task_dir}`

## Visible Inputs
- Task prompt: `{self.input_task_prompt}`
- Bundle README: `{self.input_bundle_readme}`
- Shared helper library: `{self.input_bundle_lib}`
- Starter project: `{self.starter_project_dir}`
- Starter backup: `{self.starter_backup}`
- Roster snapshot: `{self.starter_roster}`
- Rebuild helper: `{self.starter_rebuild_tool}`
- Python wrapper: `{self.python_wrapper}`

## What You Should Do
1. Read `{self.input_task_prompt}` and the starter materials under `{self.starter_project_dir}`.
2. Repair the broken Moodle-style backup in `{self.starter_backup}` by changing only the benchmark-designated editable backup files.
3. Rebuild the registrar and OneRoster outputs using the staged helper tools.
4. Write the required deliverables under `{self.remote_output_dir}`.

## Final Deliverables
- `{self.remote_output_dir}/corrected_course.mbz`
- `{self.remote_output_dir}/final_grade_export.csv`
- `{self.remote_output_dir}/final_grade_export.xml`
- `{self.remote_output_dir}/audit_report.csv`
- `{self.remote_output_dir}/audit_report.json`
- `{self.remote_output_dir}/exception_log.csv`
- `{self.remote_output_dir}/decisions.md`
- `{self.remote_output_dir}/oneroster_package/manifest.csv`
- `{self.remote_output_dir}/oneroster_package/users.csv`
- `{self.remote_output_dir}/oneroster_package/classes.csv`
- `{self.remote_output_dir}/oneroster_package/enrollments.csv`
- `{self.remote_output_dir}/oneroster_package/lineItems.csv`
- `{self.remote_output_dir}/oneroster_package/results.csv`

Use `{self.python_wrapper}` if you want the pinned Python runtime for the helper scripts.
Do not modify files under `{self.input_dir}`.
Do not rely on files outside the visible task root listed above during solve time.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.pop("reference_dir", None)
        metadata.update(
            {
                "task_id": TASK_ID,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "eval_tmp_dir": EVAL_TMP_DIR,
                "output_dir_name": self.output_dir_name,
                "input_task_prompt": self.input_task_prompt,
                "input_bundle_readme": self.input_bundle_readme,
                "input_bundle_lib": self.input_bundle_lib,
                "starter_project_dir": self.starter_project_dir,
                "starter_backup": self.starter_backup,
                "starter_roster": self.starter_roster,
                "starter_rebuild_tool": self.starter_rebuild_tool,
                "runtime_manifest": self.runtime_manifest,
                "runtime_lock": self.runtime_lock,
                "python_wrapper": self.python_wrapper,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/",
                "reference_gcs_prefix": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/reference",
            }
        )
        return metadata


config = MoodleGradebookCloseoutConfig(
    remote_output_dir=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
)


@cb.tasks_config(split="train")
def load():
    cfg = MoodleGradebookCloseoutConfig(
        remote_output_dir=os.environ.get("REMOTE_OUTPUT_DIR", "output")
    )
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    await session.interface.create_dir(meta["eval_tmp_dir"])
    try:
        tag = meta["output_dir_name"]
        score_script = f'{meta["eval_tmp_dir"]}/score_submission_{tag}.py'
        bundle_script = f'{meta["eval_tmp_dir"]}/bundle_lib_{tag}.py'
        await session.write_file(score_script, _read_script("score_outputs.py"))
        await session.write_file(bundle_script, _read_script("bundle_lib.py"))
        await session.run_command(
            f'cp "{bundle_script}" "{meta["eval_tmp_dir"]}/bundle_lib.py"',
            check=False,
        )

        reference_check = await session.run_command(
            f'test -f "{meta["reference_dir"]}/reference_contract.json"',
            check=False,
        )
        reference_rc = (
            reference_check.get("return_code", 1) if isinstance(reference_check, dict) else 1
        )
        if reference_rc != 0:
            logger.error(
                "[%s] hidden reference is not staged at %s; eval-stage must stage %s first",
                TASK_ID,
                meta["reference_dir"],
                meta.get("reference_gcs_prefix"),
            )
            return [0.0]

        command = (
            f'cd "{meta["eval_tmp_dir"]}" && '
            f'UV_CACHE_DIR="{meta["eval_tmp_dir"]}/runtime_{tag}/.uv-cache" '
            f'UV_PROJECT_ENVIRONMENT="{meta["eval_tmp_dir"]}/runtime_{tag}/.venv" '
            f'uv run --isolated --no-project --with pandas==2.2.3 --python "/usr/bin/python" '
            f'python "{score_script}" '
            f'--submission "{meta["remote_output_dir"]}" '
            f'--ground-truth "{meta["reference_dir"]}"'
        )
        result = await session.run_command(command, check=False)

        stdout = result.get("stdout", "") if isinstance(result, dict) else ""
        stderr = result.get("stderr", "") if isinstance(result, dict) else ""
        rc = result.get("return_code", 1) if isinstance(result, dict) else 1
        if stderr:
            logger.info("[%s] scorer stderr: %s", TASK_ID, stderr.strip()[:4000])
        if rc != 0:
            logger.error("[%s] scorer failed rc=%s stdout=%s", TASK_ID, rc, stdout[:4000])
            return [0.0]

        try:
            payload = _parse_json_text(stdout)
        except Exception as exc:
            logger.error("[%s] failed to parse score report: %s", TASK_ID, exc)
            return [0.0]

        raw_score = 0.0 if payload.get("hard_failed") else float(payload.get("score", 0.0))
        normalized_score = raw_score / 100.0
        logger.info(
            "[%s] raw_score=%.2f normalized_score=%.4f passed=%s",
            TASK_ID,
            raw_score,
            normalized_score,
            payload.get("passed"),
        )
        return [normalized_score]
    finally:
        await session.run_command(f'rm -rf "{meta["eval_tmp_dir"]}"', check=False)
