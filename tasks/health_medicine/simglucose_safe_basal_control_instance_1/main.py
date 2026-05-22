"""AgentHLE task: health_medicine/simglucose_safe_basal_control_instance_1."""

import json
import logging
import os
import shlex
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Optional

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
from tasks.linux_runtime import DATA_ROOT, LinuxTaskConfig


_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_hidden_summary import ScoreResult, score_hidden_summary  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "simglucose_safe_basal_control_instance_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
LINUX_REMOTE_ROOT = "/media/user/data/agenthle"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"


def _remote_join(*parts: str) -> str:
    return str(PurePosixPath(*parts))


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    check: bool = False,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, check=check, timeout=timeout)
        return await session.run_command(command, check=check)
    except TypeError:
        if timeout is not None:
            return await session.run_command(command, check=check)
        return await session.run_command(command, check=check)


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


async def _run_hidden_eval(session: cb.DesktopSession, meta: dict[str, Any]) -> dict[str, Any]:
    shell_script = f"""\
set -euo pipefail
cd {shlex.quote(meta["task_dir"])}
mkdir -p {shlex.quote(EVAL_TMP_DIR)} {shlex.quote(meta["eval_runtime_dir"])}
cp {shlex.quote(meta["runtime_pyproject"])} {shlex.quote(_remote_join(meta["eval_runtime_dir"], "pyproject.toml"))}
cp {shlex.quote(meta["runtime_lock"])} {shlex.quote(_remote_join(meta["eval_runtime_dir"], "uv.lock"))}
rm -f {shlex.quote(meta["hidden_summary_path"])} {shlex.quote(meta["eval_log_path"])}
PYTHONPATH=input:reference uv run --project {shlex.quote(meta["eval_runtime_dir"])} python {shlex.quote(meta["hidden_evaluator"])} --submission-dir {shlex.quote(meta["submission_dir"])} --output {shlex.quote(meta["hidden_summary_path"])} > {shlex.quote(meta["eval_log_path"])} 2>&1
cat {shlex.quote(meta["hidden_summary_path"])}
"""
    result = await _run_command(
        session,
        "bash -lc " + shlex.quote(shell_script),
        check=False,
        timeout=2400.0,
    )
    if result.get("return_code", 0) != 0:
        raise RuntimeError(
            "hidden eval failed\n"
            f"stdout:\n{_as_text(result.get('stdout', ''))[-4000:]}\n"
            f"stderr:\n{_as_text(result.get('stderr', ''))[-4000:]}"
        )
    try:
        return json.loads(_as_text(result.get("stdout", "")))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "failed to parse hidden summary JSON from evaluator stdout\n"
            f"stdout:\n{_as_text(result.get('stdout', ''))[-4000:]}\n"
            f"stderr:\n{_as_text(result.get('stderr', ''))[-4000:]}"
        ) from exc


class SimGlucoseSafeBasalControlConfig(LinuxTaskConfig):
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
            REMOTE_ROOT_DIR=os.environ.get("REMOTE_ROOT_DIR", LINUX_REMOTE_ROOT),
            REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def public_dir(self) -> str:
        return _remote_join(self.input_dir, "public")

    @property
    def runtime_env_dir(self) -> str:
        return _remote_join(self.input_dir, "runtime_env")

    @property
    def output_test_pos_dir(self) -> str:
        return _remote_join(self.task_dir, "output_test_pos")

    @property
    def output_test_neg_dir(self) -> str:
        return _remote_join(self.task_dir, "output_test_neg")

    @property
    def submission_dir(self) -> str:
        return _remote_join(self.remote_output_dir, "submission")

    @property
    def controller_output(self) -> str:
        return _remote_join(self.submission_dir, "controller.py")

    @property
    def metadata_output(self) -> str:
        return _remote_join(self.submission_dir, "metadata.json")

    @property
    def report_output(self) -> str:
        return _remote_join(self.submission_dir, "report.md")

    @property
    def public_task_md(self) -> str:
        return _remote_join(self.public_dir, "task.md")

    @property
    def public_task_api(self) -> str:
        return _remote_join(self.public_dir, "task_api.py")

    @property
    def public_submission_format(self) -> str:
        return _remote_join(self.public_dir, "submission_format.md")

    @property
    def public_evaluator(self) -> str:
        return _remote_join(self.public_dir, "evaluate_public.py")

    @property
    def runtime_pyproject(self) -> str:
        return _remote_join(self.runtime_env_dir, "pyproject.toml")

    @property
    def runtime_lock(self) -> str:
        return _remote_join(self.runtime_env_dir, "uv.lock")

    @property
    def hidden_evaluator(self) -> str:
        return _remote_join(self.reference_dir, "private", "evaluate_hidden.py")

    @property
    def hidden_summary_path(self) -> str:
        return _remote_join(EVAL_TMP_DIR, f"{self.REMOTE_OUTPUT_DIR}_hidden_summary.json")

    @property
    def eval_runtime_dir(self) -> str:
        return _remote_join(EVAL_TMP_DIR, "runtime_env")

    @property
    def eval_status_path(self) -> str:
        return _remote_join(EVAL_TMP_DIR, f"{self.REMOTE_OUTPUT_DIR}_hidden_eval.status")

    @property
    def eval_log_path(self) -> str:
        return _remote_join(EVAL_TMP_DIR, f"{self.REMOTE_OUTPUT_DIR}_hidden_eval.log")

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to implement a meal-unannounced basal-only glucose controller in SimGlucose.

Visible task files:
- Task brief: `{self.public_task_md}`
- Controller API: `{self.public_task_api}`
- Submission format: `{self.public_submission_format}`
- Public evaluator: `{self.public_evaluator}`
- Agent runtime manifest: `{self.runtime_pyproject}`
- Agent runtime lockfile: `{self.runtime_lock}`

Write your submission under `{self.submission_dir}`.
Required files:
- `{self.controller_output}`
- `{self.metadata_output}`

Optional supporting files:
- `{self.report_output}`
- `{self.submission_dir}/assets/...`
- `{self.submission_dir}/training_code/...`

Rules:
- Follow the public observation/action contract exactly.
- The controller must expose either `build_controller()` or `SubmissionController`.
- Use only the staged task files as your source of truth.
- If you need Python dependencies for solving, install them yourself from `{self.runtime_env_dir}`.
- Do not modify files under `input/`, `reference/`, `output_test_pos/`, or `output_test_neg/`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "public_dir": self.public_dir,
                "runtime_env_dir": self.runtime_env_dir,
                "reference_dir": self.reference_dir,
                "reference_gcs_prefix": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/reference",
                "software_dir": self.software_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "remote_output_dir": self.remote_output_dir,
                "submission_dir": self.submission_dir,
                "controller_output": self.controller_output,
                "metadata_output": self.metadata_output,
                "report_output": self.report_output,
                "public_task_md": self.public_task_md,
                "public_task_api": self.public_task_api,
                "public_submission_format": self.public_submission_format,
                "public_evaluator": self.public_evaluator,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lock": self.runtime_lock,
                "hidden_evaluator": self.hidden_evaluator,
                "hidden_summary_path": self.hidden_summary_path,
                "eval_runtime_dir": self.eval_runtime_dir,
                "eval_status_path": self.eval_status_path,
                "eval_log_path": self.eval_log_path,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = SimGlucoseSafeBasalControlConfig()


@cb.tasks_config(split="train")
def load():
    cfg = SimGlucoseSafeBasalControlConfig()
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


def _score_reason(result: ScoreResult) -> str:
    return json.dumps(
        {
            "score": result.score,
            "passed": result.passed,
            "reason": result.reason,
            "formula": result.formula,
            "episodes": result.episodes,
            "mean_tir_70_180": result.mean_tir_70_180,
            "catastrophic_episode_count": result.catastrophic_episode_count,
            "completion_ratio": result.completion_ratio,
            "eligible_for_ranking": result.eligible_for_ranking,
        },
        ensure_ascii=True,
    )


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    submission_dir = meta["submission_dir"]
    controller_output = meta["controller_output"]
    metadata_output = meta["metadata_output"]

    if not await session.exists(submission_dir):
        if meta["remote_output_dir"].endswith("/output"):
            logger.info("submission directory not present yet under default output path: %s", submission_dir)
        else:
            logger.error("submission directory missing: %s", submission_dir)
        return [0.0]
    if not await session.exists(controller_output):
        if meta["remote_output_dir"].endswith("/output"):
            logger.info("controller.py not present yet under default output path: %s", controller_output)
        else:
            logger.error("controller.py missing: %s", controller_output)
        return [0.0]
    if not await session.exists(metadata_output):
        if meta["remote_output_dir"].endswith("/output"):
            logger.info("metadata.json not present yet under default output path: %s", metadata_output)
        else:
            logger.error("metadata.json missing: %s", metadata_output)
        return [0.0]

    try:
        payload = await _run_hidden_eval(session, meta)
    except Exception as exc:
        logger.error("failed to run hidden eval for %s on active session VM: %s", submission_dir, exc)
        return [0.0]

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        logger.error("hidden summary payload missing `summary`: %s", payload)
        return [0.0]

    scored = score_hidden_summary(summary)
    logger.info("hidden summary for %s: %s", submission_dir, json.dumps(summary, ensure_ascii=True))
    logger.info("proxy score for %s: %s", submission_dir, _score_reason(scored))
    return [scored.score]
