"""AgentHLE task: ranking_node_feature_parity_recovery_instance_1."""

import json
import logging
import os
import shlex
from pathlib import Path, PurePosixPath
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

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "ranking_node_feature_parity_recovery_instance_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME="base"
LINUX_REMOTE_ROOT = "/media/user/data/agenthle"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"
WORKSPACE_ROOT = "/workspace"


def _remote_join(*parts: str) -> str:
    return str(PurePosixPath(*parts))


def _read_script(name: str) -> str:
    return (Path(__file__).resolve().parent / "scripts" / name).read_text(encoding="utf-8")


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
    raise ValueError(f"unable to parse verifier JSON from stdout: {text[:1000]}")


class RankingNodeFeatureParityRecoveryConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = "ranking_node_feature_parity_recovery_instance_1"
    VARIANT_NAME: str = "base"
    OS_TYPE: str = "linux"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_ROOT_DIR=kwargs.pop("REMOTE_ROOT_DIR", os.environ.get("REMOTE_ROOT_DIR", LINUX_REMOTE_ROOT)),
            REMOTE_OUTPUT_DIR=kwargs.pop("REMOTE_OUTPUT_DIR", os.environ.get("REMOTE_OUTPUT_DIR", "output")),
        )

    @property
    def input_workspace_archive(self) -> str:
        return _remote_join(self.input_dir, "workspace_seed.tar.gz")

    @property
    def runtime_env_dir(self) -> str:
        return _remote_join(self.input_dir, "runtime_env")

    @property
    def instruction_file(self) -> str:
        return _remote_join(self.input_dir, "instruction.md")

    @property
    def output_test_pos_dir(self) -> str:
        return _remote_join(self.task_dir, "output_test_pos")

    @property
    def output_test_neg_dir(self) -> str:
        return _remote_join(self.task_dir, "output_test_neg")

    @property
    def candidate_file(self) -> str:
        return _remote_join(self.remote_output_dir, "safe_recover.py")

    @property
    def task_description(self) -> str:
        return f"""\
You are the on-call ML platform engineer for a Linux ranking-serving node.

The evaluator will prepare a writable task workspace at:
- `{WORKSPACE_ROOT}`

Start by reading:
- `{WORKSPACE_ROOT}/instruction.md`
- `{WORKSPACE_ROOT}/config.json`
- `{WORKSPACE_ROOT}/logs/service.log`

You must repair the node by implementing:
- `{WORKSPACE_ROOT}/safe_recover.py`

Your tool must:
1. Rebuild the serving feature manifest from authoritative evidence on disk.
2. Free enough space by deleting only safe debris inside `{WORKSPACE_ROOT}/cache` and `{WORKSPACE_ROOT}/trash`.
3. Produce these files:
   - `{WORKSPACE_ROOT}/cleanup_summary.json`
   - `{WORKSPACE_ROOT}/incident_report.md`
   - `{WORKSPACE_ROOT}/state/feature_manifest.json`

When you are done, copy your final `safe_recover.py` to the output directory:
- `cp {WORKSPACE_ROOT}/safe_recover.py {self.remote_output_dir}/safe_recover.py`

The evaluator will replay your script on a fresh workspace, so make sure it works from a clean state.

Important constraints:
- Do not touch `/protected`.
- Do not modify files outside the allowed outputs listed in the task instruction.
- Do not delete anything outside `{WORKSPACE_ROOT}/cache` and `{WORKSPACE_ROOT}/trash`.
- The staged task-local pytest runtime is available via `{WORKSPACE_ROOT}/.venv/bin/pytest`.

Notes:
- Runtime dependencies are staged task-locally; do not rely on installing packages globally.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "input_workspace_archive": self.input_workspace_archive,
                "runtime_env_dir": self.runtime_env_dir,
                "instruction_file": self.instruction_file,
                "software_dir": self.software_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "remote_output_dir": self.remote_output_dir,
                "candidate_file": self.candidate_file,
                "workspace_root": WORKSPACE_ROOT,
                "canonical_gcs_root": "gs://ale-data-all/computing_math/ranking_node_feature_parity_recovery_instance_1/base/",
            }
        )
        return metadata


config = RankingNodeFeatureParityRecoveryConfig()


@cb.tasks_config(split="train")
def load():
    cfg = RankingNodeFeatureParityRecoveryConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


class _RankingNodeSetup(BaseTaskSetup):
    """Per-run /workspace rebuild from the staged archive.

    Shape B: each run must rebuild `/workspace` from the input archive
    because the agent mutates it during the run (writes safe_recover.py,
    cleans cache/trash, produces report files). Stage 1 staging cannot
    keep `/workspace` pristine across runs.
    """

    async def setup(self, task_cfg, session: cb.DesktopSession) -> None:
        meta = task_cfg.metadata
        required_paths = [
            meta["task_dir"],
            meta["input_dir"],
            meta["input_workspace_archive"],
            meta["runtime_env_dir"],
            meta["instruction_file"],
            meta["software_dir"],
            _remote_join(meta["runtime_env_dir"], "pyproject.toml"),
            _remote_join(meta["runtime_env_dir"], "uv.lock"),
        ]
        missing = [path for path in required_paths if not await session.exists(path)]
        if missing:
            raise RuntimeError("missing staged paths: " + "; ".join(missing))

        await session.makedirs(meta["remote_output_dir"])

        tool_check = await _run_command(
            session,
            "bash -lc 'command -v python3 >/dev/null && command -v uv >/dev/null && command -v sudo >/dev/null'",
            check=False,
        )
        if tool_check.get("return_code") != 0:
            raise RuntimeError("missing required Linux tools: python3, uv, or sudo")

        sudo_check = await _run_command(session, "bash -lc 'sudo -n true'", check=False)
        if sudo_check.get("return_code") != 0:
            raise RuntimeError("benchmark user needs passwordless sudo to stage /workspace")

        prepare_script_path = f"{EVAL_TMP_DIR}/prepare_workspace.py"
        await session.makedirs(EVAL_TMP_DIR)
        await session.write_file(prepare_script_path, _read_script("prepare_workspace.py"))
        prep_command = " ".join(
            [
                "python3",
                shlex.quote(prepare_script_path),
                "--input-workspace",
                shlex.quote(meta["input_workspace_archive"]),
                "--instruction-file",
                shlex.quote(meta["instruction_file"]),
                "--runtime-env-dir",
                shlex.quote(meta["runtime_env_dir"]),
            ]
        )
        prep_result = await _run_command(session, prep_command, check=False, timeout=240.0)
        if prep_result.get("return_code") != 0:
            raise RuntimeError(
                "failed to prepare /workspace: "
                + (prep_result.get("stderr") or prep_result.get("stdout") or "").strip()
            )


_setup = _RankingNodeSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    await session.makedirs(EVAL_TMP_DIR)
    verify_script_path = f"{EVAL_TMP_DIR}/verify_safe_recover.py"
    await session.write_file(verify_script_path, _read_script("verify_safe_recover.py"))

    command = " ".join(
        [
            "python3",
            shlex.quote(verify_script_path),
            "--input-workspace",
            shlex.quote(meta["input_workspace_archive"]),
            "--instruction-file",
            shlex.quote(meta["instruction_file"]),
            "--runtime-env-dir",
            shlex.quote(meta["runtime_env_dir"]),
            "--reference-dir",
            shlex.quote(meta["reference_dir"]),
            "--remote-output-dir",
            shlex.quote(meta["remote_output_dir"]),
            "--workspace-root",
            shlex.quote(meta["workspace_root"]),
        ]
    )
    result = await _run_command(session, command, check=False, timeout=480.0)

    try:
        payload = _parse_json_stdout((result.get("stdout") or "") + "\n" + (result.get("stderr") or ""))
    except ValueError:
        logger.error("verifier stdout: %s", result.get("stdout", "")[:1000])
        logger.error("verifier stderr: %s", result.get("stderr", "")[:1000])
        return [0.0]

    if result.get("return_code", 0) != 0:
        logger.error("verifier command failed: %s", payload)
        return [0.0]

    score = float(payload.get("score", 0.0))
    logger.info("ranking parity verifier payload: %s", json.dumps(payload, ensure_ascii=False)[:2000])
    return [score]
