"""AgentHLE task: engineering/openroad_sky130_ibex_pnr_signoff."""

from __future__ import annotations

import asyncio
import json
import logging
import posixpath
import shlex
from dataclasses import dataclass
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

DOMAIN_NAME = "engineering"
TASK_NAME = "openroad_sky130_ibex_pnr_signoff"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
# Keep evaluator scratch off the nearly-full VM disks. The staged ORFS trees are
# a few hundred MB, so the Ubuntu host's tmpfs has enough headroom and avoids
# the persistent-space failures seen under /media/user/data and /.
EVAL_TMP_DIR = f"/dev/shm/agenthle_eval/{TASK_NAME}"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}
VERIFIER_POLL_INTERVAL_S = 20
VERIFIER_TIMEOUT_S = 7200


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


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


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: float | None = None,
    check: bool = False,
) -> dict[str, Any]:
    # The pinned RemoteDesktopSession.run_command() API does not accept a timeout kwarg.
    # Keep the parameter for call-site readability, but do not pretend it is enforced.
    del timeout
    return await session.run_command(command, check=check)


async def _read_remote_tail(
    session: cb.DesktopSession,
    path: str,
    *,
    lines: int = 200,
) -> str:
    result = await _run_command(
        session,
        "bash -lc " + json.dumps(f"tail -n {lines} {shlex.quote(path)} 2>/dev/null"),
        check=False,
    )
    return result.get("stdout", "") or ""


async def _run_verifier_background(
    session: cb.DesktopSession,
    command_parts: list[str],
    *,
    tag: str,
    run_dir: str,
) -> tuple[str, str]:
    stdout_log = f"{run_dir}/verifier.stdout"
    stderr_log = f"{run_dir}/verifier.stderr"
    rc_file = f"{run_dir}/verifier.rc"
    verifier_command = " ".join(command_parts)
    wrapped_command = (
        f"{verifier_command} > {shlex.quote(stdout_log)} 2> {shlex.quote(stderr_log)}; "
        f"rc=$?; printf '%s\\n' \"$rc\" > {shlex.quote(rc_file)}"
    )
    launch_command = "bash -lc " + json.dumps(
        f"nohup bash -lc {shlex.quote(wrapped_command)} >/dev/null 2>&1 & echo $!"
    )
    launch = await _run_command(session, launch_command, check=False)
    if launch.get("return_code") != 0:
        raise RuntimeError(
            "failed to launch verifier: "
            f"stdout={launch.get('stdout')} stderr={launch.get('stderr')}"
        )
    pid = (launch.get("stdout") or "").strip().splitlines()[-1] if launch.get("stdout") else ""
    logger.info("[%s] verifier started in background pid=%s run_dir=%s", tag, pid or "unknown", run_dir)

    waited = 0
    while waited < VERIFIER_TIMEOUT_S:
        await asyncio.sleep(VERIFIER_POLL_INTERVAL_S)
        waited += VERIFIER_POLL_INTERVAL_S
        done = await _run_command(
            session,
            "bash -lc " + json.dumps(f"test -f {shlex.quote(rc_file)} && echo DONE || echo RUNNING"),
            check=False,
        )
        if (done.get("stdout") or "").strip() == "DONE":
            rc_text = (
                await _run_command(
                    session,
                    "bash -lc " + json.dumps(f"cat {shlex.quote(rc_file)} 2>/dev/null"),
                    check=False,
                )
            ).get("stdout", "")
            stderr_tail = await _read_remote_tail(session, stderr_log, lines=200)
            if stderr_tail.strip():
                logger.info("[%s] verifier stderr tail: %s", tag, stderr_tail.strip()[:4000])
            logger.info(
                "[%s] verifier completed rc=%s after %ss; logs under %s",
                tag,
                (rc_text or "").strip() or "unknown",
                waited,
                run_dir,
            )
            return await _read_remote_tail(session, stdout_log, lines=400), stderr_tail

    if pid.isdigit():
        await _run_command(session, f"kill {pid} 2>/dev/null || true", check=False)
    raise TimeoutError(f"verifier timed out after {VERIFIER_TIMEOUT_S}s; logs under {run_dir}")


@dataclass
class OpenroadIbexConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def input_task_instructions(self) -> str:
        return f"{self.input_dir}/task_instructions.md"

    @property
    def starter_project_dir(self) -> str:
        return f"{self.input_dir}/starter_project"

    @property
    def starter_readme(self) -> str:
        return f"{self.starter_project_dir}/README.md"

    @property
    def prepare_workspace_script(self) -> str:
        return f"{self.software_dir}/prepare_workspace.sh"

    @property
    def output_config(self) -> str:
        return f"{self.output_dir}/config.mk"

    @property
    def output_journal(self) -> str:
        return f"{self.output_dir}/JOURNAL.md"

    @property
    def output_audit_dir(self) -> str:
        return f"{self.output_dir}/flow/logs/sky130hd/ibex/base"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on Linux on an RTL-to-GDSII signoff-closure task for lowRISC ibex on sky130hd.

## Task Root
- `{self.task_dir}`

## Visible Inputs
- Task brief: `{self.input_task_instructions}`
- Starter tree: `{self.starter_project_dir}`
- Workspace helper: `{self.prepare_workspace_script}`

## What You Should Do
1. Read `{self.input_task_instructions}`.
2. Create a writable workspace by running:
   `bash {self.prepare_workspace_script}`
3. Work only inside `{self.output_dir}/workspace`.
4. Tune only the allowed files described in the staged task brief.
5. When finished, copy the required deliverables into `{self.output_dir}`.

## Final Deliverables
- `{self.output_config}`
- `{self.output_journal}`
- `{self.output_audit_dir}/config.mk.pass*` if any completed pass exists
- `{self.output_audit_dir}/pass*.stamp` if any completed pass exists

Do not modify files under `{self.input_dir}`.
Do not rely on hidden evaluator-only data.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "input_task_instructions": self.input_task_instructions,
                "starter_project_dir": self.starter_project_dir,
                "starter_readme": self.starter_readme,
                "prepare_workspace_script": self.prepare_workspace_script,
                "reference_frozen_hashes": f"{self.reference_dir}/frozen_hashes.json",
                "reference_metrics": f"{self.reference_dir}/reference_metrics.json",
                "reference_starter_zip": f"{self.reference_dir}/starter_project.zip",
                "output_config": self.output_config,
                "output_journal": self.output_journal,
                "output_audit_dir": self.output_audit_dir,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = OpenroadIbexConfig()


@cb.tasks_config(split="train")
def load():
    cfg = OpenroadIbexConfig()
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
    tag = meta["variant_name"]
    required_reference = [
        meta["reference_dir"],
        meta["reference_frozen_hashes"],
        meta["reference_metrics"],
        meta["reference_starter_zip"],
    ]
    missing_reference = [path for path in required_reference if not (await session.file_exists(path) or await session.directory_exists(path))]
    if missing_reference:
        logger.error("[%s] missing evaluator reference paths: %s", tag, missing_reference)
        return [0.0]

    if not (await session.file_exists(meta["output_dir"]) or await session.directory_exists(meta["output_dir"])):
        logger.error("[%s] missing submission directory: %s", tag, meta["output_dir"])
        return [0.0]

    await session.interface.create_dir(EVAL_TMP_DIR)
    run_dir_result = await _run_command(
        session,
        "bash -lc " + json.dumps(f"mktemp -d {shlex.quote(EVAL_TMP_DIR)}/run_XXXXXX"),
        check=False,
    )
    if run_dir_result.get("return_code") != 0:
        logger.error(
            "[%s] failed to create verifier run dir: stdout=%s stderr=%s",
            tag,
            run_dir_result.get("stdout", ""),
            run_dir_result.get("stderr", ""),
        )
        return [0.0]
    run_dir = (run_dir_result.get("stdout") or "").strip().splitlines()[-1]
    if not run_dir:
        logger.error("[%s] failed to create verifier run dir: empty mktemp output", tag)
        return [0.0]

    verifier_path = f"{run_dir}/verify_submission.py"
    await session.write_file(verifier_path, _read_script("verify_submission.py"))

    extra_args: list[str] = []
    if posixpath.basename(meta["output_dir"].rstrip("/")) != "output":
        extra_args.append("--skip-reseed")

    command_parts = [
        "python",
        shlex.quote(verifier_path),
        "--submission-dir",
        shlex.quote(meta["output_dir"]),
        "--reference-dir",
        shlex.quote(meta["reference_dir"]),
        "--work-dir",
        shlex.quote(run_dir),
    ]
    command_parts.extend(extra_args)

    try:
        stdout, stderr = await _run_verifier_background(
            session,
            command_parts,
            tag=tag,
            run_dir=run_dir,
        )
    except Exception as exc:
        logger.error("[%s] verifier execution failed: %s", tag, exc)
        return [0.0]

    try:
        payload = _parse_json_stdout(stdout)
    except Exception as exc:
        logger.error(
            "[%s] failed to parse verifier JSON: %s stdout=%s stderr=%s",
            tag,
            exc,
            stdout[:2000],
            stderr[:2000],
        )
        return [0.0]

    normalized_score = float(payload.get("normalized_score", 0.0))
    total_score = float(payload.get("total_score", 0.0))
    passed = bool(payload.get("passed", False))
    logger.info(
        "[%s] normalized_score=%.4f total_score=%.1f passed=%s",
        tag,
        normalized_score,
        total_score,
        passed,
    )
    return [max(0.0, min(1.0, normalized_score))]
