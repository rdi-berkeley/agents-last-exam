"""AgentHLE task: internal_employee_agent_instance_1."""

from __future__ import annotations

import json
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Workaround: cua_bench loads main.py via exec_module without registering
# in sys.modules, which causes @dataclass to fail. Register ourselves.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

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

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
PASS_THRESHOLD = 0.90
REMOTE_EVAL_TMP_DIR = "/tmp/agenthle_eval/internal_employee_agent_instance_1"


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

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
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
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "business_finance"
    TASK_NAME: str = "internal_employee_agent_instance_1"
    VARIANT_NAME: str = "base"
    OS_TYPE: str = "linux"

    def __init__(self) -> None:
        super().__init__(
            DOMAIN_NAME=self.DOMAIN_NAME,
            TASK_NAME=self.TASK_NAME,
            VARIANT_NAME=self.VARIANT_NAME,
            OS_TYPE=self.OS_TYPE,
            REMOTE_ROOT_DIR=os.environ.get("REMOTE_ROOT_DIR", "/media/user/data/agenthle"),
        )

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.task_dir}/output_test_neg"

    @property
    def visible_output_dir(self) -> str:
        return f"{self.task_dir}/output"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/results.json"

    @property
    def visible_output_file(self) -> str:
        return f"{self.visible_output_dir}/results.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Your Task
Act as the company's internal HR / IT assistant across all staged conversation sessions and write the final structured artifact to `{self.visible_output_file}`.

## Visible Inputs
- agent rules: `{self.input_dir}/agent_rules.md`
- HR knowledge base: `{self.input_dir}/hr_knowledge_base.md`
- IT knowledge base: `{self.input_dir}/it_knowledge_base.md`
- conversation sessions: `{self.input_dir}/queries.json`
- email / JIRA fallback stubs: `{self.input_dir}/stubs.py`
- deterministic web-search grounding for the `search_web` cases: `{self.input_dir}/web_search_grounding.json`

## What To Produce
Write exactly one JSON file at `{self.visible_output_file}` with this shape:

```json
{{
  "1.1": [{{"response": "...", "tools_used": []}}],
  "5.1": [
    {{"response": "...", "tools_used": ["draft_hr_email"]}},
    {{"response": "...", "tools_used": ["send_hr_email"]}}
  ]
}}
```

## Required Behavior
- process each test id in `queries.json` as one independent conversation session
- keep turn order within each test id
- keep session memory across turns inside the same test id
- use the exact canonical tool names from `agent_rules.md` in `tools_used`
- if you need a deterministic fallback for `search_web`, use the staged `web_search_grounding.json`
- you may use the staged `stubs.py` for local HR email / JIRA fallback behavior

## Important Constraints
- write only to `{self.visible_output_file}` inside the visible `output/` directory
- do not modify staged files under `input/`
- do not read or modify hidden evaluator directories
- keep the artifact deterministic and valid JSON
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "visible_output_dir": self.visible_output_dir,
                "remote_output_dir": self.remote_output_dir,
                "output_file": self.output_file,
                "visible_output_file": self.visible_output_file,
                "canonical_gcs_root": "gs://ale-data-all/business_finance/internal_employee_agent_instance_1/base/",
                "reference_gcs_prefix": (
                    "gs://ale-data-all/business_finance/internal_employee_agent_instance_1/base/reference"
                ),
            }
        )
        return metadata


config = TaskConfig()


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
    meta = task_cfg.metadata
    reference_dir = meta["reference_dir"]

    reference_paths = [
        reference_dir,
        f"{reference_dir}/test_suite.sh",
    ]
    for path in reference_paths:
        if not await session.exists(path):
            logger.error(
                "[%s] evaluator reference path missing at evaluate() time: %s", meta["task_name"], path
            )
            return [0.0]

    remote_results = meta["output_file"]
    if not await session.exists(remote_results):
        logger.error("[%s] results.json not found at %s", meta["task_name"], remote_results)
        return [0.0]

    await session.makedirs(REMOTE_EVAL_TMP_DIR)
    verify_script_path = f"{REMOTE_EVAL_TMP_DIR}/run_hidden_suite.py"
    await session.write_file(verify_script_path, _read_script("run_hidden_suite.py"))

    suite_path = f"{reference_dir}/test_suite.sh"
    command = (
        f"python {shlex.quote(verify_script_path)} "
        f"--results {shlex.quote(remote_results)} "
        f"--suite {shlex.quote(suite_path)} "
        f"--threshold {PASS_THRESHOLD}"
    )
    result = await _run_command(
        session,
        command,
        timeout=300.0,
        check=False,
    )

    try:
        payload = _parse_json_stdout(result.get("stdout", ""))
    except ValueError as exc:
        logger.error(
            "[%s] failed to parse VM verifier output: %s; stderr=%s",
            meta["task_name"],
            exc,
            (result.get("stderr", "") or "")[:1000],
        )
        return [0.0]

    score = float(payload.get("score", 0.0))
    logger.info(
        "[%s] hidden suite score=%.3f pass_rate=%s passed=%s failed=%s threshold=%s",
        meta["task_name"],
        score,
        payload.get("pass_rate"),
        payload.get("passed"),
        payload.get("failed"),
        payload.get("threshold"),
    )
    if result.get("return_code", 0) != 0:
        logger.error(
            "[%s] VM verifier exited non-zero (%s): %s",
            meta["task_name"],
            result.get("return_code", 0),
            (result.get("stderr", "") or "")[:1000],
        )
        return [0.0]
    return [score]
