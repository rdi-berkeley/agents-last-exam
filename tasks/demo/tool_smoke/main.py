"""Demo Task: Agent Tool Smoke Test (Linux).

Migrated from ``agenthle/tasks/demo/demo_tool_smoke_test_linux`` onto the
``LinuxTaskConfig`` base used by the new ``agents-last-exam`` framework.

The agent is asked to identify every available tool, exercise each one with the
*simplest concrete scenario that produces an observable result*, confirm the
tool actually worked by reading its returned content, and write a JSON report
summarizing which tools passed, failed, or were left untested. The evaluator
reads the report and scores: passed / total available tools identified.

This task is the fastest way to validate an agent's tool wiring — native or
external — before running real benchmark tasks. No GCS task data is staged
(``REQUIRES_TASK_DATA = False``); ``setup`` creates the output dir and clears
any stale report, and the data root is injected by the lifecycle.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)


TASK_PROMPT = """\
Goal: Test every tool that is available to you and produce a structured report.

Context: Ephemeral sandbox, no human in the loop, evaluation starts the moment
this CLI invocation returns. Default = test the tool. If a tool needs prior
session state (e.g., must be in plan mode, in a worktree, or have a running
background task), first bootstrap that state yourself, then test it. Side
effects, persistence, and "destructive" operations don't matter — the VM resets.

CRITICAL — what counts as a real test (read carefully):
A tool only counts as `passed` if you ACTUALLY EXERCISED it and OBSERVED its
return content proving it worked. Specifically, for each tool you must:
  (a) Construct the SIMPLEST concrete scenario that produces an observable,
      verifiable result. Do NOT make an empty, meaningless, or no-op call just
      to "touch" the tool. The input must be real enough that a working tool
      returns something you can check and a broken tool would not.
      Examples of the right shape:
        - a file-write tool: write a known string to a temp path, then read it
          back and confirm the bytes match;
        - a file-read tool: read a file you just created and confirm the content;
        - a shell/exec tool: run `echo <marker>` (or `printf`) and confirm the
          exact marker appears in stdout;
        - a list/search tool: list a directory you just populated and confirm
          the expected entry is present;
        - a screenshot/CUA tool: capture, then confirm you received non-empty
          image/coordinate data back.
  (b) Invoke the tool with that input.
  (c) READ / OBSERVE the tool's actual return content (stdout, file bytes,
      structured result, image, status object — whatever the tool returns).
  (d) Confirm the returned content matches the expected effect. Only then mark
      the tool `passed`.
If the call errors, returns nothing observable, or you cannot verify the result
against your expectation, mark it `failed` (with the error/observation) — never
mark a tool `passed` on a call whose result you did not actually observe and
verify.

Instructions:
1. Identify the tools you can actually see and call in this environment,
   including native tools, external tools, or MCP-provided tools if present.
   Do not invent tools, and do not include tools that are not available to you.
   IMPORTANT: if an MCP server is connected (e.g. CUA desktop tools, often
   named like `mcp__cua__*` / `cua_*` — screenshot, click, type, scroll, key,
   drag, cursor_position, mouse_move, ...), those ARE available tools: you MUST
   exercise them too and count them in your total. Do NOT skip or omit
   MCP-provided tools just because they come from an MCP server.
2. For each available tool, follow the (a)-(d) procedure above using the
   simplest safe scenario. Prefer read-only, list, status, metadata,
   validation, dry-run, or tiny local inputs — but the call must still produce
   an observable result you verify.
3. Each individual tool test should finish within 3 minutes because this task
   only checks basic tool availability.
4. If a tool's main purpose is interactive, persistent, long-running,
   destructive, irreversible, payment-related, or account-changing, do not run
   the real operation. Use a harmless dry-run, status, validation, or listing
   mode if the tool provides one — and still observe its return.
5. If a tool has no safe bounded way to produce an observable result, requires
   missing credentials or extra configuration, requires user interaction, or
   would block the task, record it in tools_untested with a short reason
   instead of calling it.
6. Record whether each tool you actually called succeeded or failed, based on
   whether you observed and verified its return content (see CRITICAL above).
7. After testing every available tool, write a JSON report to:
   {output_file}
   The report MUST have this exact structure:
   {{
     "tools_tested": ["tool_a", "tool_b", ...],
     "tools_passed": ["tool_a", ...],
     "tools_failed": {{"tool_b": "error message", ...}},
     "tools_untested": {{"tool_c": "reason not tested", ...}},
     "total": <number of available tools identified>,
     "tested": <number of tools actually called>,
     "passed": <number of called tools that passed>,
     "failed": <number of called tools that failed>,
     "untested": <number of tools not called>
   }}
   The "total", "tested", "passed", "failed", and "untested" fields are
   required integers. The "total" field must equal "tested" + "untested".
8. Exit normally when done."""


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "demo"
    TASK_NAME: str = "tool_smoke"
    VARIANT_NAME: str = "base"
    REQUIRES_TASK_DATA: bool = False

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/tool_report.json"

    @property
    def task_description(self) -> str:
        return TASK_PROMPT.format(output_file=self.output_file)

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m["output_file"] = self.output_file
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": cfg.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Create the output directory on the VM and remove any stale
    tool_report.json from a previous run (avoids re-scoring a previous
    agent's report on the persistent dev container)."""
    meta = task_cfg.metadata
    tag = meta.get("variant_name", "base")
    output_file = meta["output_file"]
    out_dir = output_file.rsplit("/", 1)[0]
    await session.run_command(f"mkdir -p {out_dir!r}", check=False)
    await session.run_command(f"rm -f {output_file!r}", check=False)
    logger.info("[%s] Output dir ready, stale report cleared: %s", tag, output_file)


def _score_report(report: dict) -> tuple[float, int, int, int, dict, dict]:
    """Extract score from the agent's tool report, tolerating minor format
    variations."""
    total = report.get("total", 0)
    tested = report.get("tested", 0)
    passed = report.get("passed", 0)
    tools_failed = report.get("tools_failed", {})
    tools_untested = report.get("tools_untested", {})
    untested = report.get("untested", 0)

    if isinstance(tools_untested, list):
        tools_untested = {name: "not tested" for name in tools_untested}

    if not untested and tools_untested:
        untested = len(tools_untested)

    if tested == 0:
        tested_list = report.get("tools_tested", [])
        if tested_list:
            tested = len(tested_list)

    if passed == 0:
        passed_list = report.get("tools_passed", [])
        if passed_list:
            passed = len(passed_list)

    if total == 0:
        total = tested + untested

    return (passed / total if total else 0.0), total, tested, passed, tools_failed, tools_untested


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Read tool_report.json and score = passed / total."""
    meta = task_cfg.metadata
    tag = meta.get("variant_name", "base")
    output_file = meta["output_file"]

    try:
        raw = await session.read_file(output_file)
        report = json.loads(raw.lstrip("﻿"))
    except Exception as exc:
        logger.error("[%s] Cannot read tool_report.json: %s", tag, exc)
        return [0.0]

    score, total, tested, passed, tools_failed, tools_untested = _score_report(report)

    if total == 0:
        logger.warning("[%s] No tools identified", tag)
        return [0.0]

    logger.info(
        "[%s] Tool smoke: %d/%d passed, %d tested (score=%.2f)",
        tag, passed, total, tested, score,
    )
    if tools_failed:
        logger.warning("[%s] Failed tools: %s", tag, tools_failed)
    if tools_untested:
        logger.warning("[%s] Untested tools: %s", tag, tools_untested)
    return [round(score, 3)]
