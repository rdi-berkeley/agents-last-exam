"""PromptBuilder — modular system prompt assembly for the OpenClaw agent harness.

Reference implementation: openclaw/src/agents/system-prompt.ts (buildAgentSystemPrompt)

Authoring rule for tool-specific prose:
    Non-obvious operational rules for a tool (polling guardrails, `target=`
    argument semantics, concurrency caps, patch-format rules, etc.) belong
    in a gated `_build_<tool>()` method here — NOT in AGENTS.md. AGENTS.md
    is injected verbatim into every prompt; putting tool-specific content
    there means a disabled tool's prose still reaches the model. A gated
    builder makes absence the signal: if `"<tool>" not in tool_summaries`,
    return `[]` and the section vanishes.

    Two layers:
      - `BaseTool.description` owns Layer 1 (one-line "what it does").
      - `_build_<tool>()` owns Layer 2 (non-obvious operational rules).

    Reference: openclaw/extensions/memory-core/src/prompt-section.ts for the
    subset-branching pattern; openclaw/src/agents/system-prompt.ts for the
    inline `if (availableTools.has(...))` gating pattern.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


BOOTSTRAP_MAX_CHARS = 12_000
"""Per-file cap for context-file injection.

Mirrors OpenClaw's DEFAULT_BOOTSTRAP_MAX_CHARS (pi-embedded-helpers/bootstrap.ts:86).
"""

BOOTSTRAP_TOTAL_MAX_CHARS = 60_000
"""Total cap across all injected context files.

Mirrors OpenClaw's DEFAULT_BOOTSTRAP_TOTAL_MAX_CHARS (pi-embedded-helpers/bootstrap.ts:87).
"""

_BOOTSTRAP_HEAD_RATIO = 0.7
_BOOTSTRAP_TAIL_RATIO = 0.2


def _trim_bootstrap_content(content: str, file_name: str, max_chars: int) -> str:
    """Trim a context file to ``max_chars`` using head/tail split with marker.

    Mirrors OpenClaw's trimBootstrapContent (pi-embedded-helpers/bootstrap.ts:126).
    """
    trimmed = content.rstrip()
    if max_chars <= 0:
        return ""
    if len(trimmed) <= max_chars:
        return trimmed

    head_chars = int(max_chars * _BOOTSTRAP_HEAD_RATIO)
    tail_chars = int(max_chars * _BOOTSTRAP_TAIL_RATIO)
    head = trimmed[:head_chars]
    tail = trimmed[-tail_chars:] if tail_chars > 0 else ""
    marker = (
        f"\n[...truncated, read {file_name} for full content...]\n"
        f"...(truncated {file_name}: kept {head_chars}+{tail_chars} chars "
        f"of {len(trimmed)})...\n"
    )
    return head + marker + tail


@dataclass
class ContextFile:
    """A file to inject into the Project Context section.

    Follows OpenClaw's contextFiles bootstrap injection pattern.
    """

    path: str  # Display label ("AGENTS.md", "TASK_MEMORY.md")
    content: str  # Full content to inject


@dataclass
class SectionConfig:
    """Toggle for an individual prompt section."""

    enabled: bool = True


@dataclass
class PromptConfig:
    """Configuration for which prompt sections to include."""

    identity: SectionConfig = field(default_factory=SectionConfig)
    time: SectionConfig = field(default_factory=SectionConfig)
    tools: SectionConfig = field(default_factory=SectionConfig)
    memory: SectionConfig = field(default_factory=SectionConfig)
    delegation: SectionConfig = field(default_factory=SectionConfig)
    project_context: SectionConfig = field(default_factory=SectionConfig)


class PromptBuilder:
    """Assembles structured system instructions from composable sections.

    Sections (in order, matching OpenClaw's system-prompt.ts):
      1. Identity — one-line agent role
      2. Tools — registered tool names with descriptions
      3. Memory Recall — when/how to use memory tools (only if memory tools present)
      4. Delegation — subagent delegation prose (only if delegation tools present)
      5. Current Date & Time — UTC timestamp (ref: OpenClaw system-prompt.ts)
      6. Project Context — bootstrap injection (AGENTS.md, TASK_MEMORY.md, etc.)

    The delegation section mirrors OpenClaw's buildAgentSystemPrompt pattern:
    absence is the signal — when a tool isn't available, its prose isn't
    emitted, and the model is not told "X is disabled."
    """

    def __init__(self, config: PromptConfig | None = None) -> None:
        self.config = config or PromptConfig()

    def build(
        self,
        *,
        tool_summaries: dict[str, str] | None = None,
        context_files: list[ContextFile] | None = None,
    ) -> str:
        """Assemble all enabled sections into a single prompt string.

        Args:
            tool_summaries: Name -> description mapping for registered tools.
                Drives the Tools section and the conditional inclusion of
                Memory Recall / Delegation subsections.
            context_files: Bootstrap files injected into the Project Context
                section (AGENTS.md, optionally TASK_MEMORY.md).
        """
        parts: list[str] = []

        if self.config.identity.enabled:
            parts.extend(self._build_identity())

        if self.config.tools.enabled and tool_summaries:
            parts.extend(self._build_tools(tool_summaries))
            exec_lines = self._build_exec(tool_summaries)
            if exec_lines:
                parts.extend(exec_lines)

        if self.config.memory.enabled and tool_summaries:
            memory_lines = self._build_memory(tool_summaries)
            if memory_lines:
                parts.extend(memory_lines)

        if self.config.delegation.enabled and tool_summaries:
            delegation_lines = self._build_delegation(tool_summaries)
            if delegation_lines:
                parts.extend(delegation_lines)

        if self.config.time.enabled:
            parts.extend(self._build_time())

        if self.config.project_context.enabled and context_files:
            parts.extend(self._build_project_context(context_files))

        return "\n".join(parts)

    def _build_identity(self) -> list[str]:
        """Build the Identity section."""
        return [
            "## Identity",
            "",
            (
                "You are an AI agent running inside the AgentHLE benchmark framework. "
                "Your role is to complete computer-use tasks on a remote Windows desktop "
                "by observing screenshots and performing mouse/keyboard actions."
            ),
            "",
        ]

    def _build_time(self) -> list[str]:
        """Build the Current Date & Time section.

        Mirrors OpenClaw's system prompt which injects the current UTC timestamp
        so the agent knows the date/time without needing a tool call.
        """
        now = datetime.now(timezone.utc)
        return [
            "## Current Date & Time",
            "",
            "- **Time zone:** UTC",
            f"- **Current:** {now.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]

    def _build_tools(self, tool_summaries: dict[str, str]) -> list[str]:
        """Build the Tools section listing registered tools.

        Tool order reflects caller's dict insertion order (Python 3.7+ stable).
        """
        lines = ["## Tools", "", "You have access to the following tools:", ""]
        for name, description in tool_summaries.items():
            lines.append(f"- **{name}**: {description}")
        lines.append("")
        return lines

    def _build_exec(self, tool_summaries: dict[str, str]) -> list[str]:
        """Build the Shell Execution section. Only included if ``exec`` is registered.

        Layer 2 operational prose for the exec tool — the
        tool-specific guardrails live in this gated builder, not AGENTS.md.
        Mirrors OpenClaw's ``describeExecTool`` prose (bash-tools.descriptions.ts)
        plus CUA-specific divergence notes (client-side timeout, cwd emulation,
        middle truncation).
        """
        if "exec" not in tool_summaries:
            return []
        gui_alternatives = "the `computer` tool"
        if "delegate_gui" in tool_summaries:
            gui_alternatives = "the `computer` tool or `delegate_gui`"
        return [
            "## Shell Execution",
            "",
            (
                "- `exec` runs a single non-GUI shell command inside the VM "
                "(cmd.exe on Windows, /bin/sh on POSIX) and returns "
                "stdout/stderr/exit_code. GUI apps launched via `exec` will "
                f"block the call until they exit — use {gui_alternatives} "
                "for GUI work."
            ),
            (
                "- Prefer one command per call. Do NOT build tight polling "
                "loops with `exec` — long-running or background work is not "
                "supported yet; use a single deterministic command instead."
            ),
            (
                "- `cwd` is emulated via a `cd` prefix and must resolve "
                "inside the task workspace (bounded by TASK_CATEGORY / TASK_TAG)."
            ),
            (
                "- `timeout` bounds the **client-side** wait only; on expiry "
                "the VM-side process may keep running. Keep timeouts tight "
                "(default 60s, max 300s)."
            ),
            (
                "- On Windows prefer direct executables (`dir`, `type`, "
                "`where`, `python3`). If you need PowerShell, write "
                "`powershell -NoProfile -Command \"...\"` explicitly — "
                "avoid wrapping in `cmd /c` or `& `."
            ),
            (
                "- Each of `stdout`/`stderr` is middle-truncated at ~200K "
                "chars; head + tail are preserved so exit/error lines stay "
                "visible. Use `read` for reviewing large file contents "
                "rather than `exec type ...`."
            ),
            "",
        ]

    def _build_memory(self, tool_summaries: dict[str, str]) -> list[str]:
        """Build the Memory Recall section. Only included if memory tools are present.

        Mirrors OpenClaw's memory-core/src/prompt-section.ts::buildPromptSection:
        each tool contributes its own behavioral line, gated on that tool being
        registered (absence-is-the-signal). Read guidance branches on the
        search/get subset; the unified `write` tool carries its own
        target='host' guidance so journaling rules live here rather than in
        AGENTS.md.
        """
        has_search = "memory_search" in tool_summaries
        has_get = "memory_get" in tool_summaries
        has_write = "write" in tool_summaries
        if not (has_search or has_get or has_write):
            return []

        lines: list[str] = ["## Memory Recall"]

        if has_search and has_get:
            lines.append(
                "Before acting on anything about prior attempts, strategies, "
                "environment observations, or task state: run memory_search on "
                "TASK_MEMORY.md + memory/session-*.md; then use memory_get to "
                "pull only the needed lines. If low confidence after search, "
                "say you checked."
            )
        elif has_search:
            lines.append(
                "Before acting on anything about prior attempts, strategies, "
                "environment observations, or task state: run memory_search on "
                "TASK_MEMORY.md + memory/session-*.md and answer from the "
                "matching results. If low confidence after search, say you checked."
            )
        elif has_get:
            lines.append(
                "Before acting on anything about prior attempts, strategies, "
                "environment observations, or task state that already points to "
                "a specific memory file or note: run memory_get to pull only the "
                "needed lines. If low confidence after reading them, say you checked."
            )

        if has_search or has_get:
            lines.append(
                "Citations: include Source: <path#line> when referencing memory snippets."
            )

        if has_write:
            lines.append(
                "Writing: use write with target='host' to journal memory. "
                "Append raw observations, actions, and errors to "
                "memory/session-NNN.md during the run. Update "
                "TASK_MEMORY.md with distilled strategies and patterns "
                "worth keeping across sessions."
            )

        lines.append("")
        return lines

    def _build_delegation(self, tool_summaries: dict[str, str]) -> list[str]:
        """Build the Delegation section.

        Emitted when any of delegate_general / delegate_gui / subagents are
        registered. Iterates the present subset and emits only the relevant
        subsections. Mirrors OpenClaw's absence-is-the-signal pattern: absent
        tools aren't described and aren't mentioned as "disabled."

        Migrated from the static openclaw/AGENTS.md Delegation section so the
        prose tracks the actual tool list rather than drifting when flags
        change (``disable_delegate_gui`` etc.).
        """
        has_general = "delegate_general" in tool_summaries
        has_gui = "delegate_gui" in tool_summaries
        has_subagents = "subagents" in tool_summaries
        if not (has_general or has_gui):
            return []

        lines: list[str] = [
            "## Delegation",
            "",
            (
                "You can delegate focused work to subagents when it helps — "
                "e.g. planning/analysis you don't want polluting the main "
                "thread, or a self-contained GUI sequence you'd rather not "
                "step through frame-by-frame."
            ),
            "",
        ]
        if has_general:
            lines.extend(
                [
                    "### `delegate_general(task, ...)` — async, auto-announces",
                    "",
                    (
                        "Spawns a general-purpose subagent session that has "
                        "**no VM access** — only memory tools and LLM "
                        "reasoning. Use for: synthesizing plans from what "
                        "you've observed; analyzing tricky text/content in "
                        "memory; deciding between multiple strategies. "
                        "Returns immediately with `{status: accepted, "
                        "run_id, note}`. Keep working — **do NOT poll**. "
                        "When the subagent finishes, its result is injected "
                        "automatically as a `[Subagent Result]` user message "
                        "on a later turn. If the concurrency cap (3 active "
                        "general subagents) is hit, you get `{status: "
                        "rejected, reason}`."
                    ),
                    "",
                ]
            )
        if has_gui:
            lines.extend(
                [
                    "### `delegate_gui(instruction, ...)` — async, auto-announces",
                    "",
                    (
                        "Spawns a GUI automation subagent driven by a vision "
                        "model. It takes over the VM for a bounded number of "
                        "steps (default 15) to perform a focused GUI "
                        "sequence — open an app, fill a form, click through "
                        "a wizard. Returns immediately with `{status: "
                        "accepted, run_id, note}`. Keep working on non-VM "
                        "tasks — **do NOT poll**. When the subagent "
                        "finishes, its result is injected as a `[Subagent "
                        "Result]` user message followed by a fresh VM "
                        "screenshot on a later turn. While the GUI subagent "
                        "is running, the VM is occupied — **do not call "
                        "`delegate_gui` again or use `computer` directly "
                        "until it completes**."
                    ),
                    "",
                ]
            )
        if has_subagents:
            steer_scope = "general or GUI" if has_gui else "general"
            lines.extend(
                [
                    "### `subagents(action=list | kill | steer, target=..., message=...)` — observability + control",
                    "",
                    (
                        "- `action=list` returns active (running/pending) and "
                        "recent (terminal) runs. **Do NOT poll** during "
                        "normal operation — results auto-announce. Use "
                        "`list` only if you suspect something is stuck."
                    ),
                    (
                        "- `action=kill` (with `target=<run_id>`) cancels a "
                        "runaway general subagent. The subagent transitions "
                        "to `killed` and no completion message will be "
                        "announced for that run."
                    ),
                    (
                        f"- `action=steer` (with `target` and `message`) "
                        f"sends a follow-up message into a **running "
                        f"subagent** ({steer_scope}) to refine or redirect "
                        f"its work mid-flight. The message is injected "
                        f"between the subagent's own turns. Target can be a "
                        f"run_id, label, run_id prefix, or `\"last\"`. Max "
                        f"4000 chars."
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                "### Rules of thumb",
                "",
                "- Don't delegate trivial things you can do in a single tool call.",
                (
                    "- Don't spawn a general subagent and then sit idle "
                    "waiting — keep making forward progress and the result "
                    "will arrive when it arrives."
                ),
                "- Don't nest delegation: subagents can't spawn further subagents.",
                "",
            ]
        )
        return lines

    def _build_project_context(self, context_files: list[ContextFile]) -> list[str]:
        """Build the Project Context section with injected file contents.

        Per-file and total char caps mirror OpenClaw's bootstrap budget
        (pi-embedded-helpers/bootstrap.ts:86-87).  Without these the system
        prompt grows linearly with TASK_MEMORY.md / AGENTS.md and re-inflates
        every turn, which was a primary driver of context bloat.

        Truncation strategy follows OpenClaw's head-70%/tail-20% split with an
        inline marker so the model knows content was elided.
        """
        if not context_files:
            return []

        lines = [
            "# Project Context",
            "",
            "The following project context files have been loaded:",
            "",
        ]
        remaining = BOOTSTRAP_TOTAL_MAX_CHARS
        for cf in context_files:
            per_file_budget = min(BOOTSTRAP_MAX_CHARS, remaining)
            content = _trim_bootstrap_content(cf.content, cf.path, per_file_budget)
            lines.append(f"### {cf.path}")
            lines.append("```")
            lines.append(content)
            lines.append("```")
            lines.append("")
            remaining -= len(content)
            if remaining <= 0:
                break
        return lines


# ---------------------------------------------------------------------------
# System-prompt composition report (observability) — merged from prompt_report.py.
# Measures total prompt size, splits project vs non-project context, and catalogs
# injected files and tool schemas. Duck-types tool objects to avoid importing BaseTool.
# ---------------------------------------------------------------------------

def build_system_prompt_report(
    *,
    system_prompt: str,
    context_files: list[Any] | None = None,
    tool_summaries: dict[str, str] | None = None,
    tools: list[Any] | None = None,
    source: str = "run",
) -> dict[str, Any]:
    """Build a report describing the system prompt composition.

    Measures total prompt size, splits project vs non-project context,
    catalogs injected files and tool schemas. Uses duck-typing for tool
    objects to avoid importing BaseTool.

    Based on OpenClaw's system prompt reporting for observability.
    """
    total_chars = len(system_prompt)

    # Split project vs non-project context at "# Project Context" header
    project_marker = "# Project Context"
    marker_pos = system_prompt.find(project_marker)
    if marker_pos >= 0:
        project_context_chars = total_chars - marker_pos
        non_project_context_chars = marker_pos
    else:
        project_context_chars = 0
        non_project_context_chars = total_chars

    # Injected files
    injected_files: list[dict[str, Any]] = []
    if context_files is not None:
        for cf in context_files:
            raw_content = getattr(cf, "content", "")
            raw_chars = len(raw_content) if raw_content else 0
            name = getattr(cf, "name", str(cf))

            # Measure how many chars actually appear in the prompt
            if raw_content and raw_content in system_prompt:
                injected_chars = len(raw_content)
            elif name in system_prompt:
                # Content was truncated; measure what's between file markers
                injected_chars = raw_chars  # fallback
            else:
                injected_chars = 0

            injected_files.append({
                "name": name,
                "raw_chars": raw_chars,
                "injected_chars": injected_chars,
                "truncated": injected_chars < raw_chars,
            })

    # Tools
    tool_entries: list[dict[str, Any]] = []
    if tools is not None:
        for tool in tools:
            name = getattr(tool, "name", str(tool))
            summary_chars = len(tool_summaries.get(name, "")) if tool_summaries else 0

            entry: dict[str, Any] = {"name": name, "summary_chars": summary_chars}

            # Duck-typed schema extraction
            if hasattr(tool, "parameters"):
                params = tool.parameters
                if isinstance(params, dict):
                    schema_str = json.dumps(params)
                    entry["schema_chars"] = len(schema_str)
                    props = params.get("properties", {})
                    entry["properties_count"] = len(props) if isinstance(props, dict) else 0
                else:
                    entry["schema_chars"] = 0
                    entry["properties_count"] = 0

            tool_entries.append(entry)
    elif tool_summaries is not None:
        for name, summary in tool_summaries.items():
            tool_entries.append({
                "name": name,
                "summary_chars": len(summary),
            })

    return {
        "source": source,
        "generated_at": time.time(),
        "system_prompt": {
            "chars": total_chars,
            "project_context_chars": project_context_chars,
            "non_project_context_chars": non_project_context_chars,
        },
        "injected_files": injected_files,
        "tools": {
            "entries": tool_entries,
        },
    }
