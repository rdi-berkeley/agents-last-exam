"""General subagent — thin wrapper around the persistent session engine.

US-SUB-002 introduced this module as a flat 5-step ``litellm.acompletion``
loop. US-SUB-008 swapped the engine for a session-backed
``GeneralSubagentSession`` (``subagent_session.py``) with its own compaction
pipeline and on-disk transcript. This module is now a thin wrapper that
handles registry lifecycle and cooperative cancellation.

The wrapper preserves the public contract used by the upcoming
``DelegateGeneralTool`` (US-SUB-005): spawn an asyncio task, get a final
text back via the registry completion queue.

Tool-filter helpers (``_filter_tools``, ``_build_subagent_system_prompt``,
``_tools_to_litellm_schema``, ``ALLOWED_TOOL_NAMES``,
``EXCLUDED_TOOL_NAMES``) now live in ``subagent_session.py`` and are
re-exported here for backwards compatibility with US-SUB-002 callers and
tests.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .subagent_registry import SubagentRegistry, SubagentUsage
from .subagent_session import (
    ALLOWED_TOOL_NAMES,
    DEFAULT_MAX_STEPS,
    EXCLUDED_TOOL_NAMES,
    GeneralSubagentSession,
    _build_subagent_system_prompt,
    _filter_tools,
    _tools_to_litellm_schema,
)

__all__ = [
    "ALLOWED_TOOL_NAMES",
    "EXCLUDED_TOOL_NAMES",
    "DEFAULT_MAX_STEPS",
    "_build_subagent_system_prompt",
    "_filter_tools",
    "_tools_to_litellm_schema",
    "run_general_subagent",
]

logger = logging.getLogger(__name__)


async def run_general_subagent(
    *,
    task: str,
    model: str,
    tools: list,
    registry: SubagentRegistry,
    run_id: str,
    summary_model: str,
    parent_session_dir: str | Path,
    memory_store: Any | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    thinking_params: dict[str, Any] | None = None,
    initial_screenshot_paths: list[str] | None = None,
) -> None:
    """Run a general subagent as a persistent session.

    Drives ``GeneralSubagentSession`` end-to-end and reports lifecycle
    transitions back to the registry. ``CancelledError`` is treated as
    cooperative cancellation (the US-SUB-005 ``SubagentsTool.kill`` path)
    and surfaces as a ``KILLED`` registry transition.

    Args:
        task: Task description for the subagent.
        model: litellm model string for the subagent's main loop.
        tools: Full tool list — filtered to ``ALLOWED_TOOL_NAMES``.
        registry: SubagentRegistry for lifecycle reporting.
        run_id: Run ID returned from ``registry.register()``.
        summary_model: litellm model string for in-session compaction.
        parent_session_dir: Main agent's session directory; the
            subagent transcript lands at ``<parent>/subagents/<run_id>/``.
        memory_store: Optional MemoryStore (forward-compat — not yet
            consumed by the engine itself; passed through for tools).
        max_steps: Safety rail for the session loop (default 50).
        thinking_params: Optional provider-specific thinking kwargs.
        initial_screenshot_paths: Optional list of absolute file paths to
            PNG screenshots to attach to the subagent's initial user
            message (US-SUB-006). Each readable path becomes an
            ``image_url`` block alongside the text task; unreadable paths
            degrade to a ``[screenshot unavailable: ...]`` text block.
    """
    session: GeneralSubagentSession | None = None
    try:
        registry.mark_running(run_id)
        session = GeneralSubagentSession(
            run_id=run_id,
            task=task,
            model=model,
            tools=tools,
            registry=registry,
            summary_model=summary_model,
            parent_session_dir=Path(parent_session_dir),
            memory_store=memory_store,
            max_steps=max_steps,
            thinking_params=thinking_params,
            initial_screenshot_paths=initial_screenshot_paths,
        )
        registry.attach_inbox(run_id, session.inbox)
        result_text = await session.run()
        log_limit = 2000
        if len(result_text) > log_limit:
            shown = (
                f"{result_text[:log_limit]}"
                f"... [truncated {len(result_text) - log_limit} chars]"
            )
        else:
            shown = result_text
        logger.info(
            "[Subagent] General subagent %s completed (%d+%d tokens)\n"
            "[Subagent:%s] result:\n%s",
            run_id, session.usage.input_tokens, session.usage.output_tokens,
            run_id, shown,
        )
        registry.complete(run_id, result_text, session.usage)
    except asyncio.CancelledError:
        # Cooperative cancellation — US-SUB-005's SubagentsTool.kill path.
        # Mark the registry KILLED but re-raise so the asyncio.Task surfaces
        # as cancelled to the supervising tool.
        logger.info("[Subagent] General subagent %s cancelled", run_id)
        registry.kill(run_id)
        raise
    except Exception as e:
        logger.warning("[Subagent] General subagent %s failed: %s", run_id, e)
        usage = session.usage if session is not None else SubagentUsage()
        registry.fail(run_id, str(e), usage)
