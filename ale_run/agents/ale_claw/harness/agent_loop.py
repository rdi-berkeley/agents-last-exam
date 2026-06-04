"""OpenClawComputerAgent — ComputerAgent subclass with mid-conversation compaction.

Overrides run() to manage a mutable message list, enabling in-place compaction
without agent rebuild. Mirrors OpenClaw's session.agent.replaceMessages() pattern
adapted for CUA's ComputerAgent lifecycle.

Design rationale:
  - OpenClaw compacts via replaceMessages() within a persistent session — messages
    are swapped in-place while the agent loop continues.
  - CUA's ComputerAgent.run() uses immutable old_items + new_items lists, requiring
    a stop-compact-resume pattern (break out, rebuild agent, restart).
  - This subclass replaces that pattern by overriding run() with a mutable items list.
    When overflow_cb.needs_compaction triggers, _compact_in_place() rewrites the list
    and the loop continues — no agent rebuild needed.

Refactoring:
  - Memory flush is called pre-API (before predict_step) via _maybe_flush_memory(),
    matching OpenClaw's runMemoryFlushIfNeeded pattern in agent-runner-memory.ts.
  - Transcript logging moved into run() via _log_step_to_transcript().
  - overflow_cb auto-injected into callbacks.

Reference:
  - agent/agent.py:658-808 — parent run() lifecycle
  - openclaw/src/agents/pi-embedded-runner/compact.ts — OpenClaw compaction orchestration
  - openclaw/src/agents/compaction.ts — chunk splitting, summarization
  - openclaw/src/auto-reply/reply/agent-runner-memory.ts — memory flush pattern
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from agent.agent import ComputerAgent, assert_callable_with, get_json, get_output_call_ids
from agent.computers.base import AsyncComputerHandler
from agent.computers.cua import cuaComputerHandler
from .model.model_config import ResolvedModel
from agent.responses import make_tool_error_item, replace_failed_computer_calls_with_function_calls
from agent.tools.base import BaseTool
from agent.types import ToolError
from core.telemetry import is_telemetry_enabled, record_event
from litellm.responses.utils import Usage


# Map of computer-action name → the param keys the action accepts.
# Used to filter the JSON arguments blob a model emits for
# ``function_call name="computer"``.
_COMPUTER_ACTION_PARAMS: Dict[str, List[str]] = {
    "screenshot": [],
    "click": ["x", "y", "button"],
    "double_click": ["x", "y"],
    "right_click": ["x", "y"],
    "type": ["text"],
    "keypress": ["keys"],
    "scroll": ["x", "y", "scroll_x", "scroll_y"],
    "move": ["x", "y"],
    "drag": ["start_x", "start_y", "end_x", "end_y"],
    "wait": ["seconds", "ms"],
    "terminate": ["status"],
}

from .model._message_shapes import _function_call_output, _image_url_block
from .canonical.canonical import normalize_to_canonical, sanitize_items
from .tools.computer_handler import OpenClawComputerHandler
from .context.context import ContextOverflowCallback, compact_messages, is_context_overflow_error
from .memory.memory import MemoryStore
from .memory.memory_flush import run_memory_flush
from .prompt import ContextFile
from .session import (
    MEMORY_FLUSH_PROMPT,
    MEMORY_FLUSH_SYSTEM_PROMPT,
    SILENT_REPLY_TOKEN,
    SessionManager,
    should_run_memory_flush,
)
from .subagent.subagent_registry import SubagentRegistry

from .agent_loop_helpers import (  # has_done_signal re-exported for deployer; rest used/re-exported
    _extract_messages_for_compaction,
    _maybe_sanitize_screenshot,
    _rewrite_input_image_to_image_url,
    _strip_images_from_messages,
    has_done_signal,
)

class OpenClawComputerAgent(ComputerAgent):
    """ComputerAgent subclass with mid-conversation compaction support.

    Overrides run() to manage a mutable message list, enabling in-place
    compaction without agent rebuild. Mirrors OpenClaw's session.agent.replaceMessages()
    pattern adapted for CUA.

    Memory flush runs pre-API (before predict_step) via _maybe_flush_memory(),
    matching OpenClaw's single-call-site pattern (runMemoryFlushIfNeeded before
    runAgentTurnWithFallback).
    """

    def __init__(
        self,
        *,
        overflow_cb: ContextOverflowCallback,
        session_mgr: SessionManager,
        memory_store: MemoryStore,
        summary_model: str,
        max_compactions: int = 10**9,
        on_compaction: Callable | None = None,
        thinking_config: Optional[Any] = None,
        resolved_model: ResolvedModel | None = None,
        summary_runtime: ResolvedModel | None = None,
        registry: SubagentRegistry | None = None,
        auto_screenshot: bool = False,
        context_files: Optional[List[ContextFile]] = None,
        # Image retention mode: "openclaw" (default) keeps all
        # images from the last N completed turns (OpenClaw-parity); "cua"
        # keeps the last N images by count (CUA-default). Both modes use
        # sticky placeholder replacement (no message-deletion cache thrash).
        # Default flipped from "count" to "openclaw" after on-task verification
        # — see develop-doc/cache-thrash-image-retention.md in the agenthle
        # repo.
        image_retention_mode: str = "openclaw",
        **kwargs,  # Pass through to ComputerAgent
    ):
        # Auto-inject overflow_cb into callbacks
        callbacks = kwargs.get("callbacks", []) or []
        if overflow_cb not in callbacks:
            callbacks = [overflow_cb] + list(callbacks)
        kwargs["callbacks"] = callbacks

        super().__init__(**kwargs)

        # Read by ``OpenClawImageAwareComputerAgent._dispatch_b2_computer_call``.
        # When False (default), only the explicit ``screenshot`` action returns
        # an image — click/type/keypress/etc. return their result as text.
        # Lives on the openclaw subclass so the SDK ``ComputerAgent`` stays
        # diff-free against ``cua-verse/cua`` upstream.
        self.auto_screenshot = bool(auto_screenshot)

        # Upgrade the auto-added SDK ImageRetentionCallback to the OpenClaw
        # variant. The SDK callback only prunes screenshots inside
        # ``computer_call_output`` items — silently no-ops for the
        # function-call shim path (Claude / GPT-5.4 / anything via
        # OpenRouter), where screenshots arrive as standalone user
        # ``image_url`` messages. The OpenClaw variant covers both paths.
        from agent.callbacks.image_retention import ImageRetentionCallback as _SDKImageRetention
        from .adapters.image_retention import OpenClawImageRetentionCallback
        for i, cb in enumerate(self.callbacks):
            if type(cb) is _SDKImageRetention:
                self.callbacks[i] = OpenClawImageRetentionCallback(
                    only_n_most_recent_images=cb.only_n_most_recent_images,
                    mode=image_retention_mode,
                )

        # Same pattern for TrajectorySaverCallback — swap the auto-added SDK
        # variant for the OpenClaw subclass, which overrides ``on_responses``
        # to skip the turn-bump that bloats trajectory dirs in the function-
        # call shim path. Removes the need for orchestration's
        # ``__init__.py`` to monkey-patch ``agent.agent.TrajectorySaverCallback``.
        from agent.callbacks.trajectory_saver import TrajectorySaverCallback as _SDKTrajectorySaver
        from .adapters.trajectory_saver import OpenClawTrajectorySaverCallback
        for i, cb in enumerate(self.callbacks):
            if type(cb) is _SDKTrajectorySaver:
                self.callbacks[i] = OpenClawTrajectorySaverCallback(
                    trajectory_dir=str(cb.trajectory_dir),
                    reset_on_run=cb.reset_on_run,
                    screenshot_dir=str(cb.screenshot_dir) if cb.screenshot_dir else None,
                )

        self.overflow_cb = overflow_cb
        self.session_mgr = session_mgr
        self.memory_store = memory_store
        self.summary_model = summary_model
        self.max_compactions = max_compactions
        self._compaction_count = 0
        self._on_compaction = on_compaction
        # Thinking config for per-call-site params
        self.thinking_config = thinking_config
        self.resolved_model = resolved_model
        self.summary_runtime = summary_runtime
        self._registry = registry
        # Stable bootstrap files re-injected post-compaction so the agent
        # re-anchors on workspace rules after the lossy summary. Mirrors
        # OpenClaw's readPostCompactionContext (post-compaction-context.ts).
        # Empty / None disables the re-injection.
        self._context_files: List[ContextFile] = list(context_files or [])

    @property
    def compaction_count(self) -> int:
        """Number of compactions performed so far in this run."""
        return self._compaction_count

    async def run(
        self,
        messages,
        stream: bool = False,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **additional_generation_kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Override ComputerAgent.run() with mutable message list + compaction.

        Replicates the full CUA run() lifecycle (callbacks, tool execution)
        but manages items in a mutable list. When overflow_cb.needs_compaction
        triggers, compacts messages in-place and continues — no agent rebuild.

        Per-step flow:
          should_continue check
          preprocessed = _on_llm_start()  <- updates current_tokens
          _maybe_flush_memory()            <- PRE-API: uses fresh current_tokens
          predict_step()                   <- API call
          yield result
          _log_step_to_transcript()        <- transcript logging
          _handle_item()
          compaction check
        """
        # MUTABLE items list — the key difference from parent run()
        items, run_kwargs, merged_kwargs = await self._run_setup(
            messages, stream, api_key, api_base, additional_generation_kwargs
        )
        new_items: List[Dict[str, Any]] = []

        # Loop until the model emits an explicit DONE signal (or the outer
        # consumer breaks out via max_steps). The upstream CUA exit condition
        # `new_items[-1].role != "assistant"` terminates on any tool-less
        # assistant turn, which silently killed runs whenever the model
        # emitted bare text (e.g., the [!silent] memory-flush sentinel after
        # a contaminated compaction). OpenClaw semantics want "stop on DONE,"
        # not "stop when tool calls stop."
        while True:
            should_continue = await self._on_run_continue(run_kwargs, items, new_items)
            if not should_continue:
                break

            combined = items + new_items
            combined = replace_failed_computer_calls_with_function_calls(combined)
            combined = self._sanitize_runtime_messages(combined)
            preprocessed = await self._on_llm_start(combined)

            # PRE-API memory flush — runs after _on_llm_start updates
            # current_tokens, before predict_step. Matches OpenClaw's single call
            # site: runMemoryFlushIfNeeded before runAgentTurnWithFallback.
            await self._maybe_flush_memory()

            loop_kwargs = {
                "messages": preprocessed,
                "model": self.model,
                "tools": self.tool_schemas,
                "stream": False,
                "computer_handler": self.computer_handler,
                "max_retries": self.max_retries,
                "use_prompt_caching": self.use_prompt_caching,
                **merged_kwargs,
            }

            # === REACTIVE OVERFLOW: try/except around predict_step ===
            try:
                result = await self.agent_loop.predict_step(
                    **loop_kwargs,
                    _on_api_start=self._on_api_start,
                    _on_api_end=self._on_api_end,
                    _on_usage=self._on_usage,
                    _on_screenshot=self._on_screenshot,
                )
            except Exception as e:
                if (
                    is_context_overflow_error(str(e))
                    and self._compaction_count < self.max_compactions
                ):
                    self.overflow_cb.force_compaction()
                    print(f"[ContextOverflow] API rejected — reactive compaction: {e}")
                    await self._compact_in_place(items, new_items)
                    items = items + new_items
                    new_items = []
                    continue
                raise

            result = get_json(result)
            result["output"] = await self._on_llm_end(result.get("output", []))
            await self._on_responses(loop_kwargs, result)

            # Sanitize truncated tool-call payloads before anything else
            # touches result["output"]. Mid-stream provider drops (sonnet-4.6
            # via OpenRouter, observed on `write` calls) leave a function_call
            # block with a half-finished JSON arguments string. If we let it
            # through, the bad string ends up persisted in transcript and in
            # the in-memory message list — fine for normal per-turn API sends,
            # but compaction's history rebuild re-parses every function_call's
            # arguments and crashes the run. Rewriting in place + appending
            # the synthetic tool_error here keeps the data self-consistent so
            # any downstream re-serializer is safe.
            self._sanitize_truncated_function_calls(result.get("output", []))

            yield result

            # Log model-emitted assistant content/tool calls to transcript.
            self._log_step_to_transcript(result)

            # Log bare assistant text to stdout (tool calls are already
            # logged by ToolLoggingCallback; text-only turns had no console
            # visibility, which masked e.g. the DONE signal).
            self._log_assistant_text(result.get("output", []))

            new_items += result.get("output", [])
            output_call_ids = get_output_call_ids(result.get("output", []))

            for item in result.get("output", []):
                partial_items = await self._handle_item(
                    item, self.computer_handler, ignore_call_ids=output_call_ids
                )
                new_items += partial_items
                if partial_items:
                    self._log_partial_items_to_transcript(partial_items)
                if partial_items:
                    yield {
                        "output": partial_items,
                        "usage": Usage(
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                        ).model_dump(),
                    }

            # === SUBAGENT COMPLETION DRAIN ===
            # Drain before the compaction check so any new user messages from
            # completed general subagents count toward this iteration's token
            # pressure.
            self._drain_completions(new_items)
            # === POST-DELEGATION SCREENSHOT DRAIN ===
            # Runs after completions so the GUI-delegation screenshot is the
            # freshest user turn heading into the next predict_step.
            self._drain_post_delegation(new_items)

            # === PROACTIVE COMPACTION INJECTION POINT ===
            if (
                self.overflow_cb.needs_compaction
                and self._compaction_count < self.max_compactions
            ):
                print("[Compaction] Proactive trigger — compacting in-place")
                await self._compact_in_place(items, new_items)
                items = items + new_items
                new_items = []
                continue

            # === DONE-BASED TERMINATION ===
            # Replaces the upstream "last item is assistant" exit. Only an
            # explicit DONE in the assistant output ends the generator from
            # inside; max_steps in the outer consumer is the other ceiling.
            if has_done_signal(result.get("output", [])):
                break

            # === BARE-TEXT TURN NUDGE (bug-2-nudge) ===
            self._maybe_nudge_bare_text(result, new_items)

        await self._on_run_end(loop_kwargs, items, new_items)

    async def _run_setup(
        self,
        messages,
        stream: bool,
        api_key: Optional[str],
        api_base: Optional[str],
        additional_generation_kwargs: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Validate config, init computers, thread API creds, and fire _on_run_start.

        Returns ``(items, run_kwargs, merged_kwargs)`` — ``items`` is the mutable
        input list that run()'s loop maintains.
        """
        if not self.agent_config_info:
            raise ValueError("Agent configuration not found")

        capabilities = self.get_capabilities()
        if "step" not in capabilities:
            raise ValueError(
                f"Agent loop {self.agent_config_info.agent_class.__name__} "
                "does not support step predictions"
            )

        await self._initialize_computers()

        # Merge kwargs and thread api credentials
        merged_kwargs = {**self.kwargs, **additional_generation_kwargs}
        if (api_key is not None) or (self.api_key is not None):
            merged_kwargs["api_key"] = api_key if api_key is not None else self.api_key
        if (api_base is not None) or (self.api_base is not None):
            merged_kwargs["api_base"] = api_base if api_base is not None else self.api_base

        items = self._process_input(messages)

        run_kwargs = {
            "messages": messages,
            "stream": stream,
            "model": self.model,
            "agent_loop": self.agent_config_info.agent_class.__name__,
            **merged_kwargs,
        }
        await self._on_run_start(run_kwargs, items)
        return items, run_kwargs, merged_kwargs

    def _maybe_nudge_bare_text(
        self,
        result: Dict[str, Any],
        new_items: List[Dict[str, Any]],
    ) -> None:
        """Append a user-role continuation when a turn produced neither a tool
        call nor a DONE marker.

        Without it, bare-text assistant turns leave the conversation ending on
        an assistant turn, which some routings reject — notably
        Anthropic-via-Vertex through OpenRouter ("This model does not support
        assistant message prefill"). Anthropic-direct and Bedrock tolerate
        assistant-tail, so the bug only surfaces under that specific routing.
        """
        output_items = result.get("output", [])
        has_tool_call = any(
            item.get("type") in ("function_call", "computer_call")
            for item in output_items
        )
        if not has_tool_call:
            nudge_text = (
                "Please continue: emit your next tool call to make "
                "progress, or output the DONE marker if the task is "
                "complete."
            )
            new_items.append({"role": "user", "content": nudge_text})
            self.session_mgr.append_message("user", nudge_text)

    async def _maybe_flush_memory(self) -> None:
        """Run memory flush if token or transcript-size threshold is exceeded.

        Single call site for memory flush — called pre-API in run().
        Matches OpenClaw's runMemoryFlushIfNeeded pattern, including the
        transcript-size force trigger from buildMemoryFlushPlan().
        """
        if self.session_mgr._state is None:
            return
        try:
            transcript_bytes = self.session_mgr.transcript_path.stat().st_size
        except OSError:
            transcript_bytes = 0
        if not should_run_memory_flush(
            self.session_mgr._state,
            current_tokens=self.overflow_cb.current_tokens,
            context_window=self.overflow_cb.context_window,
            transcript_bytes=transcript_bytes,
            compaction_ratio=self.overflow_cb.compaction_threshold_ratio,
        ):
            return
        await run_memory_flush(
            summary_model=self.summary_model,
            session_mgr=self.session_mgr,
            memory_store=self.memory_store,
            flush_prompt=MEMORY_FLUSH_PROMPT,
            flush_system_prompt=MEMORY_FLUSH_SYSTEM_PROMPT,
            silent_token=SILENT_REPLY_TOKEN,
            thinking_params=(
                self.thinking_config.flush_params(
                    self.summary_model,
                    runtime=self.summary_runtime,
                )
                if self.thinking_config is not None
                else None
            ),
            summary_runtime=self.summary_runtime,
        )

    def _drain_completions(self, new_items: List[Dict[str, Any]]) -> None:
        """Drain the subagent registry's completion queue into ``new_items``.

        Each completed/failed subagent run (general or GUI) is appended as a user
        message in ``[Subagent Result]`` format so the next LLM turn sees
        it verbatim. No-op when no registry is wired (tests, legacy paths)
        or when the queue is empty. FIFO ordering is preserved because
        ``registry.drain_completions`` drains with sequential
        ``get_nowait()`` calls.
        """
        if self._registry is None:
            return
        for run in self._registry.drain_completions():
            status = run.status.value
            body = run.result_text or run.error_message or ""
            content = (
                f"[Subagent Result] task: {run.task}\n"
                f"Status: {status}\n\n"
                f"{body}"
            )
            new_items.append({"role": "user", "content": content})
            self.session_mgr.append_message("user", content)

    def _drain_post_delegation(self, new_items: List[Dict[str, Any]]) -> None:
        """Drain the registry's post-delegation queue into ``new_items``.

        Messages are pushed by ``DelegateGUITool`` after an async GUI
        subagent run completes. They carry a fresh VM
        screenshot as ``{role: user, content: [text, image_url]}`` so the
        main agent's next ``predict_step`` sees the updated state. Pre-built
        shapes — extend verbatim, do not reformat.
        """
        if self._registry is None:
            return
        for msg in self._registry.drain_post_delegation():
            new_items.append(msg)
            self.session_mgr.append_message("user", msg.get("content", []))

    async def _initialize_computers(self) -> None:
        """Upgrade the resolved computer handler to OpenClawComputerHandler.

        Runs the SDK's normal handler resolution (BaseComputerTool path or
        ``make_computer_handler`` factory) via super(), then re-wraps a
        plain ``cuaComputerHandler`` in ``OpenClawComputerHandler`` so the
        chord-vs-sequence keypress fix applies for every
        ``OpenClawComputerAgent`` run. Default behavior for any other
        ``ComputerAgent`` subclass is unchanged — the upgrade lives at this
        cua-bench layer, not in the SDK.

        Skip when the resolved handler is a ``BaseComputerTool`` (preferred
        SDK path, different protocol) or already an ``OpenClawComputerHandler``
        subclass (e.g. orchestration's monkey-patch path) — prevents
        double-wrapping.
        """
        await super()._initialize_computers()
        handler = self.computer_handler
        if (
            isinstance(handler, cuaComputerHandler)
            and not isinstance(handler, OpenClawComputerHandler)
        ):
            upgraded = OpenClawComputerHandler(handler.cua_computer)
            await upgraded._initialize()
            self.computer_handler = upgraded

    def _sanitize_truncated_function_calls(
        self,
        output_items: List[Dict[str, Any]],
    ) -> None:
        """Rewrite truncated tool-call payloads in place + emit synthetic errors.

        Mid-stream upstream-provider drops (observed on sonnet-4.6 via
        OpenRouter, almost always on `write` calls right before the large
        ``contents`` field) leave a ``function_call`` block whose ``arguments``
        is a half-finished JSON string. Letting it through corrupts the
        canonical history: per-turn API sends tolerate the bad string (it's
        forwarded as-is and never re-parsed), but compaction's history rebuild
        in ``canonical_to_anthropic_messages`` does ``json.loads`` on every
        function_call's arguments and crashes the entire run with an empty
        "Agent error:" status.

        Sanitize at the earliest point — right after ``predict_step`` returns,
        before transcript logging or in-memory accumulation — so:
          1. The persisted ``arguments`` is always valid JSON.
          2. The synthetic ``function_call_output`` (tool_error) is paired
             with the cleaned function_call in the same transcript group.
          3. ``get_output_call_ids`` automatically picks up the synthetic
             output's call_id, so ``_handle_item`` skips dispatching the
             (now placeholder-shaped) write call. No ``ignore_call_ids``
             plumbing needed.

        The defensive try/except in ``_handle_item`` stays as a backstop in
        case some new code path slips an unsanitized item past this point.
        """
        synthetic_outputs: List[Dict[str, Any]] = []
        for item in output_items:
            if item.get("type") != "function_call":
                continue
            raw_args = item.get("arguments")
            if not isinstance(raw_args, str) or not raw_args:
                continue
            try:
                json.loads(raw_args)
                continue
            except (json.JSONDecodeError, TypeError) as e:
                tool_name = item.get("name", "<unknown>")
                call_id = item.get("call_id")
                snippet = raw_args[:200] + "..." if len(raw_args) > 200 else raw_args
                error_message = (
                    f"Malformed tool-call arguments for {tool_name!r} "
                    f"(likely truncated by upstream provider): {e!r}. "
                    f"Raw arguments: {snippet!r}. "
                    f"Please retry the call with complete arguments."
                )
                # Replace the broken arguments string with a valid placeholder
                # so any future re-serializer (compaction, replay, debug
                # tooling) can json.loads() it without exploding. The marker
                # keys make the recovery visible in transcripts/logs.
                item["arguments"] = json.dumps({
                    "_truncated_by_upstream": True,
                    "_partial_args": raw_args[:200],
                    "_original_length": len(raw_args),
                    "_recovery_note": "Original arguments truncated mid-stream; "
                                      "synthetic tool_error was returned to the model.",
                })
                synthetic_outputs.append(make_tool_error_item(error_message, call_id))
                print(
                    f"[Sanitize] Truncated tool-call args for {tool_name!r} "
                    f"(call_id={call_id}, len={len(raw_args)}); "
                    f"rewrote arguments + emitted synthetic tool_error"
                )
        # Append synthetic outputs to the same output list so they land in
        # the same transcript group, count toward output_call_ids, and reach
        # the model on the next turn as paired tool_results.
        output_items.extend(synthetic_outputs)

    async def _handle_item(
        self,
        item: Dict[str, Any],
        computer: Optional[AsyncComputerHandler] = None,
        ignore_call_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Dispatch ``function_call`` items locally; delegate everything else.

        Two openclaw-only behaviors live here so the SDK ``ComputerAgent``
        stays diff-free against ``cua-verse/cua`` upstream:

        1. ``function_call`` name=="computer": parse the JSON action, run it,
           and gate the post-action screenshot on ``self.auto_screenshot``.
           When False (default), only the explicit ``screenshot`` action
           returns an image — click/type/keypress/etc. return their tool
           result as plain text.
        2. ``function_call`` returning ``{type: "image", data, mime_type}``:
           emit a sentinel function_call_output and a separate user message
           with ``image_url`` content (used by ``ReadFileTool``).

        Other item types (``message``, ``computer_call``) delegate to
        ``super()._handle_item``, then any SDK-emitted ``input_image`` blocks
        are rewritten into LiteLLM's ``image_url`` vision-input shape.
        """
        if item.get("type") != "function_call":
            result = await super()._handle_item(item, computer, ignore_call_ids)
            return _rewrite_input_image_to_image_url(result)

        call_id = item.get("call_id")
        if ignore_call_ids and call_id and call_id in ignore_call_ids:
            return []

        try:
            return await self._dispatch_function_call(item, computer)
        except ToolError as e:
            return [make_tool_error_item(repr(e), call_id)]
        except (json.JSONDecodeError, TypeError) as e:
            # Tool-call arguments were not valid JSON. The most common cause is
            # the upstream provider truncating the streamed tool_call payload
            # mid-write (e.g. Bedrock 502 mid-stream during a large `write`).
            # Surface this back to the model as a tool_error so the agent can
            # retry on the next turn instead of crashing the run loop.
            tool_name = item.get("name", "<unknown>")
            raw_args = item.get("arguments")
            snippet = (
                raw_args[:200] + "..."
                if isinstance(raw_args, str) and len(raw_args) > 200
                else raw_args
            )
            error_message = (
                f"Malformed tool-call arguments for {tool_name!r} "
                f"(likely truncated by upstream provider): {e!r}. "
                f"Raw arguments: {snippet!r}. "
                f"Please retry the call with complete arguments."
            )
            return [make_tool_error_item(error_message, call_id)]

    async def _dispatch_function_call(
        self,
        item: Dict[str, Any],
        computer: Optional[AsyncComputerHandler],
    ) -> List[Dict[str, Any]]:
        await self._on_function_call_start(item)

        if item.get("name") == "computer" and computer:
            result = await self._dispatch_computer_function_call(item, computer)
            await self._on_function_call_end(item, result)
            return result

        # Regular function dispatch.
        function = self._get_tool(item.get("name"))
        if not function:
            raise ToolError(f"Function {item.get('name')} not found")

        args = json.loads(item.get("arguments"))
        if isinstance(function, BaseTool):
            tool_result: Any = function.call(args)
        else:
            assert_callable_with(function, **args)
            if inspect.iscoroutinefunction(function):
                tool_result = await function(**args)
            else:
                tool_result = await asyncio.to_thread(function, **args)

        if self.telemetry_enabled and is_telemetry_enabled():
            record_event(
                "agent_tool_executed",
                {"tool_type": "function", "tool_name": item.get("name")},
            )

        # Image-shaped tool return → sentinel + image_message.
        if (
            isinstance(tool_result, dict)
            and tool_result.get("type") == "image"
            and isinstance(tool_result.get("data"), str)
            and isinstance(tool_result.get("mime_type"), str)
        ):
            sentinel: Dict[str, Any] = {
                "success": tool_result.get("success", True),
                "read_image": True,
                "mime_type": tool_result["mime_type"],
            }
            if isinstance(tool_result.get("text"), str):
                sentinel["text"] = tool_result["text"]
            call_output = _function_call_output(item.get("call_id"), json.dumps(sentinel))
            image_message = {
                "role": "user",
                "content": [
                    _image_url_block(
                        f"data:{tool_result['mime_type']};base64,{tool_result['data']}"
                    )
                ],
            }
            wrapped = [call_output, image_message]
            await self._on_function_call_end(item, wrapped)
            return wrapped

        call_output = _function_call_output(item.get("call_id"), str(tool_result))
        wrapped = [call_output]
        await self._on_function_call_end(item, wrapped)
        return wrapped

    async def _dispatch_computer_function_call(
        self,
        item: Dict[str, Any],
        computer: AsyncComputerHandler,
    ) -> List[Dict[str, Any]]:
        """Route ``function_call name="computer"`` to the computer handler.

        Honors ``self.auto_screenshot``: when False, only the explicit
        ``screenshot`` action attaches an image — other actions (click,
        type, keypress, scroll, drag, wait, etc.) return their result as
        a plain text ``function_call_output``.
        """
        args = json.loads(item.get("arguments", "{}"))
        action_type = args.get("action")
        if not action_type:
            raise ToolError("Computer function call missing 'action' argument")

        relevant_params = _COMPUTER_ACTION_PARAMS.get(action_type, [])
        action_args: Dict[str, Any] = {}
        for k, v in args.items():
            if k == "action":
                continue
            if k in relevant_params or action_type not in _COMPUTER_ACTION_PARAMS:
                if v is not None and v != "" and v != []:
                    action_args[k] = v

        computer_method = getattr(computer, action_type, None)
        if not computer_method:
            raise ToolError(f"Unknown computer action: {action_type}")
        action_result = await computer_method(**action_args)

        if self.telemetry_enabled and is_telemetry_enabled():
            record_event("computer_action_executed", {"action_type": action_type})
            record_event(
                "agent_tool_executed",
                {"tool_type": "computer", "tool_name": action_type},
            )

        is_terminate = action_type == "terminate" or (
            isinstance(action_result, dict) and action_result.get("terminated")
        )

        if is_terminate:
            output_content = json.dumps(action_result if action_result else {"terminated": True})
            return [_function_call_output(item.get("call_id"), output_content)]

        # auto_screenshot=False and not an explicit screenshot —
        # return only the textual function_call_output, no image.
        if not self.auto_screenshot and action_type != "screenshot":
            output_content = json.dumps(
                action_result if action_result is not None else {"success": True}
            )
            return [_function_call_output(item.get("call_id"), output_content)]

        if self.screenshot_delay and self.screenshot_delay > 0:
            await asyncio.sleep(self.screenshot_delay)
        screenshot_base64 = await computer.screenshot()
        # resize/transcode if the screenshot exceeds OpenClaw's
        # 5 MB / 1200 px / 25 MP limits before it enters the transcript.
        sanitized_b64, sanitized_mime = _maybe_sanitize_screenshot(screenshot_base64)
        await self._on_screenshot(sanitized_b64, "screenshot_after")

        # ``action="screenshot"`` returns raw base64 — short-circuit so we
        # don't dump 58K tokens of base64 into the tool-text channel (and
        # so ``only_n_most_recent_images`` pruning still applies via the
        # ``image_url`` block we emit).
        if action_type == "screenshot" or action_result is None:
            output_content = json.dumps({"success": True, "screenshot_captured": True})
        else:
            output_content = json.dumps(action_result)
        call_output = _function_call_output(item.get("call_id"), output_content)
        image_message = {
            "role": "user",
            "content": [
                _image_url_block(f"data:{sanitized_mime};base64,{sanitized_b64}")
            ],
        }
        return [call_output, image_message]

    @staticmethod
    def _log_assistant_text(output: List[Dict[str, Any]]) -> None:
        """Print any bare assistant text to stdout.

        ToolLoggingCallback surfaces every function_call; bare text messages
        (including the DONE termination signal) had no console hook, so the
        operator only saw tool calls and infra lines. This adds one
        ``[Agent]`` line per non-empty text message, truncated to keep the
        log readable.
        """
        MAX_LEN = 500
        for item in output:
            if item.get("type") != "message":
                continue
            content = item.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("text", "")
                        if t:
                            parts.append(t)
                text = "\n".join(parts)
            else:
                text = ""
            text = text.strip()
            if not text:
                continue
            shown = text if len(text) <= MAX_LEN else text[:MAX_LEN] + f"... [+{len(text) - MAX_LEN} chars]"
            # Single-line form for grep friendliness; preserve embedded
            # newlines by replacing them with a visible marker.
            shown_inline = shown.replace("\n", " ⏎ ")
            print(f"[Agent] {shown_inline}")

    def _log_step_to_transcript(self, result: Dict[str, Any]) -> None:
        """Log a step's output to the session transcript.

        Groups output into assistant/tool turns and appends to transcript.
        Moved from perform_task() to run().
        """
        from .context.transcript import group_step_output

        step_input = result["usage"].get("input_tokens", 0)
        step_output = result["usage"].get("output_tokens", 0)

        assistant_content, tool_results = group_step_output(
            result["output"], self.trajectory_dir
        )

        if assistant_content:
            has_tools = any(
                b["type"] in ("function_call", "computer_call")
                for b in assistant_content
            )
            usage = {
                "input": step_input,
                "output": step_output,
                "total": step_input + step_output,
                "cost": result["usage"].get("response_cost", 0),
            }
            self.session_mgr.append_message(
                "assistant",
                assistant_content,
                usage=usage,
                stop_reason=result.get("stop_reason") or ("tool_use" if has_tools else None),
                api=(
                    self.resolved_model.transcript_api_label
                    if self.resolved_model is not None
                    else None
                ),
            )

        if tool_results:
            self.session_mgr.append_message("tool", tool_results)

    def _log_partial_items_to_transcript(
        self,
        output_items: List[Dict[str, Any]],
    ) -> None:
        """Log tool execution outputs emitted after `_handle_item()`.

        The main model result logs assistant text/tool calls. Actual tool outputs
        (`function_call_output`, `computer_call_output`) are yielded later via
        `partial_items`, so they must be appended separately or the transcript
        loses call/result pairing.
        """
        from .context.transcript import group_step_output

        assistant_content, tool_results = group_step_output(
            output_items, self.trajectory_dir
        )

        # Post-tool helper messages can inject a local screenshot path as a user
        # message. Persist that as user content instead of misclassifying it as
        # assistant text.
        for item in output_items:
            if item.get("type") != "message" or item.get("role") != "user":
                continue
            content = item.get("content", "")
            if isinstance(content, str):
                self.session_mgr.append_message("user", content)

        if assistant_content:
            self.session_mgr.append_message("assistant", assistant_content)
        if tool_results:
            self.session_mgr.append_message("tool", tool_results)

    async def _compact_in_place(
        self,
        items: List[Dict[str, Any]],
        new_items: List[Dict[str, Any]],
    ) -> None:
        """Run compaction on the accumulated message list.

        Modifies items/new_items to contain only the compaction summary
        + kept messages. Persists compaction entry to session transcript.

        Memory flush runs pre-API via _maybe_flush_memory(), not here.
        If compaction fires without a prior flush, log a warning (edge case
        where token estimation missed the threshold).
        """
        # Warn if no flush preceded this compaction (token estimation edge case)
        from .session import has_already_flushed_for_current_compaction
        if self.session_mgr._state is not None and not has_already_flushed_for_current_compaction(
            self.session_mgr._state
        ):
            print("[Compaction] Warning: compaction running without prior memory flush")

        # Extract messages from transcript and run compaction pipeline
        all_messages = _extract_messages_for_compaction(self.session_mgr)
        compaction_result = await compact_messages(
            all_messages,
            self.summary_model,
            self.overflow_cb.context_window,
            instructions_tokens=len(self.instructions or "") // 4,
            thinking_params=(
                self.thinking_config.compaction_params(
                    self.summary_model,
                    runtime=self.summary_runtime,
                )
                if self.thinking_config is not None
                else None
            ),
            summary_runtime=self.summary_runtime,
        )

        # Persist compaction entry with firstKeptEntryId
        history = self.session_mgr.load_history()
        msg_entries = [e for e in history if e.type == "message"]
        if compaction_result.first_kept_message_index < len(msg_entries):
            first_kept_id = msg_entries[compaction_result.first_kept_message_index].id
        else:
            first_kept_id = msg_entries[-1].id if msg_entries else "unknown"

        self.session_mgr.append_compaction(
            compaction_result.summary,
            first_kept_id,
            compaction_result.tokens_before,
        )

        # Rebuild items from compacted state
        kept_messages = all_messages[compaction_result.first_kept_message_index:]
        canonical_messages = self._build_compacted_items(
            compaction_result.summary, kept_messages
        )

        # Convert canonical → provider-specific format via the same model-aware
        # sanitize pipeline used on the normal per-turn send path.
        compacted_items = sanitize_items(
            canonical_messages,
            model=self.resolved_model or self.model,
        )

        items.clear()
        items.extend(compacted_items)
        new_items.clear()

        # Re-anchor the agent on stable workspace rules after the lossy
        # summary, and seed a byte-stable cache prefix block. Mirrors
        # OpenClaw's readPostCompactionContext (auto-reply/reply/post-
        # compaction-context.ts) called from agent-runner.ts:1565.
        post_compaction = self._build_post_compaction_message()
        if post_compaction is not None:
            items.append(post_compaction)

        # Reset and track
        self.overflow_cb.reset_after_compaction()
        self._compaction_count += 1

        if self._on_compaction:
            self._on_compaction(self._compaction_count)

        print(
            f"[Compaction] In-place compaction #{self._compaction_count} complete "
            f"({compaction_result.tokens_before}->~{len(compacted_items)} items)"
        )

    def _build_post_compaction_message(self) -> Optional[Dict[str, Any]]:
        """Build the post-compaction context-refresh user message.

        Returns ``None`` when no context files are registered (the natural
        disable knob — analogous to OpenClaw's ``postCompactionSections: []``).

        The message is framed as an explicit auto-injection so the model
        doesn't attribute it to the user. Re-reads bootstrap files verbatim
        so the resulting block is byte-identical across compactions and
        across runs — caches well as a fresh prefix point.
        """
        if not self._context_files:
            return None

        parts: List[str] = [
            "[Auto: post-compaction context refresh]",
            "",
            (
                "The conversation above was just compacted into a summary. "
                "Re-anchoring on stable workspace rules before continuing:"
            ),
            "",
        ]
        for cf in self._context_files:
            parts.append(f"## {cf.path}")
            parts.append("")
            parts.append(cf.content.rstrip())
            parts.append("")
        return {"role": "user", "content": "\n".join(parts).rstrip() + "\n"}

    def _sanitize_runtime_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize and sanitize outgoing history using the live model policy.

        This mirrors OpenClaw's runtime send-path invariant: policy resolves
        from the actual model, not just the adapter target, and the same
        sanitize pipeline applies on normal turns and compaction rebuilds.
        """
        # Live in-run history is already in provider-native flat item format for
        # OpenAI/Responses loops. Re-sanitizing that stream would downgrade
        # computer_call/computer_call_output into text and cause the model to
        # repeatedly request fresh screenshots.
        if any(
            isinstance(message, dict)
            and "type" in message
            and "role" not in message
            for message in messages
        ):
            return messages

        canonical_messages = normalize_to_canonical(messages)
        return sanitize_items(
            canonical_messages,
            model=self.resolved_model or self.model,
        )

    def _build_compacted_items(
        self, summary: str, kept_messages: List[Dict[str, Any]]
    ) -> list:
        """Build a canonical message list from compaction output.

        Produces: [user(CompactionSummaryBlock), ...normalized_kept].
        The summary provides continuity context; kept messages preserve recent
        conversation verbatim in canonical form.

        Repair runs on untyped dicts BEFORE canonical normalization — the
        existing algorithm uses stop_reason at the message level and is
        well-tested.  The sanitize_items() pipeline then applies
        canonical-level repair, ordering, and format conversion.

        Args:
            summary: The compaction summary text.
            kept_messages: Messages after first_kept_message_index from the
                original message list (the recent portion preserved by compaction).

        Returns:
            Typed canonical messages.  The caller
            (``_compact_in_place``) runs ``sanitize_items()`` to convert
            to provider-specific format.
        """
        from .canonical.canonical import (
            CanonicalMessage,
            CompactionSummaryBlock,
            normalize_to_canonical,
        )

        items: List[CanonicalMessage] = []

        # Summary as canonical message with CompactionSummaryBlock
        if summary:
            items.append(CanonicalMessage(
                role="user",
                content=[CompactionSummaryBlock(type="compaction_summary", text=summary)],
            ))

        # Kept messages: repair (role-based dicts) then normalize to canonical
        if kept_messages:
            from .context.context import repair_tool_use_result_pairing
            repair_result = repair_tool_use_result_pairing(kept_messages)
            items.extend(normalize_to_canonical(repair_result.messages))

        # Note: trailing-assistant check moved to ensure_valid_ordering()
        # in the sanitize_items() pipeline.

        return items
