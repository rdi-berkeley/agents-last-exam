"""Standalone memory flush module — pre-compaction memory persistence.

Gives the model a single turn to persist durable memories before context
is compacted. The model can call the memory_write tool to store memories,
or reply with the silent token if nothing to persist.

Design follows OpenClaw's agent-runner-memory.ts pattern: memory flush is a
standalone module, not a method on any agent class. This keeps flush logic
testable and decoupled from the agent class hierarchy.

Reference:
  - openclaw/src/auto-reply/reply/memory-flush.ts
  - openclaw/src/auto-reply/reply/agent-runner-memory.ts
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Any

from .model_config import ResolvedModel, resolve_model

from .helper_runtime import call_helper_model

if TYPE_CHECKING:
    from .memory import MemoryStore
    from .session import SessionManager


async def run_memory_flush(
    *,
    summary_model: str,
    session_mgr: SessionManager,
    memory_store: MemoryStore,
    flush_prompt: str,
    flush_system_prompt: str,
    silent_token: str,
    thinking_params: dict[str, Any] | None = None,
    summary_runtime: ResolvedModel | None = None,
) -> None:
    """Run a pre-compaction memory flush turn via litellm.

    Gives the model a single turn to persist durable memories before context
    is compacted. The model can call the memory_write tool to store memories,
    or reply with the silent token if nothing to persist.

    Based on OpenClaw's memory flush mechanism
    (openclaw/src/auto-reply/reply/memory-flush.ts).

    Args:
        summary_model: Model to use for the flush turn.
        session_mgr: Session manager for transcript access.
        memory_store: Memory store for persisting memories.
        flush_prompt: User prompt for the flush turn.
        flush_system_prompt: System prompt for the flush turn.
        silent_token: Token the model replies with when nothing to persist.
    """
    # Build memory_write tool schema for litellm
    memory_write_tool = {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": (
                "Write content to task memory. "
                "Use target='session' to append to the session log."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text content to write.",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["session", "task_memory"],
                        "description": "Where to write: 'session' (append) or 'task_memory' (overwrite).",
                    },
                },
                "required": ["content"],
            },
        },
    }

    # Build context from transcript so the model knows what to flush.
    conversation_text = _serialize_flush_context(session_mgr)

    flush_user_content = (
        f"<conversation>\n{conversation_text}\n</conversation>\n\n{flush_prompt}"
        if conversation_text
        else flush_prompt
    )

    messages = [
        {"role": "system", "content": flush_system_prompt},
        {"role": "user", "content": flush_user_content},
    ]
    resolved_summary = summary_runtime or resolve_model(summary_model)

    print(f"[MemoryFlush] Running pre-compaction memory flush turn ({len(conversation_text)} chars context)")
    try:
        # Budget: flush turns chain multiple memory_write calls and (with
        # thinking enabled) reasoning shares the output budget, so a tight cap
        # can truncate a later call's JSON args mid-string. 4096 gives headroom;
        # the model still stops when done, so cost per flush doesn't grow.
        response = await call_helper_model(
            resolved_summary,
            purpose="memory_flush",
            messages=messages,
            tools=[memory_write_tool],
            max_tokens=4096,
            temperature=1.0,
            thinking_params=thinking_params,
        )
        reply_content = response.text
        tool_calls = response.tool_calls

        # Handle tool calls — the model may call memory_write.
        # Flush prompt/reply are NOT appended to the main transcript: they are
        # an out-of-band sub-agent turn (mirrors OpenClaw's runEmbeddedPiAgent
        # with silentExpected=true). Leaking them into session history caused
        # the main agent to mimic flush behavior after compaction and emit
        # [!silent], prematurely terminating the run.
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.get("name") == "memory_write":
                    try:
                        raw_arguments = tool_call.get("arguments", "{}")
                        args = raw_arguments if isinstance(raw_arguments, dict) else _json.loads(raw_arguments)
                        content = args.get("content", "")
                        target = args.get("target", "session")
                        if content.strip():
                            if target == "task_memory":
                                memory_store.write_task_memory(content)
                                print(f"[MemoryFlush] Wrote {len(content)} chars to TASK_MEMORY.md")
                            else:
                                memory_store.append_to_session_log(content)
                                print(f"[MemoryFlush] Appended {len(content)} chars to session log")
                    except _json.JSONDecodeError as e:
                        # Log the truncated/malformed payload (bounded) so we
                        # can distinguish truncation from model-side bad JSON.
                        raw = tool_call.get("arguments", "")
                        raw_str = raw if isinstance(raw, str) else _json.dumps(raw)
                        preview = raw_str[:200]
                        suffix = "…" if len(raw_str) > 200 else ""
                        print(
                            f"[MemoryFlush] Tool call failed (JSON decode, "
                            f"likely max_tokens truncation): {e}; "
                            f"raw[{len(raw_str)}ch]={preview!r}{suffix}"
                        )
                    except Exception as e:
                        print(f"[MemoryFlush] Tool call failed: {e}")
        elif silent_token in reply_content:
            print("[MemoryFlush] Model replied silent — nothing to persist")
        else:
            print(f"[MemoryFlush] Model replied text without tool call (dropped): {reply_content[:100]}")

        session_mgr.record_memory_flush()
        print("[MemoryFlush] Flush recorded")

    except Exception as e:
        print(f"[MemoryFlush] Failed (non-fatal): {e}")
        # Record flush even on failure to prevent retry loops
        session_mgr.record_memory_flush()


def _serialize_flush_context(session_mgr: SessionManager) -> str:
    """Serialize full transcript into a text summary for the flush model.

    OpenClaw loads the full session file so the flush agent sees the entire
    conversation (agent-runner-memory.ts -> runEmbeddedPiAgent -> SessionManager.open).
    We can't pass raw transcript messages to litellm because they contain CUA-format
    content blocks (computer_call, function_call, tool_result) that require tool_call_id
    pairing. Instead, serialize the entire conversation as text so the flush model can
    read it without format constraints.
    """
    history = session_mgr.load_history()
    parts: list[str] = []
    for entry in history:
        if entry.type != "message":
            continue
        msg_data = entry.data.get("message", {})
        role = msg_data.get("role", "unknown")
        content = msg_data.get("content", "")
        block_texts = _serialize_content_blocks(content)
        if block_texts:
            parts.append(f"[{role}] {block_texts}")
    return "\n".join(parts)


def _serialize_content_blocks(content: Any) -> str:
    """Convert content (string or list of content blocks) to a text representation."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif not isinstance(block, dict):
            continue
        elif block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "computer_call":
            # Handle both "action" (computer-use-preview) and "actions" (GPT 5.4)
            actions_list = block.get("actions")  # GPT 5.4: array
            if actions_list is None:
                single = block.get("action", {})
                actions_list = [single] if single else []
            for action in actions_list:
                action_type = action.get("type", "unknown")
                detail = ""
                if action_type == "click":
                    detail = f" at ({action.get('x')}, {action.get('y')})"
                elif action_type == "keypress":
                    detail = f" {action.get('keys', [])}"
                elif action_type == "type":
                    detail = f" \"{action.get('text', '')}\""
                elif action_type == "scroll":
                    detail = f" ({action.get('x')}, {action.get('y')}) delta=({action.get('scroll_x', 0)}, {action.get('scroll_y', 0)})"
                parts.append(f"[action: {action_type}{detail}]")
        elif block.get("type") == "function_call":
            name = block.get("name", "unknown")
            args = block.get("arguments", "")
            if isinstance(args, str) and len(args) > 200:
                args = args[:200] + "..."
            parts.append(f"[tool_call: {name}({args})]")
        elif block.get("type") == "tool_result":
            result_content = block.get("content", "")
            if isinstance(result_content, str) and len(result_content) > 200:
                result_content = result_content[:200] + "..."
            parts.append(f"[tool_result: {result_content}]")
        elif block.get("type") == "thinking":
            thinking_text = block.get("thinking", "")
            if isinstance(thinking_text, str) and len(thinking_text) > 200:
                thinking_text = thinking_text[:200] + "..."
            parts.append(f"[thinking: {thinking_text}]")
        else:
            parts.append(f"[{block.get('type', 'unknown')}]")
    return " ".join(parts).strip()
