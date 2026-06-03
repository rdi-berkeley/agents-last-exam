"""
SessionManager — Cross-run session persistence for the OpenClaw agent harness.

Reproduces OpenClaw's session persistence (openclaw/src/config/sessions/transcript.ts)
adapted for CUA's single-task benchmark context. Keeps 3 of 10 OpenClaw JSONL entry types
(session, message, compaction) and drops 7 UI/multi-model types irrelevant to CUA.

Key differences from OpenClaw:
- 3 of 10 entry types (session, message, compaction)
- Single JSONL file per task (session headers mark run boundaries)
- Run numbers derived from transcript session headers (not stored in state.json)
- Task-scoped: sessions_dir/<task_id>/ vs OpenClaw's agent-scoped routing keys

Reasoning retention policy (US-OC-046):
  Thinking blocks (including thinkingSignature metadata) are retained in canonical
  session logs at write time. Sanitization — drop, preserve, or downgrade — is
  applied at replay time via TranscriptPolicy resolved from the target model, not
  at write time. This preserves transcript fidelity while allowing provider-aware
  cleanup on replay surfaces (compaction rebuild, session restore, history send).
"""

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import _normalize_actions
from .memory_flush_policy import (  # re-exported for back-compat
    DEFAULT_COMPACTION_RATIO,
    DEFAULT_MEMORY_FLUSH_FORCE_TRANSCRIPT_BYTES,
    DEFAULT_MEMORY_FLUSH_RESERVE_TOKENS_FLOOR,
    DEFAULT_MEMORY_FLUSH_SOFT_THRESHOLD_TOKENS,
    MEMORY_FLUSH_PROMPT,
    MEMORY_FLUSH_SYSTEM_PROMPT,
    SILENT_REPLY_TOKEN,
    has_already_flushed_for_current_compaction,
    should_run_memory_flush,
)
from .prompt_report import build_system_prompt_report  # re-exported for back-compat

DEFAULT_BASE_DIR = "openclaw_sessions"


@dataclass
class TokenUsage:
    """Cumulative token usage tracking.

    Uses input_tokens/output_tokens naming to match CUA SDK (OpenAI Responses API format).
    Cache token fields track Anthropic prompt caching usage.

    Note: context window size (model capacity) is stored on SessionState.context_tokens,
    NOT here. This class tracks cumulative API usage only.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def accumulate(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read += cache_read
        self.cache_write += cache_write

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenUsage":
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_read=data.get("cache_read", 0),
            cache_write=data.get("cache_write", 0),
        )


@dataclass
class SessionState:
    """Cross-run metadata persisted in state.json.

    Run numbers are derived from transcript session headers, not stored here.
    """

    task_id: str
    step_count: int = 0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    compaction_count: int = 0
    compaction_summaries: list[str] = field(default_factory=list)
    model: str = ""
    contextTokens: int = 0  # Context window size (model capacity), NOT usage. Matches OpenClaw's top-level contextTokens.
    system_prompt_report: dict[str, Any] | None = None
    memory_flush_at: str | None = None
    memory_flush_compaction_count: int | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "total_tokens": self.total_tokens.to_dict(),
            "compaction_count": self.compaction_count,
            "compaction_summaries": self.compaction_summaries,
            "model": self.model,
            "contextTokens": self.contextTokens,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.system_prompt_report is not None:
            d["system_prompt_report"] = self.system_prompt_report
        if self.memory_flush_at is not None:
            d["memory_flush_at"] = self.memory_flush_at
        if self.memory_flush_compaction_count is not None:
            d["memory_flush_compaction_count"] = self.memory_flush_compaction_count
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        tokens_data = data.get("total_tokens", {})
        return cls(
            task_id=data["task_id"],
            step_count=data.get("step_count", 0),
            total_tokens=TokenUsage.from_dict(tokens_data),
            compaction_count=data.get("compaction_count", 0),
            compaction_summaries=list(data.get("compaction_summaries", [])),
            model=data.get("model", ""),
            contextTokens=data.get("contextTokens", 0),
            system_prompt_report=data.get("system_prompt_report"),
            memory_flush_at=data.get("memory_flush_at"),
            memory_flush_compaction_count=data.get("memory_flush_compaction_count"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class TranscriptEntry:
    """A single JSONL transcript entry with parentId chain.

    Discriminated by `type`:
    - session: version, task_id, run_number, model
    - message: message.{role, content, usage?, stopReason?}
    - compaction: summary, firstKeptEntryId, tokensBefore
    """

    type: str
    id: str
    parent_id: str | None
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "type": self.type,
            "id": self.id,
            "parentId": self.parent_id,
            "timestamp": self.timestamp,
        }
        entry.update(self.data)
        return entry

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptEntry":
        core_keys = {"type", "id", "parentId", "timestamp"}
        extra = {k: v for k, v in data.items() if k not in core_keys}
        return cls(
            type=data["type"],
            id=data["id"],
            parent_id=data.get("parentId"),
            timestamp=data["timestamp"],
            data=extra,
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "entry") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class SessionManager:
    """Manages cross-run session state and JSONL transcripts for a single task.

    Storage layout:
        <base_dir>/<task_id>/state.json       — cross-run metadata
        <base_dir>/<task_id>/transcript.jsonl  — append-only conversation log

    Uses the None-sentinel pattern from MemoryStore for base_dir defaulting.
    """

    def __init__(self, task_id: str, base_dir: str | Path | None = None):
        self.task_id = task_id
        self._base_dir = Path(base_dir) if base_dir is not None else Path(DEFAULT_BASE_DIR)
        self._state: SessionState | None = None
        self._last_entry_id: str | None = None

    @property
    def task_dir(self) -> Path:
        return self._base_dir / self.task_id

    @property
    def state_path(self) -> Path:
        return self.task_dir / "state.json"

    @property
    def transcript_path(self) -> Path:
        return self.task_dir / "transcript.jsonl"

    def load_state(self) -> SessionState | None:
        """Load state from state.json. Returns None if missing or corrupt."""
        if not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return SessionState.from_dict(data)
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def save_state(self) -> None:
        """Persist current state to state.json."""
        if self._state is None:
            return
        self._state.updated_at = _now_iso()
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self._state.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    def init_session(self, model: str = "") -> SessionState:
        """Initialize a new run session.

        Loads existing state (if any), preserves cumulative step_count and
        tokens, sets model, and appends a session header entry to
        transcript.jsonl.

        Run number is derived by counting existing session headers in the
        transcript rather than being stored in state.json.

        Returns the updated SessionState.
        """
        existing = self.load_state()
        now = _now_iso()

        if existing is not None:
            self._state = existing
            self._state.model = model
            self._state.updated_at = now
            # Reset flush guard so each run gets a fresh flush opportunity.
            # Without this, a flush from the previous run blocks the current
            # run's first compaction cycle (memory_flush_compaction_count ==
            # compaction_count persists across runs). OpenClaw resets on
            # /new and /reset (session.ts:525-526).
            self._state.memory_flush_compaction_count = None
            self._state.memory_flush_at = None
        else:
            self._state = SessionState(
                task_id=self.task_id,
                step_count=0,
                model=model,
                created_at=now,
                updated_at=now,
            )

        self.save_state()

        # Derive run_number from transcript session headers
        run_number = self._count_session_headers() + 1

        # Append session header to transcript
        entry = TranscriptEntry(
            type="session",
            id=_new_id("sess"),
            parent_id=None,
            timestamp=now,
            data={
                "version": 1,
                "task_id": self.task_id,
                "run_number": run_number,
                "model": model,
            },
        )
        self._append_entry(entry)
        self._last_entry_id = entry.id

        return self._state

    def append_message(
        self,
        role: str,
        content: str | list[dict[str, Any]],
        usage: dict[str, Any] | None = None,
        stop_reason: str | None = None,
        api: str | None = None,
    ) -> TranscriptEntry:
        """Append a message entry to the transcript.

        Mirrors OpenClaw's transcript format where content is an array of typed blocks:
        text, function_call, tool_result, image, computer_call, etc.

        Args:
            role: "user", "assistant", or "tool"
            content: Text string (auto-wrapped as [{type: "text", text: ...}])
                    or a content array of typed blocks
            usage: Optional dict with input/output/total/cost keys
            stop_reason: Optional stop reason (e.g. "tool_use", "end_turn")
            api: Optional API identifier for observability (e.g. "openai-responses")

        Returns the created TranscriptEntry.
        """
        if isinstance(content, str):
            content_array = [{"type": "text", "text": content}]
        else:
            content_array = content

        message_data: dict[str, Any] = {"role": role, "content": content_array}
        if usage is not None:
            message_data["usage"] = usage
        if stop_reason is not None:
            message_data["stopReason"] = stop_reason
        if api is not None:
            message_data["api"] = api

        entry = TranscriptEntry(
            type="message",
            id=_new_id("msg"),
            parent_id=self._last_entry_id,
            timestamp=_now_iso(),
            data={"message": message_data},
        )
        self._append_entry(entry)
        self._last_entry_id = entry.id
        return entry

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
    ) -> TranscriptEntry:
        """Append a compaction entry to the transcript and update state.

        Args:
            summary: The compaction summary text
            first_kept_entry_id: ID of the first entry kept after compaction
            tokens_before: Token count before compaction

        Returns the created TranscriptEntry.
        """
        entry = TranscriptEntry(
            type="compaction",
            id=_new_id("cmp"),
            parent_id=self._last_entry_id,
            timestamp=_now_iso(),
            data={
                "summary": summary,
                "firstKeptEntryId": first_kept_entry_id,
                "tokensBefore": tokens_before,
            },
        )
        self._append_entry(entry)
        self._last_entry_id = entry.id

        # Update state
        if self._state is not None:
            self._state.compaction_count += 1
            self._state.compaction_summaries.append(summary)
            self.save_state()

        return entry

    def load_history(self, run_number: int | None = None) -> list[TranscriptEntry]:
        """Load transcript entries, optionally filtered to a specific run.

        Args:
            run_number: If provided, only return entries from this run.
                       If None, return all entries.

        Returns list of TranscriptEntry objects.
        """
        if not self.transcript_path.exists():
            return []

        entries: list[TranscriptEntry] = []
        current_run: int | None = None
        in_target_run = run_number is None  # If no filter, include everything

        for line in self.transcript_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry = TranscriptEntry.from_dict(data)

            if entry.type == "session":
                current_run = entry.data.get("run_number")
                if run_number is not None:
                    in_target_run = current_run == run_number

            if in_target_run:
                entries.append(entry)

        return entries

    def update_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token usage and persist."""
        if self._state is not None:
            self._state.total_tokens.accumulate(input_tokens, output_tokens)
            self.save_state()

    def get_step_count(self) -> int:
        """Return the current cumulative step count."""
        if self._state is not None:
            return self._state.step_count
        return 0

    def update_step_count(self, step: int) -> None:
        """Update the current step count and persist."""
        if self._state is not None:
            self._state.step_count = step
            self.save_state()

    def add_compaction_summary(self, summary: str) -> None:
        """Add a compaction summary to state.json (without a transcript entry)."""
        if self._state is not None:
            self._state.compaction_count += 1
            self._state.compaction_summaries.append(summary)
            self.save_state()

    def get_compaction_summaries(self) -> list[str]:
        """Return compaction summaries from state."""
        if self._state is not None:
            return list(self._state.compaction_summaries)
        loaded = self.load_state()
        if loaded is not None:
            return list(loaded.compaction_summaries)
        return []

    def record_memory_flush(self) -> None:
        """Record that a memory flush occurred at the current compaction count.

        Sets memory_flush_at to now and memory_flush_compaction_count to the
        current compaction_count, enforcing a one-flush-per-compaction-cycle
        invariant via has_already_flushed_for_current_compaction().

        Based on OpenClaw's memory flush tracking
        (openclaw/src/auto-reply/reply/memory-flush.ts:176-182).
        """
        if self._state is not None:
            self._state.memory_flush_at = _now_iso()
            self._state.memory_flush_compaction_count = self._state.compaction_count
            self.save_state()

    def set_system_prompt_report(self, report: dict[str, Any]) -> None:
        """Store a system prompt report in state.json."""
        if self._state is not None:
            self._state.system_prompt_report = report
            self.save_state()

    def _count_session_headers(self) -> int:
        """Count the number of session header entries in the transcript."""
        if not self.transcript_path.exists():
            return 0
        count = 0
        for line in self.transcript_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "session":
                count += 1
        return count

    def _append_entry(self, entry: TranscriptEntry) -> None:
        """Append a single JSONL line to the transcript file."""
        self.task_dir.mkdir(parents=True, exist_ok=True)
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")


# Memory-flush policy (constants + should_run_memory_flush / has_already_flushed_
# for_current_compaction) moved to memory_flush_policy.py; re-exported below.


# ---------------------------------------------------------------------------
# Transcript Replay — Cross-run continuity (US-OC-012)
# Converts transcript entries to API messages, sanitizes stale data,
# and limits history to prevent context overflow.
#
# Based on OpenClaw's replay pipeline (pi-embedded-runner/sanitizeSessionHistory,
# limitHistoryTurns, sanitizeToolUseResultPairing).
# ---------------------------------------------------------------------------

_THINKING_BLOCK_RE = re.compile(r"<THINKING>.*?</THINKING>", re.DOTALL)

# Base64 data URL pattern (matches data:image/...;base64,...)
_BASE64_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}")


def _get_openai_reasoning_signature(value: Any) -> dict[str, str] | None:
    """Parse a persisted OpenAI thinkingSignature back into id/type metadata."""
    if not value:
        return None
    candidate = None
    if isinstance(value, str):
        trimmed = value.strip()
        if not (trimmed.startswith("{") and trimmed.endswith("}")):
            return None
        try:
            candidate = json.loads(trimmed)
        except (json.JSONDecodeError, ValueError):
            return None
    elif isinstance(value, dict):
        candidate = value
    else:
        return None

    if not isinstance(candidate, dict):
        return None
    item_id = candidate.get("id")
    item_type = candidate.get("type")
    if not isinstance(item_id, str) or not isinstance(item_type, str):
        return None
    return {"id": item_id, "type": item_type}


def build_replay_messages(entries: list[TranscriptEntry]) -> list[dict[str, Any]]:
    """Convert transcript entries into API message dicts for replay.

    Handles three entry types:
    - message: extracts role/content from entry.data["message"], strips stale metadata
    - compaction: replaces all messages before firstKeptEntryId with a single
      assistant summary message
    - session: skipped (run boundary markers, not API messages)

    Based on OpenClaw's replaceMessages pipeline
    (openclaw/pi-embedded-runner/sanitizeSessionHistory).
    """
    messages: list[dict[str, Any]] = []
    # Map entry IDs to their index in the messages list for compaction lookups
    entry_id_to_msg_index: dict[str, int] = {}

    for entry in entries:
        if entry.type == "session":
            continue

        if entry.type == "compaction":
            summary = entry.data.get("summary", "")
            first_kept_id = entry.data.get("firstKeptEntryId")

            if first_kept_id and first_kept_id in entry_id_to_msg_index:
                # Replace all messages before firstKeptEntryId with a summary
                cut_index = entry_id_to_msg_index[first_kept_id]
                kept = messages[cut_index:]
                messages = [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"[Compaction summary] {summary}"}],
                    }
                ] + kept
                # Rebuild index since positions shifted
                entry_id_to_msg_index = {}
                # We can't easily remap, but subsequent compactions will
                # reference entries added after this point
            else:
                # No matching entry found — just prepend the summary
                messages.insert(0, {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"[Compaction summary] {summary}"}],
                })
            continue

        if entry.type == "message":
            msg_data = entry.data.get("message", {})
            role = msg_data.get("role", "user")
            content = msg_data.get("content", "")

            # Map OpenClaw roles to standard API roles
            if role == "toolResult":
                role = "user"  # tool results are user messages in the API

            # Strip stale metadata (usage, stopReason, api)
            message: dict[str, Any] = {"role": role, "content": content}
            entry_id_to_msg_index[entry.id] = len(messages)
            messages.append(message)

    return messages


def sanitize_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize replayed messages for API compatibility.

    Five sanitization passes:
    1. Strip base64 images (huge, stale screenshots from prior runs)
    2. Strip thinking blocks (<THINKING>...</THINKING>) from assistant text
    3. Repair orphaned tool results (drop tool_result without matching call, and vice versa)
    4. Strip stale usage keys from all messages
    5. Ensure user-first ordering (Gemini compatibility)

    Based on OpenClaw's sanitizeSessionHistory and sanitizeToolUseResultPairing
    (openclaw/pi-embedded-runner/).
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        result.append(_sanitize_single_message(msg))

    # Pass 3: Repair orphaned tool results / calls
    result = _repair_orphaned_tool_pairs(result)

    # Pass 5: Ensure user-first ordering
    if result and result[0].get("role") != "user":
        result.insert(0, {"role": "user", "content": "[session history follows]"})

    return result


def _sanitize_single_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Apply per-message sanitization (passes 1, 2, 4)."""
    role = msg.get("role", "")
    content = msg.get("content", "")

    # Pass 4: Strip stale usage
    sanitized: dict[str, Any] = {"role": role}
    for k, v in msg.items():
        if k not in ("usage", "stopReason", "api", "role", "content"):
            sanitized[k] = v

    if isinstance(content, str):
        # Pass 1: Strip base64 images from text
        content = _BASE64_IMAGE_RE.sub("[image removed]", content)
        # Pass 2: Strip thinking blocks from assistant messages
        if role == "assistant":
            content = _THINKING_BLOCK_RE.sub("", content).strip()
        sanitized["content"] = content  # keep even if empty
        return sanitized

    if isinstance(content, list):
        sanitized_blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type", "")

            # Pass 1: Strip image blocks entirely
            if block_type in ("image", "image_url"):
                sanitized_blocks.append({
                    "type": "text",
                    "text": "[screenshot from prior run]",
                })
                continue

            # Pass 1: Strip base64 from source blocks
            if (
                isinstance(block.get("source"), dict)
                and block["source"].get("type") == "base64"
            ):
                sanitized_blocks.append({
                    "type": "text",
                    "text": "[screenshot from prior run]",
                })
                continue

            # Pass 1: Strip base64 data URLs from text blocks
            if block_type == "text":
                text = block.get("text", "")
                text = _BASE64_IMAGE_RE.sub("[image removed]", text)
                # Pass 2: Strip thinking blocks
                if role == "assistant":
                    text = _THINKING_BLOCK_RE.sub("", text).strip()
                sanitized_blocks.append({"type": "text", "text": text})
                continue

            # Keep other blocks as-is (function_call, computer_call, tool_result, etc.)
            sanitized_blocks.append(block)

        sanitized["content"] = sanitized_blocks
        return sanitized

    sanitized["content"] = content
    return sanitized


def _repair_orphaned_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove orphaned tool results and tool calls without matching pairs.

    For each tool_result content block, verify a matching function_call or
    computer_call with the same ID exists in an earlier assistant message.
    Drop orphaned results. Also drop function_call/computer_call blocks
    with no matching result in a later message.

    Based on OpenClaw's sanitizeToolUseResultPairing.
    """
    # Collect all tool call IDs from assistant messages
    call_ids: set[str] = set()
    result_ids: set[str] = set()

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = block.get("type", "")
            if block_type in ("function_call", "computer_call"):
                call_id = block.get("id", "")
                if call_id:
                    call_ids.add(call_id)
            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                if tool_use_id:
                    result_ids.add(tool_use_id)

    # IDs that have both a call and a result
    paired_ids = call_ids & result_ids

    # Filter messages: remove orphaned blocks
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            cleaned.append(msg)
            continue

        filtered_blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type", "")
            if block_type in ("function_call", "computer_call"):
                call_id = block.get("id", "")
                if call_id in paired_ids:
                    filtered_blocks.append(block)
                # else: orphaned call — drop
            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                if tool_use_id in paired_ids:
                    filtered_blocks.append(block)
                # else: orphaned result — drop
            else:
                filtered_blocks.append(block)

        if filtered_blocks:
            cleaned.append({**msg, "content": filtered_blocks})
        # If all blocks were removed, skip the message entirely

    return cleaned


def limit_history_turns(
    messages: list[dict[str, Any]], limit: int | None = None
) -> list[dict[str, Any]]:
    """Keep the last N user turns (and their associated responses).

    Iterates backwards, counting role="user" messages. When the count exceeds
    the limit, slices from that point forward.

    Based on OpenClaw's limitHistoryTurns (pi-embedded-runner/).

    Args:
        messages: List of API message dicts
        limit: Max number of user turns to keep. None or <= 0 means keep all.

    Returns:
        Sliced message list containing only the last `limit` user turns
        and their associated assistant/tool responses.
    """
    if limit is None or limit <= 0:
        return messages

    # Find the indices of all user messages
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]

    if len(user_indices) <= limit:
        return messages

    # Keep from the (limit)th-from-last user message onward
    cut_index = user_indices[-limit]
    return messages[cut_index:]


def _string_message_item(role: str, content: str) -> dict[str, Any]:
    """A Responses API message item from a plain string — assistant emits
    ``output_text``; every other role emits ``input_text``."""
    if role == "assistant":
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        }
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": content}],
    }


def convert_to_responses_api_items(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert replay messages from Chat Completions format to Responses API items.

    CUA SDK agent loops (anthropic.py, openai.py) dispatch on top-level ``type``
    fields (Responses API item format), not on ``role`` with nested content blocks
    (Chat Completions format).  Our transcript stores the latter, so without this
    conversion the loops drop function_call / computer_call blocks and ignore
    ``role: "tool"`` messages entirely.

    Unnests each Chat Completions message into one or more flat Responses API
    items that the loops can dispatch correctly:

    - User message → ``{type: "message", role: "user", content: [{type: "input_text", …}]}``
    - Assistant text → ``{type: "message", role: "assistant", content: [{type: "output_text", …}]}``
    - function_call block → ``{type: "function_call", call_id, name, arguments}``
    - computer_call block → ``{type: "computer_call", call_id, action}``
    - tool_result block → ``{type: "function_call_output", …}`` or
      ``{type: "computer_call_output", …}`` (matched by call_id to original call type)

    The ``id`` → ``call_id`` mapping matches what ``group_step_output`` stores
    (``"id"`` key in transcript) versus what the Responses API expects
    (``"call_id"`` key).

    Applied as the LAST step in the replay pipeline (after sanitize_history +
    limit_history_turns) so that orphaned-pair repair still works on the nested
    format with ``id`` / ``tool_use_id`` fields.

    US-OC-022: Replay Format Fix.
    """
    items: list[dict[str, Any]] = []
    # Track call types so tool results emit the correct output type
    # (computer_call → computer_call_output, function_call → function_call_output)
    call_type_map: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # --- String content ---
        if isinstance(content, str):
            items.append(_string_message_item(role, content))
            continue

        # --- List content (content blocks) ---
        if isinstance(content, list):
            if role == "assistant":
                unnested = _unnest_assistant_blocks(content, call_type_map)
                # Record call types for later tool result matching
                for item in unnested:
                    itype = item.get("type", "")
                    if itype in ("function_call", "computer_call"):
                        cid = item.get("call_id", "")
                        if cid:
                            call_type_map[cid] = itype
                items.extend(unnested)
            elif role in ("tool", "user"):
                # Check if this is a tool-result message (all blocks are tool_result)
                has_tool_results = any(
                    b.get("type") == "tool_result" for b in content if isinstance(b, dict)
                )
                if role == "tool" or (role == "user" and has_tool_results):
                    items.extend(_unnest_tool_blocks(content, call_type_map))
                else:
                    items.append(_wrap_user_content(content))
            else:
                # Unknown role — treat as user
                items.append(_wrap_user_content(content))
            continue

        # --- Fallback: wrap as user ---
        items.append(_string_message_item("user", str(content)))

    return _ensure_tool_adjacency(items)


def _ensure_tool_adjacency(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder items so each tool call is immediately followed by its output.

    Memory flush messages (user prompt + assistant reply) can be interleaved
    in the transcript between a tool call and its result, because the flush
    fires per-step while tool results may be logged in a later step.  The
    Anthropic API requires ``tool_result`` immediately after ``tool_use``,
    so we defer any non-output items that appear between a call and its
    matching output, then flush them after the output.
    """
    result: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    pending_call_ids: set[str] = set()

    for item in items:
        t = item.get("type", "")

        if t in ("function_call", "computer_call"):
            pending_call_ids.add(item.get("call_id", ""))
            result.append(item)
        elif t in ("function_call_output", "computer_call_output"):
            call_id = item.get("call_id", "")
            pending_call_ids.discard(call_id)
            result.append(item)
            # All pending calls resolved → flush deferred items
            if not pending_call_ids:
                result.extend(deferred)
                deferred = []
        elif pending_call_ids:
            # Non-tool item while tool calls are pending → defer
            deferred.append(item)
        else:
            result.append(item)

    # Flush any remaining deferred items
    result.extend(deferred)
    return result


def _flush_pending_text(
    items: list[dict[str, Any]],
    pending_text: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Emit accumulated text blocks as one assistant message (if any) and return
    a fresh, empty pending-text list."""
    if pending_text:
        items.append({
            "type": "message",
            "role": "assistant",
            "content": pending_text,
        })
    return []


def _unnest_assistant_blocks(
    blocks: list[dict[str, Any]],
    call_type_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Unnest an assistant message's content blocks into Responses API items.

    Text blocks are collected into a single ``message`` item; structured blocks
    (function_call, computer_call) become top-level items.  Text that precedes a
    structured block is flushed as a separate message item so ordering is
    preserved.
    """
    if call_type_map is None:
        call_type_map = {}
    items: list[dict[str, Any]] = []
    pending_text: list[dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "")
            if text:
                pending_text.append({"type": "output_text", "text": text})

        elif btype == "function_call":
            # Flush pending text before emitting the structured item
            pending_text = _flush_pending_text(items, pending_text)
            items.append({
                "type": "function_call",
                "call_id": block.get("id", block.get("call_id", "")),
                "name": block.get("name", ""),
                "arguments": block.get("arguments", ""),
            })

        elif btype == "computer_call":
            pending_text = _flush_pending_text(items, pending_text)
            call_id = block.get("id", block.get("call_id", ""))
            if call_id:
                call_type_map[call_id] = "computer_call"
            actions = _normalize_actions(block)
            action_desc = json.dumps(actions)[:200] if actions else "details unavailable"
            items.append({
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": f"[computer action: {action_desc}]",
                }],
            })

        elif btype == "thinking":
            signature = _get_openai_reasoning_signature(block.get("thinkingSignature"))
            if signature is None:
                continue
            pending_text = _flush_pending_text(items, pending_text)
            summary = []
            if block.get("thinking"):
                summary = [{
                    "type": "summary_text",
                    "text": block.get("thinking", ""),
                }]
            items.append({
                "type": signature["type"],
                "id": signature["id"],
                "summary": summary,
            })

        else:
            # Unknown block type — keep as text
            pending_text.append({"type": "output_text", "text": str(block)})

    # Flush remaining text
    _flush_pending_text(items, pending_text)

    return items


def _unnest_tool_blocks(
    blocks: list[dict[str, Any]],
    call_type_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert tool_result content blocks into top-level Responses API output items.

    Uses ``call_type_map`` (call_id → "function_call" | "computer_call") to emit
    the correct output type:
    - computer_call → ``computer_call_output`` with text (screenshot from prior
      turn is no longer available; OpenAI validates image data so we can't use
      a placeholder PNG)
    - function_call → ``function_call_output`` with the original text content
    """
    if call_type_map is None:
        call_type_map = {}
    items: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "tool_result":
            call_id = block.get("tool_use_id", block.get("call_id", ""))
            if call_type_map.get(call_id) == "computer_call":
                items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": f"[computer result: {block.get('content', '')[:200]}]",
                    }],
                })
            else:
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": block.get("content", ""),
                })
        elif btype == "computer_call_output":
            items.append({
                "type": "computer_call_output",
                "call_id": block.get("tool_use_id", block.get("call_id", "")),
                "output": block.get("content", ""),
            })
        # else: skip non-tool blocks in tool messages
    return items


def _wrap_user_content(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap user content blocks into a Responses API message item."""
    converted: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            converted.append({"type": "input_text", "text": block.get("text", "")})
        elif btype in ("input_text", "input_image"):
            converted.append(block)  # already correct format
        else:
            # Unknown — convert to text
            converted.append({"type": "input_text", "text": str(block)})
    return {
        "type": "message",
        "role": "user",
        "content": converted or [{"type": "input_text", "text": ""}],
    }


# build_system_prompt_report moved to prompt_report.py; re-exported below.
