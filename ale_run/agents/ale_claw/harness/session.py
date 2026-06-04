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

Reasoning retention policy:
  Thinking blocks (including thinkingSignature metadata) are retained in canonical
  session logs at write time. Sanitization — drop, preserve, or downgrade — is
  applied at replay time via TranscriptPolicy resolved from the target model, not
  at write time. This preserves transcript fidelity while allowing provider-aware
  cleanup on replay surfaces (compaction rebuild, session restore, history send).
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .memory.memory_flush_policy import (  # re-exported for back-compat
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
from .prompt import build_system_prompt_report  # re-exported for back-compat
from .context.replay import (  # re-exported for back-compat
    build_replay_messages,
    convert_to_responses_api_items,
    limit_history_turns,
    sanitize_history,
)

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


# Transcript Replay pipeline (build_replay_messages / sanitize_history /
# limit_history_turns / convert_to_responses_api_items + helpers) moved to
# replay.py; public entry points re-exported above.

# build_system_prompt_report lives in prompt.py (merged from prompt_report.py); re-exported below.
