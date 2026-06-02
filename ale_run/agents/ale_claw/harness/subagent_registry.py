"""
SubagentRegistry — Lifecycle tracking for spawned subagent runs with optional
disk persistence (append-only JSONL).

Reproduces OpenClaw's subagent registry (openclaw/src/agents/subagent-registry.ts,
subagent-registry.types.ts, subagent-registry.store.ts) adapted for CUA's
single-process asyncio model.

Key differences from OpenClaw:
- Append-only JSONL persistence (vs. full-map JSON overwrite).
- Direct method calls instead of gateway lifecycle event listeners.
- asyncio.Queue instead of steer()-based mid-stream injection for result delivery.
- Depth-1 only (no recursive delegation) for V1.
- No versioned format migration — forward-compatible via from_dict() defaults.

Two subagent types share this registry:
- general: async one-shot workers (planning, analysis, memory). Results pushed
  to completion queue and drained between main agent steps.
- gui: blocking VM relay loops. Results returned directly via tool call,
  never pushed to the completion queue.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class SubagentType(str, Enum):
    """Discriminator for the two subagent execution models."""

    GENERAL = "general"
    GUI = "gui"


class SubagentStatus(str, Enum):
    """Lifecycle status for a subagent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    KILLED = "killed"


_TERMINAL_STATUSES = frozenset({
    SubagentStatus.COMPLETE,
    SubagentStatus.ERROR,
    SubagentStatus.KILLED,
})


def _subagent_transcript_path(base: Path, run_id: str) -> Path:
    """On-disk transcript JSONL for a subagent run: ``<base>/subagents/<run_id>/transcript.jsonl``."""
    return base / "subagents" / run_id / "transcript.jsonl"


@dataclass
class SubagentUsage:
    """Token usage tracking for a single subagent run."""

    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubagentUsage":
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )


@dataclass
class SubagentRun:
    """A single subagent run record.

    Based on OpenClaw's SubagentRunRecord (subagent-registry.types.ts) — keeps
    the 11 fields needed for lifecycle tracking and result delivery, drops 20+
    fields related to gateway routing, announce retry, and disk persistence.
    """

    run_id: str
    type: SubagentType
    task: str
    label: str
    model: str
    status: SubagentStatus = SubagentStatus.PENDING
    result_text: str | None = None
    error_message: str | None = None
    created_at: str = ""
    ended_at: str | None = None
    usage: SubagentUsage = field(default_factory=SubagentUsage)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "type": self.type.value,
            "task": self.task,
            "label": self.label,
            "model": self.model,
            "status": self.status.value,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "usage": self.usage.to_dict(),
        }
        if self.result_text is not None:
            d["result_text"] = self.result_text
        if self.error_message is not None:
            d["error_message"] = self.error_message
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubagentRun":
        """Reconstruct a SubagentRun from a dict (inverse of to_dict)."""
        return cls(
            run_id=data["run_id"],
            type=SubagentType(data["type"]),
            task=data.get("task", ""),
            label=data.get("label", ""),
            model=data.get("model", ""),
            status=SubagentStatus(data["status"]),
            result_text=data.get("result_text"),
            error_message=data.get("error_message"),
            created_at=data.get("created_at", ""),
            ended_at=data.get("ended_at"),
            usage=SubagentUsage.from_dict(data.get("usage", {})),
        )


class SubagentLimitError(Exception):
    """Raised when the concurrency limit for general subagents is reached."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return f"sub-{uuid.uuid4().hex[:12]}"


class SubagentRegistry:
    """Registry tracking all subagent runs with optional JSONL disk persistence.

    The completion queue (asyncio.Queue) collects finished general subagent runs.
    GUI subagent runs are blocking and return results directly — they are tracked
    in the registry for observability but never pushed to the queue.

    Concurrency limit (max_concurrent) applies only to general subagents.

    When ``persist_path`` is provided, every state transition appends the affected
    run's record as a single JSONL line. On restore, last entry per run_id wins
    (natural dedup for append-only writes).
    """

    def __init__(
        self, max_concurrent: int = 3, persist_path: Path | None = None
    ) -> None:
        self._runs: dict[str, SubagentRun] = {}
        self._completion_queue: asyncio.Queue[SubagentRun] = asyncio.Queue()
        self._post_delegation_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task] = {}
        self._inboxes: dict[str, "asyncio.Queue[str]"] = {}
        self._max_concurrent = max_concurrent
        self._persist_path = persist_path

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def persist_path(self) -> Path | None:
        return self._persist_path

    @property
    def completion_queue(self) -> "asyncio.Queue[SubagentRun]":
        """Read-only access to the completion queue for introspection/tests."""
        return self._completion_queue

    def _persist_run(self, run_id: str) -> None:
        """Append a single run's current state as one JSONL line.

        No-op when persist_path is None or run_id is unknown.
        """
        if self._persist_path is None:
            return
        run = self._runs.get(run_id)
        if run is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run.to_dict()) + "\n")

    def register(
        self,
        *,
        type: SubagentType,
        task: str,
        label: str = "",
        model: str = "",
    ) -> SubagentRun:
        """Create and register a new subagent run.

        For GENERAL type, raises SubagentLimitError if active_count() >= max_concurrent.
        GUI type is exempt from the concurrency limit (blocking, at most 1 at a time
        enforced by the tool handler, not the registry).
        """
        if type == SubagentType.GENERAL and self.active_count() >= self._max_concurrent:
            raise SubagentLimitError(
                f"Cannot spawn general subagent: {self.active_count()} active "
                f"(limit {self._max_concurrent})"
            )

        run = SubagentRun(
            run_id=_new_run_id(),
            type=type,
            task=task,
            label=label,
            model=model,
            status=SubagentStatus.PENDING,
            created_at=_now_iso(),
        )
        self._runs[run.run_id] = run
        self._persist_run(run.run_id)
        return run

    def mark_running(self, run_id: str) -> None:
        """Transition a PENDING run to RUNNING."""
        run = self._runs.get(run_id)
        if run is None:
            return
        if run.status == SubagentStatus.PENDING:
            run.status = SubagentStatus.RUNNING
            self._persist_run(run_id)

    def complete(
        self,
        run_id: str,
        result_text: str,
        usage: SubagentUsage | None = None,
    ) -> None:
        """Mark a run as successfully completed.

        Sets result_text, status, ended_at. Pushes the run to the completion
        queue for drain_completions().
        """
        run = self._runs.get(run_id)
        if run is None or run.status in _TERMINAL_STATUSES:
            return
        run.status = SubagentStatus.COMPLETE
        run.result_text = result_text
        run.ended_at = _now_iso()
        if usage is not None:
            run.usage = usage
        self._inboxes.pop(run_id, None)
        self._persist_run(run_id)
        self._completion_queue.put_nowait(run)

    def fail(
        self,
        run_id: str,
        error_message: str,
        usage: SubagentUsage | None = None,
    ) -> None:
        """Mark a run as failed with an error message.

        Pushes to the completion queue so the main loop can report the failure.
        """
        run = self._runs.get(run_id)
        if run is None or run.status in _TERMINAL_STATUSES:
            return
        run.status = SubagentStatus.ERROR
        run.error_message = error_message
        run.ended_at = _now_iso()
        if usage is not None:
            run.usage = usage
        self._inboxes.pop(run_id, None)
        self._persist_run(run_id)
        self._completion_queue.put_nowait(run)

    def kill(self, run_id: str) -> None:
        """Mark a run as killed.

        Does NOT cancel the asyncio.Task — that's the caller's responsibility
        (US-SUB-002/005). This only updates the registry record.
        """
        run = self._runs.get(run_id)
        if run is None or run.status in _TERMINAL_STATUSES:
            return
        run.status = SubagentStatus.KILLED
        run.ended_at = _now_iso()
        self._inboxes.pop(run_id, None)
        self._persist_run(run_id)

    def attach_task(self, run_id: str, task: asyncio.Task) -> None:
        """Associate an asyncio.Task with a run so ``kill_run`` can cancel it.

        Idempotent: silently ignores unknown run_ids so the caller (US-SUB-005
        ``DelegateGeneralTool``) does not need to re-check the registry after
        ``register``.
        """
        if run_id in self._runs:
            self._tasks[run_id] = task

    def attach_inbox(self, run_id: str, inbox: "asyncio.Queue[str]") -> None:
        """Associate a steer inbox with a run (US-SUB-009).

        Idempotent: silently ignores unknown run_ids.
        """
        if run_id in self._runs:
            self._inboxes[run_id] = inbox

    def get_inbox(self, run_id: str) -> "asyncio.Queue[str] | None":
        """Return the steer inbox for a run, or None if not attached / unknown."""
        return self._inboxes.get(run_id)

    def kill_run(self, run_id: str) -> bool:
        """Cancel the underlying asyncio.Task and transition the run to KILLED.

        Returns True if a kill signal was issued, False if the run is unknown
        or already in a terminal status. The wrapper's
        ``except asyncio.CancelledError`` path also calls ``self.kill(run_id)``
        after cancellation propagates; the second call is a no-op because
        ``kill`` bails on terminal statuses.
        """
        run = self._runs.get(run_id)
        if run is None or run.status in _TERMINAL_STATUSES:
            return False

        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
        self.kill(run_id)
        return True

    def active_count(self) -> int:
        """Count of PENDING + RUNNING general subagent runs."""
        return sum(
            1
            for run in self._runs.values()
            if run.type == SubagentType.GENERAL
            and run.status in (SubagentStatus.PENDING, SubagentStatus.RUNNING)
        )

    def get_run(self, run_id: str) -> SubagentRun | None:
        """Look up a single run by ID."""
        return self._runs.get(run_id)

    def list_runs(
        self, *, status_filter: SubagentStatus | None = None
    ) -> list[SubagentRun]:
        """List all runs, optionally filtered by status."""
        if status_filter is None:
            return list(self._runs.values())
        return [r for r in self._runs.values() if r.status == status_filter]

    def drain_completions(self) -> list[SubagentRun]:
        """Return all completed/failed general subagent runs from the queue.

        Non-blocking — uses get_nowait() to drain everything available.
        Returns an empty list if no completions are pending.
        """
        results: list[SubagentRun] = []
        while True:
            try:
                results.append(self._completion_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return results

    def enqueue_post_delegation(self, message: dict[str, Any]) -> None:
        """Enqueue a pre-built user-message dict for post-delegation injection.

        Used by ``DelegateGUITool`` to hand a fresh VM screenshot back to the
        main agent as a ``{role: user, content: [text, image_url]}`` message
        (US-SUB-006). The main agent loop drains this queue at the same seam
        as ``drain_completions`` so the message lands in ``new_items`` before
        the next ``predict_step``.
        """
        self._post_delegation_queue.put_nowait(message)

    def drain_post_delegation(self) -> list[dict[str, Any]]:
        """Return all pending post-delegation user messages from the queue.

        Non-blocking FIFO drain — mirrors ``drain_completions`` shape.
        Messages are already in final ``{role, content}`` form; the loop
        extends ``new_items`` with them verbatim.
        """
        results: list[dict[str, Any]] = []
        while True:
            try:
                results.append(self._post_delegation_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return results

    def restore(self) -> int:
        """Load runs from the JSONL persist file; mark orphaned non-terminal runs.

        Deduplication: last entry for each run_id wins (append-only + last-write-wins).
        Orphan detection: runs with status pending|running from a prior session are
        transitioned to error with a descriptive message.

        Returns the number of prior-session runs loaded (0 if no file or no path).
        """
        if self._persist_path is None or not self._persist_path.exists():
            return 0

        loaded: dict[str, SubagentRun] = {}
        for line in self._persist_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            try:
                run = SubagentRun.from_dict(data)
            except (KeyError, ValueError):
                continue
            loaded[run.run_id] = run

        for run in loaded.values():
            if run.status in (SubagentStatus.PENDING, SubagentStatus.RUNNING):
                run.status = SubagentStatus.ERROR
                run.error_message = "stalled: prior session ended before completion"
                run.ended_at = _now_iso()

        count = len(loaded)
        loaded.update(self._runs)
        self._runs = loaded
        return count

    def completed_runs(self) -> list[dict[str, Any]]:
        """Return all runs with terminal status, enriched with transcript paths.

        Each returned dict is ``run.to_dict()`` plus a ``transcript_path`` key
        (str or None) pointing to the subagent's on-disk transcript JSONL.
        """
        results: list[dict[str, Any]] = []
        for run in self._runs.values():
            if run.status not in _TERMINAL_STATUSES:
                continue
            d = run.to_dict()
            if self._persist_path is not None:
                transcript = _subagent_transcript_path(
                    self._persist_path.parent, run.run_id
                )
                d["transcript_path"] = str(transcript) if transcript.exists() else None
            else:
                d["transcript_path"] = None
            results.append(d)
        return results

    def to_snapshot(self) -> list[dict[str, Any]]:
        """Serialize all runs for transcript observability."""
        return [run.to_dict() for run in self._runs.values()]
