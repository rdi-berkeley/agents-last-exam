"""Canonical message type definitions (TypedDicts).

Leaf module split out of ``canonical.py``: the typed content-block and
role-based message shapes that the whole canonical pipeline (adapters + the
sanitize passes) operates on. Has no harness dependencies, so any canonical
submodule can import these without an import cycle.

Field conventions match OpenClaw / Anthropic:
  - ``id`` on FunctionCallBlock / ComputerCallBlock (not ``call_id``)
  - ``tool_use_id`` on ToolResultBlock
  - ``call_id`` is Responses API only — adapters map ``id`` → ``call_id``
"""

from __future__ import annotations

from typing import Any, Literal, Union

from typing_extensions import NotRequired, TypedDict


# ---------------------------------------------------------------------------
# Content block types
# ---------------------------------------------------------------------------


class TextBlock(TypedDict):
    """Plain text content block."""

    type: Literal["text"]
    text: str


class FunctionCallBlock(TypedDict):
    """Function (tool) call issued by the assistant."""

    type: Literal["function_call"]
    id: str
    name: str
    arguments: str  # JSON string


class ComputerCallBlock(TypedDict):
    """Computer action call issued by the assistant.

    ``actions`` is always a list — singular ``action`` dicts are normalized
    to ``[action]`` at ingestion by :func:`normalize_to_canonical`.
    """

    type: Literal["computer_call"]
    id: str
    actions: list[dict[str, Any]]


class ToolResultBlock(TypedDict):
    """Result of a function or computer call."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: str
    is_error: NotRequired[bool]


class ThinkingBlock(TypedDict):
    """Provider-specific reasoning / thinking block.

    ``thinkingSignature`` is a tamper-proof token validated by the API on
    re-submission — if missing, malformed, or from a different provider the
    API rejects the request.
    """

    type: Literal["thinking"]
    thinking: str
    thinkingSignature: NotRequired[str]


class CompactionSummaryBlock(TypedDict):
    """Summary of compacted (older) conversation history.

    Distinct from TextBlock so downstream passes can identify and skip
    compaction summaries during repair / sanitization.
    """

    type: Literal["compaction_summary"]
    text: str


ContentBlock = Union[
    TextBlock,
    FunctionCallBlock,
    ComputerCallBlock,
    ToolResultBlock,
    ThinkingBlock,
    CompactionSummaryBlock,
]


# ---------------------------------------------------------------------------
# Canonical message
# ---------------------------------------------------------------------------


class CanonicalMessage(TypedDict):
    """Role-based message with typed content blocks.

    Mirrors OpenClaw's AgentMessage: role + content array + optional
    stop_reason.  All pipeline passes (repair, sanitization, format
    conversion) operate on ``list[CanonicalMessage]``.
    """

    role: Literal["user", "assistant", "tool", "system"]
    content: list[ContentBlock]
    stop_reason: NotRequired[str]
