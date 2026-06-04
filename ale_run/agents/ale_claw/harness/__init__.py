"""OpenClaw agent modules — modular components for the OpenClaw agent harness.

Components:
  - OpenClawComputerAgent: ComputerAgent subclass with mid-loop compaction
  - PromptBuilder: assembles structured system instructions from composable sections
  - PromptConfig / SectionConfig: section toggle configuration
  - ContextFile: bootstrap file injection container
  - MemoryStore / SearchResult: task-workspace persistent memory storage
  - MemorySearchTool / MemoryGetTool / MemoryWriteTool: agent memory tools
  - SessionManager / SessionState / TokenUsage / TranscriptEntry: session persistence
  - has_already_flushed_for_current_compaction / should_run_memory_flush: memory flush guards
  - MEMORY_FLUSH_PROMPT / MEMORY_FLUSH_SYSTEM_PROMPT / SILENT_REPLY_TOKEN: flush prompts
  - ContextOverflowCallback / is_context_overflow_error: context overflow detection
  - CompactionResult / compact_messages: compaction pipeline
  - ToolPairingRepairReport / repair_tool_use_result_pairing: tool pairing repair
  - split_preserved_recent_turns: recent turns preservation
  - build_tools / get_tool_summaries / ToolLoggingCallback: tool registry & logging
  - build_replay_messages / sanitize_history / limit_history_turns: transcript replay
  - run_memory_flush: standalone memory flush module
  - ThinkLevel / ThinkingConfig / resolve_thinking_default: thinking level system
  - CanonicalMessage / ContentBlock types / converters: canonical message format
  - sanitize_items / repair_orphaned_pairs / ensure_valid_ordering: sanitize pipeline
"""

from .adapters import (
    OpenClawImageRetentionCallback,
    OpenClawTrajectorySaverCallback,
)
from .agent_loop import OpenClawComputerAgent
from .tools.analyze_image import AnalyzeImageTool
from .model.cache_policy import (
    OPENCLAW_CACHE_BOUNDARY,
    apply_openclaw_cache_markers,
    supports_anthropic_cache,
)
from .tools.computer_handler import OpenClawComputerHandler
from .canonical.canonical import (
    CanonicalMessage,
    CompactionSummaryBlock,
    ComputerCallBlock,
    ContentBlock,
    FunctionCallBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    canonical_to_anthropic_messages,
    canonical_to_responses_api,
    ensure_valid_ordering,
    normalize_to_canonical,
    repair_orphaned_pairs,
    sanitize_items,
)
from .context.context import (
    CompactionResult,
    ContextOverflowCallback,
    ToolPairingRepairReport,
    compact_messages,
    is_context_overflow_error,
    repair_tool_use_result_pairing,
    split_preserved_recent_turns,
)
from .memory.memory import (
    MemoryGetTool,
    MemorySearchTool,
    MemoryStore,
    MemoryWriteTool,
    SearchResult,
)
from .memory.memory_flush import run_memory_flush
from .prompt import ContextFile, PromptBuilder, PromptConfig, SectionConfig
from .session import (
    DEFAULT_MEMORY_FLUSH_RESERVE_TOKENS_FLOOR,
    MEMORY_FLUSH_PROMPT,
    MEMORY_FLUSH_SYSTEM_PROMPT,
    SILENT_REPLY_TOKEN,
    SessionManager,
    SessionState,
    TokenUsage,
    TranscriptEntry,
    build_replay_messages,
    build_system_prompt_report,
    convert_to_responses_api_items,
    has_already_flushed_for_current_compaction,
    limit_history_turns,
    sanitize_history,
    should_run_memory_flush,
)
from .subagent.subagent_registry import (
    SubagentLimitError,
    SubagentRegistry,
    SubagentRun,
    SubagentStatus,
    SubagentType,
    SubagentUsage,
)
from .subagent.subagent_tools import DelegateGeneralTool, DelegateGUITool, SubagentsTool
from .model.thinking import ThinkingConfig, ThinkLevel, resolve_thinking_default
from .tools.tools import ToolLoggingCallback, build_tools, get_tool_summaries
from .tools.tools_fs import EditFileTool, ReadFileTool, WriteFileTool
from .tools.tools_shell import ExecTool
from .tools.tools_web import WebFetchTool, WebSearchTool
from .context.transcript import group_step_output

# Side-effect import — registers the OpenRouter unified loop with
# agent.decorators._AGENT_REGISTRY. Lives here (rather than agent/loops/)
# so sparse-checkout consumers that only pull the openclaw subpackage
# still get the chat-completions OpenRouter route instead of falling
# through to loops/openai.py (Responses API).
from .model import unified_loop  # noqa: F401

__all__ = [
    "OpenClawComputerAgent",
    "OpenClawComputerHandler",
    "OpenClawImageRetentionCallback",
    "OpenClawTrajectorySaverCallback",
    "CanonicalMessage",
    "CompactionSummaryBlock",
    "ComputerCallBlock",
    "ContentBlock",
    "FunctionCallBlock",
    "TextBlock",
    "ThinkingBlock",
    "ToolResultBlock",
    "canonical_to_anthropic_messages",
    "canonical_to_responses_api",
    "ensure_valid_ordering",
    "normalize_to_canonical",
    "repair_orphaned_pairs",
    "sanitize_items",
    "CompactionResult",
    "ContextFile",
    "DEFAULT_MEMORY_FLUSH_RESERVE_TOKENS_FLOOR",
    "ContextOverflowCallback",
    "ToolLoggingCallback",
    "ToolPairingRepairReport",
    "build_replay_messages",
    "build_tools",
    "convert_to_responses_api_items",
    "compact_messages",
    "get_tool_summaries",
    "limit_history_turns",
    "MemoryGetTool",
    "MemorySearchTool",
    "MemoryStore",
    "MemoryWriteTool",
    "PromptBuilder",
    "PromptConfig",
    "sanitize_history",
    "SearchResult",
    "SectionConfig",
    "SessionManager",
    "SessionState",
    "TokenUsage",
    "TranscriptEntry",
    "build_system_prompt_report",
    "has_already_flushed_for_current_compaction",
    "is_context_overflow_error",
    "repair_tool_use_result_pairing",
    "run_memory_flush",
    "split_preserved_recent_turns",
    "MEMORY_FLUSH_PROMPT",
    "MEMORY_FLUSH_SYSTEM_PROMPT",
    "SILENT_REPLY_TOKEN",
    "should_run_memory_flush",
    "ThinkingConfig",
    "ThinkLevel",
    "resolve_thinking_default",
    "group_step_output",
    "SubagentLimitError",
    "SubagentRegistry",
    "SubagentRun",
    "SubagentStatus",
    "SubagentType",
    "SubagentUsage",
    "DelegateGeneralTool",
    "DelegateGUITool",
    "SubagentsTool",
    "AnalyzeImageTool",
    "OPENCLAW_CACHE_BOUNDARY",
    "apply_openclaw_cache_markers",
    "supports_anthropic_cache",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "ExecTool",
    "WebSearchTool",
    "WebFetchTool",
]
