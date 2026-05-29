"""GeminiCliConfig: per-episode knobs for the Gemini CLI deployer.

Routing is provider-driven via :attr:`GeminiCliConfig.provider`:

- ``"openrouter"`` (default): the deployer sets ``OPENROUTER_API_KEY`` and
  ``OPENROUTER_COMPRESSION_MODEL`` and keeps the bare model id. The
  ``cua-verse/gemini-cli#agenthle`` fork maps ``gemini-*`` model names to
  ``google/gemini-*`` on the OpenRouter request and forwards tool-result
  content correctly (functionResponse.id linkage + streaming tool-call
  accumulation), so native file tools work over OpenRouter.
- ``"google"``: direct Google API. The deployer reads ``GEMINI_API_KEY``
  (or ``GOOGLE_API_KEY``) and uses the bare model id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig

# Native filesystem tools are kept enabled. The agenthle fork forwards their
# tool-result content correctly over OpenRouter, so the agent can read/write
# files directly rather than routing everything through run_shell_command.
_ALLOWED_TOOLS = (
    "run_shell_command",
    "write_file",
    "read_file",
    "list_directory",
    "glob",
    "grep_search",
    "replace",
    "read_many_files",
    "list_background_processes",
    "read_background_output",
)

# Only web / persistent-state / interactive / tracker tools are disabled —
# they don't fit headless benchmark runs. Matches agenthle's
# GEMINI_DEFAULT_DISABLED_TOOLS.
_DISABLED_TOOLS = (
    "google_web_search",
    "web_fetch",
    "save_memory",
    "activate_skill",
    "get_internal_docs",
    "write_todos",
    "ask_user",
    "enter_plan_mode",
    "exit_plan_mode",
    "update_topic",
    "complete_task",
    "tracker_create_task",
    "tracker_update_task",
    "tracker_get_task",
    "tracker_list_tasks",
    "tracker_add_dependency",
    "tracker_visualize",
)


@dataclass
class GeminiCliConfig(BaseAgentConfig):
    """Tunables for :class:`GeminiCliDeployer`."""

    name: ClassVar[str] = "gemini-cli"

    model: str = "gemini-3.1-pro-preview"
    provider: str = "openrouter"
    approval_mode: str = "yolo"
    allowed_tools: tuple[str, ...] = _ALLOWED_TOOLS
    disabled_tools: tuple[str, ...] = _DISABLED_TOOLS
    npm_package: str = "https://github.com/cua-verse/gemini-cli/releases/download/v0.38.1-agenthle/google-gemini-cli-0.38.1.tgz"
    compression_model: str = "google/gemini-3-flash-preview"
