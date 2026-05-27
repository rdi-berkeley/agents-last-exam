"""GeminiCliConfig: per-episode knobs for the Gemini CLI deployer.

API keys live in the operator's shell env.  The deployer reads
``GEMINI_API_KEY`` (direct Google) or ``OPENROUTER_API_KEY``
(routed via OpenRouter fork).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig

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
    approval_mode: str = "yolo"
    allowed_tools: tuple[str, ...] = _ALLOWED_TOOLS
    disabled_tools: tuple[str, ...] = _DISABLED_TOOLS
    npm_package: str = "@google/gemini-cli"
