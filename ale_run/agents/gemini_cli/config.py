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

# Deny-only tool policy: we list ONLY the tools to disable and never an allow
# list (an allow+deny pair is confusing, and the full per-agent tool inventory
# is hard to pin down). Native filesystem tools therefore stay enabled — the
# agenthle fork forwards their tool-result content correctly over OpenRouter.
#
# Persistent-state / interactive / tracker tools are disabled (they don't fit
# headless benchmark runs). Web tools (google_web_search, web_fetch) are
# intentionally left ENABLED: the benchmark allows internet, so web access is
# harmonized to "on" across agents that can support it.
_DISABLED_TOOLS = (
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
class GeminiCliConfig:
    """Tunables for :class:`GeminiCliDeployer`.

    Standalone config (no shared base). The episode wall-budget is
    orchestration-owned; ``timeout_s`` is no longer an agent knob.
    """

    name: ClassVar[str] = "gemini-cli"

    # agenthle gemini_cli.yaml model is ``google/gemini-3.1-pro-preview``;
    # the cua-verse fork maps the bare ``gemini-*`` name to ``google/gemini-*``
    # on the OpenRouter request, so we keep the bare id here.
    model: str = "gemini-3.1-pro-preview"
    provider: str = "openrouter"
    approval_mode: str = "yolo"

    # agenthle gemini_cli.yaml: max_session_turns: -1 (unbounded; wall-clock
    # timeout is the real cap). Written into settings.json as maxSessionTurns.
    max_session_turns: int = -1

    disabled_tools: tuple[str, ...] = _DISABLED_TOOLS
    npm_package: str = "https://github.com/cua-verse/gemini-cli/releases/download/v0.38.8-agenthle/google-gemini-cli-0.38.8.tgz"
    compression_model: str = "google/gemini-3-flash-preview"
