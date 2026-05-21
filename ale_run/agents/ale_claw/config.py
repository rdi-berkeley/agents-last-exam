"""AleClawConfig: per-episode knobs for the OpenClaw native agent deployer.

Inherits :class:`BaseAgentConfig` for the standard surface
(``model`` / ``max_turns`` / ``timeout_s``) and adds OpenClaw-specific
knobs below.

**API keys live in the operator's shell env**, not in this config. The
deployer never touches ``os.environ`` — litellm (the harness's LLM
client) reads ``OPENROUTER_API_KEY`` / ``ANTHROPIC_API_KEY`` /
``OPENAI_API_KEY`` directly from the process's env vars, which the
operator populates via shell ``source`` of an ``.env`` / ``.envrc``.
For docker / VM runtimes those vars are propagated by
:mod:`ale.runtime._env`.

Typical usage::

    # In shell:
    #   export OPENROUTER_API_KEY=...
    #   source ~/.config/agenthle/eval.env

    cfg = AleClawConfig(
        model="openrouter/anthropic/claude-sonnet-4-20250514",
        max_turns=100,
        timeout_s=3600,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig


@dataclass
class AleClawConfig(BaseAgentConfig):
    """Tunables for :class:`AleClawDeployer`."""

    name: ClassVar[str] = "ale-claw"

    # ---- override base defaults ----
    model: str = "openrouter/anthropic/claude-sonnet-4.6"
    """LiteLLM-format model id. Maps to OpenClaw's ``model`` kwarg verbatim.
    OpenRouter routes work via the vendored ``unified_loop`` (registered for
    ``openrouter/.*`` regex)."""

    max_turns: int | None = 100
    """Mapped to OpenClaw's ``max_steps``. Hard ceiling on the agent run loop."""

    timeout_s: float = 3600.0
    """Wall-clock budget for the whole episode. Enforced via
    :func:`asyncio.wait_for` around the harness's ``agent.run`` loop."""

    # ---- model variants ----
    summary_model: str | None = None
    """Model for compaction + memory_flush. None → ``lightweight_model`` if set,
    else ``model``. Cheaper sibling for cost savings."""

    gui_model: str | None = None
    """Model for the ``delegate_gui`` subagent. None → ``lightweight_model``
    if set, else falls back to main."""

    lightweight_model: str | None = None
    """Optional cheap-sibling model exposed to delegate tools. ALE convention:
    no auto-magic sibling lookup; caller opts in explicitly."""

    # ---- loop control ----
    max_history_turns: int | None = None
    """Truncate replay-message history when restoring a transcript. None = unlimited."""

    disable_main_computer: bool = False
    """If True, the main agent has no ``computer`` tool — all GUI work goes
    through ``delegate_gui``. Mutually exclusive with :attr:`disable_delegate_gui`."""

    disable_delegate_gui: bool = False
    """If True, no GUI subagent — main agent uses its own ``computer``."""

    disabled_tools: list[str] = field(default_factory=lambda: ["web_search"])
    """Tools to drop from the assembled tool list (matched by ``BaseTool.name``).
    Defaults to ``["web_search"]`` because BRAVE_API_KEY is rarely provisioned;
    set to ``[]`` to opt back in (and ensure ``BRAVE_API_KEY`` is exported in
    your shell)."""

    # ---- thinking levels (off | low | medium | high) ----
    thinking_level: str | None = None
    """Base thinking level. None → resolved-default for the model
    (see ``harness.thinking.resolve_thinking_default``)."""

    flush_thinking_level: str | None = None
    """Memory flush thinking. None → inherit :attr:`thinking_level`."""

    compaction_thinking_level: str | None = None
    """Compaction-rebuild thinking. None → inherit :attr:`thinking_level`."""

    vision_thinking_level: str = "off"
    """Vision/screenshot summarization thinking. Default off (cost)."""

    gui_thinking_level: str = "off"
    """``delegate_gui`` subagent thinking. Default off."""

    # ---- image retention ----
    image_retention_mode: str = "openclaw"
    """``openclaw`` (default — last N completed turns) or ``cua`` (last N images
    by count). OpenClaw mode reduces cache thrash on multi-screenshot turns."""

    # ---- documentation ----
    upstream_version: str = "openclaw-cua@a830cae2"
    """Source upstream commit for the vendored ``harness/`` tree.
    Surfaced via :attr:`AleClawDeployer.version`."""

    # ---- v2 (NOT in v1 — always per-run) ----
    # memory_base_dir: str | None = None
    # session_base_dir: str | None = None

    def __post_init__(self) -> None:
        if self.disable_main_computer and self.disable_delegate_gui:
            raise ValueError(
                "Both disable_main_computer and disable_delegate_gui set — "
                "agent has no way to interact with the VM."
            )
        for level_field, value in [
            ("thinking_level", self.thinking_level),
            ("flush_thinking_level", self.flush_thinking_level),
            ("compaction_thinking_level", self.compaction_thinking_level),
            ("vision_thinking_level", self.vision_thinking_level),
            ("gui_thinking_level", self.gui_thinking_level),
        ]:
            if value is not None and value not in ("off", "low", "medium", "high"):
                raise ValueError(
                    f"AleClawConfig.{level_field}={value!r} not in "
                    f"{{off, low, medium, high}}"
                )
