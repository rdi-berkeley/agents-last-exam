"""Model, tool and session settings for ALE-Claw."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class AleClawConfig:
    """Model, tool and session settings."""

    name: ClassVar[str] = "ale-claw"

    model: str = "openrouter/anthropic/claude-sonnet-4.6"
    """LiteLLM-format model id. Maps to OpenClaw's ``model`` kwarg verbatim.
    OpenRouter routes work via the vendored ``unified_loop`` (registered for
    ``openrouter/.*`` regex)."""

    # ---- custom OpenAI-compatible gateway (optional) ----
    base_url: str | None = None
    """Custom OpenAI-compatible endpoint. When set, it is passed straight to
    ``agent.run(api_base=...)`` → ``litellm`` as ``api_base``, so LiteLLM hits
    ``<base_url>/responses`` (or ``/chat/completions``) instead of the provider
    default. Threaded as a direct call kwarg — it does NOT touch ``os.environ``,
    so it cannot leak into other runs' OpenAI traffic. ``None`` ⇒ provider
    default. Pair with an ``openai/<model>`` model id to route via LiteLLM's
    OpenAI-compatible client at this base."""

    api_key: str | None = field(default=None, repr=False)
    """Optional key passed directly to LiteLLM. When omitted, LiteLLM uses
    the credentials configured for the selected provider."""

    extra_generation_kwargs: dict = field(default_factory=dict)
    """Extra model options. The main loop receives all options. Compatible
    options also reach helpers and subagents: gateway-specific options stay on
    the same provider, while sampling options follow an explicitly different
    provider. Helper-owned token limits, timeouts, and Responses-only truncation
    are not overridden. Per-helper thinking settings take precedence."""

    max_turns: int | None = 100
    """Mapped to OpenClaw's ``max_steps``. Hard ceiling on the agent run loop."""

    # ---- model variants ----
    summary_model: str | None = None
    """Model for compaction + memory_flush. None → ``auxiliary_model`` if set,
    else ``model``. Cheaper sibling for cost savings."""

    gui_model: str | None = None
    """Model for the ``delegate_gui`` subagent. None → ``auxiliary_model``
    if set, else falls back to main."""

    auxiliary_model: str | None = None
    """Optional cheaper sibling model offered to subagents, and the fallback for
    ``summary_model`` / ``gui_model`` when those are unset. Caller opts in
    explicitly; there is no automatic sibling lookup."""

    # ---- loop control ----
    max_history_turns: int | None = None
    """Truncate replay-message history when restoring a transcript. None = unlimited."""

    disable_main_computer: bool = False
    """If True, the main agent has no ``computer`` tool — all GUI work goes
    through ``delegate_gui``. Mutually exclusive with :attr:`disable_delegate_gui`."""

    disable_delegate_gui: bool = False
    """If True, no GUI subagent — main agent uses its own ``computer``."""

    disabled_tools: list[str] = field(default_factory=list)
    """Tools to drop from the assembled tool list (matched by ``BaseTool.name``).
    Empty by default; ``web_search`` requires ``BRAVE_API_KEY`` to work."""

    # ---- pointer coordinate space ----
    coordinate_space: str | None = None
    """Coordinate space the GUI model emits for pointer actions.

    - ``None`` (default): auto-detect from :attr:`model` — Gemini-family models
      ground clicks to a normalized ``[0, 1000)`` grid, everything else emits
      screen pixels (see ``harness.tools.computer_handler
      .coordinate_space_for_model``).
    - ``"pixel"``: coords are already screen pixels; pass through unchanged.
    - ``"normalized"``: coords are on a ``[0, 1000)`` grid; the computer handler
      rescales them to pixels using the live screen size before acting.

    Gemini computer-use models ground to a fixed 0-1000 grid regardless of the
    "Screen resolution: WxH pixels" hint in the ``computer`` tool description, so
    their raw clicks land in the wrong place unless rescaled. When main and GUI
    subagent models differ, the space is chosen from the model that actually
    drives the computer handler (``gui_model`` when ``disable_main_computer``,
    else ``model``); set this explicitly to override if they disagree."""

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

    def __post_init__(self) -> None:
        if self.disable_main_computer and self.disable_delegate_gui:
            raise ValueError(
                "Both disable_main_computer and disable_delegate_gui set — "
                "agent has no way to interact with the VM."
            )
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError("AleClawConfig.max_turns must be positive")
        reserved = {"api_key", "api_base", "max_turns"} & self.extra_generation_kwargs.keys()
        if reserved:
            raise ValueError(f"Use dedicated configuration fields instead of extra_generation_kwargs: {sorted(reserved)}")
        if self.coordinate_space is not None and self.coordinate_space not in (
            "pixel",
            "normalized",
        ):
            raise ValueError(
                f"AleClawConfig.coordinate_space={self.coordinate_space!r} "
                "not in {pixel, normalized, None}"
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
