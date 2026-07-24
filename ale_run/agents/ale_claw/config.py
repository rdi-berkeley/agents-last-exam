"""AleClawConfig: per-episode knobs for the OpenClaw native agent deployer.

Standalone config (no shared base). Declares ``model`` / ``max_turns``
(mapped to OpenClaw's ``max_steps``) plus the OpenClaw-specific knobs
below. The episode wall-budget is orchestration-owned, so this config no
longer carries ``timeout_s``.

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
    #   for f in secret/eval_time/*.env; do source "$f"; done

    cfg = AleClawConfig(
        model="openrouter/anthropic/claude-sonnet-4-20250514",
        max_turns=100,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class AleClawConfig:
    """Tunables for :class:`AleClawDeployer`."""

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

    api_key: str | None = None
    """Literal API key for ``base_url``. When set (typically via
    ``api_key: ${env:...}`` in the agent yaml, resolved host-side), it is passed
    to ``agent.run(api_key=...)`` → ``litellm`` as a direct kwarg and also
    satisfies the install() credential gate — so no global ``OPENAI_API_KEY`` is
    needed and no key leaks into ``os.environ``. ``None`` ⇒ litellm resolves
    creds from env as before."""

    extra_generation_kwargs: dict = field(default_factory=dict)
    """Extra kwargs merged into every model request (passed to ``agent.run`` →
    ``litellm``). An escape hatch for gateway-specific quirks the loop hardcodes
    otherwise. Later keys win, so this overrides the loop's defaults. Example for
    Meta's Responses gateway (which rejects the loop's default ``truncation:
    auto``): ``{truncation: disabled}``. Empty ⇒ loop defaults unchanged."""

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

    disabled_tools: list[str] = field(default_factory=lambda: ["web_search"])
    """Tools to drop from the assembled tool list (matched by ``BaseTool.name``).
    Defaults to ``["web_search"]`` because BRAVE_API_KEY is rarely provisioned;
    set to ``[]`` to opt back in (and ensure ``BRAVE_API_KEY`` is exported in
    your shell)."""

    # ---- substrate transport ----
    substrate_transport: str = "mcp"
    """How the non-GUI tools (``read``/``write``/``edit``/``exec``) reach the VM.

    - ``"mcp"`` (default): route through the ``vm_mcp_server`` bridge — the agent
      consumes the same MCP substrate as installed agents. Tool granularity is
      unchanged; only the transport moves off ``RemoteDesktopSession``.
    - ``"session"``: legacy direct ``session.interface`` RPC. Retained as a
      debug / parity escape hatch; may be removed once the MCP path is validated.

    GUI (the ``computer`` tool) is governed separately by :attr:`gui_transport`."""

    gui_transport: str | None = None
    """How the GUI ``computer`` tool reaches the VM (Phase 2).

    - ``None`` (default): follow :attr:`substrate_transport` — so GUI is ``"mcp"``
      by default, and a ``substrate_transport="session"`` run stays all-session
      without having to flip this too.
    - ``"mcp"``: route GUI through the ``cua_mcp_server`` bridge — clicks/keys/
      screenshots become MCP tool calls (pixel↔[0,1000] conversion in the
      handler). Requires ``substrate_transport="mcp"``.
    - ``"session"``: the cua ``RemoteDesktopSession`` handler (pixel coords).

    With both transports on ``"mcp"`` (the default), ale_claw never touches
    ``RemoteDesktopSession`` for tool I/O."""

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
        if self.substrate_transport not in ("mcp", "session"):
            raise ValueError(
                f"AleClawConfig.substrate_transport={self.substrate_transport!r} "
                "not in {mcp, session}"
            )
        if self.gui_transport is not None and self.gui_transport not in ("mcp", "session"):
            raise ValueError(
                f"AleClawConfig.gui_transport={self.gui_transport!r} not in {{mcp, session, None}}"
            )
        # An explicit gui=mcp on a session substrate is a genuine conflict; the
        # None default instead *follows* substrate (so session mode just works).
        if self.gui_transport == "mcp" and self.substrate_transport != "mcp":
            raise ValueError(
                "AleClawConfig.gui_transport='mcp' requires substrate_transport='mcp'"
            )
        if self.gui_transport is None:
            self.gui_transport = self.substrate_transport
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
