"""OpenClawCliConfig: per-episode knobs for the OpenClaw CLI deployer.

OpenClaw is installed from a fork tarball (not public npm).
CUA bridge is the native OpenClaw plugin (not MCP).

Auth: API keys set via auth-profiles.json.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

CUA_TOOL_NAMES = (
    "screenshot",
    "click",
    "type",
    "key",
    "key_down",
    "key_up",
    "hold_key",
    "mouse_move",
    "mouse_down",
    "mouse_up",
    "drag",
    "scroll",
    "wait",
    "cursor_position",
)

# Mirrors the 12-entry deny list shipped in every agenthle
# openclaw_*_openrouter*.yaml. The first seven are backend / late-binding
# tools whose core impl reaches for a plugin we don't load (parity with the
# gateway runner); the last five are gateway-only tools that raise
# `1006 abnormal closure` under `agent --local` because there is no live
# gateway WebSocket. Verified OS-agnostic on the 2026-05-03
# demo_tool_smoke_test smoke matrix.
_TOOLS_DENY = (
    # backend / late-binding (parity with gateway runner)
    "web_search",
    "web_fetch",
    "image_generate",
    "video_generate",
    "music_generate",
    "memory_search",
    "sessions_yield",
    # gateway-only tools — `agent --local` has no live gateway WS
    "sessions_list",
    "sessions_history",
    "sessions_spawn",
    "sessions_send",
    "cron",
)


@dataclass
class OpenClawCliConfig:
    """Tunables for :class:`OpenClawCliDeployer`.

    Standalone config (no shared base). The episode wall-budget is
    orchestration-owned; ``timeout_s`` is no longer an agent knob.
    """

    name: ClassVar[str] = "openclaw-cli"

    # agenthle openclaw_cli_openrouter_gpt-5_4.yaml: openai/gpt-5.4.
    model: str = "openai/gpt-5.4"

    # ---- primary-model routing ----
    provider: str = "openrouter"
    """Routing provider, drives auth-profile + model-prefix setup
    explicitly (not a key-presence heuristic):
      - ``"openrouter"`` (default) → openrouter auth profile via
        OPENROUTER_API_KEY; model is prefixed ``openrouter/<model>``.
        Requires OPENROUTER_API_KEY.
      - ``"direct"`` → native-provider auth profile selected by the
        model's vendor: an OpenAI model (``gpt-*`` / ``openai/...``) uses
        the ``openai`` provider + OPENAI_API_KEY; an Anthropic model
        (``claude-*`` / ``anthropic/...``) uses the ``anthropic`` provider
        + ANTHROPIC_API_KEY. OPENROUTER_API_KEY is dropped from the launch
        env so it cannot override the chosen direct provider.
      - ``"zai"`` → Z.AI's native provider via ZAI_API_KEY,
        Z_AI_API_KEY, or GLM_API_KEY; a custom CN/global endpoint can be
        set with ``base_url``.
      - ``"custom"`` → a named OpenAI-compatible provider declared through
        ``provider_id``, ``model_id``, ``base_url``, and ``api_key_env``.
    Missing the required key for the chosen provider is a hard error."""

    provider_id: str | None = None
    """OpenClaw provider id used when ``provider: custom``."""

    model_id: str | None = None
    """Provider-facing model/deployment id. The top-level ``model`` remains
    the human-facing model name recorded in experiment logs."""

    base_url: str | None = None
    """Custom OpenAI-compatible base URL for the resolved provider, written
    into openclaw.json as ``models.providers.<provider>.baseUrl``. ``None`` ⇒
    the provider's built-in default endpoint. With ``provider: openrouter`` set
    to ``https://ark.cn-beijing.volces.com/api/v3`` this routes the (chat-
    completions) openrouter path at Volcengine Ark — Ark's chat/completions API
    is fully OpenAI-compatible, unlike its stricter Responses API. The model is
    the gateway's id (e.g. an Ark ``ep-...`` endpoint)."""

    api_key: str | None = None
    """Literal API key for the ``base_url`` gateway. ``None`` ⇒ the key is read
    from the provider's env var (e.g. OPENROUTER_API_KEY). When set (typically
    via ``api_key: ${env:ARK_API_KEY}`` in the agent yaml, resolved host-side),
    it is used directly in auth-profiles.json — so the secret travels with the
    serialized config, needs no env-passthrough whitelist change, and does not
    collide with a real OPENROUTER_API_KEY in the shell env."""

    api_key_env: str | None = None
    """Environment variable containing the key for ``provider: custom``."""

    provider_api: str = "openai-completions"
    """OpenClaw transport used by a custom provider."""

    supports_usage_in_streaming: bool = True
    """Request usage metadata in streamed responses. OpenAI-compatible
    endpoints normally support this via ``stream_options.include_usage``;
    disable only for an endpoint that rejects that request field."""

    model_params: dict[str, object] | None = None
    """Provider-specific parameters written to the primary model's
    ``agents.defaults.models.<provider/model>.params`` entry. Use
    ``extra_body`` for request-body fields that the OpenClaw transport does
    not natively expose."""

    # OpenClaw's OWN internal run budget, written into openclaw.json
    # (``timeoutSeconds``) and passed as ``agent --local --timeout <s>``.
    # This is an agent-consumed knob (the CLI enforces it itself), distinct
    # from the orchestration episode budget. agenthle
    # openclaw_cli_openrouter_gpt-5_4.yaml: timeout_seconds: 600.
    agent_timeout_s: int = 600

    # Provider-specific accepted values are validated by OpenClaw.
    thinking: str = "high"
    # agenthle openclaw_cli_openrouter_gpt-5_4.yaml sets vision_model to the
    # same id as model; the dataclass default was None but every operational
    # openclaw_cli yaml pins it, so default to the gpt-5.4 operational value.
    vision_model: str | None = "openai/gpt-5.4"
    """Image model used by both the explicit ``image`` tool and automatic
    media understanding."""

    vision_provider: str | None = None
    """Routing mode for ``vision_model``. ``None`` inherits ``provider``.
    Set ``"direct"`` to route a vision model through its native provider
    while the primary model uses a gateway or Z.AI endpoint."""

    vision_base_url: str | None = None
    """Optional endpoint override for the vision route. When
    ``vision_provider`` is unset, ``None`` inherits ``base_url``."""

    vision_api_key: str | None = None
    """Optional API key for the vision route. Native-provider environment
    variables are used when omitted."""
    tools_deny: tuple[str, ...] = _TOOLS_DENY
    # Matches the agenthle openclaw_*_openrouter*.yaml plugins.allow: the
    # `memory-core` plugin provides the harmless `memory_get` file reader
    # (`memory_search` is denied above). The resolved auth provider
    # (`openrouter` / `openai` / `anthropic`) is appended at write time by
    # the deployer, so it need not be listed here.
    plugins_allow: tuple[str, ...] = ("cua", "openrouter", "memory-core")
    plugins_deny: tuple[str, ...] = ()
    heartbeat_every: str = "never"

    tarball_path: str = "/opt/ale/openclaw-fork.tgz"
    """Path to openclaw fork tarball inside the sandbox."""

    tarball_url: str = ""
    """GitHub Release URL for the fork tarball. Used as fallback when
    tarball_path does not exist on disk."""

    cua_plugin_path: str = "/opt/ale/openclaw-cua-plugin"
    """Path to CUA plugin source directory inside the sandbox."""

    cua_plugin_repo: str = "https://github.com/cua-verse/openclaw.git"
    """Git URL to clone CUA plugin source from when cua_plugin_path is missing."""

    cua_plugin_branch: str = "agenthle"
    """Branch to clone for CUA plugin source."""
