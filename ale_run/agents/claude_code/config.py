"""ClaudeCodeConfig: per-episode knobs for the Claude Code CLI deployer.

**API keys live in the operator's shell env**, never in this config.
The deployer's VM-side bash script reads ``ANTHROPIC_API_KEY`` /
``OPENROUTER_API_KEY`` from the inherited env (propagated host → VM by
:mod:`ale.runtime._env`). OpenRouter routing auto-detects: if
``ANTHROPIC_API_KEY`` is unset but ``OPENROUTER_API_KEY`` is set, the
script remaps to ``ANTHROPIC_AUTH_TOKEN`` + ``ANTHROPIC_BASE_URL``.

Typical usage::

    # In shell:
    #   export ANTHROPIC_API_KEY=sk-ant-...    # direct
    #   # OR
    #   export OPENROUTER_API_KEY=sk-or-...    # routed (auto-detected)

    cfg = ClaudeCodeConfig(
        model="claude-opus-4-7",
        max_budget_usd=5.0,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class ClaudeCodeConfig:
    """Tunables for :class:`ClaudeCodeDeployer`.

    Standalone config (no shared base). The episode wall-budget is
    orchestration-owned; ``timeout_s`` is no longer an agent knob.
    """

    name: ClassVar[str] = "claude-code"

    # agenthle claude_code_openrouter.yaml: anthropic/claude-opus-4.6
    # (direct claude_code.yaml: claude-sonnet-4-6).
    model: str = "anthropic/claude-opus-4.6"

    # agenthle claude_code_openrouter.yaml: max_turns: -1 (≡ unlimited; the
    # deployer omits --max-turns when < 0). Direct claude_code.yaml: 300.
    # The base class used to leave this None → --max-turns was always
    # skipped; pinning -1 restores the explicit-unlimited intent.
    max_turns: int | None = -1

    # ---- routing (no secrets — API keys come from shell env) ----
    provider: str = "openrouter"
    """Routing provider, drives env setup explicitly (not key-presence
    heuristics):
      - ``"openrouter"`` → ANTHROPIC_BASE_URL=openrouter, AUTH_TOKEN=
        OPENROUTER_API_KEY, ANTHROPIC_API_KEY="". Requires OPENROUTER_API_KEY.
      - ``"direct"`` → uses ANTHROPIC_API_KEY against anthropic.com (or
        ``base_url`` if set). Requires ANTHROPIC_API_KEY.
    Missing the required key for the chosen provider is a hard error."""

    base_url: str | None = None
    """Custom Anthropic-compatible base URL. Overrides the provider's
    default ``ANTHROPIC_BASE_URL``."""

    api_key: str | None = None
    """Literal API key, used in place of the provider's env-var key. ``None`` ⇒
    read from the env (OPENROUTER_API_KEY for ``openrouter``, ANTHROPIC_API_KEY
    for ``direct``). When set (typically ``api_key: ${env:...}`` in the agent
    yaml, resolved host-side), it travels with the serialized config — no env
    passthrough whitelist change needed and no collision with a real shell key.
    For ``openrouter`` it becomes ANTHROPIC_AUTH_TOKEN; for ``direct``,
    ANTHROPIC_API_KEY."""

    extra_headers: dict[str, str] = field(default_factory=dict)
    """Extra HTTP headers for every API request, joined into Claude Code's
    ``ANTHROPIC_CUSTOM_HEADERS`` (newline-separated ``Name: value`` lines).
    For gateways that authenticate by header (e.g. a Cloudflare Access service
    token in front of an Anthropic-compatible endpoint). Values may use
    ``${env:VAR}`` in the yaml; they travel with the serialized config."""

    session_header: str | None = None
    """Name of a header to send with a per-run random id (``ale-<hex>``) so a
    gateway with session affinity pins the run's requests to one backend
    (prefix-cache locality across turns). ``None`` sends no such header."""

    extra_env: dict[str, str] = field(default_factory=dict)
    """Extra environment variables for the CLI process (applied last, after the
    provider routing). E.g. ``ANTHROPIC_SMALL_FAST_MODEL`` so Claude Code's
    auxiliary calls hit the same custom model, or ``DISABLE_TELEMETRY``."""

    api_proxy: bool = False
    """Run a local Anthropic-Messages proxy inside the sandbox
    (``api_proxy.py``) and point the CLI at it. The proxy forwards to
    ``base_url`` with ``extra_headers`` (so the CLI never sees gateway
    credentials), converts ``document`` blocks (PDF reads) to text, calls the
    upstream non-streaming and re-samples on malformed-output / 5xx / 429
    errors, then re-emits SSE to the CLI. For self-hosted gateways that only
    partially implement the Anthropic Messages API."""

    resume_on_api_error: int = 0
    """When the CLI ends its session with an ``API Error`` or ``Request timed
    out`` result (the CLI does not retry those), resume the same session
    (``--resume``) with a "continue" prompt up to this many times. 0 disables.
    Three consecutive failures within a minute stop the loop early."""

    # ---- CLI knobs ----
    max_budget_usd: float | None = None
    context_window: int | None = None
    auto_compact_token_limit: int | None = None
    max_output_tokens: int | None = None
    """Maximum output tokens requested from the model through Claude Code's
    ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` environment variable. ``None`` leaves the
    CLI default unchanged."""
    disabled_tools: tuple[str, ...] = ()
    dangerously_skip_permissions: bool = True

    otel_enabled: bool = True
    """Capture Claude Code's built-in OpenTelemetry stream for every run. The
    deployer starts a run-local OTLP/HTTP receiver and points the CLI at it
    (``CLAUDE_CODE_ENABLE_TELEMETRY=1`` + ``OTEL_*`` env vars), storing raw OTLP
    batches, flattened per-event logs (API request timing/tokens/cost/request-id,
    tool arguments/duration, prompts), cumulative metrics, and a summary in
    ``work_dir``. No upstream Claude Code changes are required. Set ``false`` to
    run the CLI with telemetry off."""

    max_thinking_tokens: int | None = 31999
    """Extended-thinking token budget, passed to the CLI via the
    ``MAX_THINKING_TOKENS`` env var (Claude Code's documented knob —
    see https://code.claude.com/docs/en/costs#adjust-extended-thinking).
    Claude Code enables extended thinking by default; ``31999`` is the
    default-high ("ultrathink") cap. We set it explicitly so the reasoning
    level is pinned + visible rather than relying on the CLI default
    (parity with codex's ``reasoning_effort=high``). Lower it (e.g. 8000)
    to cut cost, or set 0 to disable thinking.

    ``null``/``None`` ⇒ the deployer sets ``CLAUDE_CODE_DISABLE_THINKING=1``
    so the CLI OMITS the thinking param from API requests — required for
    adaptive-thinking models (e.g. claude-fable-5) over OpenRouter: the CLI
    otherwise sends ``thinking: {type: "adaptive"}``, which OpenRouter
    mangles into ``thinking: {type: "disabled"}`` upstream, and those
    models reject it with 400 ("thinking.type.disabled is not supported").
    With the param absent the provider applies the model's native adaptive
    thinking — the model still thinks; this is not a thinking-off switch.
    (Merely unsetting MAX_THINKING_TOKENS is NOT enough — the CLI still
    sends adaptive. Verified by request capture + e2e tool flow.)"""

    effort_level: str | None = None
    """Reasoning-effort level for adaptive-thinking models (Claude Opus 4.7/4.8,
    Fable 5, ...), passed to the CLI as ``--effort <level>``. This is the
    Anthropic-native control that maps to the API's ``output_config.effort`` —
    DISTINCT from ``max_thinking_tokens``/``budget_tokens`` (the legacy extended-
    thinking budget, which is removed → 400 on Opus 4.7/4.8). Accepted values:
    ``low | medium | high | xhigh | max`` (verified against the pinned CLI's
    ``--effort <level>`` option). Claude Code only applies effort to supported
    Claude models; it's a no-op on non-Claude endpoints.

    ``None`` ⇒ don't pass ``--effort`` (let the CLI use its default). When SET,
    the deployer ALSO skips ``MAX_THINKING_TOKENS``/``CLAUDE_CODE_DISABLE_THINKING``
    entirely: effort is the reasoning control for these models, and pairing it
    with the legacy budget env is wrong (budget_tokens 400s on 4.7/4.8). Thinking
    is left to the model's adaptive default."""

    # ---- documentation ----
    cli_version: str = "@anthropic-ai/claude-code@2.1.280"
    """Full npm spec of the pinned CLI. The deployer probes the installed
    ``claude --version`` against the version encoded here and (re)installs
    via npm when missing or mismatched — so bumping this is sufficient to
    upgrade images with an older claude pre-baked."""

    def __post_init__(self) -> None:
        if self.context_window is not None and self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.auto_compact_token_limit is not None and self.auto_compact_token_limit <= 0:
            raise ValueError("auto_compact_token_limit must be positive")
