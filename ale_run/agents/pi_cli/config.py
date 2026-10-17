"""Configuration for the pi (earendil-works/pi) CLI deployer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class PiCliConfig:
    """Per-episode pi CLI settings."""

    name: ClassVar[str] = "pi_cli"

    model: str = "anthropic/claude-sonnet-4.6"
    """Model id. For ``provider="direct"`` this must be ``<vendor>/<id>``
    (e.g. ``anthropic/claude-sonnet-4.6``) — the vendor prefix is stripped and
    used to pick both the pi ``--provider`` flag and the API-key env var. For
    ``"openrouter"``/``"custom"`` it is passed through as-is."""

    provider: str = "openrouter"
    """Routing mode:
      - ``"openrouter"`` -> ``--provider openrouter --model <model>``, key
        from ``api_key`` or ``OPENROUTER_API_KEY``.
      - ``"direct"`` -> ``<vendor>/<id>`` split into ``--provider <vendor>
        --model <id>``, key from ``api_key`` or the vendor's own env var
        (``ANTHROPIC_API_KEY``/``OPENAI_API_KEY``/``GEMINI_API_KEY``/
        ``XAI_API_KEY``/``HF_TOKEN``).
      - ``"custom"`` -> ``--provider custom --model <model>``, requires
        ``base_url``; writes a ``models.json`` entry pi reads its request
        shape from (see ``max_tokens``), key from ``api_key`` or
        ``OPENROUTER_API_KEY``/``HF_TOKEN``."""

    base_url: str | None = None
    """OpenAI-compatible endpoint. Required (and only used) when
    ``provider="custom"``."""

    api_key: str | None = field(default=None, repr=False)
    """Literal API key, used in place of the provider's env-var key for any
    ``provider`` mode. ``None`` falls back to the env var documented under
    ``provider``."""

    cli_version: str = "0.85.1"
    """Version of ``@earendil-works/pi-coding-agent`` to install via npm."""

    thinking_level: str | None = None
    """Passed as ``--thinking <level>`` when set; provider-specific accepted
    values are validated by pi itself."""

    max_tokens: int = 65536
    """Output limit for custom providers, written as models.json maxTokens.
    Direct and OpenRouter providers use Pi's model catalog limits."""

    input_modalities: list[str] = field(default_factory=lambda: ["text", "image"])
    """Accepted custom-model inputs; use ["text"] for text-only endpoints."""

    disabled_tools: list[str] = field(default_factory=list)
    """Tool names passed as ``--exclude-tools a,b,c``."""

    extra_args: list[str] = field(default_factory=list)
    """Free-form flags appended verbatim to the pi CLI invocation."""

    def __post_init__(self) -> None:
        if not self.input_modalities or any(
            modality not in {"text", "image"} for modality in self.input_modalities
        ):
            raise ValueError("input_modalities must contain text and/or image")
