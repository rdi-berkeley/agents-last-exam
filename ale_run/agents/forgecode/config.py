"""ForgecodeConfig: per-episode knobs for the forgecode (tailcallhq/forge) deployer.

Auth: forge reads its API key from environment variables.  When using
OpenRouter, ``ANTHROPIC_API_KEY`` is set to the OpenRouter key and
``ANTHROPIC_BASE_URL`` is pointed at ``https://openrouter.ai/api/v1``.
For direct providers the standard env var is exported as-is.

forge.toml ``[session]`` block pins ``provider_id`` + ``model_id`` so
multi-key environments cannot accidentally route through the wrong vendor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig


@dataclass
class ForgecodeConfig(BaseAgentConfig):
    """Tunables for :class:`ForgecodeDeployer`."""

    name: ClassVar[str] = "forgecode"

    model: str = "anthropic/claude-sonnet-4-6"
    timeout_s: float = 600

    # Sampling / model knobs surfaced through forge.toml.
    temperature: float | None = 0.7

    # Reserved for future use: forge tool names to disable.  forge itself
    # does not have a config-driven disable-list, so this is a forward-
    # compatible placeholder; the deployer translates this into a comment
    # in forge.toml only.
    disabled_tools: tuple[str, ...] = ()

    # Provider routing fields.  The model prefix (``anthropic/...``,
    # ``openai/...``) determines which env var to export, unless the user
    # is going through OpenRouter in which case we always use the
    # OPENROUTER_API_KEY.
    provider: str = "openrouter"
    """``"openrouter"`` or ``"direct"``."""

    @property
    def is_openrouter(self) -> bool:
        return self.provider.lower() in ("openrouter", "open_router")

    # ------------------------------------------------------------------
    # forge.toml rendering
    # ------------------------------------------------------------------

    def forge_provider_id(self) -> str:
        """Return the ``provider_id`` for ``forge.toml``'s ``[session]``.

        forge's canonical IDs: ``open_router``, ``anthropic``, ``openai``,
        etc.  (``crates/forge_domain/src/provider.rs``).
        """
        if self.is_openrouter:
            return "open_router"
        prefix = self.model.split("/", 1)[0].lower()
        if prefix == "anthropic":
            return "anthropic"
        if prefix == "openai" or prefix.startswith("gpt"):
            return "openai"
        return prefix or "open_router"

    def forge_model_id(self) -> str:
        """Return the ``model_id`` for ``forge.toml``'s ``[session]``.

        OpenRouter keeps the ``vendor/model`` form; direct providers strip
        the vendor prefix (e.g. ``anthropic/claude-sonnet-4`` becomes
        ``claude-sonnet-4`` under ``provider_id = "anthropic"``).
        """
        if self.is_openrouter:
            return self.model
        provider = self.forge_provider_id()
        prefix = self.model.split("/", 1)[0].lower()
        if prefix == provider and "/" in self.model:
            return self.model.split("/", 1)[1]
        return self.model

    def render_forge_toml(self) -> str:
        """Render the ``~/.forge/.forge.toml`` content.

        Always sets ``auto_dump = "json"`` so forge writes a timestamped
        dump.json on ``TaskComplete``.  Deliberately omits
        ``max_requests_per_turn`` and ``max_tool_failure_per_turn`` so
        forge's interactive "continue anyway?" prompt never fires.
        """
        provider = self.forge_provider_id()
        model = self.forge_model_id()
        lines = [
            'auto_dump = "json"',
            "",
            "[session]",
            f'provider_id = "{provider}"',
            f'model_id = "{model}"',
        ]
        if self.temperature is not None:
            lines.extend(["", f"temperature = {float(self.temperature)}"])
        return "\n".join(lines) + "\n"
