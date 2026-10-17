"""Configuration for source-built ZCode and a custom OpenAI-compatible provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class ZCodeConfig:
    """Tunables for :class:`ZCodeDeployer`.

    Standalone config (no shared base, no LiteLLM). ``model``/``base_url``/
    ``api_key`` describe a single OpenAI-chat-completions-compatible
    provider that the deployer registers with ZCode as a custom provider
    named ``provider_id`` -- there is no vendor-name heuristic, just this
    one endpoint.
    """

    name: ClassVar[str] = "zcode"

    model: str = "Qwen/Qwen3.5-9B"
    """Model id as the upstream OpenAI-compatible endpoint expects it
    (passed through verbatim in the provider config's ``personalModelIds``
    and ``defaultModelSelection.modelId`` -- no prefix rewriting)."""

    provider_id: str = "hf-router"
    """Arbitrary id for the custom provider entry ZCode registers this
    endpoint under (``defaultModelSelection.providerId`` and the
    provider-config's ``providerId``). Cosmetic; must just be unique and
    URL/id-safe."""

    base_url: str = "https://router.huggingface.co/v1"
    """OpenAI-chat-completions-compatible base URL
    (``provider.config.api.baseUrl``)."""

    api_key: str | None = field(default=None, repr=False)
    """Literal API key (``provider.config.access.apiKey``). Required --
    ZCode's custom-provider path has no natural env-var fallback name to
    default to (unlike the LiteLLM-routed agents in this repo), so a
    missing key is a hard error at install time. Set via ``api_key:
    ${env:HF_INFERENCE_TOKEN}`` in the agent yaml."""

    cli_ref: str = "aac4755666d09fdcd70272fcf063c077a639015f"
    """Git ref of https://github.com/zai-org/ZCode to build."""

    context_window: int = 200000
    """Context capacity metadata, independent of the output-token override."""

    max_output_tokens: int | None = None
    """Explicit output-token override. None preserves the pinned CLI's own
    model metadata and request mapping without adding an adapter output limit.
    """

    reasoning_request: dict[str, object] = field(default_factory=dict)
    """Provider-specific reasoning fields, e.g. {"reasoning_effort": "low"}."""

    extra_envs: dict[str, str] = field(default_factory=dict)
    """Free-form passthrough env vars exported to the ``zcode`` subprocess
    (e.g. proxy settings). Keys here override anything the deployer would
    otherwise compute."""
