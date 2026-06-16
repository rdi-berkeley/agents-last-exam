"""CodexConfig: per-episode knobs for the OpenAI Codex CLI deployer.

The deployer ensures the running codex is exactly the pinned **fork** build
(``fork_version``, e.g. ``codex-cli 0.0.0-agenthle-20260614``): it compares
``codex --version`` and, when the binary is missing/stale/stock, installs stock
``@openai/codex`` from NPM (if needed) and overlays the fork native binary from
the GitHub Release; a matching version skips the download. The fork = openai/codex
``main`` merged in, plus two carries: the Windows ``apply_patch.exe`` hardlink
fix and the OpenRouter MCP adaptation (namespaced-tool flatten + dispatch
remap). Built from cua-verse/codex ``agenthle``; published as the release below.

Auth: OpenRouter routing uses ``OPENROUTER_API_KEY`` injected via env.
Direct OpenAI routing uses ``OPENAI_API_KEY``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)

# Reasoning-effort variants the fork binary can deserialize. The merged build
# accepts the full set incl. ``max`` (a newer reasoning level — upstream
# represents it as ReasoningEffort::Custom, so any string parses). Kept as a
# guard so a much older baked fork still loads the catalog; entries with an
# unknown effort are stripped at load time (logged, not silent).
_FORK_KNOWN_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)

# NPM fallback version: only used when NO codex is on PATH (rare — images bake
# the fork). The stock package is installed then the fork binary is overlaid, so
# this is just the stock base the overlay replaces. Pin a real published stock
# version so the fallback install succeeds.
_DEFAULT_CODEX_VERSION = "0.114.0"

# GitHub Release of the fork native binary, per OS. Downloaded and overlaid over
# the vendor binary whenever the running version != ``fork_version``.
# Both are glibc/MSVC x86-64 release builds from cua-verse/codex
# ``agenthle`` (openai/main merged + Windows apply_patch fix + OpenRouter MCP
# adaptation). Empty string = skip overlay. Linux asset ``codex``; Windows asset
# ``codex-x86_64-pc-windows-msvc.exe`` (distinct targets, separate assets).
_RELEASE_BASE = (
    "https://github.com/cua-verse/codex/releases/download/v0.0.0-agenthle-20260614"
)
_DEFAULT_PATCHED_BINARY_URL = f"{_RELEASE_BASE}/codex"
_DEFAULT_PATCHED_BINARY_URL_WINDOWS = (
    f"{_RELEASE_BASE}/codex-x86_64-pc-windows-msvc.exe"
)


@dataclass
class CodexConfig:
    """Tunables for :class:`CodexDeployer`.

    Standalone config (no shared base). The episode wall-budget is
    orchestration-owned; ``timeout_s`` is no longer an agent knob.
    """

    name: ClassVar[str] = "codex"

    # agenthle codex_openrouter.yaml: openai/gpt-5.4 (direct codex.yaml: gpt-5.4).
    model: str = "openai/gpt-5.4"

    # ---- routing (no secrets — API keys come from shell env) ----
    provider: str = "openrouter"
    """Routing provider, drives env + config.toml setup explicitly (not a
    model-name heuristic):
      - ``"openrouter"`` → config.toml ``model_provider = "openrouter"`` +
        openrouter model_providers block, auth via OPENROUTER_API_KEY.
        Requires OPENROUTER_API_KEY.
      - ``"direct"`` → direct OpenAI routing via OPENAI_API_KEY.
        Requires OPENAI_API_KEY.
    Missing the required key for the chosen provider is a hard error."""

    # Codex sandbox policy: "danger-full-access" is the only meaningful
    # option for headless eval on an already-isolated VM.
    sandbox_mode: str = "danger-full-access"

    # Bypass all interactive approval prompts (headless exec).
    yolo: bool = True

    # Codex CLI's model_reasoning_effort. Codex 0.114 sends this through
    # the Responses-API wire as ``reasoning.effort``.
    reasoning_effort: str = "high"

    # NPM package version to install.
    codex_version: str = _DEFAULT_CODEX_VERSION

    # GitHub Release URL for the patched native binary (Linux musl x86-64).
    # Empty string = skip binary replacement (use npm's bundled binary).
    patched_binary_url: str = _DEFAULT_PATCHED_BINARY_URL

    # GitHub Release URL for the patched Windows binary (codex.exe,
    # x86_64-pc-windows-msvc). The deployer downloads this instead of
    # ``patched_binary_url`` when running on Windows. Empty string = skip
    # replacement on Windows (use npm's bundled codex.exe).
    patched_binary_url_windows: str = _DEFAULT_PATCHED_BINARY_URL_WINDOWS

    # Pinned fork version the running ``codex`` must report (``codex --version``).
    # The deployer ensures the engine is exactly this build: if no codex is on
    # PATH it installs stock + overlays the fork; if a codex is present but its
    # version doesn't match this string, it downloads + overlays the fork; if it
    # already matches, it skips the download. This replaces the old
    # baked-fork-by-sentinel skip, which couldn't tell an old fork from a new one
    # (both reported plain ``0.0.0``). Must match the version baked into the
    # release at ``patched_binary_url`` (see codex-rs workspace Cargo.toml).
    fork_version: str = "0.0.0-agenthle-20260614"

    # ---- model catalog (models not in codex's bundled catalog) ----
    # Host path to a Codex model-catalog JSON (an external catalog
    # file). Needed to run models the bundled catalog
    # does not know about. Resolved relative to the
    # process cwd (same convention as ``secret_file``). Empty = no catalog
    # (use codex's bundled catalog). When set, the file is read + sanitised
    # host-side in __post_init__ and the content is shipped to the sandbox via
    # ``model_catalog_content`` below; the deployer writes it to
    # ``~/.codex/model_catalog.json`` and points config.toml's
    # ``model_catalog_json`` at it.
    model_catalog_path: str = ""

    # Auto-populated from ``model_catalog_path`` (do not set by hand). Carries
    # the (sanitised) catalog JSON text into the sandbox through the serialized
    # config kwargs — the deployer runs in-sandbox where ``model_catalog_path``
    # is not reachable, so the content must travel with the config.
    model_catalog_content: str = ""

    # ---- codex feature flags (== tool surface) ----
    # Override codex's default ``[features]`` map. A ``{feature_key: bool}`` dict
    # written verbatim into ``~/.codex/config.toml`` ``[features]``. ``true``
    # force-enables, ``false`` force-disables — both directions in one map
    # (codex's own features table is a single bool map). Empty = inherit codex's
    # current defaults unchanged. Only headless-meaningful keys are worth setting;
    # the consolidated codex.yaml preset documents them all (commented). Example:
    # ``{"multi_agent_v2": True, "multi_agent": False}``.
    feature_overrides: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Load the catalog host-side (build_config) and embed its content so it
        # reaches the in-sandbox deployer. On the in-sandbox reconstruction the
        # content kwarg is already populated → we skip the (host-only) file read.
        if self.model_catalog_path and not self.model_catalog_content:
            try:
                raw = Path(self.model_catalog_path).read_text(encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(
                    f"codex: model_catalog_path {self.model_catalog_path!r} "
                    f"could not be read: {exc}"
                ) from exc
            self.model_catalog_content = _sanitise_catalog_for_fork(raw)


def _sanitise_catalog_for_fork(raw: str) -> str:
    """Drop reasoning-effort variants the pinned fork binary cannot parse.

    The fork's ``ReasoningEffort`` enum predates ``max``; an entry listing it
    (or any other unknown variant) makes the whole catalog fail to load. We
    remove those entries — and rewrite a model's ``default_reasoning_level`` to
    ``high`` if it pointed at a dropped variant — and log exactly what changed.
    Returns the JSON text unchanged when nothing needed stripping.
    """
    try:
        catalog = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"codex: model catalog is not valid JSON: {exc}") from exc

    dropped: list[str] = []
    for model in catalog.get("models", []):
        if not isinstance(model, dict):
            continue
        slug = model.get("slug", "?")
        levels = model.get("supported_reasoning_levels")
        if isinstance(levels, list):
            kept = []
            for entry in levels:
                effort = entry.get("effort") if isinstance(entry, dict) else entry
                if effort in _FORK_KNOWN_REASONING_EFFORTS:
                    kept.append(entry)
                else:
                    dropped.append(f"{slug}.supported_reasoning_levels[{effort}]")
            model["supported_reasoning_levels"] = kept
        default = model.get("default_reasoning_level")
        if default is not None and default not in _FORK_KNOWN_REASONING_EFFORTS:
            dropped.append(f"{slug}.default_reasoning_level={default}->high")
            model["default_reasoning_level"] = "high"

    if not dropped:
        return raw
    logger.warning(
        "codex: stripped %d catalog field(s) the pinned fork cannot parse "
        "(rebuild the fork to use them): %s",
        len(dropped), ", ".join(dropped),
    )
    return json.dumps(catalog)
