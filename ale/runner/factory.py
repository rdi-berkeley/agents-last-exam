"""Build Provider / Agent instances from spec dataclasses.

The registries below are the canonical way to add a new provider or agent
to the experiment yaml surface. Shortcut → fqdn mapping keeps yamls short
without sacrificing the explicit ``module.Class`` form for new agents.
"""
from __future__ import annotations

import importlib
from typing import Any

from ale.agents.base import BaseAgentDeployer
from ale.core.provider import Provider

from .spec import AgentSpec, ProviderSpec


# =============================================================================
# Provider registry — kind → (provider class, config class)
# =============================================================================

PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "gcs_direct": (
        "ale.providers.gcs_direct.GCSDirectProvider",
        "ale.providers.gcs_direct.GCSDirectConfig",
    ),
    "static": (
        "ale.providers.static.StaticProvider",
        "ale.providers.static.StaticProviderConfig",
    ),
}


def build_provider(spec: ProviderSpec) -> Provider:
    if spec.kind not in PROVIDER_REGISTRY:
        raise KeyError(
            f"unknown provider.kind={spec.kind!r}; "
            f"available: {sorted(PROVIDER_REGISTRY)}"
        )
    prov_path, cfg_path = PROVIDER_REGISTRY[spec.kind]
    prov_cls = _import_name(prov_path)
    cfg_cls = _import_name(cfg_path)
    try:
        cfg = cfg_cls(**spec.config)
    except TypeError as exc:
        raise TypeError(
            f"provider.kind={spec.kind!r} got bad config: {exc}; "
            f"check unknown / missing keys"
        ) from exc
    return prov_cls(cfg)


# =============================================================================
# Agent registry — shortcut → (deployer class, config class)
# =============================================================================

AGENT_REGISTRY: dict[str, tuple[str, str]] = {
    "claude_code": (
        "ale.agents.claude_code.deployer.ClaudeCodeDeployer",
        "ale.agents.claude_code.config.ClaudeCodeConfig",
    ),
    # Add more here as deployers come online:
    # "openclaw_cli": ("ale...OpenClawDeployer", "ale...OpenClawConfig"),
    # "codex":        ("ale...CodexDeployer",   "ale...CodexConfig"),
}


def build_agent(spec: AgentSpec) -> BaseAgentDeployer:
    """Return a configured deployer ready for ``deployer.run(env)``.

    ``spec.class_`` may be either:
      - a shortcut key from :data:`AGENT_REGISTRY` (e.g. ``"claude_code"``)
      - a fully-qualified deployer class path. In that case we infer the
        config class from the deployer's constructor annotation.
    """
    if spec.class_ in AGENT_REGISTRY:
        dep_path, cfg_path = AGENT_REGISTRY[spec.class_]
        dep_cls = _import_name(dep_path)
        cfg_cls = _import_name(cfg_path)
    else:
        # Treat class_ as fully-qualified deployer path; infer config.
        dep_cls = _import_name(spec.class_)
        cfg_cls = _infer_config_class(dep_cls)
    try:
        cfg = cfg_cls(**spec.config)
    except TypeError as exc:
        raise TypeError(
            f"agent id={spec.id!r} (class {spec.class_}) got bad config: {exc}"
        ) from exc
    return dep_cls(cfg)


# =============================================================================
# Helpers
# =============================================================================

def _import_name(dotted_path: str) -> Any:
    """``"pkg.mod.Class"`` → the class object."""
    module_path, _, attr = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"invalid import path: {dotted_path!r}")
    module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        raise AttributeError(f"{module_path} has no attribute {attr!r}")
    return getattr(module, attr)


def _infer_config_class(deployer_cls: type) -> type:
    """Pull the Config class out of the deployer's ``__init__(self, config: X)`` signature."""
    import inspect
    sig = inspect.signature(deployer_cls.__init__)
    if "config" not in sig.parameters:
        raise TypeError(
            f"{deployer_cls.__name__} has no 'config' parameter; "
            f"cannot infer Config class — register the shortcut explicitly."
        )
    ann = sig.parameters["config"].annotation
    if ann is inspect.Parameter.empty:
        raise TypeError(
            f"{deployer_cls.__name__}.__init__ has untyped 'config' parameter; "
            f"add type annotation or register the shortcut explicitly."
        )
    return ann
