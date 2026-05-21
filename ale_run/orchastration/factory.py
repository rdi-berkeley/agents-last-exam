"""AGENT_REGISTRY + ``build_provider`` + resolver helpers.

Two factory surfaces:

  - :data:`AGENT_REGISTRY`             shortcut → BaseAgentDeployer subclass
  - :func:`build_provider`             ProviderSpec → Provider instance
  - :func:`resolve_agent`              AgentSpec → (deployer_cls, config_cls)
  - :func:`build_config`               instantiate the deployer's config dataclass
                                       from yaml kwargs.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..base_interface import Provider
    from .experiment_spec import AgentSpec, ProviderSpec


# ----------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------


# Registered shortcut → ``"<module>.<class>"`` FQN. Deferred imports so a
# ``--dry-run`` parse doesn't pull pydantic / cua_bench / litellm just to
# check that an agent shortcut is recognized.
_AGENT_FQNS: dict[str, str] = {
    "claude_code": "ale_run.agents.claude_code.deployer.ClaudeCodeDeployer",
    "ale_claw": "ale_run.agents.ale_claw.deployer.AleClawDeployer",
}


class _LazyRegistryView:
    """Dict-like view: ``"key" in AGENT_REGISTRY`` works without importing classes.

    ``AGENT_REGISTRY[name]`` triggers the actual import; iteration only
    yields names.
    """

    def __contains__(self, key: object) -> bool:
        return key in _AGENT_FQNS

    def __getitem__(self, key: str) -> type:
        fqn = _AGENT_FQNS[key]
        mod_name, attr = fqn.rsplit(".", 1)
        return getattr(importlib.import_module(mod_name), attr)

    def __iter__(self):
        return iter(_AGENT_FQNS)

    def keys(self):
        return _AGENT_FQNS.keys()

    def __repr__(self) -> str:
        return f"_LazyRegistryView({sorted(_AGENT_FQNS)})"


AGENT_REGISTRY = _LazyRegistryView()


def _config_class_for(deployer_cls: type) -> type:
    """The deployer's matching Config dataclass.

    Convention: a deployer in ``ale_run.agents.<pkg>.deployer`` has its
    config in ``ale_run.agents.<pkg>.config`` as ``<DeployerStem>Config``
    where ``<DeployerStem>`` is the deployer's class name with ``Deployer``
    stripped.
    """
    stem = deployer_cls.__name__.removesuffix("Deployer")
    pkg = deployer_cls.__module__.rsplit(".", 1)[0]
    config_mod = importlib.import_module(f"{pkg}.config")
    cls_name = f"{stem}Config"
    cls = getattr(config_mod, cls_name, None)
    if cls is None:
        raise RuntimeError(
            f"agent factory: deployer {deployer_cls.__name__} has no "
            f"matching {cls_name} in {pkg}.config"
        )
    return cls


def resolve_agent(spec: "AgentSpec") -> tuple[type, type]:
    """Resolve an AgentSpec to ``(deployer_cls, config_cls)``.

    Raises if ``spec.class_`` isn't registered AND isn't an importable FQN.
    """
    cls_key = spec.class_
    if cls_key in AGENT_REGISTRY:
        deployer_cls = AGENT_REGISTRY[cls_key]
    else:
        if "." not in cls_key:
            raise KeyError(
                f"agent class {cls_key!r} not in AGENT_REGISTRY "
                f"({sorted(_AGENT_FQNS)}) and not a fully-qualified module path"
            )
        mod_name, attr = cls_key.rsplit(".", 1)
        deployer_cls = getattr(importlib.import_module(mod_name), attr)

    # Validate runtime against the deployer's supported set + the
    # framework's runtime registry (every Runtime subclass that lifecycle
    # can build).
    from ..environments.runtime import RUNTIME_REGISTRY

    supported = getattr(deployer_cls, "supported_runtimes", frozenset())
    if spec.runtime is not None and spec.runtime not in supported:
        raise ValueError(
            f"agent {cls_key!r}: runtime {spec.runtime!r} not in "
            f"supported set {sorted(supported)}"
        )
    chosen = spec.runtime or next(iter(supported), None)
    if chosen is None:
        raise ValueError(
            f"agent {cls_key!r} declares no supported_runtimes — set the "
            "ClassVar on the deployer subclass."
        )
    if chosen not in RUNTIME_REGISTRY:
        raise NotImplementedError(
            f"agent {cls_key!r}: chosen runtime {chosen!r} not in "
            f"RUNTIME_REGISTRY ({sorted(RUNTIME_REGISTRY)})"
        )

    return deployer_cls, _config_class_for(deployer_cls)


def build_config(config_cls: type, raw: dict[str, Any]) -> Any:
    """Instantiate a deployer's config dataclass from a yaml kwargs dict.

    Filters unknown keys (so a profile-style yaml with extras doesn't
    raise TypeError) and coerces ``tuple``-typed fields from yaml ``list``.
    """
    if not dataclasses.is_dataclass(config_cls):
        raise TypeError(f"{config_cls.__name__} is not a dataclass")

    field_map = {f.name: f for f in dataclasses.fields(config_cls)}
    kwargs: dict[str, Any] = {}
    for k, v in raw.items():
        f = field_map.get(k)
        if f is None:
            continue
        if isinstance(v, list) and getattr(f.type, "__origin__", None) is tuple:
            v = tuple(v)
        kwargs[k] = v
    return config_cls(**kwargs)


# ----------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------


def build_provider(spec: "ProviderSpec") -> "Provider":
    """Build a Provider instance from a ProviderSpec."""
    kind = spec.kind
    if kind == "gcloud":
        from ..environments.providers.gcloud import GcloudProvider
        return GcloudProvider(spec.config)
    if kind == "static":
        from ..environments.providers.static import StaticProvider
        return StaticProvider(spec.config)
    raise NotImplementedError(f"provider kind {kind!r} is not implemented")
