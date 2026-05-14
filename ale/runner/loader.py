"""YAML loader for ExperimentSpec.

Handles:
- ``${env:VAR}`` substitution at parse time (KeyError if VAR unset).
- Defaults: artifacts (empty), concurrency=1, output.root=".logs/ale".
- Schema validation: required fields, unknown keys → loud TypeError.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .spec import (
    AgentSpec,
    ArtifactsSpec,
    ExperimentSpec,
    OutputSpec,
    ProviderSpec,
    TaskSpec,
)


# ${env:VARNAME} → os.environ[VARNAME]
_ENV_RE = re.compile(r"\$\{env:([A-Z_][A-Z0-9_]*)\}")


# =============================================================================
# Public API
# =============================================================================

def load_experiment(path: str | Path) -> ExperimentSpec:
    """Read a yaml file, substitute env vars, build :class:`ExperimentSpec`."""
    text = Path(path).read_text()
    text = _substitute_env(text)
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"experiment yaml root must be a mapping, got {type(raw).__name__}")
    return _build_experiment(raw)


# =============================================================================
# Env substitution
# =============================================================================

def _substitute_env(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        var = m.group(1)
        val = os.environ.get(var)
        if val is None:
            raise KeyError(
                f"experiment yaml references ${{env:{var}}} but {var} is not set"
            )
        return val
    return _ENV_RE.sub(repl, text)


# =============================================================================
# Builders (raw dict → dataclasses)
# =============================================================================

def _build_experiment(raw: dict[str, Any]) -> ExperimentSpec:
    _require(raw, "name", "provider", "agents", "tasks")
    output = _build_output(raw.get("output") or {})
    provider = _build_provider(raw["provider"])
    agents = [_build_agent(a) for a in raw["agents"]]
    tasks = [_build_task(t) for t in raw["tasks"]]
    artifacts = _build_artifacts(raw.get("artifacts") or {})
    concurrency = int(raw.get("concurrency", 1))
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if not agents:
        raise ValueError("experiment must declare at least one agent")
    if not tasks:
        raise ValueError("experiment must declare at least one task")
    return ExperimentSpec(
        name=str(raw["name"]),
        output=output,
        provider=provider,
        agents=agents,
        tasks=tasks,
        artifacts=artifacts,
        concurrency=concurrency,
    )


def _build_output(raw: dict[str, Any]) -> OutputSpec:
    return OutputSpec(root=str(raw.get("root", ".logs/ale")))


def _build_provider(raw: dict[str, Any]) -> ProviderSpec:
    if not isinstance(raw, dict):
        raise TypeError(f"provider must be a mapping, got {type(raw).__name__}")
    _require(raw, "kind")
    kind = str(raw["kind"])
    cfg = {k: v for k, v in raw.items() if k != "kind"}
    return ProviderSpec(kind=kind, config=cfg)


def _build_agent(raw: dict[str, Any]) -> AgentSpec:
    _require(raw, "id", "class")
    return AgentSpec(
        id=str(raw["id"]),
        class_=str(raw["class"]),
        config=dict(raw.get("config") or {}),
    )


def _build_task(raw: dict[str, Any]) -> TaskSpec:
    _require(raw, "path")
    variants = raw.get("variants", [0])
    if not isinstance(variants, list) or not all(isinstance(v, int) for v in variants):
        raise TypeError(f"task.variants must be a list of ints, got {variants!r}")
    return TaskSpec(path=str(raw["path"]), variants=list(variants))


def _build_artifacts(raw: dict[str, Any]) -> ArtifactsSpec:
    return ArtifactsSpec(
        gcs_bucket=raw.get("gcs_bucket") or None,
        gcs_local_key_file=raw.get("gcs_local_key_file") or None,
        gcs_vm_key_file=raw.get("gcs_vm_key_file") or None,
        fallback_to_cua=bool(raw.get("fallback_to_cua", True)),
    )


def _require(raw: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in raw]
    if missing:
        raise KeyError(f"missing required field(s): {missing}")
