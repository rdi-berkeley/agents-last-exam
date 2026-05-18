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
    # Concurrency knobs. Old single `concurrency:` field was renamed to
    # `run_concurrency:` so the dual-knob semantics are explicit. We accept
    # the old key with a deprecation warning rather than silent rename, so
    # operators see the change.
    if "concurrency" in raw and "run_concurrency" in raw:
        raise ValueError(
            "experiment yaml has both `concurrency:` (legacy) and "
            "`run_concurrency:` — pick one (run_concurrency wins long-term)."
        )
    if "concurrency" in raw:
        import warnings
        warnings.warn(
            "experiment yaml `concurrency:` is deprecated — rename to "
            "`run_concurrency:` (semantics unchanged for the single-knob "
            "case). Set `provision_concurrency:` to override the matching "
            "VM-acquire cap.",
            DeprecationWarning, stacklevel=2,
        )
        run_concurrency = int(raw["concurrency"])
    else:
        run_concurrency = int(raw.get("run_concurrency", 1))
    if run_concurrency < 1:
        raise ValueError(f"run_concurrency must be >= 1, got {run_concurrency}")
    provision_raw = raw.get("provision_concurrency")
    provision_concurrency: int | None
    if provision_raw is None:
        provision_concurrency = None
    else:
        provision_concurrency = int(provision_raw)
        if provision_concurrency < 1:
            raise ValueError(
                f"provision_concurrency must be >= 1, got {provision_concurrency}"
            )
    eval_timeout_s = float(raw.get("eval_timeout_s", 3600.0))
    if eval_timeout_s <= 0:
        raise ValueError(f"eval_timeout_s must be > 0, got {eval_timeout_s}")
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
        run_concurrency=run_concurrency,
        provision_concurrency=provision_concurrency,
        eval_timeout_s=eval_timeout_s,
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
    runtime = raw.get("runtime")
    if runtime is not None and not isinstance(runtime, str):
        raise TypeError(f"agent.runtime must be a string, got {type(runtime).__name__}")
    return AgentSpec(
        id=str(raw["id"]),
        class_=str(raw["class"]),
        config=dict(raw.get("config") or {}),
        runtime=runtime,
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
