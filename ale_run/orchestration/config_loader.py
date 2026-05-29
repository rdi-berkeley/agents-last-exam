"""YAML loader for ExperimentSpec.

The public-facing yaml is minimal — name / agent / environment / tasks — with
long-tail knobs pulled in via ``profile:`` paths under ``configs/``. This
loader normalizes that shape into the internal ``ExperimentSpec`` dataclass,
which the Runner consumes.

Three shape conveniences relative to the dataclass:

* ``agent: {...}`` single-dict form (most experiments). Lowered into the
  internal ``agents: [...]`` matrix as a single entry. Legacy
  ``agents: [...]`` lists still work for matrix runs.
* ``tasks: <path>`` string form. ``.txt`` → one task path per line
  (variant 0); ``.yaml`` → list of ``{path, variants}`` entries. The
  legacy inline list form still works.
* ``profile: <relative-path>`` on agent / environment, plus top-level
  ``run_profile: <relative-path>``. Profiles are partial dicts; the main
  yaml's keys win on conflict (deep-merged one level for ``config:``).

Other behavior:

* ``${env:VAR}`` substitution at parse time (KeyError if VAR unset).
* All relative paths resolve against the main yaml's directory.
* Unknown / missing keys raise loudly.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .factory import AGENT_REGISTRY
from .experiment_spec import (
    AgentSpec,
    ArtifactsSpec,
    ExperimentSpec,
    OutputSpec,
    ProviderSpec,
    TaskSpec,
)

logger = logging.getLogger(__name__)


# ${env:VARNAME} → os.environ[VARNAME]
_ENV_RE = re.compile(r"\$\{env:([A-Z_][A-Z0-9_]*)\}")

# Top-level keys consumed by the loader (anything else → TypeError).
_TOP_LEVEL_KEYS = frozenset({
    "name", "secret_file", "agent", "agents", "environment", "tasks",
    "run_profile",
    # run-level fields (also accepted inline at the top, override run_profile):
    "output", "artifacts_path",
    "concurrency",
    "cleanup_mode",
})


# =============================================================================
# Public API
# =============================================================================

def load_experiment(path: str | Path) -> ExperimentSpec:
    """Read a yaml file, substitute env vars, build :class:`ExperimentSpec`.

    Two-pass parse: first pass extracts ``secret_file:`` (if present) without
    ``${env:VAR}`` substitution; that file is loaded into ``os.environ``
    (shell env wins on conflict); second pass parses for real, with
    substitution. If ``secret_file:`` is unset, falls back to auto-loading
    ``<main yaml's directory>/secret/.env`` (then ``.../.env``) when they
    exist. Profile yamls referenced from the main file inherit the same
    ``os.environ``, so their ``${env:...}`` refs resolve the same way.
    """
    main_path = Path(path).resolve()
    base_dir = main_path.parent

    text = main_path.read_text()
    # First pass: parse raw text (no substitution) just to find secret_file.
    try:
        pre = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"experiment yaml {main_path} is not valid yaml: {exc}") from exc
    if not isinstance(pre, dict):
        raise TypeError(f"experiment yaml root must be a mapping, got {type(pre).__name__}")
    secret_file = pre.get("secret_file")
    if secret_file is not None:
        _load_dotenv(_resolve_path(secret_file, base_dir), override=True)
    else:
        # Convenience auto-detect. `secret/.env` is the canonical location
        # (alongside the checked-in `secret/.env.example` template); legacy
        # `.env` at the yaml's directory is still honored as a fallback.
        _load_dotenv(base_dir / "secret" / ".env")
        _load_dotenv(base_dir / ".env")

    # Second pass: substitute + parse for real.
    raw = yaml.safe_load(_substitute_env(text)) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"experiment yaml root must be a mapping, got {type(raw).__name__}")
    return _build_experiment(raw, base_dir=base_dir)


# =============================================================================
# Env + yaml helpers
# =============================================================================

def _load_dotenv(path: Path, *, override: bool = False) -> None:
    """Hand-rolled dotenv parser. Format: ``KEY=value`` per line; ``#``
    starts a line comment; surrounding single/double quotes are stripped;
    blank lines and unparseable lines are skipped (logged at DEBUG).

    When ``override=False`` (default / auto-detected files), shell env
    wins — never overwrites an already-set os.environ key.  When
    ``override=True`` (explicit ``secret_file:`` in the experiment yaml),
    the file is the authoritative source and overwrites shell values.

    Missing file is silently ignored — most setups have one, some don't.
    """
    if not path.is_file():
        return
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            logger.debug("dotenv %s:%d skipped (no '='): %r", path, line_no, raw)
            continue
        key = key.strip()
        value = value.strip()
        # Strip an inline comment after an unquoted value: `FOO=bar # note`.
        if value and value[0] not in ("'", '"'):
            value = value.split(" #", 1)[0].rstrip()
        # Strip matching quotes around the value.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key.isidentifier():
            logger.debug("dotenv %s:%d skipped (bad key): %r", path, line_no, raw)
            continue
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def _read_yaml(path: Path) -> Any:
    text = path.read_text()
    text = _substitute_env(text)
    return yaml.safe_load(text) or {}


def _substitute_env(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        var = m.group(1)
        val = os.environ.get(var)
        if val is None:
            raise KeyError(
                f"yaml references ${{env:{var}}} but {var} is not set"
            )
        return val
    return _ENV_RE.sub(repl, text)


def _resolve_path(p: str | Path, base_dir: Path) -> Path:
    """Relative paths resolve under ``base_dir``; absolute / ``~`` honored."""
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    return (base_dir / pp).resolve()


def _merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge: overlay wins. ``config`` sub-key is deep-merged one
    level so an agent profile's ``config.timeout_s`` survives even if the
    main yaml overrides ``config.model``."""
    out: dict[str, Any] = dict(base)
    for k, v in overlay.items():
        if k == "config" and isinstance(v, dict) and isinstance(out.get("config"), dict):
            out["config"] = {**out["config"], **v}
        else:
            out[k] = v
    return out


# =============================================================================
# Top-level builder
# =============================================================================

def _build_experiment(raw: dict[str, Any], *, base_dir: Path) -> ExperimentSpec:
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise TypeError(f"unknown top-level keys: {sorted(unknown)}")
    _require(raw, "name")

    # Apply optional run_profile under the top-level dict, with main yaml
    # winning. Result is the effective top-level for output / artifacts /
    # concurrency.
    if (rp := raw.get("run_profile")) is not None:
        prof = _read_yaml(_resolve_path(rp, base_dir))
        if not isinstance(prof, dict):
            raise TypeError(f"run_profile {rp!r} must be a mapping")
        effective = _merge_dict(prof, {k: v for k, v in raw.items() if k != "run_profile"})
    else:
        effective = dict(raw)

    output = _build_output(effective.get("output") or {})
    artifacts = _build_artifacts(effective.get("artifacts_path") or {})
    concurrency = _build_concurrency(effective)
    cleanup_mode = _build_cleanup_mode(effective)

    if "agent" in raw and "agents" in raw:
        raise ValueError("set either `agent:` (single) or `agents:` (list), not both")
    if "agent" in raw:
        agents = [_build_agent_single(raw["agent"], base_dir=base_dir)]
    elif "agents" in raw:
        agents = [_build_agent_single(a, base_dir=base_dir) for a in raw["agents"]]
    else:
        raise KeyError("missing required field: `agent` (or `agents`)")

    provider = _build_environment(raw.get("environment") or {}, base_dir=base_dir)

    if "tasks" not in raw:
        raise KeyError("missing required field: `tasks`")
    tasks = _build_tasks(raw["tasks"], base_dir=base_dir)

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
        cleanup_mode=cleanup_mode,
    )


# =============================================================================
# Section builders
# =============================================================================

def _build_output(raw: dict[str, Any]) -> OutputSpec:
    return OutputSpec(root=str(raw.get("root", ".logs/ale")))


def _build_concurrency(eff: dict[str, Any]) -> int:
    n = int(eff.get("concurrency", 1))
    if n < 1:
        raise ValueError(f"concurrency must be >= 1, got {n}")
    return n


_VALID_CLEANUP_MODES = frozenset({"delete", "stop", "keep"})


def _build_cleanup_mode(eff: dict[str, Any]) -> str:
    raw = str(eff.get("cleanup_mode") or "delete")
    if raw not in _VALID_CLEANUP_MODES:
        raise ValueError(
            f"cleanup_mode must be one of {sorted(_VALID_CLEANUP_MODES)}, got {raw!r}"
        )
    return raw


_VALID_OUTPUT_PATH_LITERALS = frozenset({"local"})
_VALID_TASK_DATA_LITERALS = frozenset({"baked_in_sandbox"})


def _build_artifacts(raw: dict[str, Any]) -> ArtifactsSpec:
    defaults = ArtifactsSpec()

    output_path = raw.get("output_path")
    if output_path is not None:
        op = str(output_path).strip()
        if op and op not in _VALID_OUTPUT_PATH_LITERALS and not op.startswith("gs://"):
            raise ValueError(
                f"artifacts_path.output_path must be null, 'local', or a "
                f"'gs://...' bucket path; got {output_path!r}"
            )
        output_path = op or None

    task_data_source = raw.get("task_data_source") or defaults.task_data_source
    tdp = str(task_data_source).strip()
    if (tdp not in _VALID_TASK_DATA_LITERALS
            and not tdp.startswith("gs://")
            and not tdp.startswith("hf://")):
        raise ValueError(
            f"artifacts_path.task_data_source must be 'baked_in_sandbox', "
            f"'gs://<bucket>', or 'hf://<dataset>'; got {task_data_source!r}"
        )

    return ArtifactsSpec(
        task_data_source=tdp,
        output_path=output_path,
    )


# ---- agent ---------------------------------------------------------------

# Shape of an agent block (after profile merge):
#   harness | class : str          — registry shortcut OR fqdn
#   model            : str          — sugar → config.model
#   id               : str | None   — defaults to harness/class short name
#   executor         : str | None   — vm | local | docker (deployer default if None)
#   profile          : str | None   — consumed by the merger; not in AgentSpec
#   config           : dict         — passed verbatim to the deployer Config
_AGENT_TOP_KEYS = frozenset({"harness", "class", "model", "id", "executor", "profile", "config"})


def _build_agent_single(raw: dict[str, Any], *, base_dir: Path) -> AgentSpec:
    if not isinstance(raw, dict):
        raise TypeError(f"agent must be a mapping, got {type(raw).__name__}")

    merged: dict[str, Any] = dict(raw)
    if (prof_path := merged.pop("profile", None)) is not None:
        prof = _read_yaml(_resolve_path(prof_path, base_dir))
        if not isinstance(prof, dict):
            raise TypeError(f"agent profile {prof_path!r} must be a mapping")
        prof_unknown = set(prof) - _AGENT_TOP_KEYS
        if prof_unknown:
            raise TypeError(
                f"agent profile {prof_path!r} has unknown keys: {sorted(prof_unknown)}"
            )
        merged = _merge_dict(prof, merged)

    unknown = set(merged) - _AGENT_TOP_KEYS
    if unknown:
        raise TypeError(f"agent has unknown keys: {sorted(unknown)}")

    if "harness" in merged and "class" in merged:
        raise ValueError("set either `harness:` (shortcut) or `class:` (fqdn), not both")
    cls = merged.get("harness") or merged.get("class")
    if not cls:
        raise KeyError("agent missing required field: `harness` (or `class`)")
    cls = str(cls)

    config: dict[str, Any] = dict(merged.get("config") or {})
    # `model:` at the agent level is sugar for `config.model:`. Main-yaml
    # value wins (this lambda is only ever called after _merge_dict has
    # already settled the per-level priority).
    if (m := merged.get("model")) is not None:
        config["model"] = m

    agent_id = merged.get("id")
    if agent_id is None:
        # Default to the registry shortcut, or the unqualified tail of a fqdn.
        agent_id = cls if cls in AGENT_REGISTRY else cls.rsplit(".", 1)[-1]

    executor = merged.get("executor")
    if executor is not None and not isinstance(executor, str):
        raise TypeError(f"agent.executor must be a string, got {type(executor).__name__}")

    return AgentSpec(id=str(agent_id), class_=cls, config=config, executor=executor)


# ---- environment ---------------------------------------------------------

# Environment block accepts any flat key besides `provider` / `profile` — those
# extras become ProviderSpec.config kwargs (passed straight to the
# provider-specific Config dataclass).
_ENVIRONMENT_RESERVED = frozenset({"provider", "profile"})


def _build_environment(raw: dict[str, Any], *, base_dir: Path) -> ProviderSpec:
    if not isinstance(raw, dict):
        raise TypeError(f"environment must be a mapping, got {type(raw).__name__}")

    merged: dict[str, Any] = dict(raw)
    if (prof_path := merged.pop("profile", None)) is not None:
        prof = _read_yaml(_resolve_path(prof_path, base_dir))
        if not isinstance(prof, dict):
            raise TypeError(f"environment profile {prof_path!r} must be a mapping")
        if "provider" in prof:
            raise ValueError(
                f"environment profile {prof_path!r} must not set `provider:` — "
                f"`provider` lives in the main yaml so swapping profiles can't "
                f"silently change the backend."
            )
        merged = _merge_dict(prof, merged)

    if "provider" not in merged:
        raise KeyError("environment missing required field: `provider`")
    provider = str(merged["provider"])
    cfg = {k: v for k, v in merged.items() if k not in _ENVIRONMENT_RESERVED}

    # `service_account_key` is a user-friendly path field — expand `~`
    # before handing to the Config dataclass.
    if (sa := cfg.get("service_account_key")) is not None:
        cfg["service_account_key"] = str(Path(str(sa)).expanduser())

    _validate_provider_required(provider, cfg)
    return ProviderSpec(kind=provider, config=cfg)


def _validate_provider_required(provider: str, cfg: dict[str, Any]) -> None:
    """Surface friendly errors for the most common missing-field mistakes
    BEFORE the deployer Config dataclass raises a less-readable TypeError."""
    if provider == "gcloud":
        for k in ("project", "service_account_key"):
            if not cfg.get(k):
                raise KeyError(
                    f"environment.provider=gcloud missing required field `{k}` "
                    f"(set it in the main yaml; profile shouldn't carry it)"
                )
    elif provider == "static":
        if not cfg.get("endpoint"):
            raise KeyError("environment.provider=static missing required field `endpoint`")


# ---- tasks ---------------------------------------------------------------

def _build_tasks(raw: Any, *, base_dir: Path) -> list[TaskSpec]:
    """Three accepted shapes:

    * ``str`` ending in ``.txt``  → one `<path>` per line, variant 0.
    * ``str`` ending in ``.yaml`` → list of ``{path, variants}`` entries.
    * ``list``                    → legacy inline form, each entry a
                                    ``{path, variants}`` dict.
    """
    if isinstance(raw, str):
        return _load_tasks_file(_resolve_path(raw, base_dir))
    if isinstance(raw, list):
        return [_build_task_entry(t) for t in raw]
    raise TypeError(f"tasks must be a string path or a list, got {type(raw).__name__}")


def _load_tasks_file(path: Path) -> list[TaskSpec]:
    if not path.exists():
        raise FileNotFoundError(f"tasks file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        out: list[TaskSpec] = []
        for raw_line in path.read_text().splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            out.append(TaskSpec(path=line, variants=[0]))
        return out
    if suffix in (".yaml", ".yml"):
        data = _read_yaml(path)
        if not isinstance(data, list):
            raise TypeError(f"{path}: top-level must be a list of task entries")
        return [_build_task_entry(t) for t in data]
    raise ValueError(f"unsupported tasks-file suffix: {path.suffix!r} (use .txt or .yaml)")


def _build_task_entry(raw: dict[str, Any]) -> TaskSpec:
    if not isinstance(raw, dict):
        raise TypeError(f"task entry must be a mapping, got {type(raw).__name__}")
    _require(raw, "path")
    variants = raw.get("variants", [0])
    if not isinstance(variants, list) or not all(isinstance(v, int) for v in variants):
        raise TypeError(f"task.variants must be a list of ints, got {variants!r}")
    return TaskSpec(path=str(raw["path"]), variants=list(variants))


# ---- shared --------------------------------------------------------------

def _require(raw: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in raw]
    if missing:
        raise KeyError(f"missing required field(s): {missing}")
