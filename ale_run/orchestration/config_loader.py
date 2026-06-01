"""YAML loader for ExperimentSpec.

The experiment yaml is deliberately thin: it wires together externalized
config files and carries only run-level knobs. Agent and environment
configuration live in their own files under ``configs/``; the experiment
references them by path.

Shape:

* ``agents: [<path>, ...]`` — a list of paths to agent config yamls under
  ``configs/agents/``. Each referenced file is a full agent preset
  (``harness`` + ``model`` + ``config``). Listing more than one runs the
  agent matrix (every agent over every task). ``agent: <path>`` is accepted
  as the single-agent shorthand. The agent's ``id`` defaults to the preset
  filename stem so two presets of the same harness don't collide.
* ``environment: <path>`` — a single path to an environment yaml under
  ``configs/environments/``. The file carries ``provider:`` plus its
  provider-specific knobs. Exactly one environment per experiment.
* ``tasks: <path>`` string form. ``.txt`` → one task path per line
  (variant 0); ``.yaml`` → list of ``{path, variants}`` entries. An inline
  list of ``{path, variants}`` is also accepted.
* Run-level keys live at the experiment top level: ``output``,
  ``artifacts_path``, ``concurrency``, ``cleanup_mode``, ``prompt_suffix``.

Other behavior:

* ``${env:VAR}`` substitution at parse time (KeyError if VAR unset) — applied
  to the experiment yaml AND every referenced config file.
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

from .experiment_spec import (
    AgentSpec,
    ArtifactsSpec,
    EnvironmentSpec,
    ExperimentSpec,
    OutputSpec,
    ProviderSpec,
    TaskSpec,
)

# Fallback container ref if a docker-routed image family has no registered
# docker_image (every shipped linux family does).
_DEFAULT_DOCKER_IMAGE = "agentslastexam/ale-kasm:latest"

logger = logging.getLogger(__name__)


# ${env:VARNAME} → os.environ[VARNAME]
_ENV_RE = re.compile(r"\$\{env:([A-Z_][A-Z0-9_]*)\}")

# Top-level keys consumed by the loader (anything else → TypeError).
_TOP_LEVEL_KEYS = frozenset({
    "name", "secret_file", "agent", "agents", "environment", "tasks",
    # run-level fields:
    "output",
    "concurrency",
    "cleanup_mode",
    "prompt_suffix",
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


# =============================================================================
# Top-level builder
# =============================================================================

def _build_experiment(raw: dict[str, Any], *, base_dir: Path) -> ExperimentSpec:
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise TypeError(f"unknown top-level keys: {sorted(unknown)}")
    _require(raw, "name")

    # Run-level knobs live directly at the experiment top level. `output.root`
    # is the host-side run-log dir (a workflow choice); artifact SOURCING and
    # output MIRRORING (task_data_source / output_path) are substrate-coupled
    # and live in the environment yaml instead (see _build_environment_from_path).
    output = _build_output(raw.get("output") or {})
    concurrency = _build_concurrency(raw)
    cleanup_mode = _build_cleanup_mode(raw)
    prompt_suffix = str(raw.get("prompt_suffix") or "")

    if "agent" in raw and "agents" in raw:
        raise ValueError("set either `agent:` (single path) or `agents:` (list of paths), not both")
    if "agent" in raw:
        agents = [_build_agent_from_path(raw["agent"], base_dir=base_dir)]
    elif "agents" in raw:
        if not isinstance(raw["agents"], list):
            raise TypeError(f"agents must be a list of yaml paths, got {type(raw['agents']).__name__}")
        agents = [_build_agent_from_path(p, base_dir=base_dir) for p in raw["agents"]]
    else:
        raise KeyError("missing required field: `agents` (list of paths) or `agent` (single path)")

    if "environment" not in raw:
        raise KeyError("missing required field: `environment` (path to an environment yaml)")
    environment, artifacts = _build_environment_from_path(raw["environment"], base_dir=base_dir)

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
        environment=environment,
        agents=agents,
        tasks=tasks,
        artifacts=artifacts,
        concurrency=concurrency,
        cleanup_mode=cleanup_mode,
        prompt_suffix=prompt_suffix,
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

# Shape of an agent config yaml (configs/agents/<preset>.yaml):
#   harness | class : str          — registry shortcut OR fqdn
#   model            : str          — sugar → config.model
#   id               : str | None   — defaults to the preset filename stem
#   executor         : str | None   — vm | local | docker (deployer default if None)
#   config           : dict         — passed verbatim to the deployer Config
_AGENT_TOP_KEYS = frozenset({"harness", "class", "model", "id", "executor", "config"})


def _build_agent_from_path(path: Any, *, base_dir: Path) -> AgentSpec:
    """Load one agent preset from a ``configs/agents/<preset>.yaml`` path."""
    if not isinstance(path, str):
        raise TypeError(
            f"each agent entry must be a path string to a config yaml, "
            f"got {type(path).__name__}: {path!r}"
        )
    resolved = _resolve_path(path, base_dir)
    raw = _read_yaml(resolved)
    if not isinstance(raw, dict):
        raise TypeError(f"agent config {path!r} must be a mapping")

    unknown = set(raw) - _AGENT_TOP_KEYS
    if unknown:
        raise TypeError(f"agent config {path!r} has unknown keys: {sorted(unknown)}")

    if "harness" in raw and "class" in raw:
        raise ValueError(
            f"agent config {path!r}: set either `harness:` (shortcut) or "
            f"`class:` (fqdn), not both"
        )
    cls = raw.get("harness") or raw.get("class")
    if not cls:
        raise KeyError(f"agent config {path!r} missing required field: `harness` (or `class`)")
    cls = str(cls)

    config: dict[str, Any] = dict(raw.get("config") or {})
    # `model:` at the agent level is sugar for `config.model:`.
    if (m := raw.get("model")) is not None:
        config["model"] = m

    # Default the agent id to the preset filename stem so two presets of the
    # same harness in one `agents:` matrix get distinct, readable ids.
    agent_id = raw.get("id") or resolved.stem

    executor = raw.get("executor")
    if executor is not None and not isinstance(executor, str):
        raise TypeError(f"agent config {path!r}: executor must be a string, got {type(executor).__name__}")

    return AgentSpec(id=str(agent_id), class_=cls, config=config, executor=executor)


# ---- environment ---------------------------------------------------------

# Keys at the env-yaml top level that are NOT provider config.
_ENVIRONMENT_TOP_RESERVED = frozenset({
    "provider", "providers", "snapshots", "task_data_source", "output_path",
})


def _build_environment_from_path(
    path: Any, *, base_dir: Path,
) -> tuple[EnvironmentSpec, ArtifactsSpec]:
    """Load the environment from a ``configs/environments/<env>.yaml``.

    Two shapes are accepted:

    * **Per-snapshot** (``snapshots:`` present) — each task-card snapshot maps
      to ``{provider, image, <provider>: {knobs}}``. Shared per-provider config
      (creds/network) lives under ``providers:``. The loader reshapes this into
      one :class:`ProviderSpec` per provider kind (gcloud gets its snapshots
      subset in the format its Config expects; docker gets the resolved image +
      sizing) plus a snapshot→kind routing table.
    * **Single provider** (``provider:`` at top level, no ``snapshots:``) — the
      dev/``static`` shape: one provider serves every snapshot (it is a fixed
      attached box, so there is nothing to map).

    Returns the resolved :class:`EnvironmentSpec` plus the artifact-path config.
    """
    if not isinstance(path, str):
        raise TypeError(
            f"environment must be a single path string to an environment yaml, "
            f"got {type(path).__name__}: {path!r}"
        )
    raw = _read_yaml(_resolve_path(path, base_dir))
    if not isinstance(raw, dict):
        raise TypeError(f"environment config {path!r} must be a mapping")

    artifacts = _build_artifacts({
        "task_data_source": raw.get("task_data_source"),
        "output_path": raw.get("output_path"),
    })

    if "snapshots" in raw:
        env = _build_per_snapshot_env(raw, path)
    elif "provider" in raw:
        env = _build_single_provider_env(raw, path)
    else:
        raise KeyError(
            f"environment config {path!r} must declare either `snapshots:` "
            f"(per-snapshot provider mapping) or `provider:` (single provider)"
        )
    return env, artifacts


def _build_single_provider_env(raw: dict[str, Any], path: str) -> EnvironmentSpec:
    """Single-provider env (e.g. static dev attach): one provider, all snapshots."""
    provider = str(raw["provider"])
    cfg = {k: v for k, v in raw.items() if k not in _ENVIRONMENT_TOP_RESERVED}
    if (sa := cfg.get("service_account_key")) is not None:
        cfg["service_account_key"] = str(Path(str(sa)).expanduser())
    _validate_provider_required(provider, cfg, path)
    return EnvironmentSpec(
        provider_specs={provider: ProviderSpec(kind=provider, config=cfg)},
        snapshot_kind={},
        default_kind=provider,
    )


def _build_per_snapshot_env(raw: dict[str, Any], path: str) -> EnvironmentSpec:
    """Per-snapshot env: reshape snapshot→{provider,image,knobs} into one
    ProviderSpec per kind + a snapshot→kind routing table."""
    from ..environments.images import get as get_image

    shared = raw.get("providers") or {}
    if not isinstance(shared, dict):
        raise TypeError(f"environment {path!r}: `providers:` must be a mapping")
    snapshots = raw["snapshots"]
    if not isinstance(snapshots, dict) or not snapshots:
        raise TypeError(f"environment {path!r}: `snapshots:` must be a non-empty mapping")

    snapshot_kind: dict[str, str] = {}
    gcloud_snaps: dict[str, Any] = {}   # tag -> {image, gpu, zones}
    docker_cfg: dict[str, Any] | None = None

    for tag, entry in snapshots.items():
        if not isinstance(entry, dict):
            raise TypeError(f"environment {path!r}: snapshot {tag!r} must be a mapping")
        kind = entry.get("provider")
        image = entry.get("image")
        if not kind:
            raise KeyError(f"environment {path!r}: snapshot {tag!r} missing `provider`")
        if not image:
            raise KeyError(f"environment {path!r}: snapshot {tag!r} missing `image`")
        knobs = entry.get(kind) or {}       # provider-specific block (e.g. gcloud:/docker:)
        snapshot_kind[str(tag)] = str(kind)

        if kind == "gcloud":
            gcloud_snaps[str(tag)] = {"image": str(image), **knobs}
        elif kind == "docker":
            # docker has one container image; resolve the registry family to its
            # published container ref. Multiple docker snapshots must agree.
            ref = get_image(str(image)).docker_image or _DEFAULT_DOCKER_IMAGE
            this = {"image": ref, "image_family": str(image), **knobs}
            if docker_cfg is not None and docker_cfg != this:
                raise ValueError(
                    f"environment {path!r}: multiple docker snapshots with "
                    f"differing config is not supported ({docker_cfg} vs {this})"
                )
            docker_cfg = this
        elif kind == "static":
            raise ValueError(
                f"environment {path!r}: `static` cannot be a per-snapshot "
                f"provider (it is a fixed attached box). Use a single-provider "
                f"static env (e.g. static_dev.yaml) instead."
            )
        else:
            raise NotImplementedError(
                f"environment {path!r}: snapshot {tag!r} provider {kind!r} not supported"
            )

    provider_specs: dict[str, ProviderSpec] = {}
    if gcloud_snaps:
        gc = dict(shared.get("gcloud") or {})
        if (sa := gc.get("service_account_key")) is not None:
            gc["service_account_key"] = str(Path(str(sa)).expanduser())
        gc["snapshots"] = gcloud_snaps
        _validate_provider_required("gcloud", gc, path)
        provider_specs["gcloud"] = ProviderSpec(kind="gcloud", config=gc)
    if docker_cfg is not None:
        dk = {**(shared.get("docker") or {}), **docker_cfg}
        provider_specs["docker"] = ProviderSpec(kind="docker", config=dk)

    return EnvironmentSpec(
        provider_specs=provider_specs,
        snapshot_kind=snapshot_kind,
        default_kind=None,
    )


def _validate_provider_required(provider: str, cfg: dict[str, Any], path: str = "") -> None:
    """Surface friendly errors for the most common missing-field mistakes
    BEFORE the deployer Config dataclass raises a less-readable TypeError."""
    where = f" in {path!r}" if path else ""
    if provider == "gcloud":
        for k in ("project", "service_account_key"):
            if not cfg.get(k):
                raise KeyError(
                    f"environment provider=gcloud missing required field `{k}`{where} "
                    f"(set it under `providers.gcloud`)"
                )
    elif provider == "static":
        if not cfg.get("endpoint"):
            raise KeyError(f"environment provider=static missing required field `endpoint`{where}")


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
