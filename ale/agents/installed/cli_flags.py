"""Declarative CLI flag / env var descriptors (harbor-inspired).

Each :class:`CliFlag` / :class:`EnvVar` maps one config dataclass field to
a CLI argument or env var. The deployer declares a list of these and calls
:func:`build_cli_args` + :func:`build_env` to render — no hand-rolled
``if cfg.foo: cmd.append(...)`` chains.

Usage::

    CLI_FLAGS = [
        CliFlag("model", "--model"),
        CliFlag(
            "max_turns", "--max-turns",
            when=lambda v: v is not None and v >= 0,
        ),
        CliFlag(
            "dangerously_skip_permissions",
            "--dangerously-skip-permissions",
            kind="bool_flag",      # no value, emitted only when True
        ),
        CliFlag(
            "disabled_tools", "--disallowedTools",
            kind="space_joined",   # tuple → "a b c"
            when=lambda v: bool(v),
        ),
    ]

    ENV_VARS = [
        EnvVar("anthropic_api_key", "ANTHROPIC_API_KEY",
               when=lambda cfg: not cfg.openrouter_api_key),
        EnvVar("openrouter_api_key", "ANTHROPIC_AUTH_TOKEN",
               when=lambda cfg: bool(cfg.openrouter_api_key)),
    ]

    args = build_cli_args(cfg, CLI_FLAGS)
    env_lines = build_env(cfg, ENV_VARS)

Kinds:
    ``value``       — ``--flag <value>`` (default)
    ``equals``      — ``--flag=<value>``
    ``bool_flag``   — ``--flag`` (no value), only emitted when truthy
    ``multi_value`` — ``--flag <a> <b> <c>`` (one flag, separate args per item)
    ``repeated``    — ``--flag <a> --flag <b>`` (one flag per item)
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal


Kind = Literal["value", "equals", "bool_flag", "multi_value", "repeated"]


@dataclass(frozen=True)
class CliFlag:
    """One CLI argument derived from a config field."""

    field_name: str
    flag: str
    kind: Kind = "value"
    when: Callable[[Any], bool] | None = None
    """Predicate on the field's value. Default: emit iff value is not None
    and (for collections) non-empty and (for bools) True."""
    formatter: Callable[[Any], str] | None = None
    """Optional value→str converter applied to scalars (``value``/``equals``)
    or to each element (``repeated``). Ignored for ``bool_flag``."""


@dataclass(frozen=True)
class EnvVar:
    """One env-var derived from a config field."""

    field_name: str
    env_var: str
    when: Callable[[Any], bool] | None = None
    """Predicate on the *config object* (not just the field value). Useful
    when one field gates whether another env var is emitted."""
    formatter: Callable[[Any], str] | None = None
    """Optional value→str. Defaults to ``str()``."""


# =============================================================================
# Builders
# =============================================================================

def build_cli_args(config: Any, descriptors: Iterable[CliFlag]) -> list[str]:
    """Render flag descriptors against ``config``. Returns a flat ``argv`` list.

    Caller is responsible for shell-quoting (use :func:`shlex.quote` per token
    when assembling a single shell command string).
    """
    args: list[str] = []
    for d in descriptors:
        if not hasattr(config, d.field_name):
            raise AttributeError(
                f"CliFlag {d.flag} references unknown config field {d.field_name!r}"
            )
        value = getattr(config, d.field_name)
        if not _should_emit_flag(d, value):
            continue
        args.extend(_render_flag(d, value))
    return args


def build_env(config: Any, descriptors: Iterable[EnvVar]) -> dict[str, str]:
    """Render env descriptors against ``config``. Returns ``{name: value}``."""
    out: dict[str, str] = {}
    for d in descriptors:
        if not hasattr(config, d.field_name):
            raise AttributeError(
                f"EnvVar {d.env_var} references unknown config field {d.field_name!r}"
            )
        value = getattr(config, d.field_name)
        if d.when is not None and not d.when(config):
            continue
        if d.when is None and (value is None or value == ""):
            continue
        out[d.env_var] = (d.formatter or str)(value)
    return out


def render_env_lines(env: dict[str, str], *, shell: str = "bash") -> str:
    """Render a dict as shell ``export`` lines. Single-quotes for safety."""
    if shell != "bash":
        raise NotImplementedError(f"only bash supported for now, got {shell!r}")
    return "".join(
        f"export {k}={shlex.quote(v)}\n" for k, v in env.items()
    )


# =============================================================================
# Internals
# =============================================================================

def _should_emit_flag(d: CliFlag, value: Any) -> bool:
    if d.when is not None:
        return bool(d.when(value))
    # Default predicates per kind.
    if d.kind == "bool_flag":
        return bool(value)
    if value is None:
        return False
    if d.kind in ("multi_value", "repeated") and len(value) == 0:
        return False
    return True


def _render_flag(d: CliFlag, value: Any) -> list[str]:
    fmt = d.formatter or str
    if d.kind == "bool_flag":
        return [d.flag]
    if d.kind == "value":
        return [d.flag, fmt(value)]
    if d.kind == "equals":
        return [f"{d.flag}={fmt(value)}"]
    if d.kind == "multi_value":
        return [d.flag, *(fmt(v) for v in value)]
    if d.kind == "repeated":
        out: list[str] = []
        for v in value:
            out.extend([d.flag, fmt(v)])
        return out
    raise ValueError(f"unknown CliFlag kind: {d.kind!r}")
