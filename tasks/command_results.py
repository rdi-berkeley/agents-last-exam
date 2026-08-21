"""Shared fail-closed predicates for evaluator command results."""

from __future__ import annotations

from collections.abc import Mapping


RETURN_CODE_ALIASES = ("return_code", "returncode", "exit_code", "rc")
_MISSING = object()


def extract_return_code(result: object) -> int | None:
    """Return one unambiguous exact-integer command result, or fail closed."""

    values: list[object] = []
    try:
        if isinstance(result, Mapping):
            values = [result[name] for name in RETURN_CODE_ALIASES if name in result]
        else:
            for name in RETURN_CODE_ALIASES:
                value = getattr(result, name, _MISSING)
                if value is not _MISSING:
                    values.append(value)
    except Exception:
        return None

    if not values or any(type(value) is not int for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def is_exact_zero_return_code(value: object) -> bool:
    """Accept only a present, non-boolean integer zero return code."""

    return type(value) is int and value == 0
