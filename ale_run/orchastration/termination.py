"""run.json termination resolver per LOG_SPEC §4.

Three pure helpers:

  - :func:`classify_error`   substring-match an exception against a small
                             category vocabulary.
  - :func:`resolve_phase`    prefer ``env.current_phase`` over the
                             lifecycle's coarse tracker.
  - :func:`redact_config`    redact ``*_api_key`` / ``api_key`` from an
                             agent's yaml config before logging.
"""

from __future__ import annotations

import asyncio
from typing import Any

_CATEGORY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("rate_limited", ("rate limit", "ratelimit", "429", "too many requests")),
    (
        "vm_quota_exhausted",
        (
            "quota",
            "stockout",
            "resource_exhausted",
            "does not have enough resources",
            "cpus_per_vm_family",
        ),
    ),
    (
        "auth_failed",
        (
            "401",
            "403",
            "authentication_failed",
            "permission denied",
            "unauthorized",
            "forbidden",
            "llm auth failed",
            "user not found",
            "invalid api key",
        ),
    ),
    (
        "gcs_missing",
        ("matched no objects", "no urls matched", "bucketnotfoundexception", "no such object"),
    ),
    (
        "transport_error",
        (
            "connection reset",
            "connection refused",
            "503",
            "service unavailable",
            "deadline exceeded",
            "broken pipe",
            "remote end closed connection",
        ),
    ),
    ("rpc_timeout", ("timeout", "timed out")),
]


def classify_error(exc: BaseException) -> str | None:
    """Return a LOG_SPEC §4 termination.category for ``exc``, or None.

    ``KeyboardInterrupt`` / ``asyncio.CancelledError`` always yield None.
    """
    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
        return None
    if isinstance(exc, asyncio.TimeoutError) or isinstance(exc, TimeoutError):
        return "rpc_timeout"
    msg = str(exc).lower()
    for category, substrings in _CATEGORY_PATTERNS:
        if any(s in msg for s in substrings):
            return category
    return None


def resolve_phase(env_phase: str | None, fallback: str | None) -> str | None:
    """Pick the phase reported by the env if present, else the lifecycle's tracker.

    Returns None for clean completions (callers should pass None for both).
    """
    return env_phase or fallback


_REDACTED_KEYS = frozenset(
    {
        "anthropic_api_key",
        "openrouter_api_key",
        "openai_api_key",
        "brave_api_key",
        "api_key",
    }
)


def redact_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if k.lower() in _REDACTED_KEYS and isinstance(v, str) and v:
            out[k] = f"***{v[-4:]}" if len(v) >= 4 else "***"
        else:
            out[k] = v
    return out


def err_dict(exc: BaseException) -> dict[str, Any]:
    """Build the LOG_SPEC §4 termination.error payload from an exception."""
    import traceback

    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
