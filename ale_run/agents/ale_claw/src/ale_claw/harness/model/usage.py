"""Record out-of-band helper model usage without storing prompts or credentials."""

from __future__ import annotations

import contextvars
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HelperUsageRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def record(self, response: Any, *, model: str, purpose: str) -> None:
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            usage = {}
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        if hasattr(details, "model_dump"):
            details = details.model_dump()
        if not isinstance(details, dict):
            details = {}
        hidden = getattr(response, "_hidden_params", None) or {}
        cost = usage.get("cost")
        if not isinstance(cost, (int, float)) and isinstance(hidden, dict):
            cost = hidden.get("response_cost")
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "model": model,
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
            "cache_read_input_tokens": details.get("cached_tokens"),
            "cache_write_input_tokens": details.get("cache_write_tokens"),
            "cost_usd": cost if isinstance(cost, (int, float)) else None,
        }
        with self._lock, self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")


_recorder: contextvars.ContextVar[HelperUsageRecorder | None] = contextvars.ContextVar(
    "ale_claw_helper_usage", default=None
)


def start_helper_usage(path: Path) -> contextvars.Token:
    return _recorder.set(HelperUsageRecorder(path))


def stop_helper_usage(token: contextvars.Token) -> None:
    _recorder.reset(token)


def record_helper_usage(response: Any, *, model: str, purpose: str) -> None:
    recorder = _recorder.get()
    if recorder is not None:
        recorder.record(response, model=model, purpose=purpose)
