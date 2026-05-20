"""Rate-limit detector for agent stderr streams.

Ported from simprun/monitor.py. Standalone helper — the lifecycle starts a
background task that tails the agent's stderr file and calls
:meth:`RateLimitDetector.check` periodically.
"""

from __future__ import annotations

import re
import time

_PATTERNS = [
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"\b429\b"),
    re.compile(r"\b503\b"),
    re.compile(r"overloaded", re.IGNORECASE),
    re.compile(r"too.?many.?requests", re.IGNORECASE),
    re.compile(r"throttl", re.IGNORECASE),
]

_TRIGGER_THRESHOLD = 3
_WINDOW_SECONDS = 60.0


class RateLimitDetector:
    def __init__(
        self,
        threshold: int = _TRIGGER_THRESHOLD,
        window: float = _WINDOW_SECONDS,
    ):
        self._threshold = threshold
        self._window = window
        self._hits: list[float] = []
        self._seen_bytes = 0

    def check(self, text: str, timestamp: float | None = None) -> bool:
        ts = timestamp or time.monotonic()
        new_text = text[self._seen_bytes:]
        self._seen_bytes = len(text)
        if not new_text:
            return self.is_triggered

        for pat in _PATTERNS:
            if pat.search(new_text):
                self._hits.append(ts)
                break

        cutoff = ts - self._window
        self._hits = [t for t in self._hits if t > cutoff]
        return self.is_triggered

    @property
    def is_triggered(self) -> bool:
        return len(self._hits) >= self._threshold

    def reset(self) -> None:
        self._hits.clear()
        self._seen_bytes = 0
