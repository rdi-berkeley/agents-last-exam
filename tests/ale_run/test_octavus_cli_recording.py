"""OctavusCliDeployer._read_thread waiting for the run's recording.

The execution recording is stopped and uploaded after the run ends, so a finished
thread can still report it `processing`. These tests lock in that the host-side read
keeps going until the recording is final, gives up after its own window, and returns
at once when there is no recording to wait for. The HTTP read and the clock are faked.
"""

from __future__ import annotations

import io
import json

import pytest

from ale_run.agents.octavus_cli import OctavusCliDeployer
from ale_run.agents.octavus_cli import deployer as deployer_module

_USAGE = {"costUsd": 0.5, "inputTokens": 10, "outputTokens": 2}
_PROCESSING = {"status": "processing", "visibility": "public", "url": None, "error": None}
_READY = {
    "status": "ready",
    "visibility": "public",
    "url": "https://cdn.example.com/recordings/rec.mp4",
    "error": None,
}


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _read(monkeypatch: pytest.MonkeyPatch, responses: list[dict]) -> tuple[dict | None, int, float]:
    """Run _read_thread against ``responses`` (the last one repeats).

    Returns the thread it settled on, how many reads it made, and the time it waited.
    """
    clock = _Clock()
    reads = {"n": 0}

    def fake_urlopen(request: object, timeout: float = 30) -> io.BytesIO:
        body = responses[min(reads["n"], len(responses) - 1)]
        reads["n"] += 1
        return io.BytesIO(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(deployer_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(deployer_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(deployer_module.time, "sleep", clock.sleep)
    thread = OctavusCliDeployer._read_thread(
        "https://platform.example", "agent1", "t1", "oct_agt_x"
    )
    return thread, reads["n"], clock.now - 1000.0


def test_read_thread_waits_for_the_recording_to_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    thread, reads, _waited = _read(
        monkeypatch,
        [
            {"status": "completed", "usage": _USAGE, "recording": _PROCESSING},
            {"status": "completed", "usage": _USAGE, "recording": _READY},
        ],
    )

    assert thread is not None
    assert thread["recording"] == _READY
    assert reads == 2


def test_read_thread_gives_up_on_a_recording_that_never_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread, _reads, waited = _read(
        monkeypatch, [{"status": "completed", "usage": _USAGE, "recording": _PROCESSING}]
    )

    assert thread is not None
    assert thread["recording"]["status"] == "processing"
    assert deployer_module._RECORDING_GRACE_S <= waited < deployer_module._RECORDING_GRACE_S + 10


def test_read_thread_without_a_recording_returns_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    thread, reads, waited = _read(
        monkeypatch, [{"status": "completed", "usage": _USAGE, "recording": None}]
    )

    assert thread is not None
    assert reads == 1
    assert waited == 0
