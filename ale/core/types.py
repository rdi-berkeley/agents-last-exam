"""Core types: Actions, AgenthleObservation, AgenthleState.

Subclass OpenEnv Pydantic bases. No Variant class — agenthle tasks
declare variants by returning a ``list[cb.Task]`` from their decorated
``load()``; we address them by integer index.
"""
from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation, State


# =============================================================================
# Actions
# =============================================================================

class Submit(Action):
    """Agent's final submission. Triggers the task's ``evaluate()``.

    ``payload`` is opaque to AgenthleEnv — the task code reads what it needs
    from the VM via the DesktopSession, not from this payload. The field is
    kept for future agents that want to attach structured submission data.
    """

    payload: dict[str, Any] = {}


class RunCommand(Action):
    """Execute a shell command in the VM (ad-hoc env.step pass-through)."""

    cmd: str | list[str]
    timeout: float | None = None


class ReadFile(Action):
    """Read a file from the VM."""

    path: str


class WriteFile(Action):
    """Write a file in the VM."""

    path: str
    data: bytes | str


class Screenshot(Action):
    """Capture a PNG screenshot of the VM's desktop."""


# =============================================================================
# Observation
# =============================================================================

class AgenthleObservation(Observation):
    """Sparse observation. Only fields relevant to the producing action populate.

    Inherits ``done`` / ``reward`` / ``metadata`` from OpenEnv. After a Submit,
    ``reward`` carries the score returned by the task's ``evaluate()``.
    """

    instruction: str | None = None

    # populated by RunCommand
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None

    # populated by ReadFile
    file_data: bytes | None = None

    # populated by Screenshot
    screenshot_png: bytes | None = None

    # populated by Submit — eval execution telemetry
    eval_status: str | None = None
    """``"success"`` (evaluate ran and returned) / ``"failed"`` (evaluate
    raised) / ``None`` for non-Submit observations. ``"not_executed"`` is
    set on the EpisodeResult when Submit never fired at all."""

    eval_duration_s: float | None = None
    """Wall time of the ``task.evaluate()`` call. None if it didn't run."""

    eval_error: dict[str, Any] | None = None
    """``{"type", "message", "traceback"}`` when ``eval_status == "failed"``."""


# =============================================================================
# State
# =============================================================================

class AgenthleState(State):
    """Per-session state."""

    task_path: str | None = None
    variant_index: int | None = None
    vm_id: str | None = None
