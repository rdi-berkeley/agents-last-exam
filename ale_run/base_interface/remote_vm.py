"""Remote-VM addressing + range-download result shape.

Pure data shapes that travel through everywhere talking to the cua-server.
The HTTP primitives that consume these (``run_remote``, ``upload_file``,
``download_file_range``, ...) live in :mod:`ale_run.environments.remote`
— this module is just the contract.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RemoteVMConfig:
    """How to address a cua-server VM (URL + OS) plus optional run/task ids.

    The HTTP helpers in :mod:`ale_run.environments.remote` all take one
    of these. Identity fields (``run_id`` / ``task_id``) are best-effort
    and only flow through to logs / structured upload paths — they're
    not part of the protocol.
    """

    server_url: str
    os_type: str = "windows"
    run_id: str | None = None
    task_id: str | None = None

    @property
    def is_linux(self) -> bool:
        return self.os_type == "linux"


@dataclass
class RangeResult:
    """Outcome of an incremental file fetch (see ``download_file_range``).

    ``remote_size = -1`` means the remote file was missing at probe time
    (the helper still returns ``success=True`` with an empty delta so
    the puller can record the absence and try again next tick).
    """

    success: bool
    remote_size: int = 0
    delta: bytes = b""
    error: str | None = None
