"""ALE-Claw computer-use agent."""

from typing import TYPE_CHECKING

from .config import AleClawConfig

if TYPE_CHECKING:
    from .runner import RunResult, run

__all__ = ["AleClawConfig", "RunResult", "run"]


def __getattr__(name: str):
    # The CLI routes SDK output to stderr before importing the runtime.
    if name in ("RunResult", "run"):
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
