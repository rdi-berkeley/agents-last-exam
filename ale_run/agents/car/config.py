"""CarConfig: per-episode knobs for the Common Agent Runtime (CAR) deployer.

Standalone config (no shared base), per the ALE convention. Declares the model
and turn budget plus the few CAR-specific knobs the deployer consumes.

CAR is driven out-of-sandbox: the deployer launches the `car run-task` headless
runner on the host, hands it stdio MCP bridges that reach the eval VM's
cua-server, and CAR runs its own propose -> validate -> execute loop. The runner
emits a JSONL transcript the deployer converts to an ATIF trajectory.

**API keys live in the operator's shell env**, not in this config. CAR reads
``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``GOOGLE_API_KEY`` from the process
env (the deployer passes ``os.environ`` straight through to the runner).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class CarConfig:
    """Tunables for :class:`CarDeployer`."""

    name: ClassVar[str] = "car"

    model: str | None = None
    """CAR model id, resolved by CAR's own registry/router (e.g.
    ``claude-sonnet-4-6``). NOTE: this is CAR's catalog id, NOT the LiteLLM
    ``provider/model`` form the other ALE harnesses use. Required: ``install``
    raises if unset, since a benchmark run must pin its model. Passed verbatim to
    ``car run-task --model``."""

    max_turns: int = 100
    """Hard ceiling on CAR's agent loop, passed to ``--max-turns``."""

    car_bin: str = "car"
    """Path to the ``car`` CLI binary. Override if it is not on PATH."""

    gui: bool = True
    """Wire the cua (GUI) MCP bridge in addition to the vm (shell/fs) bridge, so
    CAR can drive the desktop. Set False for shell/file-only tasks."""

    eventlog: bool = True
    """Also write CAR's engine event journal alongside the transcript. The
    journal is metadata-only (action ids + durations); the transcript carries the
    content. Useful for debugging validator/policy decisions."""

    extra_env: dict[str, str] = field(default_factory=dict)
    """Extra environment variables to inject into the ``car run-task`` process
    (merged over ``os.environ``)."""
