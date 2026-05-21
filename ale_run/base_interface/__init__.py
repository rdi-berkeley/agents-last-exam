"""``base_interface`` — every settled contract the framework agrees on.

This package has **zero internal dependencies**: it imports only stdlib,
pydantic, and openenv. Every other top-level layer (``agents/``,
``environments/``, ``tasks/``, ``orchastration/``) depends on it; nothing
in those layers cross-references each other's internals.

Contents:

* :class:`BaseAgentDeployer`, :class:`BaseAgentConfig`,
  :class:`AgentRunResult`, :class:`EpisodeResult` — agent contract.
* :class:`BaseRuntime` — substrate adapter contract.
* :class:`Provider`, :class:`EnvSpec`, :class:`VMHandle`,
  :data:`ReleaseMode` — VM provisioning contract.
* :class:`TaskDataSpec` — task data-staging contract.
* :class:`Trajectory`, :class:`TrajectoryBuilder`, :class:`Step`,
  :class:`ToolCall`, :class:`ToolResult`, :class:`Observation`,
  :class:`StepMetrics`, :class:`FinalMetrics`, :class:`AgentInfo`,
  :class:`ContentPart`, :class:`ImageSource` — ATIF trajectory format.
* :class:`RemoteVMConfig`, :class:`RangeResult` — cua-server addressing
  + range-download result shape.

The only intra-package coupling is the ``BaseAgentDeployer`` ↔
``BaseRuntime`` reference, which both files manage via TYPE_CHECKING.
Outside ``base_interface/`` the cycle is invisible.
"""
from __future__ import annotations

from .deployer import (
    AgentRunResult,
    BaseAgentConfig,
    BaseAgentDeployer,
    EpisodeResult,
)
from .provider import (
    EnvSpec,
    OS,
    Provider,
    ReleaseMode,
    VMHandle,
)
from .remote_vm import RangeResult, RemoteVMConfig
from .runtime import BaseRuntime
from .task import TaskDataSpec
from .trajectory import (
    AgentInfo,
    ContentPart,
    FinalMetrics,
    ImageSource,
    Observation,
    SCHEMA_VERSION,
    Source,
    Step,
    StepMetrics,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryBuilder,
)

__all__ = [
    # deployer.py
    "AgentRunResult",
    "BaseAgentConfig",
    "BaseAgentDeployer",
    "EpisodeResult",
    # provider.py
    "EnvSpec",
    "OS",
    "Provider",
    "ReleaseMode",
    "VMHandle",
    # remote_vm.py
    "RangeResult",
    "RemoteVMConfig",
    # runtime.py
    "BaseRuntime",
    # task.py
    "TaskDataSpec",
    # trajectory.py
    "AgentInfo",
    "ContentPart",
    "FinalMetrics",
    "ImageSource",
    "Observation",
    "SCHEMA_VERSION",
    "Source",
    "Step",
    "StepMetrics",
    "ToolCall",
    "ToolResult",
    "Trajectory",
    "TrajectoryBuilder",
]
