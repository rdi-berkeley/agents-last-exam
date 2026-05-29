"""``base_interface`` — every settled contract the framework agrees on.

This package has **zero internal dependencies**: it imports only stdlib,
pydantic, openenv, and requests. Every other top-level layer
(``agents/``, ``environments/``, ``tasks/``, ``orchestration/``,
``executors/``) depends on it; nothing in those layers cross-references
each other's internals.

Contents:

* :class:`BaseAgentDeployer`, :class:`BaseAgentConfig`,
  :class:`AgentRunResult`, :class:`EpisodeResult` — agent contract.
* :class:`BaseExecutor` — executor (substrate adapter) contract.
* :class:`SandboxHandle`, :class:`SandboxSpec`, :class:`Provider`,
  :data:`OS`, :data:`ReleaseMode`, :class:`RangeResult`,
  :class:`SandboxUnreachableError` — sandbox (the cua-server target)
  data + API + provisioning contract.
* :class:`TaskDataSpec` — task data-staging contract.
* :class:`Trajectory`, :class:`TrajectoryBuilder`, :class:`Step`,
  :class:`ToolCall`, :class:`ToolResult`, :class:`Observation`,
  :class:`StepMetrics`, :class:`FinalMetrics`, :class:`AgentInfo`,
  :class:`ContentPart`, :class:`ImageSource` — ATIF trajectory format.

The only intra-package coupling is the ``BaseAgentDeployer`` ↔
``BaseExecutor`` reference, which both files manage via TYPE_CHECKING.
"""
from __future__ import annotations

from .agent_deployer import (
    AgentRunResult,
    BaseAgentConfig,
    BaseAgentDeployer,
    EpisodeResult,
)
from .executor import BaseExecutor, GatherReport
from .sandbox import (
    OS,
    Provider,
    RangeResult,
    ReleaseMode,
    SandboxHandle,
    SandboxSpec,
    SandboxUnreachableError,
)
from .task_data import TaskDataSpec
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
    # agent_deployer.py
    "AgentRunResult",
    "BaseAgentConfig",
    "BaseAgentDeployer",
    "EpisodeResult",
    # executor.py
    "BaseExecutor",
    "GatherReport",
    # sandbox.py
    "OS",
    "Provider",
    "RangeResult",
    "ReleaseMode",
    "SandboxHandle",
    "SandboxSpec",
    "SandboxUnreachableError",
    # task_data.py
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
