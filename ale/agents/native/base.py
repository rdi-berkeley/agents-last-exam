"""NativeAgent ABC — orchestrator-side step loop.

Subclasses implement ``act(obs, state) -> list[Action]``: one LLM turn,
zero or more actions. The framework owns the loop, max_turns enforcement,
trajectory population, and final Submit. Subclasses focus only on the
LLM-call ↔ Action translation.

This module **defines the contract only** — no concrete LLM agent ships
here yet. A concrete implementation (e.g. wrapping cua-bench's
``ComputerAgent``) lives in a sibling subpackage and is the topic of a
later slice.
"""
from __future__ import annotations

import abc
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ale.agents.base import BaseAgent, BaseAgentConfig, EpisodeResult
from ale.agents.trajectory import (
    Observation as TrajObservation,
    Step,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)
from ale.core.env import AgenthleEnv
from ale.core.types import (
    AgenthleObservation,
    ReadFile,
    RunCommand,
    Screenshot,
    Submit,
    WriteFile,
)
from openenv.core.env_server.types import Action


# =============================================================================
# Config + State
# =============================================================================

@dataclass
class NativeAgentConfig(BaseAgentConfig):
    """Tunables for any orchestrator-side agent.

    Subclasses (e.g. an LLM-specific agent) extend this with provider-specific
    fields (``temperature``, ``thinking_budget``, etc.).
    """

    name: ClassVar[str] = "native"

    provider: str = "anthropic"
    """LLM provider id: ``anthropic`` | ``openai`` | ``openrouter`` | ...
    Concrete subclass uses this to pick the SDK."""

    temperature: float | None = None
    max_tokens: int | None = None


@dataclass
class NativeAgentState:
    """Mutable per-episode state passed to every ``act()`` call.

    Most LLM agents need to carry:
    - the running conversation (``messages``)
    - which tool calls are still awaiting results (``pending_tool_results``)
    - cumulative token counters

    Subclasses may attach additional state via ``extra``.
    """

    step_count: int = 0                           # action steps taken (env.step calls)
    turn_count: int = 0                           # LLM turns (act() calls)
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_results: dict[str, Any] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base native agent
# =============================================================================

class NativeAgent(BaseAgent, abc.ABC):
    """Orchestrator-side step-loop agent.

    Subclass contract: implement :meth:`act`. The base class drives the loop,
    trajectory, and submit.

    Lifecycle::

        env.reset_async(...)                              # step_id 1: user/instruction
        while not done and turn < max_turns:
            actions = await self.act(obs, state)          # 1 LLM call
            if not actions: break                         # agent says done
            for action in actions:                        # may be parallel tool calls
                obs = await env.step_async(action)        # 1 env step
                trajectory.add_step(...)
                if obs.done: break
        if not obs.done:
            obs = await env.step_async(Submit())          # force evaluate

    Default ``max_turns`` (when config doesn't set one) is 100 turns of
    ``act()`` — a soft safety stop.
    """

    DEFAULT_MAX_TURNS: ClassVar[int] = 100

    def __init__(self, config: NativeAgentConfig):
        self._config = config

    @property
    def config(self) -> NativeAgentConfig:
        return self._config

    # ---- subclass contract ---------------------------------------------------

    @abc.abstractmethod
    async def act(
        self,
        obs: AgenthleObservation,
        state: NativeAgentState,
    ) -> list[Action]:
        """Produce the actions to take this turn.

        Returns:
            A list of actions. Parallel tool calls go in one list.
            Return ``[]`` if the agent decides it's done (framework will
            issue ``Submit`` automatically).

        Implementations should:
            - update ``state.messages`` with the new assistant turn
            - update ``state.total_*_tokens`` from the LLM response usage
            - **not** call ``env`` directly (framework does it)
        """

    def _initial_state(self) -> NativeAgentState:
        """Hook for subclasses to seed messages / extra state. Default: empty."""
        return NativeAgentState()

    @property
    def agent_version(self) -> str | None:
        """Subclasses can override to surface their own version string."""
        return None

    # ---- framework loop ------------------------------------------------------

    async def run(
        self,
        env: AgenthleEnv,
        *,
        variant_index: int = 0,
    ) -> EpisodeResult:
        max_turns = self._config.max_turns or self.DEFAULT_MAX_TURNS
        task_path = env.task_path
        builder = TrajectoryBuilder(
            episode_id=uuid.uuid4().hex,
            agent_name=self._config.name,
            agent_version=self.agent_version,
            model=self._config.model or None,
            task_path=task_path,
            variant_index=variant_index,
        )
        t0 = time.monotonic()
        instruction: str | None = None
        status = "completed"
        error: str | None = None
        final_reward: float | None = None

        try:
            obs = await env.reset_async(variant_index=variant_index)
            instruction = obs.instruction or ""
            builder._traj.instruction = instruction         # noqa: SLF001 (builder owns this)
            builder.add_step(source="user", message=instruction)

            state = self._initial_state()

            while True:
                # Termination: env said we're done in a prior iteration.
                if obs.done:
                    final_reward = obs.reward
                    break

                # Termination: turn budget exhausted.
                if state.turn_count >= max_turns:
                    status = "timeout"
                    error = f"max_turns={max_turns} exceeded"
                    builder.add_step(
                        source="system",
                        message=error,
                        extra={"reason": "max_turns_exceeded"},
                    )
                    obs = await env.step_async(Submit())
                    final_reward = obs.reward
                    break

                # One LLM turn.
                actions = await self.act(obs, state)
                state.turn_count += 1

                # Empty list = agent says it's done; force Submit to score.
                if not actions:
                    obs = await env.step_async(Submit())
                    final_reward = obs.reward
                    break

                # Otherwise execute each action; record both agent and env steps.
                self._record_agent_step(builder, state, actions)
                for action in actions:
                    obs = await env.step_async(action)
                    state.step_count += 1
                    self._record_env_step(builder, action, obs)
                    if obs.done:
                        final_reward = obs.reward
                        break

            traj = builder.finalize(reward=final_reward, status=status)
            return EpisodeResult(
                reward=final_reward,
                status=status,
                error=error,
                instruction=instruction,
                trajectory=traj,
                duration_s=time.monotonic() - t0,
                task_path=task_path,
                variant_index=variant_index,
            )

        except Exception as exc:
            traj = builder.finalize(
                reward=None, status="failed",
            )
            return EpisodeResult(
                reward=None,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                instruction=instruction,
                trajectory=traj,
                duration_s=time.monotonic() - t0,
                task_path=task_path,
                variant_index=variant_index,
            )

    # ---- trajectory helpers --------------------------------------------------

    @staticmethod
    def _record_agent_step(
        builder: TrajectoryBuilder,
        state: NativeAgentState,
        actions: list[Action],
    ) -> None:
        """Emit one ``agent``-source Step summarizing this LLM turn."""
        tool_calls = [
            ToolCall(name=type(a).__name__, arguments=_action_to_dict(a))
            for a in actions
        ]
        metrics = StepMetrics(
            input_tokens=state.total_input_tokens or None,
            output_tokens=state.total_output_tokens or None,
            cost_usd=state.total_cost_usd or None,
        )
        # Pull the latest assistant message off state.messages if present.
        message: str | None = None
        if state.messages and state.messages[-1].get("role") == "assistant":
            content = state.messages[-1].get("content")
            if isinstance(content, str):
                message = content
        builder.add_step(
            source="agent",
            message=message,
            tool_calls=tool_calls,
            metrics=metrics,
        )

    @staticmethod
    def _record_env_step(
        builder: TrajectoryBuilder,
        action: Action,
        obs: AgenthleObservation,
    ) -> None:
        """Emit one ``environment``-source Step for the env's response to ``action``."""
        # Build a tool result string from the parts of the observation that
        # actually got populated by env.step (sparse — only one or two fields).
        parts: list[str] = []
        if obs.stdout is not None:
            parts.append(f"stdout:\n{obs.stdout}")
        if obs.stderr:
            parts.append(f"stderr:\n{obs.stderr}")
        if obs.exit_code is not None:
            parts.append(f"exit_code: {obs.exit_code}")
        if obs.file_data is not None:
            parts.append(f"file_data: {len(obs.file_data)} bytes")
        if obs.reward is not None:
            parts.append(f"reward: {obs.reward}")
        message = "\n".join(parts) if parts else None

        builder.add_step(
            source="environment",
            message=message,
            observation=TrajObservation(),
            extra={"action_type": type(action).__name__},
        )


# =============================================================================
# Helpers
# =============================================================================

def _action_to_dict(action: Action) -> dict[str, Any]:
    """Drop heavy fields (e.g. WriteFile.data when bytes) for trajectory summary."""
    d = action.model_dump()
    # Strip the bytes payload of WriteFile from the trajectory summary.
    if isinstance(action, WriteFile) and isinstance(action.data, bytes):
        d["data"] = f"<{len(action.data)} bytes>"
    elif isinstance(action, ReadFile):
        pass  # path is fine
    elif isinstance(action, Screenshot):
        pass
    elif isinstance(action, RunCommand):
        pass
    elif isinstance(action, Submit):
        # payload may be large; keep keys only.
        if "payload" in d and isinstance(d["payload"], dict):
            d["payload_keys"] = list(d["payload"].keys())
            d.pop("payload", None)
    return d
