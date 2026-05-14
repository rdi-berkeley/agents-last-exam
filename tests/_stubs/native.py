"""StubNativeAgent: NativeAgent that emits a fixed action sequence.

Validates the NativeAgent framework loop without needing a real LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale.agents.native.base import NativeAgent, NativeAgentConfig, NativeAgentState
from ale.core.types import AgenthleObservation
from openenv.core.env_server.types import Action


@dataclass
class StubNativeAgentConfig(NativeAgentConfig):
    name: ClassVar[str] = "stub-native"
    model: str = "stub-llm"
    provider: str = "stub"


class StubNativeAgent(NativeAgent):
    """Replays a fixed list-of-lists of actions, one list per ``act()`` call.

    Example::

        agent = StubNativeAgent(
            actions_per_turn=[
                [WriteFile(path="...", data="hello")],   # turn 1
                [],                                      # turn 2 → framework Submits
            ],
        )
    """

    def __init__(
        self,
        *,
        actions_per_turn: list[list[Action]],
        config: StubNativeAgentConfig | None = None,
    ):
        super().__init__(config or StubNativeAgentConfig())
        self._script = list(actions_per_turn)
        self._cursor = 0

    @property
    def agent_version(self) -> str | None:
        return "stub-0.1"

    async def act(
        self,
        obs: AgenthleObservation,
        state: NativeAgentState,
    ) -> list[Action]:
        if self._cursor >= len(self._script):
            return []
        actions = self._script[self._cursor]
        self._cursor += 1
        return actions
