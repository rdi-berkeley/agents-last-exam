"""Per-instance selection of the CUA tool protocol."""

from agent import ComputerAgent
from agent.types import AgentConfigInfo


class ToolCallingComputerAgent(ComputerAgent):
    def __init__(self, *, tool_loop: AgentConfigInfo | None = None, **kwargs):
        if tool_loop is not None:
            kwargs["custom_loop"] = tool_loop.agent_class()
        super().__init__(**kwargs)
        if tool_loop is not None:
            # CUA leaves this empty for custom loops, but run() needs its tool metadata.
            self.agent_config_info = tool_loop
