import pytest
from agent.decorators import find_agent_config
from ale_claw import AleClawConfig
from ale_claw.harness.model.tool_loop import ToolCallingComputerAgent
from ale_claw.harness.subagent.subagent_gui import _GUIAgent


@pytest.mark.parametrize(
    "model", ["openai/gpt-4.1", "openai/example-model", "openrouter/vendor/model"]
)
@pytest.mark.parametrize("agent_cls", [ToolCallingComputerAgent, _GUIAgent])
def test_explicit_protocol_is_per_instance(model, agent_cls):
    automatic = find_agent_config(model)
    for mode, expected in [
        ("function_calling", "UnifiedAgentConfig"),
        ("prompt", "GenericVlmConfig"),
    ]:
        info = AleClawConfig(model=model, tool_calling=mode).tool_loop_override
        agent = agent_cls(model=model, tool_loop=info, telemetry_enabled=False)
        assert agent.model == model
        assert agent.agent_config_info is info
        assert type(agent.agent_loop).__name__ == expected
        assert "step" in agent.get_capabilities()
        assert "tool_calling" not in agent.kwargs
    assert AleClawConfig(model=model).tool_loop_override is None
    assert agent_cls(model=model, telemetry_enabled=False).agent_config_info is automatic


def test_invalid_tool_calling_fails_before_session():
    with pytest.raises(ValueError, match="tool_calling"):
        AleClawConfig(tool_calling="unknown")
