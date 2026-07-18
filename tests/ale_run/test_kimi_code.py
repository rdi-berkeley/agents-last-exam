from __future__ import annotations

import json
import base64
from pathlib import Path
from types import SimpleNamespace

from ale_run.agents.kimi_code.config import KimiCodeConfig
from ale_run.agents.kimi_code.deployer import (
    KimiCodeDeployer,
    _expected_version,
)
from ale_run.base_interface import AgentRunResult, TrajectoryBuilder
from ale_run.orchestration.factory import AGENT_REGISTRY


def _builder() -> TrajectoryBuilder:
    return TrajectoryBuilder(
        agent_name="kimi-code",
        model="kimi-k3",
        task_path="demo/seecheck",
        variant_index=0,
    )


def test_kimi_code_is_registered() -> None:
    assert AGENT_REGISTRY["kimi_code"] is KimiCodeDeployer


def test_expected_version_parses_scoped_npm_spec() -> None:
    assert _expected_version("@moonshot-ai/kimi-code@0.27.0") == "0.27.0"
    assert _expected_version("@moonshot-ai/kimi-code") is None


def test_build_env_uses_ephemeral_model_overlay(tmp_path: Path) -> None:
    executor = SimpleNamespace(
        config=KimiCodeConfig(),
        env={"MOONSHOT_API_KEY": "test-key"},
    )
    deployer = KimiCodeDeployer(executor)

    env = deployer._build_env(executor.config, work_dir=tmp_path)

    assert env["KIMI_CODE_HOME"] == str(tmp_path / "kimi-home")
    assert env["KIMI_MODEL_NAME"] == "kimi-k3"
    assert env["KIMI_MODEL_API_KEY"] == "test-key"
    assert env["KIMI_MODEL_CAPABILITIES"] == "image_in,thinking"
    assert env["KIMI_DISABLE_TELEMETRY"] == "1"
    assert env["KIMI_CODE_NO_AUTO_UPDATE"] == "1"


def test_parse_wire_preserves_tools_images_usage_and_latency(tmp_path: Path) -> None:
    wire = (
        tmp_path
        / "kimi-home"
        / "sessions"
        / "workspace"
        / "session"
        / "agents"
        / "main"
        / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    records = [
        {
            "type": "llm.request",
            "provider": "kimi",
            "model": "kimi-k3",
            "turnStep": "0.1",
        },
        {
            "type": "context.append_loop_event",
            "event": {"type": "step.begin", "uuid": "step-1"},
        },
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "content.part",
                "part": {"type": "think", "think": "Inspect the screen."},
            },
        },
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "tool.call",
                "toolCallId": "call-1",
                "name": "mcp__cua__screenshot",
                "args": {},
            },
        },
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "tool.result",
                "toolCallId": "call-1",
                "result": {
                    "output": [
                        {"type": "text", "text": "Screenshot captured."},
                        {
                            "type": "image_url",
                            "imageUrl": {"url": "data:image/png;base64,aW1hZ2U="},
                        },
                    ]
                },
            },
        },
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "content.part",
                "part": {"type": "text", "text": "The desktop is visible."},
            },
        },
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "step.end",
                "usage": {
                    "inputOther": 100,
                    "inputCacheRead": 200,
                    "inputCacheCreation": 30,
                    "output": 40,
                },
                "llmFirstTokenLatencyMs": 500,
                "llmStreamDurationMs": 250,
                "messageId": "message-1",
            },
        },
    ]
    wire.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    builder = _builder()

    KimiCodeDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=KimiCodeConfig(),
        run_result=AgentRunResult(status="completed", exit_code=0),
        builder=builder,
    )

    agent_step, environment_step = builder.trajectory.steps
    assert agent_step.reasoning == "Inspect the screen."
    assert agent_step.message == "The desktop is visible."
    assert agent_step.tool_calls[0].name == "mcp__cua__screenshot"
    assert agent_step.metrics.input_tokens == 100
    assert agent_step.metrics.cache_read_tokens == 200
    assert agent_step.metrics.cache_creation_tokens == 30
    assert agent_step.metrics.output_tokens == 40
    assert agent_step.metrics.duration_ms == 750
    image = environment_step.observation.results[0].content[1].image
    assert image.type == "base64"
    assert image.media_type == "image/png"
    assert image.data == "aW1hZ2U="
    assert builder.trajectory.extra["kimi_code"]["llm_requests"][0]["model"] == "kimi-k3"


def test_parse_wire_rehydrates_kimi_blobref_images(tmp_path: Path) -> None:
    agent_dir = tmp_path / "kimi-home" / "sessions" / "workspace" / "session" / "agents" / "main"
    wire = agent_dir / "wire.jsonl"
    blobs = agent_dir / "blobs"
    blobs.mkdir(parents=True)
    image_bytes = b"png-bytes"
    (blobs / "image-hash").write_bytes(image_bytes)
    records = [
        {
            "type": "context.append_loop_event",
            "event": {"type": "step.begin", "uuid": "step-1"},
        },
        {
            "type": "context.append_loop_event",
            "event": {
                "type": "tool.result",
                "toolCallId": "call-1",
                "result": {
                    "output": [
                        {
                            "type": "image_url",
                            "imageUrl": {"url": "blobref:image/png;image-hash"},
                        }
                    ]
                },
            },
        },
        {
            "type": "context.append_loop_event",
            "event": {"type": "step.end", "usage": {}},
        },
    ]
    wire.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    builder = _builder()

    KimiCodeDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=KimiCodeConfig(),
        run_result=AgentRunResult(status="completed", exit_code=0),
        builder=builder,
    )

    image = builder.trajectory.steps[-1].observation.results[0].content[0].image
    assert image.type == "base64"
    assert image.media_type == "image/png"
    assert image.data == base64.b64encode(image_bytes).decode("ascii")


def test_parse_transcript_fallback(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "Bash",
                                    "arguments": '{"command":"pwd"}',
                                },
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "content": "/tmp/work\n",
                    }
                ),
                json.dumps({"role": "assistant", "content": "/tmp/work"}),
            ]
        ),
        encoding="utf-8",
    )
    builder = _builder()

    KimiCodeDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=KimiCodeConfig(),
        run_result=AgentRunResult(status="completed", exit_code=0),
        builder=builder,
    )

    assert builder.trajectory.steps[0].tool_calls[0].arguments == {"command": "pwd"}
    assert builder.trajectory.steps[1].observation.results[0].content[0].text == "/tmp/work\n"
    assert builder.trajectory.steps[2].message == "/tmp/work"
