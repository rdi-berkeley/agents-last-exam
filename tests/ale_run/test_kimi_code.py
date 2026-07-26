from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

from ale_run.agents.kimi_code.config import KimiCodeConfig
from ale_run.agents.kimi_code.deployer import KimiCodeDeployer, _expected_version
from ale_run.base_interface import AgentRunResult, TrajectoryBuilder
from ale_run.orchestration.factory import AGENT_REGISTRY


def _builder() -> TrajectoryBuilder:
    return TrajectoryBuilder(
        agent_name="kimi-code",
        model="kimi-k3",
        task_path="demo/seecheck",
        variant_index=0,
    )


def _parse(tmp_path: Path) -> TrajectoryBuilder:
    builder = _builder()
    KimiCodeDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=KimiCodeConfig(),
        run_result=AgentRunResult(status="completed", exit_code=0),
        builder=builder,
    )
    return builder


def test_kimi_code_is_registered() -> None:
    assert AGENT_REGISTRY["kimi_code"] is KimiCodeDeployer


def test_config_matches_tested_k3_settings() -> None:
    config = KimiCodeConfig()

    assert config.model == "kimi-k3"
    assert config.max_context_size == 1_048_576
    assert config.thinking_effort == "max"
    assert config.otel_enabled is True
    assert config.cli_version == "@moonshot-ai/kimi-code@0.27.0"


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
    assert env["KIMI_MODEL_MAX_CONTEXT_SIZE"] == "1048576"
    assert env["KIMI_MODEL_CAPABILITIES"] == "image_in,thinking"
    assert env["KIMI_MODEL_THINKING_EFFORT"] == "max"
    assert env["KIMI_MODEL_OUTPUT_FORMAT"] == "stream-json"
    assert env["KIMI_DISABLE_TELEMETRY"] == "1"
    assert env["KIMI_CODE_NO_AUTO_UPDATE"] == "1"


def test_build_env_injects_run_local_otel(tmp_path: Path) -> None:
    executor = SimpleNamespace(
        config=KimiCodeConfig(),
        env={"MOONSHOT_API_KEY": "test-key", "NODE_OPTIONS": "--trace-warnings"},
    )
    deployer = KimiCodeDeployer(executor)
    bootstrap = tmp_path / "node_modules" / "ale-kimi-otel" / "bootstrap.mjs"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text("", encoding="utf-8")

    env = deployer._build_env(
        executor.config,
        work_dir=tmp_path,
        otel_endpoint="http://127.0.0.1:4318",
        otel_bootstrap=bootstrap,
    )

    import_option = f"--import={bootstrap.resolve().as_uri()}"
    assert env["ALE_KIMI_OTEL_ENDPOINT"] == "http://127.0.0.1:4318"
    assert env["ALE_KIMI_OTEL_IMPORT_OPTION"] == import_option
    assert env["NODE_OPTIONS"] == f"--trace-warnings {import_option}"


def test_raw_telemetry_wal_is_incrementally_mirrored() -> None:
    assert "otel_requests.jsonl" in KimiCodeDeployer.hot_artifacts
    assert "telemetry.jsonl" not in KimiCodeDeployer.hot_artifacts


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
            },
        },
    ]
    wire.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    builder = _parse(tmp_path)

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


def test_parse_wire_rehydrates_blobref_image(tmp_path: Path) -> None:
    agent_dir = tmp_path / "kimi-home" / "sessions" / "workspace" / "session" / "agents" / "main"
    blobs = agent_dir / "blobs"
    blobs.mkdir(parents=True)
    image_bytes = b"png-bytes"
    (blobs / "image-hash").write_bytes(image_bytes)
    (agent_dir / "wire.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "type": "context.append_loop_event",
                    "event": {"type": "step.begin"},
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
        )
        + "\n",
        encoding="utf-8",
    )

    image = _parse(tmp_path).trajectory.steps[-1].observation.results[0].content[0].image

    assert image.type == "base64"
    assert image.data == base64.b64encode(image_bytes).decode("ascii")


def test_missing_blob_uses_json_string_transcript_image(tmp_path: Path) -> None:
    agent_dir = tmp_path / "kimi-home" / "sessions" / "workspace" / "session" / "agents" / "main"
    agent_dir.mkdir(parents=True)
    (agent_dir / "wire.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "type": "context.append_loop_event",
                    "event": {"type": "step.begin"},
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
                                    "imageUrl": {"url": "blobref:image/png;already-deleted"},
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
        )
        + "\n",
        encoding="utf-8",
    )
    inline_image = base64.b64encode(b"transcript-image").decode("ascii")
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps(
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": json.dumps(
                    [
                        {
                            "type": "image_url",
                            "imageUrl": {"url": f"data:image/png;base64,{inline_image}"},
                        }
                    ]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    image = _parse(tmp_path).trajectory.steps[-1].observation.results[0].content[0].image

    assert image.type == "base64"
    assert image.data == inline_image


def test_transcript_only_json_string_image_is_preserved(tmp_path: Path) -> None:
    inline_image = base64.b64encode(b"transcript-only-image").decode("ascii")
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps(
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": json.dumps(
                    [
                        {"type": "text", "text": "Screenshot captured."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{inline_image}"},
                        },
                    ]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    content = _parse(tmp_path).trajectory.steps[0].observation.results[0].content

    assert content[0].text == "Screenshot captured."
    assert content[1].image.type == "base64"
    assert content[1].image.data == inline_image


def test_transcript_blobref_without_blob_directory_is_not_persisted(
    tmp_path: Path,
) -> None:
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps(
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": json.dumps(
                    [
                        {
                            "type": "image_url",
                            "imageUrl": {"url": "blobref:image/png;already-deleted"},
                        }
                    ]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    content = _parse(tmp_path).trajectory.steps[0].observation.results[0].content

    assert content == []
