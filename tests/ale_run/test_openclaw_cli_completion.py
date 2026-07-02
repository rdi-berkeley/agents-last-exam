from __future__ import annotations

import asyncio
import base64
import json
from http.client import HTTPConnection
from types import SimpleNamespace

from ale_run.agents.openclaw_cli import OpenClawCliConfig, OpenClawCliDeployer
from ale_run.agents.openclaw_cli import deployer as deployer_module
from ale_run.agents.openclaw_cli import vision as vision_module
from ale_run.base_interface.agent_deployer import AgentRunResult
from ale_run.base_interface.trajectory import StepMetrics, TrajectoryBuilder


class _LingeringOpenClawProcess:
    pid = 4242
    returncode = None

    def __init__(self, *, stderr, envelope: dict) -> None:
        stderr.write(json.dumps(envelope).encode())
        stderr.flush()

    def poll(self) -> None:
        return None


def test_launch_finishes_when_result_envelope_is_complete(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(deployer_module, "_POLL_INTERVAL_S", 0)

    config = OpenClawCliConfig()
    executor = SimpleNamespace(config=config, work_dir=str(tmp_path), env={})
    deployer = OpenClawCliDeployer(executor)
    monkeypatch.setattr(deployer, "_complete_workspace_bootstrap", lambda: None)
    monkeypatch.setattr(deployer, "_launch_prefix", lambda: ["openclaw"])
    monkeypatch.setattr(deployer, "_build_env", lambda *_: {})

    envelope = {
        "payloads": [{"text": "done", "mediaUrl": None}],
        "meta": {
            "durationMs": 1234,
            "agentMeta": {"sessionId": "session-1"},
            "stopReason": "stop",
        },
    }
    process = None

    def fake_popen(*_, stderr, **__) -> _LingeringOpenClawProcess:
        nonlocal process
        process = _LingeringOpenClawProcess(
            stderr=stderr,
            envelope=envelope,
        )
        return process

    monkeypatch.setattr(deployer_module.subprocess, "Popen", fake_popen)

    result = asyncio.run(deployer.launch("finish the task"))

    assert process is not None
    assert process.poll() is None
    assert result.status == "completed"
    assert result.exit_code is None


def test_parse_stderr_json_accepts_diagnostics_after_envelope() -> None:
    stderr = "\n".join(
        [
            "[agent/embedded] run complete",
            json.dumps(
                {
                    "payloads": [],
                    "meta": {"durationMs": 25},
                },
                indent=2,
            ),
            "[plugin] lingering background handle",
        ]
    )

    assert deployer_module._parse_stderr_json(stderr) == {
        "payloads": [],
        "meta": {"durationMs": 25},
    }


def test_envelope_usage_overrides_partial_transcript_totals(tmp_path) -> None:
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {
                        "input": 10,
                        "output": 5,
                        "cacheRead": 20,
                        "cacheWrite": 2,
                    },
                },
            }
        )
    )
    (tmp_path / "stderr.log").write_text(
        json.dumps(
            {
                "payloads": [{"text": "done"}],
                "meta": {
                    "durationMs": 25,
                    "agentMeta": {
                        "usage": {
                            "input": 100,
                            "output": 50,
                            "cacheRead": 200,
                            "cacheWrite": 20,
                        }
                    },
                },
            }
        )
    )
    builder = TrajectoryBuilder(
        agent_name="openclaw-cli",
        model="test-model",
        task_path="demo/seecheck",
        variant_index=0,
    )

    OpenClawCliDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=OpenClawCliConfig(),
        run_result=AgentRunResult(
            status="completed",
            pid=1,
            exit_code=0,
            transcript_path=str(tmp_path / "transcript.jsonl"),
            stderr_path=str(tmp_path / "stderr.log"),
            duration_s=1,
        ),
        builder=builder,
    )
    final_metrics = builder.finalize(reward=1).final_metrics

    assert final_metrics is not None
    assert final_metrics.total_input_tokens == 100
    assert final_metrics.total_output_tokens == 50
    assert final_metrics.total_cache_read_tokens == 200
    assert final_metrics.total_cache_creation_tokens == 20


def test_empty_envelope_usage_does_not_erase_step_totals(tmp_path) -> None:
    (tmp_path / "stderr.log").write_text(
        json.dumps(
            {
                "payloads": [],
                "meta": {
                    "durationMs": 25,
                    "agentMeta": {
                        "usage": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        }
                    },
                },
            }
        )
    )
    builder = TrajectoryBuilder(
        agent_name="openclaw-cli",
        model="test-model",
        task_path="demo/seecheck",
        variant_index=0,
    )
    builder.add_step(
        source="agent",
        metrics=StepMetrics(input_tokens=10, output_tokens=5),
    )

    OpenClawCliDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=OpenClawCliConfig(),
        run_result=AgentRunResult(
            status="completed",
            pid=1,
            exit_code=0,
            transcript_path=str(tmp_path / "transcript.jsonl"),
            stderr_path=str(tmp_path / "stderr.log"),
            duration_s=1,
        ),
        builder=builder,
    )
    final_metrics = builder.finalize(reward=1).final_metrics

    assert final_metrics is not None
    assert final_metrics.total_input_tokens == 10
    assert final_metrics.total_output_tokens == 5


def test_extract_provider_usage_supports_responses_json_and_sse() -> None:
    responses_usage = vision_module.extract_provider_usage(
        json.dumps(
            {
                "usage": {
                    "input_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 80},
                    "output_tokens": 17,
                }
            }
        ).encode()
    )
    assert responses_usage == {
        "requests_with_usage": 1,
        "input_tokens": 120,
        "input_write_tokens": 40,
        "cache_read_tokens": 80,
        "cache_creation_tokens": 0,
        "output_tokens": 17,
    }

    streaming_usage = vision_module.extract_provider_usage(
        b'data: {"choices":[],"usage":{"prompt_tokens":31,'
        b'"prompt_tokens_details":{"cached_tokens":11},'
        b'"completion_tokens":7}}\n\ndata: [DONE]\n'
    )
    assert streaming_usage == {
        "requests_with_usage": 1,
        "input_tokens": 31,
        "input_write_tokens": 20,
        "cache_read_tokens": 11,
        "cache_creation_tokens": 0,
        "output_tokens": 7,
    }


def test_vision_usage_proxy_forwards_response_and_records_usage(
    tmp_path,
    monkeypatch,
) -> None:
    response_body = json.dumps(
        {
            "id": "response-1",
            "usage": {
                "input_tokens": 45,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens": 6,
            },
        }
    ).encode()
    requests = []

    class FakeResponse:
        status = 200
        reason = "OK"

        def read(self) -> bytes:
            return response_body

        def getheaders(self) -> list[tuple[str, str]]:
            return [("Content-Type", "application/json")]

    class FakeHttpsConnection:
        def __init__(self, host, port, timeout) -> None:
            assert host == "api.openai.com"
            assert port == 443
            assert timeout == 120

        def request(self, method, path, body, headers) -> None:
            requests.append((method, path, body, headers))

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            return

    monkeypatch.setattr(
        vision_module.http.client,
        "HTTPSConnection",
        FakeHttpsConnection,
    )
    usage_log = tmp_path / "vision-usage.jsonl"
    proxy = vision_module.VisionUsageProxy(
        upstream_url="https://api.openai.com/v1",
        usage_log=usage_log,
        provider="openai",
        model="gpt-5.4",
    )
    proxy.start()
    try:
        connection = HTTPConnection(
            "127.0.0.1",
            int(proxy.base_url.rsplit(":", 1)[1].split("/", 1)[0]),
            timeout=5,
        )
        connection.request(
            "POST",
            "/v1/responses",
            body=b'{"model":"gpt-5.4"}',
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        assert response.read() == response_body
        connection.close()
    finally:
        proxy.stop()

    assert requests[0][0:3] == (
        "POST",
        "/v1/responses",
        b'{"model":"gpt-5.4"}',
    )
    assert requests[0][3]["Authorization"] == "Bearer test-key"
    assert requests[0][3]["Accept-Encoding"] == "identity"
    assert json.loads(usage_log.read_text()) == {
        "provider": "openai",
        "model": "gpt-5.4",
        "requests_with_usage": 1,
        "input_tokens": 45,
        "input_write_tokens": 40,
        "cache_read_tokens": 5,
        "cache_creation_tokens": 0,
        "output_tokens": 6,
    }


def test_parse_keeps_image_usage_in_openclaw_metadata_only(tmp_path) -> None:
    run_dir = tmp_path / "run"
    work_dir = run_dir / "origin_log" / "openclaw-cli"
    work_dir.mkdir(parents=True)
    image_bytes = b"\x89PNG\r\n\x1a\nsmall-test-image"
    encoded = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/png;base64,{encoded}"
    events = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "image-call",
                        "name": "image",
                        "arguments": {"image": data_url},
                    }
                ],
                "usage": {
                    "input": 100,
                    "output": 20,
                    "cacheRead": 40,
                    "cacheWrite": 3,
                },
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "image-call",
                "content": [
                    {"type": "text", "text": "visible"},
                    {
                        "type": "image",
                        "mimeType": "image/png",
                        "data": encoded,
                    },
                ],
            },
        },
    ]
    (work_dir / "transcript.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    (work_dir / "vision-usage.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "openai",
                        "model": "gpt-5.4",
                        "requests_with_usage": 1,
                        "input_tokens": 60,
                        "input_write_tokens": 50,
                        "cache_read_tokens": 10,
                        "cache_creation_tokens": 0,
                        "output_tokens": 5,
                    }
                ),
                json.dumps(
                    {
                        "provider": "openai",
                        "model": "gpt-5.4",
                        "requests_with_usage": 1,
                        "input_tokens": 40,
                        "input_write_tokens": 40,
                        "cache_read_tokens": 0,
                        "cache_creation_tokens": 0,
                        "output_tokens": 4,
                    }
                ),
            ]
        )
        + "\n"
    )
    builder = TrajectoryBuilder(
        agent_name="openclaw-cli",
        model="primary-model",
        task_path="demo/seecheck",
        variant_index=0,
    )

    OpenClawCliDeployer.parse_artifacts(
        work_dir=work_dir,
        config=OpenClawCliConfig(),
        run_result=AgentRunResult(
            status="completed",
            pid=1,
            exit_code=0,
            transcript_path=str(work_dir / "transcript.jsonl"),
            stderr_path=str(work_dir / "stderr.log"),
            duration_s=1,
        ),
        builder=builder,
    )
    trajectory = builder.finalize(reward=1)

    assert trajectory.extra["openclaw_cli"]["image_model_usage"] == {
        "provider": "openai",
        "model": "gpt-5.4",
        "requests_with_usage": 2,
        "input_tokens": 100,
        "input_write_tokens": 90,
        "cache_read_tokens": 10,
        "cache_creation_tokens": 0,
        "output_tokens": 9,
    }
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_input_tokens == 100
    assert trajectory.final_metrics.total_output_tokens == 20
    assert "model_usage" not in trajectory.final_metrics.model_dump()

    raw_transcript = (work_dir / "transcript.jsonl").read_text()
    assert "base64," not in raw_transcript
    assert encoded not in raw_transcript
    screenshots = list((run_dir / "screenshots").iterdir())
    assert len(screenshots) == 1
    assert screenshots[0].read_bytes() == image_bytes
    assert trajectory.steps[0].tool_calls[0].arguments["image"].startswith(
        "screenshots/"
    )
    image_part = trajectory.steps[1].observation.results[0].content[1]
    assert image_part.image is not None
    assert image_part.image.type == "path"
    assert image_part.image.path == trajectory.steps[0].tool_calls[0].arguments[
        "image"
    ]


def test_stage_transcript_file_images_for_gather(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / ".openclaw" / "workspace"
    workspace.mkdir(parents=True)
    source = workspace / "screen.png"
    source.write_bytes(b"image-file")
    work_dir = tmp_path / "artifacts"
    work_dir.mkdir()
    transcript = work_dir / "transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "image-call",
                            "name": "image",
                            "arguments": {"image": "screen.png"},
                        }
                    ],
                },
            }
        )
        + "\n"
    )

    vision_module.stage_transcript_file_images(transcript, work_dir)

    event = json.loads(transcript.read_text())
    relative = event["message"]["content"][0]["arguments"]["image"]
    assert relative.startswith("screenshots/openclaw-input-")
    assert (work_dir / relative).read_bytes() == b"image-file"
