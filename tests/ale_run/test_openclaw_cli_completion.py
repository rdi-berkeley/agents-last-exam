from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ale_run.agents.openclaw_cli import OpenClawCliConfig, OpenClawCliDeployer
from ale_run.agents.openclaw_cli import deployer as deployer_module
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
