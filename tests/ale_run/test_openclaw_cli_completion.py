from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ale_run.agents.openclaw_cli import OpenClawCliConfig, OpenClawCliDeployer
from ale_run.agents.openclaw_cli import deployer as deployer_module


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
