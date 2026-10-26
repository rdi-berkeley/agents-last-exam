from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ale_run.agents.claude_code import api_proxy
from ale_run.agents.claude_code.config import ClaudeCodeConfig
from ale_run.agents.claude_code.deployer import ClaudeCodeDeployer


def _env(tmp_path: Path, config: ClaudeCodeConfig) -> dict[str, str]:
    executor = SimpleNamespace(config=config, work_dir=str(tmp_path), env={})
    return ClaudeCodeDeployer(executor)._build_env(config)


def test_gateway_headers_and_extra_env(tmp_path: Path) -> None:
    config = ClaudeCodeConfig(
        provider="openrouter", base_url="https://gateway.example", api_key="not-needed",
        extra_headers={"CF-Access-Client-Id": "id", "CF-Access-Client-Secret": "secret"},
        session_header="X-Session-Id",
        extra_env={"ANTHROPIC_SMALL_FAST_MODEL": "served/model", "DISABLE_TELEMETRY": "1"},
    )
    env = _env(tmp_path, config)
    lines = env["ANTHROPIC_CUSTOM_HEADERS"].split("\n")
    assert lines[:2] == ["CF-Access-Client-Id: id", "CF-Access-Client-Secret: secret"]
    assert lines[2].startswith("X-Session-Id: ale-")
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "served/model"
    assert env["DISABLE_TELEMETRY"] == "1"


def test_no_gateway_knobs_means_no_custom_headers(tmp_path: Path) -> None:
    env = _env(tmp_path, ClaudeCodeConfig(provider="direct", api_key="test"))
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env


def test_session_id_is_stable_per_deployer(tmp_path: Path) -> None:
    config = ClaudeCodeConfig(provider="direct", api_key="test", session_header="X-Session-Id")
    deployer = ClaudeCodeDeployer(SimpleNamespace(config=config, work_dir=str(tmp_path), env={}))
    first = deployer._build_env(config)["ANTHROPIC_CUSTOM_HEADERS"]
    assert deployer._build_env(config)["ANTHROPIC_CUSTOM_HEADERS"] == first


def _events(sse: bytes) -> list[tuple[str, dict]]:
    out = []
    for chunk in sse.decode().split("\n\n"):
        if not chunk.strip():
            continue
        event_line, data_line = chunk.split("\n", 1)
        out.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return out


def test_proxy_re_emits_message_as_sse() -> None:
    msg = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "dir"}},
        ],
    }
    events = _events(api_proxy._to_sse(msg))
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    assert events[0][1]["message"]["content"] == []
    deltas = [data["delta"] for name, data in events if name == "content_block_delta"]
    assert deltas[0] == {"type": "text_delta", "text": "hello"}
    assert "".join(d["partial_json"] for d in deltas[1:]) == json.dumps({"command": "dir"})
    assert next(data for name, data in events if name == "message_delta")["delta"]["stop_reason"] == "tool_use"


def test_proxy_normalize_flattens_text_tool_results_and_drops_cache_control() -> None:
    body = {
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": [
                    {"type": "text", "text": "a"}, {"type": "text", "text": "b"},
                ]},
            ]},
            {"role": "assistant", "content": []},
        ],
    }
    out = api_proxy._normalize(body)
    assert out["messages"][0]["content"][0]["content"] == "a\nb"
    assert out["messages"][1]["content"] == "(empty)"
    assert "cache_control" not in json.dumps(out)


def test_proxy_converts_document_blocks_to_text() -> None:
    block = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AAAA"}}
    out = api_proxy._convert_document(block)
    assert out["type"] == "text"
    assert out["text"]
