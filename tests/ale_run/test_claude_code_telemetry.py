from __future__ import annotations

import json
import socket
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from ale_run.agents.claude_code.deployer import ClaudeCodeDeployer
from ale_run.agents.claude_code.telemetry import (
    ClaudeCodeOtelCollector,
    recover_telemetry_artifacts,
)


def _record(name: str, timestamp: str, sequence: int, **attributes: object) -> dict:
    """Build a Claude Code OTLP logRecord (body = prefixed name, attrs = short)."""
    items = [
        {"key": "event.name", "value": {"stringValue": name}},
        {"key": "event.timestamp", "value": {"stringValue": timestamp}},
        {"key": "event.sequence", "value": {"intValue": sequence}},
    ]
    for key, value in attributes.items():
        if isinstance(value, bool):
            encoded = {"boolValue": value}
        elif isinstance(value, int):
            encoded = {"intValue": value}
        elif isinstance(value, float):
            encoded = {"doubleValue": value}
        else:
            encoded = {"stringValue": str(value)}
        items.append({"key": key, "value": encoded})
    return {
        "timeUnixNano": "1783556555442000000",
        "observedTimeUnixNano": "1783556555442000000",
        "body": {"stringValue": f"claude_code.{name}"},
        "attributes": items,
    }


def _logs_payload(*records: dict) -> dict:
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "claude-code"}}
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "com.anthropic.claude_code.events"},
                        "logRecords": list(records),
                    }
                ],
            }
        ]
    }


def _metrics_payload() -> dict:
    def _sum(name: str, points: list[tuple[float, dict]]) -> dict:
        return {
            "name": name,
            "sum": {
                "dataPoints": [
                    {
                        "asDouble": value,
                        "attributes": [
                            {"key": k, "value": {"stringValue": v}}
                            for k, v in attrs.items()
                        ],
                    }
                    for value, attrs in points
                ]
            },
        }

    return {
        "resourceMetrics": [
            {
                "scopeMetrics": [
                    {
                        "scope": {"name": "com.anthropic.claude_code"},
                        "metrics": [
                            _sum(
                                "claude_code.token.usage",
                                [
                                    (525.0, {"type": "input", "model": "m"}),
                                    (10.0, {"type": "output", "model": "m"}),
                                ],
                            ),
                            _sum(
                                "claude_code.cost.usage",
                                [(0.0125, {"model": "m"})],
                            ),
                        ],
                    }
                ]
            }
        ]
    }


def _post_json(url: str, payload: dict) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200


def _post_chunked(url: str, payload: dict) -> None:
    """POST with Transfer-Encoding: chunked and no Content-Length.

    This is exactly how Claude Code's OpenTelemetry JS exporter frames requests;
    a Content-Length-only reader captures zero bytes.
    """
    parsed = urlparse(url)
    body = json.dumps(payload).encode()
    frame = b"%x\r\n%s\r\n0\r\n\r\n" % (len(body), body)
    request = (
        f"POST {parsed.path} HTTP/1.1\r\n"
        f"Host: {parsed.hostname}:{parsed.port}\r\n"
        "Content-Type: application/json\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: close\r\n\r\n"
    ).encode() + frame
    with socket.create_connection((parsed.hostname, parsed.port), timeout=5) as sock:
        sock.sendall(request)
        response = b""
        while True:
            data = sock.recv(4096)
            if not data:
                break
            response += data
    assert b"200" in response.split(b"\r\n", 1)[0]


def test_collector_persists_and_summarizes_logs_and_metrics(tmp_path: Path) -> None:
    collector = ClaudeCodeOtelCollector(tmp_path)
    collector.start()
    logs = _logs_payload(
        _record("user_prompt", "2026-07-02T10:00:00.000Z", 0,
                prompt_length=37, prompt="do the thing", **{"prompt.id": "p1"}),
        _record("api_request", "2026-07-02T10:00:02.000Z", 1,
                model="claude-sonnet-4-6", duration_ms=1415,
                input_tokens=3, output_tokens=5, cache_read_tokens=16885,
                cost_usd=0.0364, request_id="req_abc", query_source="sdk"),
        _record("tool_result", "2026-07-02T10:00:03.000Z", 2,
                tool_name="Bash", tool_use_id="toolu_1", duration_ms=250,
                success=True, tool_input='{"command":"echo hi"}'),
        _record("tool_decision", "2026-07-02T10:00:03.100Z", 3,
                tool_name="Bash", decision="accept", source="config"),
    )
    _post_json(collector.endpoint + "/v1/logs", logs)
    _post_json(collector.endpoint + "/v1/metrics", _metrics_payload())
    collector.stop()

    raw_records = [
        json.loads(line) for line in collector.raw_path.read_text().splitlines()
    ]
    assert raw_records[0]["record_type"] == "otlp_request_chunk"
    assert {r["signal"] for r in raw_records} == {"logs", "metrics"}

    events = [json.loads(line) for line in collector.events_path.read_text().splitlines()]
    assert len(events) == 4

    summary = json.loads(collector.summary_path.read_text())
    assert summary["event_count"] == 4
    assert summary["events_by_name"] == {
        "api_request": 1, "tool_decision": 1, "tool_result": 1, "user_prompt": 1,
    }

    call = summary["api_calls"][0]
    assert call["model"] == "claude-sonnet-4-6"
    assert call["duration_ms"] == 1415
    assert call["request_id"] == "req_abc"
    assert call["query_source"] == "sdk"
    assert call["input_tokens"] == 3

    assert summary["tool_executions"][0]["tool_name"] == "Bash"
    assert summary["tool_executions"][0]["duration_ms"] == 250
    assert summary["tool_executions"][0]["tool_input"] == '{"command":"echo hi"}'
    assert summary["tool_decisions"][0]["decision"] == "accept"
    assert summary["user_prompts"][0]["prompt"] == "do the thing"

    assert summary["metric_totals"]["tokens_by_type"] == {"input": 525, "output": 10}
    assert summary["metric_totals"]["total_cost_usd"] == 0.0125


def test_collector_reads_chunked_transfer_encoding(tmp_path: Path) -> None:
    # Claude Code's exporter sends chunked bodies with no Content-Length; the
    # receiver must de-chunk or it captures nothing.
    collector = ClaudeCodeOtelCollector(tmp_path)
    collector.start()
    payload = _logs_payload(
        _record("api_request", "2026-07-02T10:00:02.000Z", 0,
                model="m", duration_ms=100, request_id="req_chunk")
    )
    _post_chunked(collector.endpoint + "/v1/logs", payload)
    collector.stop()

    events = [json.loads(line) for line in collector.events_path.read_text().splitlines()]
    assert len(events) == 1
    summary = json.loads(collector.summary_path.read_text())
    assert summary["api_calls"][0]["request_id"] == "req_chunk"


def test_large_raw_request_is_chunked_and_recovers_derived_files(tmp_path: Path) -> None:
    large_output = "x" * (2 * 1024 * 1024)
    payload = _logs_payload(
        _record("tool_result", "2026-07-02T10:00:03.000Z", 0,
                tool_name="Bash", tool_use_id="toolu_large", duration_ms=250,
                success=True, tool_input=large_output)
    )
    collector = ClaudeCodeOtelCollector(tmp_path)
    collector.start()
    _post_json(collector.endpoint + "/v1/logs", payload)
    collector.stop()

    raw_lines = collector.raw_path.read_bytes().splitlines()
    assert len(raw_lines) > 1
    assert max(map(len, raw_lines)) < 1024 * 1024

    collector.events_path.unlink()
    collector.summary_path.unlink()
    assert recover_telemetry_artifacts(tmp_path) == 1

    recovered = json.loads(collector.events_path.read_text().strip())
    assert recovered["attributes"]["tool_input"] == large_output
    summary = json.loads(collector.summary_path.read_text())
    assert summary["event_count"] == 1
    assert summary["tool_executions"][0]["tool_input"] == large_output


def test_claude_code_incrementally_mirrors_raw_telemetry_wal() -> None:
    assert "otel_requests.jsonl" in ClaudeCodeDeployer.hot_artifacts
    assert "telemetry.jsonl" not in ClaudeCodeDeployer.hot_artifacts


def test_recovery_ignores_incomplete_trailing_request(tmp_path: Path) -> None:
    collector = ClaudeCodeOtelCollector(tmp_path)
    complete = _logs_payload(
        _record("api_request", "2026-07-02T10:00:00.000Z", 0,
                model="m", duration_ms=123, request_id="req_ok")
    )
    collector._store_request("logs", {}, json.dumps(complete).encode())
    collector._store_request("logs", {}, b'{"resourceLogs":"' + b"x" * (2 * 1024 * 1024))

    raw_lines = collector.raw_path.read_text().splitlines()
    collector.raw_path.write_text("\n".join(raw_lines[:-1]) + "\n")

    assert recover_telemetry_artifacts(tmp_path) == 1
    summary = json.loads(collector.summary_path.read_text())
    assert summary["event_count"] == 1
    assert summary["events_by_name"] == {"api_request": 1}


def test_events_sorted_by_sequence(tmp_path: Path) -> None:
    # Batches can arrive out of order; the derived event log is sorted by the
    # CLI's monotonic event.sequence.
    collector = ClaudeCodeOtelCollector(tmp_path)
    later = _logs_payload(
        _record("api_request", "2026-07-02T10:00:05.000Z", 5, model="m")
    )
    earlier = _logs_payload(
        _record("user_prompt", "2026-07-02T10:00:00.000Z", 0, prompt_length=1)
    )
    collector._store_request("logs", {}, json.dumps(later).encode())
    collector._store_request("logs", {}, json.dumps(earlier).encode())

    # Finalization rebuilds telemetry.jsonl from the raw WAL, sorted by sequence.
    assert recover_telemetry_artifacts(tmp_path) == 2
    events = [json.loads(line) for line in collector.events_path.read_text().splitlines()]
    assert [e["attributes"]["event.sequence"] for e in events] == [0, 5]
