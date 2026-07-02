from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from ale_run.agents.codex.deployer import CodexDeployer
from ale_run.agents.codex.telemetry import (
    CodexOtelCollector,
    recover_telemetry_artifacts,
)


def _record(name: str, timestamp: str, **attributes: object) -> dict:
    items = [
        {"key": "event.name", "value": {"stringValue": name}},
        {"key": "event.timestamp", "value": {"stringValue": timestamp}},
    ]
    for key, value in attributes.items():
        if isinstance(value, bool):
            encoded = {"boolValue": value}
        elif isinstance(value, int):
            encoded = {"intValue": str(value)}
        else:
            encoded = {"stringValue": str(value)}
        items.append({"key": key, "value": encoded})
    return {
        "observedTimeUnixNano": "123",
        "severityNumber": 9,
        "severityText": "INFO",
        "attributes": items,
    }


def test_collector_persists_and_summarizes_otel_logs(tmp_path: Path) -> None:
    collector = CodexOtelCollector(tmp_path)
    collector.start()
    payload = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "codex_exec"},
                        }
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "codex_otel.log_only"},
                        "logRecords": [
                            _record(
                                "codex.api_request",
                                "2026-07-02T10:00:00.500Z",
                                duration_ms=500,
                                endpoint="/responses",
                                attempt=0,
                            ),
                            _record(
                                "codex.sse_event",
                                "2026-07-02T10:00:02.000Z",
                                **{"event.kind": "response.completed"},
                            ),
                            _record(
                                "codex.tool_result",
                                "2026-07-02T10:00:03.000Z",
                                tool_name="exec_command",
                                call_id="call_123",
                                duration_ms=250,
                                success=True,
                                arguments='{"cmd":"echo test"}',
                                output="test",
                            ),
                        ],
                    }
                ],
            }
        ]
    }
    request = urllib.request.Request(
        collector.endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200
    collector.stop()

    raw_records = [
        json.loads(line) for line in collector.raw_path.read_text().splitlines()
    ]
    assert raw_records[0]["record_type"] == "otlp_request_chunk"
    assert raw_records[0]["chunk_count"] == 1

    events = [json.loads(line) for line in collector.events_path.read_text().splitlines()]
    assert len(events) == 3
    assert events[2]["attributes"]["arguments"] == '{"cmd":"echo test"}'
    assert events[2]["attributes"]["output"] == "test"

    summary = json.loads(collector.summary_path.read_text())
    assert summary["event_count"] == 3
    assert summary["api_calls"][0]["transport_duration_ms"] == 500
    assert summary["api_calls"][0]["response_duration_ms"] == 2000
    assert summary["tool_executions"][0]["duration_ms"] == 250
    assert summary["tool_executions"][0]["arguments"] == '{"cmd":"echo test"}'
    assert summary["tool_executions"][0]["output"] == "test"


def test_large_raw_request_is_chunked_and_recovers_derived_files(tmp_path: Path) -> None:
    large_output = "x" * (2 * 1024 * 1024)
    payload = {
        "resourceLogs": [
            {
                "scopeLogs": [
                    {
                        "logRecords": [
                            _record(
                                "codex.tool_result",
                                "2026-07-02T10:00:03.000Z",
                                tool_name="exec_command",
                                call_id="call_large",
                                duration_ms=250,
                                success=True,
                                output=large_output,
                            )
                        ]
                    }
                ]
            }
        ]
    }
    collector = CodexOtelCollector(tmp_path)
    collector.start()
    request = urllib.request.Request(
        collector.endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200
    collector.stop()

    raw_lines = collector.raw_path.read_bytes().splitlines()
    assert len(raw_lines) > 1
    assert max(map(len, raw_lines)) < 1024 * 1024

    collector.events_path.unlink()
    collector.summary_path.unlink()
    assert recover_telemetry_artifacts(tmp_path) == 1

    recovered = json.loads(collector.events_path.read_text().strip())
    assert recovered["attributes"]["output"] == large_output
    summary = json.loads(collector.summary_path.read_text())
    assert summary["event_count"] == 1
    assert summary["tool_executions"][0]["output"] == large_output


def test_codex_incrementally_mirrors_raw_telemetry_wal() -> None:
    assert "otel_requests.jsonl" in CodexDeployer.hot_artifacts
    assert "telemetry.jsonl" not in CodexDeployer.hot_artifacts


def test_recovery_ignores_incomplete_trailing_request(tmp_path: Path) -> None:
    collector = CodexOtelCollector(tmp_path)
    complete_payload = {
        "resourceLogs": [
            {
                "scopeLogs": [
                    {
                        "logRecords": [
                            _record(
                                "codex.turn_ttft",
                                "2026-07-02T10:00:00.000Z",
                                duration_ms=123,
                            )
                        ]
                    }
                ]
            }
        ]
    }
    collector._store_request({}, json.dumps(complete_payload).encode())
    collector._store_request({}, b'{"resourceLogs":"' + b"x" * (2 * 1024 * 1024))

    raw_lines = collector.raw_path.read_text().splitlines()
    collector.raw_path.write_text("\n".join(raw_lines[:-1]) + "\n")

    assert recover_telemetry_artifacts(tmp_path) == 1
    summary = json.loads(collector.summary_path.read_text())
    assert summary["event_count"] == 1
    assert summary["events_by_name"] == {"codex.turn_ttft": 1}
