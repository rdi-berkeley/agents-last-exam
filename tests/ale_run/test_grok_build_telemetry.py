from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from google.protobuf.json_format import ParseDict
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
)
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)

from ale_run.agents.grok_build.otel import (
    GrokBuildOtelCollector,
    recover_otel_artifacts,
)
from ale_run.agents.grok_build.telemetry import recover_native_telemetry_artifacts


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_native_telemetry_captures_llm_and_tool_timings(tmp_path: Path) -> None:
    events = tmp_path / "session_events.jsonl"
    updates = tmp_path / "session_updates.jsonl"
    _write_jsonl(
        events,
        [
            {
                "ts": "2026-07-26T01:00:00.000Z",
                "type": "turn_started",
                "model_id": "kimi-k2.5",
            },
            {
                "ts": "2026-07-26T01:00:00.100Z",
                "type": "loop_started",
                "loop_index": 0,
            },
            {"ts": "2026-07-26T01:00:00.600Z", "type": "first_token"},
            {
                "ts": "2026-07-26T01:00:01.000Z",
                "type": "tool_started",
                "tool_name": "use_tool",
            },
            {
                "ts": "2026-07-26T01:00:01.010Z",
                "type": "mcp_tool_call_started",
                "server_name": "cua",
                "tool_name": "screenshot",
                "call_id": "cua__screenshot",
            },
            {
                "ts": "2026-07-26T01:00:01.260Z",
                "type": "mcp_tool_call_completed",
                "server_name": "cua",
                "tool_name": "screenshot",
                "call_id": "cua__screenshot",
                "duration_ms": 250,
                "success": True,
                "is_timeout": False,
            },
            {
                "ts": "2026-07-26T01:00:01.300Z",
                "type": "tool_completed",
                "tool_name": "use_tool",
                "duration_ms": 300,
                "outcome": "success",
            },
        ],
    )
    _write_jsonl(
        updates,
        [
            {
                "method": "session/update",
                "params": {
                    "sessionId": "session-1",
                    "update": {
                        "sessionUpdate": "turn_completed",
                        "usage": {
                            "inputTokens": 900,
                            "cachedReadTokens": 700,
                            "outputTokens": 50,
                            "reasoningTokens": 20,
                            "totalTokens": 950,
                            "modelCalls": 1,
                            "apiDurationMs": 800,
                        },
                    },
                },
            }
        ],
    )

    assert (
        recover_native_telemetry_artifacts(
            tmp_path,
            events_file=events,
            updates_file=updates,
        )
        == 7
    )

    normalized = [
        json.loads(line) for line in (tmp_path / "native_telemetry.jsonl").read_text().splitlines()
    ]
    summary = json.loads((tmp_path / "native_telemetry_summary.json").read_text())
    assert len(normalized) == 7
    assert summary["session_id"] == "session-1"
    assert summary["usage"]["input_tokens"] == 200
    assert summary["usage"]["cache_read_tokens"] == 700
    assert summary["usage"]["api_duration_ms"] == 800
    assert summary["llm_calls"][0]["first_token_ms"] == 500
    assert summary["llm_calls"][0]["duration_ms"] == 900
    assert summary["tool_executions"][0]["tool_name"] == "cua__screenshot"
    assert summary["tool_executions"][0]["duration_ms"] == 250
    assert summary["tool_executions"][1]["tool_name"] == "use_tool"
    assert summary["tool_executions"][1]["success"] is True


def test_missing_native_logs_still_produces_recoverable_summary(tmp_path: Path) -> None:
    count = recover_native_telemetry_artifacts(
        tmp_path,
        events_file=None,
        updates_file=None,
        terminal_event={
            "sessionId": "session-fallback",
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 20,
                "output_tokens": 5,
            },
        },
    )

    summary = json.loads((tmp_path / "native_telemetry_summary.json").read_text())
    assert count == 0
    assert summary["session_id"] == "session-fallback"
    assert summary["usage"]["input_tokens"] == 10
    assert summary["event_count"] == 0


def _attribute(key: str, value: object) -> dict:
    if isinstance(value, bool):
        encoded = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    else:
        encoded = {"stringValue": str(value)}
    return {"key": key, "value": encoded}


def _log_record(name: str, sequence: int, **attributes: object) -> dict:
    return {
        "timeUnixNano": str(1_800_000_000_000_000_000 + sequence),
        "eventName": name,
        "attributes": [
            _attribute("event.sequence", sequence),
            *[_attribute(key, value) for key, value in attributes.items()],
        ],
    }


def _post_protobuf(url: str, message: object) -> None:
    request = urllib.request.Request(
        url,
        data=message.SerializeToString(),  # type: ignore[attr-defined]
        headers={"Content-Type": "application/x-protobuf"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200


def test_external_otel_collector_captures_protobuf_logs_and_metrics(
    tmp_path: Path,
) -> None:
    logs = ParseDict(
        {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            _attribute("service.name", "grok-cli"),
                            _attribute("grok_code.schema.version", "v1"),
                        ]
                    },
                    "scopeLogs": [
                        {
                            "scope": {"name": "ai.xai.grok_code"},
                            "logRecords": [
                                _log_record(
                                    "grok_code.api_request",
                                    1,
                                    model="grok-4.5",
                                    duration_ms=800,
                                    input_tokens=100,
                                    output_tokens=20,
                                ),
                                _log_record(
                                    "grok_code.tool_result",
                                    2,
                                    tool_name="web_search",
                                    duration_ms=250,
                                    success=True,
                                    outcome="completed",
                                ),
                            ],
                        }
                    ],
                }
            ]
        },
        ExportLogsServiceRequest(),
    )
    metrics = ParseDict(
        {
            "resourceMetrics": [
                {
                    "scopeMetrics": [
                        {
                            "scope": {"name": "ai.xai.grok_code"},
                            "metrics": [
                                {
                                    "name": "grok_code.token.usage",
                                    "unit": "{token}",
                                    "sum": {
                                        "aggregationTemporality": 1,
                                        "isMonotonic": True,
                                        "dataPoints": [
                                            {
                                                "asInt": "100",
                                                "attributes": [
                                                    _attribute("type", "input"),
                                                    _attribute("model", "grok-4.5"),
                                                ],
                                            },
                                            {
                                                "asInt": "20",
                                                "attributes": [
                                                    _attribute("type", "output"),
                                                    _attribute("model", "grok-4.5"),
                                                ],
                                            },
                                        ],
                                    },
                                }
                            ],
                        }
                    ]
                }
            ]
        },
        ExportMetricsServiceRequest(),
    )
    collector = GrokBuildOtelCollector(tmp_path)
    collector.start()
    _post_protobuf(collector.endpoint + "/v1/logs", logs)
    _post_protobuf(collector.endpoint + "/v1/metrics", metrics)
    collector.stop()

    raw = [json.loads(line) for line in collector.raw_path.read_text().splitlines()]
    assert {record["signal"] for record in raw} == {"logs", "metrics"}
    events = [json.loads(line) for line in collector.events_path.read_text().splitlines()]
    assert [event["attributes"]["event.name"] for event in events] == [
        "grok_code.api_request",
        "grok_code.tool_result",
    ]
    summary = json.loads(collector.summary_path.read_text())
    assert summary["api_calls"][0]["duration_ms"] == 800
    assert summary["tool_executions"][0]["tool_name"] == "web_search"
    assert summary["metric_totals"]["tokens_by_type"] == {
        "input": 100,
        "output": 20,
    }


def test_external_otel_wal_recovers_derived_files(tmp_path: Path) -> None:
    logs = ParseDict(
        {
            "resourceLogs": [
                {
                    "scopeLogs": [
                        {
                            "logRecords": [
                                _log_record("grok_code.session_start", 1, model="grok-4.5")
                            ]
                        }
                    ]
                }
            ]
        },
        ExportLogsServiceRequest(),
    )
    collector = GrokBuildOtelCollector(tmp_path)
    collector._store_request("logs", {}, logs.SerializeToString())

    assert recover_otel_artifacts(tmp_path) == 1
    summary = json.loads(collector.summary_path.read_text())
    assert summary["events_by_name"] == {"grok_code.session_start": 1}
