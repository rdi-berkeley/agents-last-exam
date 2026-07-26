from __future__ import annotations

import json
import socket
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from ale_run.agents.kimi_code.telemetry import (
    KimiCodeOtelCollector,
    OTEL_BOOTSTRAP_SOURCE,
    recover_telemetry_artifacts,
)


def _trace_payload(*spans: dict) -> dict:
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "kimi-code"},
                        }
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "@opentelemetry/instrumentation-undici",
                            "version": "0.30.0",
                        },
                        "spans": list(spans),
                    }
                ],
            }
        ]
    }


def _http_span(
    *,
    url: str = "https://api.moonshot.ai/v1/chat/completions",
    start_ns: int = 1_784_407_264_183_000_000,
    end_ns: int = 1_784_407_270_091_376_893,
    large_value: str | None = None,
) -> dict:
    attributes = [
        {"key": "http.request.method", "value": {"stringValue": "POST"}},
        {"key": "url.full", "value": {"stringValue": url}},
        {"key": "http.response.status_code", "value": {"intValue": 200}},
    ]
    if large_value is not None:
        attributes.append({"key": "test.large", "value": {"stringValue": large_value}})
    return {
        "traceId": "4b47440a5c43f2ef9130810c88275e8a",
        "spanId": "2e8c9e1560d79b14",
        "name": "POST",
        "kind": 3,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": attributes,
        "status": {"code": 0},
        "events": [],
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
        while data := sock.recv(4096):
            response += data
    assert b"200" in response.split(b"\r\n", 1)[0]


def test_collector_reads_chunked_trace_and_builds_summary(tmp_path: Path) -> None:
    collector = KimiCodeOtelCollector(tmp_path)
    collector.start()
    _post_chunked(collector.endpoint + "/v1/traces", _trace_payload(_http_span()))
    collector.stop()

    spans = [json.loads(line) for line in collector.events_path.read_text().splitlines()]
    assert len(spans) == 1
    summary = json.loads(collector.summary_path.read_text())
    call = summary["http_calls"][0]
    assert summary["span_count"] == 1
    assert call["url"] == "https://api.moonshot.ai/v1/chat/completions"
    assert call["status_code"] == 200
    assert call["duration_ms"] == 5908.377


def test_large_trace_wal_records_are_bounded_and_recoverable(
    tmp_path: Path,
) -> None:
    large_value = "x" * (2 * 1024 * 1024)
    collector = KimiCodeOtelCollector(tmp_path)
    collector.start()
    _post_json(
        collector.endpoint + "/v1/traces",
        _trace_payload(_http_span(large_value=large_value)),
    )
    collector.stop()

    raw_lines = collector.raw_path.read_bytes().splitlines()
    assert len(raw_lines) > 1
    assert max(map(len, raw_lines)) < 1024 * 1024

    collector.events_path.unlink()
    collector.summary_path.unlink()
    assert recover_telemetry_artifacts(tmp_path) == 1
    recovered = json.loads(collector.events_path.read_text())
    assert recovered["attributes"]["test.large"] == large_value


def test_recovery_ignores_incomplete_final_otlp_request(tmp_path: Path) -> None:
    collector = KimiCodeOtelCollector(tmp_path)
    collector._store_request({}, json.dumps(_trace_payload(_http_span())).encode())
    collector._store_request({}, b'{"resourceSpans":"' + b"x" * (2 * 1024 * 1024))
    lines = collector.raw_path.read_text().splitlines()
    collector.raw_path.write_text("\n".join(lines[:-1]) + "\n")

    assert recover_telemetry_artifacts(tmp_path) == 1
    assert json.loads(collector.summary_path.read_text())["span_count"] == 1


def test_summary_merges_http_usage_latency_and_tool_timing(tmp_path: Path) -> None:
    collector = KimiCodeOtelCollector(tmp_path)
    collector._store_request({}, json.dumps(_trace_payload(_http_span())).encode())
    wire = tmp_path / "wire.jsonl"
    wire.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "type": "llm.request",
                    "model": "kimi-k3",
                    "provider": "kimi",
                    "thinkingEffort": "max",
                    "turnStep": "0.1",
                    "time": 1_784_407_264_183,
                },
                {
                    "type": "context.append_loop_event",
                    "event": {
                        "type": "tool.call",
                        "toolCallId": "Bash_0",
                        "name": "Bash",
                        "args": {"command": "sleep 1"},
                    },
                    "time": 1_784_407_269_000,
                },
                {
                    "type": "context.append_loop_event",
                    "event": {
                        "type": "tool.result",
                        "toolCallId": "Bash_0",
                        "result": {"output": "ok"},
                    },
                    "time": 1_784_407_270_010,
                },
                {
                    "type": "context.append_loop_event",
                    "event": {
                        "type": "step.end",
                        "finishReason": "tool_use",
                        "messageId": "chatcmpl-test",
                        "llmFirstTokenLatencyMs": 4575,
                        "llmStreamDurationMs": 914,
                        "llmRequestBuildMs": 15,
                        "usage": {
                            "inputOther": 7095,
                            "inputCacheRead": 18944,
                            "inputCacheCreation": 0,
                            "output": 53,
                        },
                    },
                    "time": 1_784_407_270_091,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert recover_telemetry_artifacts(tmp_path, wire_file=wire) == 1
    summary = json.loads(collector.summary_path.read_text())
    llm = summary["llm_calls"][0]
    assert llm["model"] == "kimi-k3"
    assert llm["message_id"] == "chatcmpl-test"
    assert llm["first_token_ms"] == 4575
    assert llm["cache_read_tokens"] == 18944
    assert llm["duration_ms"] == 5908.377
    tool = summary["tool_executions"][0]
    assert tool["tool_name"] == "Bash"
    assert tool["duration_ms"] == 1010
    assert tool["success"] is True


def test_bootstrap_uses_json_otlp_without_console_output() -> None:
    assert "UndiciInstrumentation" in OTEL_BOOTSTRAP_SOURCE
    assert "OTLPTraceExporter" in OTEL_BOOTSTRAP_SOURCE
    assert "SimpleSpanProcessor" in OTEL_BOOTSTRAP_SOURCE
    assert "ConsoleSpanExporter" not in OTEL_BOOTSTRAP_SOURCE
