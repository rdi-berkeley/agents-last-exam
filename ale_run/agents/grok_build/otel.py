"""Run-local OTLP/HTTP protobuf receiver for Grok Build."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import threading
import uuid
from collections import Counter
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
    ExportLogsServiceResponse,
)
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
    ExportMetricsServiceResponse,
)

_RAW_CHUNK_BYTES = 512 * 1024
_SIGNAL_BY_PATH = {
    "/v1/logs": "logs",
    "/v1/metrics": "metrics",
}


class GrokBuildOtelCollector:
    """Receive Grok Build's external OTEL stream and persist complete batches."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.raw_path = work_dir / "otel_requests.jsonl"
        self.events_path = work_dir / "telemetry.jsonl"
        self.metrics_path = work_dir / "telemetry_metrics.jsonl"
        self.summary_path = work_dir / "telemetry_summary.json"
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        if self._server is None:
            raise RuntimeError("Grok Build OTel collector has not been started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.raw_path,
            self.events_path,
            self.metrics_path,
            self.summary_path,
        ):
            path.unlink(missing_ok=True)

        collector = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                signal = _SIGNAL_BY_PATH.get(self.path)
                if signal is None:
                    self.send_error(404)
                    return

                body = self._read_body()
                if self.headers.get("Content-Encoding", "").lower() == "gzip":
                    try:
                        body = gzip.decompress(body)
                    except (OSError, EOFError):
                        pass

                collector._store_request(signal, dict(self.headers), body)
                response = _response_body(signal)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def _read_body(self) -> bytes:
                if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                    return _read_chunked_body(self.rfile)
                length = int(self.headers.get("Content-Length", "0"))
                return self.rfile.read(length)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="grok-build-otel-collector",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
        self._write_derived()

    def _store_request(self, signal: str, headers: dict[str, str], body: bytes) -> None:
        payload = _decode_payload(signal, body)
        events = _flatten_otlp_logs(payload) if signal == "logs" else []
        metrics = _flatten_otlp_metrics(payload) if signal == "metrics" else []
        request_id = uuid.uuid4().hex
        chunks = [
            body[offset : offset + _RAW_CHUNK_BYTES]
            for offset in range(0, len(body), _RAW_CHUNK_BYTES)
        ] or [b""]
        payload_sha256 = hashlib.sha256(body).hexdigest()

        with self._lock:
            with self.raw_path.open("a", encoding="utf-8") as file:
                for index, chunk in enumerate(chunks):
                    record: dict[str, Any] = {
                        "record_type": "otlp_request_chunk",
                        "signal": signal,
                        "request_id": request_id,
                        "chunk_index": index,
                        "chunk_count": len(chunks),
                        "payload_encoding": "base64",
                        "payload": base64.b64encode(chunk).decode("ascii"),
                    }
                    if index == 0:
                        record["headers"] = headers
                        record["payload_bytes"] = len(body)
                        record["payload_sha256"] = payload_sha256
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())

            _append_jsonl(self.events_path, events)
            _append_jsonl(self.metrics_path, metrics)

    def _write_derived(
        self,
        events: list[dict[str, Any]] | None = None,
        metrics: list[dict[str, Any]] | None = None,
    ) -> None:
        events = _read_jsonl(self.events_path) if events is None else events
        metrics = _read_jsonl(self.metrics_path) if metrics is None else metrics
        events.sort(key=_event_sort_key)
        names = Counter(
            str(event.get("attributes", {}).get("event.name") or "unknown") for event in events
        )
        summary = {
            "event_count": len(events),
            "events_by_name": dict(sorted(names.items())),
            "api_calls": _events_named(events, "grok_code.api_request"),
            "api_errors": _events_named(events, "grok_code.api_error"),
            "tool_executions": _events_named(events, "grok_code.tool_result"),
            "tool_decisions": _events_named(events, "grok_code.tool_decision"),
            "mcp_connections": _events_named(events, "grok_code.mcp_server_connection"),
            "metric_totals": _summarize_metrics(metrics),
            "artifacts": {
                "raw_otlp_requests": self.raw_path.name,
                "events": self.events_path.name,
                "metrics": self.metrics_path.name,
            },
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def recover_otel_artifacts(work_dir: Path) -> int:
    """Rebuild derived external-OTEL artifacts from the raw request WAL."""
    raw_path = work_dir / "otel_requests.jsonl"
    if not raw_path.exists():
        return 0

    events: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for signal, payload in _read_otlp_payloads(raw_path):
        if signal == "metrics":
            metrics.extend(_flatten_otlp_metrics(payload))
        else:
            events.extend(_flatten_otlp_logs(payload))

    events.sort(key=_event_sort_key)
    _write_jsonl_atomic(work_dir / "telemetry.jsonl", events)
    _write_jsonl_atomic(work_dir / "telemetry_metrics.jsonl", metrics)
    GrokBuildOtelCollector(work_dir)._write_derived(events, metrics)
    return len(events)


def _response_body(signal: str) -> bytes:
    if signal == "metrics":
        return ExportMetricsServiceResponse().SerializeToString()
    return ExportLogsServiceResponse().SerializeToString()


def _decode_payload(signal: str, body: bytes) -> dict[str, Any]:
    request = ExportMetricsServiceRequest() if signal == "metrics" else ExportLogsServiceRequest()
    try:
        request.ParseFromString(body)
    except DecodeError:
        return {}
    return MessageToDict(request)


def _read_chunked_body(rfile: Any) -> bytes:
    body = bytearray()
    while True:
        size_line = rfile.readline()
        if not size_line:
            break
        try:
            size = int(size_line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            break
        if size == 0:
            rfile.readline()
            break
        body += rfile.read(size)
        rfile.readline()
    return bytes(body)


def _read_otlp_payloads(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    current_request_id: str | None = None
    current_state: dict[str, Any] | None = None

    for record in _iter_jsonl(path):
        if record.get("record_type") != "otlp_request_chunk":
            continue
        request_id = record.get("request_id")
        chunk_index = record.get("chunk_index")
        chunk_count = record.get("chunk_count")
        encoded = record.get("payload")
        if (
            not isinstance(request_id, str)
            or not isinstance(chunk_index, int)
            or not isinstance(chunk_count, int)
            or chunk_count < 1
            or not isinstance(encoded, str)
        ):
            continue

        if request_id != current_request_id:
            if current_state is not None:
                decoded = _decode_chunked_payload(current_state)
                if decoded is not None:
                    yield decoded
            current_request_id = request_id
            current_state = {
                "signal": str(record.get("signal") or "logs"),
                "chunk_count": chunk_count,
                "chunks": {},
                "payload_sha256": record.get("payload_sha256"),
            }

        if current_state["chunk_count"] != chunk_count or not 0 <= chunk_index < chunk_count:
            current_state["invalid"] = True
            continue
        try:
            current_state["chunks"][chunk_index] = base64.b64decode(
                encoded,
                validate=True,
            )
        except (TypeError, ValueError):
            current_state["invalid"] = True

    if current_state is not None:
        decoded = _decode_chunked_payload(current_state)
        if decoded is not None:
            yield decoded


def _decode_chunked_payload(
    state: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    chunks = state["chunks"]
    if state.get("invalid") or len(chunks) != state["chunk_count"]:
        return None
    body = b"".join(chunks[index] for index in range(state["chunk_count"]))
    expected_sha256 = state.get("payload_sha256")
    if expected_sha256 and hashlib.sha256(body).hexdigest() != expected_sha256:
        return None
    signal = str(state.get("signal") or "logs")
    return signal, _decode_payload(signal, body)


def _flatten_otlp_logs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for resource_log in payload.get("resourceLogs", []):
        resource = _attributes(resource_log.get("resource", {}).get("attributes", []))
        for scope_log in resource_log.get("scopeLogs", []):
            scope_data = scope_log.get("scope", {})
            scope = {
                "name": scope_data.get("name"),
                "version": scope_data.get("version"),
                "attributes": _attributes(scope_data.get("attributes", [])),
            }
            for record in scope_log.get("logRecords", []):
                attributes = _attributes(record.get("attributes", []))
                event_name = record.get("eventName")
                if isinstance(event_name, str) and event_name:
                    attributes.setdefault("event.name", event_name)
                events.append(
                    {
                        "time_unix_nano": record.get("timeUnixNano"),
                        "observed_time_unix_nano": record.get("observedTimeUnixNano"),
                        "severity_number": record.get("severityNumber"),
                        "severity_text": record.get("severityText"),
                        "body": _any_value(record.get("body")),
                        "attributes": attributes,
                        "resource": resource,
                        "scope": scope,
                        "trace_id": record.get("traceId"),
                        "span_id": record.get("spanId"),
                        "flags": record.get("flags"),
                    }
                )
    return events


def _flatten_otlp_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for resource_metric in payload.get("resourceMetrics", []):
        resource = _attributes(resource_metric.get("resource", {}).get("attributes", []))
        for scope_metric in resource_metric.get("scopeMetrics", []):
            scope = scope_metric.get("scope", {})
            for metric in scope_metric.get("metrics", []):
                for kind in ("sum", "gauge", "histogram"):
                    container = metric.get(kind)
                    if not isinstance(container, dict):
                        continue
                    for point in container.get("dataPoints", []):
                        points.append(
                            {
                                "name": metric.get("name"),
                                "description": metric.get("description"),
                                "unit": metric.get("unit"),
                                "kind": kind,
                                "value": _metric_value(point),
                                "start_time_unix_nano": point.get("startTimeUnixNano"),
                                "time_unix_nano": point.get("timeUnixNano"),
                                "attributes": _attributes(point.get("attributes", [])),
                                "resource": resource,
                                "scope": {
                                    "name": scope.get("name"),
                                    "version": scope.get("version"),
                                },
                            }
                        )
    return points


def _metric_value(point: dict[str, Any]) -> int | float | None:
    for key in ("asInt", "asDouble", "sum", "count"):
        value = point.get(key)
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value) if "." in value else int(value)
            except ValueError:
                return None
    return None


def _attributes(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(item["key"]): _any_value(item.get("value"))
        for item in items
        if isinstance(item, dict) and item.get("key")
    }


def _any_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "bytesValue" in value:
        return value["bytesValue"]
    if "arrayValue" in value:
        return [_any_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return _attributes(value["kvlistValue"].get("values", []))
    return value


def _events_named(
    events: list[dict[str, Any]],
    event_name: str,
) -> list[dict[str, Any]]:
    return [
        event.get("attributes", {})
        for event in events
        if event.get("attributes", {}).get("event.name") == event_name
    ]


def _summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {}
    token_usage: dict[str, float] = {}
    for point in metrics:
        name = point.get("name")
        value = point.get("value")
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            continue
        totals[name] = totals.get(name, 0.0) + value
        if name == "grok_code.token.usage":
            token_type = str(point.get("attributes", {}).get("type") or "unknown")
            token_usage[token_type] = token_usage.get(token_type, 0.0) + value
    return {
        "by_metric": {name: _clean_number(value) for name, value in sorted(totals.items())},
        "tokens_by_type": {
            name: _clean_number(value) for name, value in sorted(token_usage.items())
        },
    }


def _clean_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _event_sort_key(event: dict[str, Any]) -> tuple[int, str]:
    sequence = event.get("attributes", {}).get("event.sequence")
    return (
        sequence if isinstance(sequence, int) else 2**63 - 1,
        str(event.get("time_unix_nano") or event.get("observed_time_unix_nano") or ""),
    )


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
