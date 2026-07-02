"""Per-run OpenTelemetry receiver for Codex CLI observability."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import threading
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

_RAW_CHUNK_BYTES = 512 * 1024


class CodexOtelCollector:
    """Receive Codex OTLP/HTTP JSON logs and persist complete run telemetry."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.raw_path = work_dir / "otel_requests.jsonl"
        self.events_path = work_dir / "telemetry.jsonl"
        self.summary_path = work_dir / "telemetry_summary.json"
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        if self._server is None:
            raise RuntimeError("Codex OTel collector has not been started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1/logs"

    def start(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.raw_path, self.events_path, self.summary_path):
            path.unlink(missing_ok=True)

        collector = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != "/v1/logs":
                    self.send_error(404)
                    return

                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                if self.headers.get("Content-Encoding", "").lower() == "gzip":
                    body = gzip.decompress(body)

                collector._store_request(dict(self.headers), body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="codex-otel-collector",
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
        self._write_summary()

    def _store_request(self, headers: dict[str, str], body: bytes) -> None:
        received_at = datetime.now(timezone.utc).isoformat()
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        events = (
            list(_flatten_otlp_logs(payload, received_at))
            if isinstance(payload, dict)
            else []
        )
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
                        "request_id": request_id,
                        "received_at": received_at,
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
            if events:
                with self.events_path.open("a", encoding="utf-8") as file:
                    for event in events:
                        file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_summary(self, events: list[dict[str, Any]] | None = None) -> None:
        if events is None:
            events = _read_jsonl(self.events_path)
        events.sort(
            key=lambda event: (
                event.get("attributes", {}).get("event.timestamp")
                or event.get("observed_time_unix_nano")
                or ""
            )
        )

        names = Counter(
            event.get("attributes", {}).get("event.name", "unknown") for event in events
        )
        summary = {
            "event_count": len(events),
            "events_by_name": dict(sorted(names.items())),
            "api_calls": _summarize_api_calls(events),
            "transport_connections": [
                event
                for event in events
                if event.get("attributes", {}).get("event.name") == "codex.websocket_connect"
            ],
            "time_to_first_token": [
                event
                for event in events
                if event.get("attributes", {}).get("event.name") == "codex.turn_ttft"
            ],
            "tool_executions": _summarize_tools(events),
            "artifacts": {
                "raw_otlp_requests": self.raw_path.name,
                "events": self.events_path.name,
            },
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def recover_telemetry_artifacts(work_dir: Path) -> int:
    """Rebuild derived telemetry files from the incrementally mirrored raw WAL."""
    raw_path = work_dir / "otel_requests.jsonl"
    if not raw_path.exists():
        return 0

    events: list[dict[str, Any]] = []
    for received_at, payload in _read_otlp_payloads(raw_path):
        events.extend(_flatten_otlp_logs(payload, received_at))

    events_path = work_dir / "telemetry.jsonl"
    _write_jsonl_atomic(events_path, events)
    collector = CodexOtelCollector(work_dir)
    collector._write_summary(events)
    return len(events)


def _read_otlp_payloads(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    current_request_id: str | None = None
    current_state: dict[str, Any] | None = None

    for record in _iter_jsonl(path):
        if record.get("record_type") != "otlp_request_chunk":
            if current_state is not None:
                decoded = _decode_chunked_payload(current_state)
                if decoded is not None:
                    yield decoded
                current_request_id = None
                current_state = None
            legacy_body = record.get("body", {})
            if isinstance(legacy_body, dict) and isinstance(legacy_body.get("json"), dict):
                yield str(record.get("received_at", "")), legacy_body["json"]
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
                "received_at": str(record.get("received_at", "")),
                "chunk_count": chunk_count,
                "chunks": {},
                "payload_sha256": record.get("payload_sha256"),
            }

        if current_state["chunk_count"] != chunk_count or not 0 <= chunk_index < chunk_count:
            current_state["invalid"] = True
            continue
        try:
            current_state["chunks"][chunk_index] = base64.b64decode(
                encoded, validate=True,
            )
        except (ValueError, TypeError):
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
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return state["received_at"], payload


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
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _flatten_otlp_logs(payload: dict[str, Any], received_at: str) -> list[dict[str, Any]]:
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
                events.append(
                    {
                        "received_at": received_at,
                        "time_unix_nano": record.get("timeUnixNano"),
                        "observed_time_unix_nano": record.get("observedTimeUnixNano"),
                        "severity_number": record.get("severityNumber"),
                        "severity_text": record.get("severityText"),
                        "body": _any_value(record.get("body")),
                        "attributes": _attributes(record.get("attributes", [])),
                        "resource": resource,
                        "scope": scope,
                        "trace_id": record.get("traceId"),
                        "span_id": record.get("spanId"),
                        "flags": record.get("flags"),
                    }
                )
    return events


def _attributes(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {item.get("key", ""): _any_value(item.get("value")) for item in items if item.get("key")}


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


def _summarize_api_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: deque[dict[str, Any]] = deque()
    calls: list[dict[str, Any]] = []

    for event in events:
        attributes = event.get("attributes", {})
        event_name = attributes.get("event.name")
        if event_name in {"codex.api_request", "codex.websocket_request"}:
            completed_at = _parse_timestamp(attributes.get("event.timestamp"))
            transport_duration = _int_or_none(attributes.get("duration_ms"))
            started_at = None
            if completed_at is not None and transport_duration is not None:
                started_at = completed_at.timestamp() * 1000 - transport_duration
            call = {
                "sequence": len(calls) + 1,
                "transport": ("http" if event_name == "codex.api_request" else "websocket"),
                "started_at": _format_epoch_ms(started_at),
                "transport_completed_at": attributes.get("event.timestamp"),
                "response_completed_at": None,
                "transport_duration_ms": transport_duration,
                "response_duration_ms": None,
                "status_code": attributes.get("http.response.status_code"),
                "attempt": attributes.get("attempt"),
                "endpoint": attributes.get("endpoint"),
                "attributes": attributes,
            }
            calls.append(call)
            pending.append(call)
            continue

        if (
            event_name == "codex.sse_event"
            and attributes.get("event.kind") == "response.completed"
            and pending
        ):
            call = pending.popleft()
            completed_at = _parse_timestamp(attributes.get("event.timestamp"))
            started_at = _parse_timestamp(call["started_at"])
            call["response_completed_at"] = attributes.get("event.timestamp")
            if completed_at is not None and started_at is not None:
                call["response_duration_ms"] = round(
                    (completed_at - started_at).total_seconds() * 1000
                )
            call["response_completed_event"] = event

    return calls


def _summarize_tools(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for event in events:
        attributes = event.get("attributes", {})
        if attributes.get("event.name") != "codex.tool_result":
            continue
        completed_at = _parse_timestamp(attributes.get("event.timestamp"))
        duration_ms = _int_or_none(attributes.get("duration_ms"))
        started_at = None
        if completed_at is not None and duration_ms is not None:
            started_at = completed_at.timestamp() * 1000 - duration_ms
        tools.append(
            {
                "sequence": len(tools) + 1,
                "tool_name": attributes.get("tool_name"),
                "call_id": attributes.get("call_id"),
                "started_at": _format_epoch_ms(started_at),
                "completed_at": attributes.get("event.timestamp"),
                "duration_ms": duration_ms,
                "success": attributes.get("success"),
                "arguments": attributes.get("arguments"),
                "output": attributes.get("output"),
                "mcp_server": attributes.get("mcp_server"),
                "mcp_server_origin": attributes.get("mcp_server_origin"),
                "event": event,
            }
        )
    return tools


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_epoch_ms(value: float | None) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value / 1000, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
