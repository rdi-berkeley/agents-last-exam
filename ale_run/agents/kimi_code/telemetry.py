"""Run-local OpenTelemetry capture and recovery for Kimi Code."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

_RAW_CHUNK_BYTES = 512 * 1024
_TRACE_PATH = "/v1/traces"

OTEL_NPM_PACKAGE_VERSIONS = {
    "@opentelemetry/sdk-node": "0.220.0",
    "@opentelemetry/sdk-trace-base": "2.9.0",
    "@opentelemetry/exporter-trace-otlp-http": "0.220.0",
    "@opentelemetry/instrumentation-undici": "0.30.0",
}
OTEL_NPM_PACKAGES = tuple(
    f"{name}@{version}" for name, version in OTEL_NPM_PACKAGE_VERSIONS.items()
)

OTEL_BOOTSTRAP_SOURCE = """\
import { NodeSDK } from "@opentelemetry/sdk-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { UndiciInstrumentation } from "@opentelemetry/instrumentation-undici";

const endpoint = process.env.ALE_KIMI_OTEL_ENDPOINT;
if (endpoint) {
  const importOption = process.env.ALE_KIMI_OTEL_IMPORT_OPTION;
  delete process.env.ALE_KIMI_OTEL_ENDPOINT;
  delete process.env.ALE_KIMI_OTEL_IMPORT_OPTION;
  if (importOption && process.env.NODE_OPTIONS) {
    process.env.NODE_OPTIONS = process.env.NODE_OPTIONS.replace(importOption, "").trim();
  }
  const sdk = new NodeSDK({
    serviceName: "kimi-code",
    spanProcessors: [new SimpleSpanProcessor(
      new OTLPTraceExporter({ url: `${endpoint}/v1/traces` })
    )],
    instrumentations: [new UndiciInstrumentation()],
  });
  sdk.start();
  let stopping = false;
  process.once("beforeExit", async () => {
    if (stopping) return;
    stopping = true;
    try { await sdk.shutdown(); } catch {}
  });
}
"""


class KimiCodeOtelCollector:
    """Receive OTLP/HTTP traces and persist a recoverable bounded-record WAL."""

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
            raise RuntimeError("Kimi Code OTel collector is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.raw_path, self.events_path, self.summary_path):
            path.unlink(missing_ok=True)
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != _TRACE_PATH:
                    self.send_error(404)
                    return
                body = self._read_body()
                if self.headers.get("Content-Encoding", "").lower() == "gzip":
                    try:
                        body = gzip.decompress(body)
                    except (OSError, EOFError):
                        pass
                collector._store_request(dict(self.headers), body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def _read_body(self) -> bytes:
                if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                    return _read_chunked_body(self.rfile)
                return self.rfile.read(int(self.headers.get("Content-Length", "0")))

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="kimi-code-otel-collector",
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
        recover_telemetry_artifacts(self.work_dir)

    def _store_request(self, headers: dict[str, str], body: bytes) -> None:
        request_id = uuid.uuid4().hex
        chunks = [
            body[offset : offset + _RAW_CHUNK_BYTES]
            for offset in range(0, len(body), _RAW_CHUNK_BYTES)
        ] or [b""]
        with self._lock, self.raw_path.open("a", encoding="utf-8") as file:
            for index, chunk in enumerate(chunks):
                record: dict[str, Any] = {
                    "record_type": "otlp_request_chunk",
                    "request_id": request_id,
                    "received_at": datetime.now(timezone.utc).isoformat(),
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "payload": base64.b64encode(chunk).decode("ascii"),
                }
                if index == 0:
                    record.update(
                        headers=headers,
                        payload_bytes=len(body),
                        payload_sha256=hashlib.sha256(body).hexdigest(),
                    )
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())


def write_otel_bootstrap(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OTEL_BOOTSTRAP_SOURCE, encoding="utf-8")


def recover_telemetry_artifacts(
    work_dir: Path,
    *,
    wire_file: Path | None = None,
) -> int:
    """Rebuild normalized spans and summaries from the incrementally mirrored WAL."""
    spans: list[dict[str, Any]] = []
    raw_path = work_dir / "otel_requests.jsonl"
    for received_at, payload in _read_otlp_payloads(raw_path):
        spans.extend(_flatten_otlp_traces(payload, received_at))
    spans.sort(key=lambda span: int(span.get("start_time_unix_nano") or 0))
    _write_jsonl_atomic(work_dir / "telemetry.jsonl", spans)

    http_calls = [_summarize_http_span(span, i) for i, span in enumerate(spans, 1)]
    wire = _summarize_wire(wire_file)
    model_http_calls = [call for call in http_calls if _is_model_endpoint(call.get("url"))]
    llm_calls: list[dict[str, Any]] = []
    for index in range(max(len(model_http_calls), len(wire["llm_calls"]))):
        merged: dict[str, Any] = {"sequence": index + 1}
        if index < len(model_http_calls):
            merged.update(model_http_calls[index])
        if index < len(wire["llm_calls"]):
            merged.update(wire["llm_calls"][index])
        merged["sequence"] = index + 1
        llm_calls.append(merged)

    summary = {
        "span_count": len(spans),
        "http_calls": http_calls,
        "llm_calls": llm_calls,
        "tool_executions": wire["tool_executions"],
        "artifacts": {
            "raw_otlp_requests": raw_path.name,
            "spans": "telemetry.jsonl",
            "wire": str(wire_file) if wire_file else None,
        },
    }
    (work_dir / "telemetry_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(spans)


def _read_chunked_body(stream: Any) -> bytes:
    body = bytearray()
    while size_line := stream.readline():
        try:
            size = int(size_line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            break
        if size == 0:
            stream.readline()
            break
        body += stream.read(size)
        stream.readline()
    return bytes(body)


def _read_otlp_payloads(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    state: dict[str, Any] | None = None
    for record in _iter_jsonl(path):
        if record.get("record_type") != "otlp_request_chunk":
            continue
        request_id = record.get("request_id")
        if state is None or state["request_id"] != request_id:
            if state is not None:
                decoded = _decode_payload(state)
                if decoded:
                    yield decoded
            state = {
                "request_id": request_id,
                "received_at": record.get("received_at", ""),
                "count": record.get("chunk_count"),
                "sha256": record.get("payload_sha256"),
                "chunks": {},
            }
        try:
            state["chunks"][int(record["chunk_index"])] = base64.b64decode(
                record["payload"],
                validate=True,
            )
        except (KeyError, TypeError, ValueError):
            state["invalid"] = True
    if state is not None:
        decoded = _decode_payload(state)
        if decoded:
            yield decoded


def _decode_payload(state: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    count = state.get("count")
    chunks = state["chunks"]
    if state.get("invalid") or not isinstance(count, int) or len(chunks) != count:
        return None
    try:
        body = b"".join(chunks[index] for index in range(count))
    except KeyError:
        return None
    if state.get("sha256") and hashlib.sha256(body).hexdigest() != state["sha256"]:
        return None
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return (str(state["received_at"]), payload) if isinstance(payload, dict) else None


def _flatten_otlp_traces(payload: dict[str, Any], received_at: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for resource_span in payload.get("resourceSpans", []):
        resource = _attributes(resource_span.get("resource", {}).get("attributes", []))
        for scope_span in resource_span.get("scopeSpans", []):
            scope_data = scope_span.get("scope", {})
            scope = {"name": scope_data.get("name"), "version": scope_data.get("version")}
            for span in scope_span.get("spans", []):
                spans.append(
                    {
                        "received_at": received_at,
                        "trace_id": span.get("traceId"),
                        "span_id": span.get("spanId"),
                        "parent_span_id": span.get("parentSpanId"),
                        "name": span.get("name"),
                        "kind": span.get("kind"),
                        "start_time_unix_nano": span.get("startTimeUnixNano"),
                        "end_time_unix_nano": span.get("endTimeUnixNano"),
                        "attributes": _attributes(span.get("attributes", [])),
                        "status": span.get("status"),
                        "events": span.get("events", []),
                        "resource": resource,
                        "scope": scope,
                    }
                )
    return spans


def _summarize_http_span(span: dict[str, Any], sequence: int) -> dict[str, Any]:
    attrs = span.get("attributes", {})
    start = _int_or_none(span.get("start_time_unix_nano"))
    end = _int_or_none(span.get("end_time_unix_nano"))
    return {
        "sequence": sequence,
        "method": attrs.get("http.request.method"),
        "url": attrs.get("url.full"),
        "status_code": attrs.get("http.response.status_code"),
        "started_at": _format_time(start, 1_000_000_000),
        "completed_at": _format_time(end, 1_000_000_000),
        "duration_ms": round((end - start) / 1_000_000, 3)
        if start is not None and end is not None
        else None,
        "trace_id": span.get("trace_id"),
        "span_id": span.get("span_id"),
    }


def _summarize_wire(wire_file: Path | None) -> dict[str, list[dict[str, Any]]]:
    llm_calls: list[dict[str, Any]] = []
    tool_executions: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    tools: dict[str, dict[str, Any]] = {}
    if wire_file is None:
        return {"llm_calls": llm_calls, "tool_executions": tool_executions}
    for record in _iter_jsonl(wire_file):
        if record.get("type") == "llm.request":
            requests.append(record)
            continue
        if record.get("type") != "context.append_loop_event":
            continue
        event = record.get("event") or {}
        event_type = event.get("type")
        if event_type == "step.end":
            request = requests.pop(0) if requests else {}
            usage = event.get("usage") or {}
            llm_calls.append(
                {
                    "model": request.get("model"),
                    "provider": request.get("provider"),
                    "thinking_effort": request.get("thinkingEffort"),
                    "turn_step": request.get("turnStep"),
                    "request_started_at": _format_time(
                        _int_or_none(request.get("time")),
                        1000,
                    ),
                    "request_completed_at": _format_time(
                        _int_or_none(record.get("time")),
                        1000,
                    ),
                    "message_id": event.get("messageId"),
                    "finish_reason": event.get("finishReason"),
                    "first_token_ms": event.get("llmFirstTokenLatencyMs"),
                    "stream_duration_ms": event.get("llmStreamDurationMs"),
                    "input_tokens": usage.get("inputOther"),
                    "cache_read_tokens": usage.get("inputCacheRead"),
                    "cache_creation_tokens": usage.get("inputCacheCreation"),
                    "output_tokens": usage.get("output"),
                }
            )
        elif event_type == "tool.call":
            tool_id = event.get("toolCallId") or event.get("uuid")
            if isinstance(tool_id, str):
                tools[tool_id] = {
                    "tool_call_id": tool_id,
                    "tool_name": event.get("name"),
                    "arguments": event.get("args"),
                    "started_at": _format_time(_int_or_none(record.get("time")), 1000),
                    "_started_ms": _int_or_none(record.get("time")),
                }
        elif event_type == "tool.result":
            tool_id = event.get("toolCallId") or event.get("parentUuid")
            if not isinstance(tool_id, str):
                continue
            tool = tools.pop(tool_id, {"tool_call_id": tool_id, "_started_ms": None})
            completed = _int_or_none(record.get("time"))
            started = tool.pop("_started_ms")
            result = event.get("result") or {}
            tool.update(
                sequence=len(tool_executions) + 1,
                completed_at=_format_time(completed, 1000),
                duration_ms=completed - started
                if completed is not None and started is not None
                else None,
                success=not bool(result.get("isError")),
                error=result.get("error"),
            )
            tool_executions.append(tool)
    return {"llm_calls": llm_calls, "tool_executions": tool_executions}


def _is_model_endpoint(value: Any) -> bool:
    return isinstance(value, str) and value.split("?", 1)[0].rstrip("/").endswith(
        ("/chat/completions", "/responses", "/messages")
    )


def _attributes(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {item["key"]: _any_value(item.get("value")) for item in items if item.get("key")}


def _any_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "boolValue", "doubleValue", "bytesValue"):
        if key in value:
            return value[key]
    if "intValue" in value:
        return _int_or_none(value["intValue"])
    if "arrayValue" in value:
        return [_any_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return _attributes(value["kvlistValue"].get("values", []))
    return value


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
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _format_time(value: int | None, divisor: int) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value / divisor, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
