"""Per-run OpenTelemetry receiver for Claude Code CLI observability.

Claude Code emits structured OpenTelemetry **logs** (one record per event:
``api_request``, ``api_error``, ``tool_result``, ``tool_decision``,
``user_prompt``, ...) and **metrics** (cumulative token/cost/session counters)
when ``CLAUDE_CODE_ENABLE_TELEMETRY=1``. The events carry operational detail the
``transcript.jsonl`` does not: per-request model latency (``duration_ms``), the
upstream Anthropic ``request_id``, ``query_source`` (main vs. auxiliary calls
such as session-title generation), per-tool duration/success, and prompt text.

This module stands up a run-local OTLP/HTTP receiver, points the CLI at it via
env vars (see the deployer), and persists complete run telemetry in ``work_dir``:

* ``otel_requests.jsonl`` — a write-ahead log of every raw OTLP batch, chunked
  and SHA-256 tagged so large tool payloads stay as bounded JSONL records and a
  truncated final batch never invalidates earlier complete ones. This file is on
  the deployer's ``hot_artifacts`` list so it is mirrored incrementally off the
  sandbox during long tasks, not only at the final directory gather.
* ``telemetry.jsonl`` — flattened log events, one JSON object per record.
* ``telemetry_metrics.jsonl`` — flattened metric data points.
* ``telemetry_summary.json`` — API calls (timing/tokens/cost/request_id), API
  errors, tool executions, tool decisions, user prompts, and metric totals.

The Claude Code exporter is the OpenTelemetry JS SDK, which sends OTLP/HTTP with
``Transfer-Encoding: chunked`` and **no** ``Content-Length`` — the receiver must
de-chunk the request body (a ``read(Content-Length)`` reader would capture
nothing). No Claude Code upstream source changes are required.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import threading
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

_RAW_CHUNK_BYTES = 512 * 1024

# OTLP/HTTP signal → request path. Claude Code (OTel JS SDK) appends these to
# OTEL_EXPORTER_OTLP_ENDPOINT.
_SIGNAL_BY_PATH = {"/v1/logs": "logs", "/v1/metrics": "metrics"}


class ClaudeCodeOtelCollector:
    """Receive Claude Code OTLP/HTTP JSON and persist complete run telemetry."""

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
        """Base OTLP endpoint; the CLI appends ``/v1/logs`` and ``/v1/metrics``."""
        if self._server is None:
            raise RuntimeError("Claude Code OTel collector has not been started")
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
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def _read_body(self) -> bytes:
                # The OTel JS exporter streams the body with
                # Transfer-Encoding: chunked and no Content-Length, so a plain
                # read(Content-Length) reads zero bytes. Handle both framings.
                if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                    return _read_chunked_body(self.rfile)
                length = int(self.headers.get("Content-Length", "0"))
                return self.rfile.read(length)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="claude-code-otel-collector",
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
        received_at = datetime.now(timezone.utc).isoformat()
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None

        events: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            if signal == "logs":
                events = _flatten_otlp_logs(payload, received_at)
            elif signal == "metrics":
                metrics = _flatten_otlp_metrics(payload, received_at)

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
            if metrics:
                with self.metrics_path.open("a", encoding="utf-8") as file:
                    for point in metrics:
                        file.write(json.dumps(point, ensure_ascii=False) + "\n")

    def _write_derived(
        self,
        events: list[dict[str, Any]] | None = None,
        metrics: list[dict[str, Any]] | None = None,
    ) -> None:
        if events is None:
            events = _read_jsonl(self.events_path)
        if metrics is None:
            metrics = _read_jsonl(self.metrics_path)
        events.sort(key=_event_sort_key)

        names = Counter(
            event.get("attributes", {}).get("event.name", "unknown") for event in events
        )
        summary = {
            "event_count": len(events),
            "events_by_name": dict(sorted(names.items())),
            "api_calls": _summarize_api_calls(events),
            "api_errors": _summarize_api_errors(events),
            "tool_executions": _summarize_tools(events),
            "tool_decisions": _summarize_tool_decisions(events),
            "user_prompts": _summarize_prompts(events),
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


def recover_telemetry_artifacts(work_dir: Path) -> int:
    """Rebuild derived telemetry files from the incrementally mirrored raw WAL.

    Returns the number of recovered log events. Used at finalization so a task
    whose final directory gather was interrupted still yields complete telemetry
    from the incrementally-pulled ``otel_requests.jsonl``.
    """
    raw_path = work_dir / "otel_requests.jsonl"
    if not raw_path.exists():
        return 0

    events: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for signal, received_at, payload in _read_otlp_payloads(raw_path):
        if signal == "metrics":
            metrics.extend(_flatten_otlp_metrics(payload, received_at))
        else:
            events.extend(_flatten_otlp_logs(payload, received_at))

    events.sort(key=_event_sort_key)
    _write_jsonl_atomic(work_dir / "telemetry.jsonl", events)
    _write_jsonl_atomic(work_dir / "telemetry_metrics.jsonl", metrics)
    collector = ClaudeCodeOtelCollector(work_dir)
    collector._write_derived(events, metrics)
    return len(events)


def _read_chunked_body(rfile: Any) -> bytes:
    """Decode an HTTP/1.1 ``Transfer-Encoding: chunked`` request body."""
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
            rfile.readline()  # consume trailing CRLF after the last chunk
            break
        body += rfile.read(size)
        rfile.readline()  # consume the CRLF terminating this chunk
    return bytes(body)


def _read_otlp_payloads(path: Path) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(signal, received_at, payload)`` for each complete WAL request."""
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
                "signal": record.get("signal", "logs"),
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
) -> tuple[str, str, dict[str, Any]] | None:
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
    return str(state.get("signal", "logs")), state["received_at"], payload


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


def _flatten_otlp_metrics(payload: dict[str, Any], received_at: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for resource_metric in payload.get("resourceMetrics", []):
        resource = _attributes(
            resource_metric.get("resource", {}).get("attributes", [])
        )
        for scope_metric in resource_metric.get("scopeMetrics", []):
            scope_name = scope_metric.get("scope", {}).get("name")
            for metric in scope_metric.get("metrics", []):
                name = metric.get("name")
                unit = metric.get("unit")
                for kind in ("sum", "gauge", "histogram"):
                    container = metric.get(kind)
                    if not isinstance(container, dict):
                        continue
                    for data_point in container.get("dataPoints", []):
                        points.append(
                            {
                                "received_at": received_at,
                                "metric": name,
                                "kind": kind,
                                "unit": unit,
                                "value": _data_point_value(data_point),
                                "attributes": _attributes(
                                    data_point.get("attributes", [])
                                ),
                                "time_unix_nano": data_point.get("timeUnixNano"),
                                "start_time_unix_nano": data_point.get(
                                    "startTimeUnixNano"
                                ),
                                "scope": scope_name,
                                "resource": resource,
                            }
                        )
    return points


def _data_point_value(data_point: dict[str, Any]) -> Any:
    if "asInt" in data_point:
        try:
            return int(data_point["asInt"])
        except (TypeError, ValueError):
            return data_point["asInt"]
    if "asDouble" in data_point:
        return data_point["asDouble"]
    # Histogram data points carry sum/count rather than a scalar value.
    if "sum" in data_point or "count" in data_point:
        return {"sum": data_point.get("sum"), "count": data_point.get("count")}
    return None


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


def _event_sort_key(event: dict[str, Any]) -> tuple[int, int, str]:
    """Order events by Claude Code's monotonic ``event.sequence`` then time."""
    attributes = event.get("attributes", {})
    sequence = attributes.get("event.sequence")
    has_sequence = 0 if isinstance(sequence, int) else 1
    return (
        has_sequence,
        sequence if isinstance(sequence, int) else 0,
        attributes.get("event.timestamp")
        or event.get("time_unix_nano")
        or event.get("observed_time_unix_nano")
        or "",
    )


def _summarize_api_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for event in events:
        attributes = event.get("attributes", {})
        if attributes.get("event.name") != "api_request":
            continue
        completed_at = attributes.get("event.timestamp")
        duration_ms = _int_or_none(attributes.get("duration_ms"))
        started_at = None
        parsed = _parse_timestamp(completed_at)
        if parsed is not None and duration_ms is not None:
            started_at = parsed.timestamp() * 1000 - duration_ms
        calls.append(
            {
                "sequence": len(calls) + 1,
                "started_at": _format_epoch_ms(started_at),
                "completed_at": completed_at,
                "duration_ms": duration_ms,
                "model": attributes.get("model"),
                "request_id": attributes.get("request_id"),
                "query_source": attributes.get("query_source"),
                "input_tokens": _int_or_none(attributes.get("input_tokens")),
                "output_tokens": _int_or_none(attributes.get("output_tokens")),
                "cache_read_tokens": _int_or_none(attributes.get("cache_read_tokens")),
                "cache_creation_tokens": _int_or_none(
                    attributes.get("cache_creation_tokens")
                ),
                "cost_usd": attributes.get("cost_usd"),
                "effort": attributes.get("effort"),
                "prompt_id": attributes.get("prompt.id"),
                "event": event,
            }
        )
    return calls


def _summarize_api_errors(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for event in events:
        attributes = event.get("attributes", {})
        if attributes.get("event.name") != "api_error":
            continue
        errors.append(
            {
                "sequence": len(errors) + 1,
                "timestamp": attributes.get("event.timestamp"),
                "model": attributes.get("model"),
                "error": attributes.get("error") or attributes.get("error_type"),
                "status_code": attributes.get("status_code"),
                "attempt": _int_or_none(attributes.get("attempt")),
                "duration_ms": _int_or_none(attributes.get("duration_ms")),
                "request_id": attributes.get("request_id"),
                "event": event,
            }
        )
    return errors


def _summarize_tools(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for event in events:
        attributes = event.get("attributes", {})
        if attributes.get("event.name") != "tool_result":
            continue
        completed_at = attributes.get("event.timestamp")
        duration_ms = _int_or_none(attributes.get("duration_ms"))
        started_at = None
        parsed = _parse_timestamp(completed_at)
        if parsed is not None and duration_ms is not None:
            started_at = parsed.timestamp() * 1000 - duration_ms
        tools.append(
            {
                "sequence": len(tools) + 1,
                "tool_name": attributes.get("tool_name"),
                "tool_use_id": attributes.get("tool_use_id"),
                "started_at": _format_epoch_ms(started_at),
                "completed_at": completed_at,
                "duration_ms": duration_ms,
                "success": attributes.get("success"),
                "error_type": attributes.get("error_type"),
                # Tool arguments (present with OTEL_LOG_TOOL_DETAILS=1). The CLI
                # emits them under `tool_input`; `tool_parameters` is accepted as a
                # fallback for other/older builds.
                "tool_input": attributes.get("tool_input")
                or attributes.get("tool_parameters"),
                "tool_input_size_bytes": _int_or_none(
                    attributes.get("tool_input_size_bytes")
                ),
                "tool_result_size_bytes": _int_or_none(
                    attributes.get("tool_result_size_bytes")
                ),
                "event": event,
            }
        )
    return tools


def _summarize_tool_decisions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for event in events:
        attributes = event.get("attributes", {})
        if attributes.get("event.name") != "tool_decision":
            continue
        decisions.append(
            {
                "sequence": len(decisions) + 1,
                "timestamp": attributes.get("event.timestamp"),
                "tool_name": attributes.get("tool_name"),
                "decision": attributes.get("decision"),
                "source": attributes.get("source"),
                "tool_use_id": attributes.get("tool_use_id"),
            }
        )
    return decisions


def _summarize_prompts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for event in events:
        attributes = event.get("attributes", {})
        if attributes.get("event.name") != "user_prompt":
            continue
        prompts.append(
            {
                "sequence": len(prompts) + 1,
                "timestamp": attributes.get("event.timestamp"),
                "prompt_id": attributes.get("prompt.id"),
                "prompt_length": _int_or_none(attributes.get("prompt_length")),
                "command_name": attributes.get("command_name"),
                "prompt": attributes.get("prompt"),
            }
        )
    return prompts


def _summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce metric data points to the latest cumulative value per series.

    Claude Code metrics are monotonic cumulative sums, re-sent every export
    interval, so the final value for each (metric, attribute) series is the
    run total. Keyed on the sorted attribute tuple.
    """
    latest: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}
    order: list[tuple[str, tuple[tuple[str, Any], ...]]] = []
    for point in metrics:
        name = point.get("metric")
        if not name:
            continue
        attr_key = tuple(sorted(point.get("attributes", {}).items()))
        key = (name, attr_key)
        if key not in latest:
            order.append(key)
        latest[key] = point.get("value")

    tokens_by_type: dict[str, int] = defaultdict(int)
    cost_total = 0.0
    series: list[dict[str, Any]] = []
    for name, attr_key in order:
        value = latest[(name, attr_key)]
        attributes = dict(attr_key)
        series.append({"metric": name, "attributes": attributes, "value": value})
        if name == "claude_code.token.usage" and isinstance(value, (int, float)):
            tokens_by_type[str(attributes.get("type", "unknown"))] += int(value)
        elif name == "claude_code.cost.usage" and isinstance(value, (int, float)):
            cost_total += float(value)

    return {
        "series": series,
        "tokens_by_type": dict(tokens_by_type),
        "total_cost_usd": round(cost_total, 6) if cost_total else cost_total,
    }


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
