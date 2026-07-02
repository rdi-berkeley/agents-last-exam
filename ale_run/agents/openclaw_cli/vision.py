"""OpenClaw-specific vision usage capture and transcript image artifacts."""
from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import json
import logging
import mimetypes
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


logger = logging.getLogger(__name__)

_DATA_IMAGE_RE = re.compile(
    r"data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\r\n]+)"
)
_HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
})
_USAGE_FIELDS = (
    "requests_with_usage",
    "input_tokens",
    "input_write_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "output_tokens",
)


class VisionUsageProxy:
    """Transparent loopback proxy that records provider response usage."""

    def __init__(
        self,
        *,
        upstream_url: str,
        usage_log: Path,
        provider: str,
        model: str,
    ) -> None:
        parsed = urlsplit(upstream_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(f"unsupported vision upstream URL: {upstream_url}")
        self._parsed = parsed
        self._usage_log = usage_log
        self._provider = provider
        self._model = model
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._write_lock = threading.Lock()

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("vision usage proxy is not running")
        base_path = self._parsed.path.rstrip("/")
        return f"http://127.0.0.1:{self._server.server_port}{base_path}"

    def start(self) -> None:
        self._usage_log.unlink(missing_ok=True)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                return

            def do_GET(self) -> None:
                self._forward()

            def do_POST(self) -> None:
                self._forward()

            def _forward(self) -> None:
                content_length = int(self.headers.get("content-length", "0"))
                body = self.rfile.read(content_length) if content_length else None
                headers = {
                    name: value
                    for name, value in self.headers.items()
                    if name.lower() not in _HOP_BY_HOP_HEADERS
                }
                headers["Accept-Encoding"] = "identity"
                connection = http.client.HTTPSConnection(
                    proxy._parsed.hostname,
                    proxy._parsed.port or 443,
                    timeout=120,
                )
                try:
                    connection.request(
                        self.command,
                        self.path,
                        body=body,
                        headers=headers,
                    )
                    response = connection.getresponse()
                    response_body = response.read()
                    self.send_response(response.status, response.reason)
                    for name, value in response.getheaders():
                        lowered = name.lower()
                        if (
                            lowered not in _HOP_BY_HOP_HEADERS
                            and lowered != "content-encoding"
                        ):
                            self.send_header(name, value)
                    self.send_header("Content-Length", str(len(response_body)))
                    self.end_headers()
                    self.wfile.write(response_body)
                    proxy._record_usage(response_body)
                except Exception as exc:
                    payload = json.dumps({
                        "error": {
                            "message": f"ALE vision proxy failed: {exc}",
                            "type": "ale_vision_proxy_error",
                        }
                    }).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                finally:
                    connection.close()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="ale-openclaw-vision-usage",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def _record_usage(self, response_body: bytes) -> None:
        usage = extract_provider_usage(response_body)
        if not usage:
            return
        record = {
            "provider": self._provider,
            "model": self._model,
            **usage,
        }
        with self._write_lock:
            with self._usage_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _first_nonnegative_int(mapping: dict, *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
        ):
            return int(value)
    return 0


def extract_provider_usage(body: bytes) -> dict[str, int] | None:
    """Extract aggregate usage from JSON or SSE provider responses."""
    text = body.decode("utf-8", errors="replace")
    payloads: list[object] = []
    try:
        payloads.append(json.loads(text))
    except json.JSONDecodeError:
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payloads.append(json.loads(data))
            except json.JSONDecodeError:
                continue

    candidates: list[dict] = []

    def _walk(value: object) -> None:
        if isinstance(value, dict):
            usage = value.get("usage")
            if isinstance(usage, dict):
                candidates.append(usage)
            for child in value.values():
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    for payload in payloads:
        _walk(payload)
    if not candidates:
        return None

    usage = candidates[-1]
    input_tokens = _first_nonnegative_int(
        usage,
        "input_tokens",
        "prompt_tokens",
        "input",
    )
    output_tokens = _first_nonnegative_int(
        usage,
        "output_tokens",
        "completion_tokens",
        "output",
    )
    details = (
        usage.get("input_tokens_details")
        or usage.get("prompt_tokens_details")
        or {}
    )
    cache_read = (
        _first_nonnegative_int(details, "cached_tokens")
        if isinstance(details, dict)
        else 0
    )
    if not cache_read:
        cache_read = _first_nonnegative_int(
            usage,
            "cacheRead",
            "cache_read_tokens",
        )
    cache_creation = _first_nonnegative_int(
        usage,
        "cacheWrite",
        "cache_creation_tokens",
    )
    if not any((input_tokens, output_tokens, cache_read, cache_creation)):
        return None
    return {
        "requests_with_usage": 1,
        "input_tokens": input_tokens,
        "input_write_tokens": max(0, input_tokens - cache_read),
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
        "output_tokens": output_tokens,
    }


def read_image_model_usage(path: Path) -> dict | None:
    if not path.is_file():
        return None
    totals = {key: 0 for key in _USAGE_FIELDS}
    provider = None
    model = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        provider = record.get("provider") or provider
        model = record.get("model") or model
        for key in totals:
            totals[key] += _first_nonnegative_int(record, key)
    if not any(totals.values()):
        return None
    return {
        "provider": provider,
        "model": model,
        **totals,
    }


def run_dir_for_artifacts(work_dir: Path) -> Path:
    if work_dir.parent.name == "origin_log":
        return work_dir.parent.parent
    return work_dir


def stage_transcript_file_images(
    transcript_file: Path,
    work_dir: Path,
) -> None:
    """Copy image-tool file inputs into the gathered artifact directory."""
    workspace = Path.home() / ".openclaw" / "workspace"
    path_map: dict[str, str] = {}
    changed = False

    def _stage(value: object) -> object:
        nonlocal changed
        if isinstance(value, dict):
            return {key: _stage(child) for key, child in value.items()}
        if isinstance(value, list):
            return [_stage(child) for child in value]
        if not isinstance(value, str):
            return value
        if (
            value.startswith(("data:image/", "http://", "https://"))
            or value.startswith("screenshots/")
        ):
            return value
        cached = path_map.get(value)
        if cached:
            return cached
        source_path = Path(value)
        candidates = (
            [source_path]
            if source_path.is_absolute()
            else [workspace / source_path, work_dir / source_path]
        )
        source = next(
            (candidate for candidate in candidates if candidate.is_file()),
            None,
        )
        if source is None:
            return value
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        suffix = source.suffix.lower() or ".png"
        relative = f"screenshots/openclaw-input-{digest}{suffix}"
        destination = work_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        path_map[value] = relative
        changed = True
        return relative

    output: list[str] = []
    for line in transcript_file.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            output.append(line)
            continue
        message = event.get("message") if isinstance(event, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") in ("toolCall", "tool_use")
                    and block.get("name") == "image"
                ):
                    key = "arguments" if "arguments" in block else "input"
                    block[key] = _stage(block.get(key))
        if isinstance(message, dict) and message.get("toolName") == "image":
            message["details"] = _stage(message.get("details"))
        output.append(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
    if changed:
        transcript_file.write_text("\n".join(output) + "\n", encoding="utf-8")


def persist_transcript_images(
    transcript_file: Path,
    run_dir: Path,
) -> int:
    """Persist transcript image payloads and replace them with path refs."""
    screenshots_dir = run_dir / "screenshots"
    staged_dir = transcript_file.parent / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    if staged_dir.is_dir() and staged_dir != screenshots_dir:
        for source in staged_dir.iterdir():
            if not source.is_file():
                continue
            destination = screenshots_dir / source.name
            if not destination.exists():
                shutil.move(str(source), str(destination))
                written += 1
            else:
                source.unlink()
        try:
            staged_dir.rmdir()
        except OSError as exc:
            logger.warning(
                "openclaw_cli: failed to remove staged screenshot directory "
                "%s: %s",
                staged_dir,
                exc,
            )

    known = {
        hashlib.sha256(path.read_bytes()).hexdigest(): f"screenshots/{path.name}"
        for path in screenshots_dir.iterdir()
        if path.is_file()
    }

    def _store(media_type: str, encoded: str) -> str:
        nonlocal written
        payload = encoded.strip()
        payload += "=" * (-len(payload) % 4)
        try:
            raw = base64.b64decode(payload)
        except (binascii.Error, ValueError):
            return ""
        digest = hashlib.sha256(raw).hexdigest()
        if digest in known:
            return known[digest]
        extension = mimetypes.guess_extension(media_type) or ".png"
        relative = f"screenshots/openclaw-inline-{digest[:16]}{extension}"
        (run_dir / relative).write_bytes(raw)
        known[digest] = relative
        written += 1
        return relative

    def _rewrite(value: object) -> object:
        if isinstance(value, dict):
            if (
                value.get("type") == "image"
                and isinstance(value.get("data"), str)
            ):
                media_type = str(value.get("mimeType") or "image/png")
                relative = _store(media_type, value["data"])
                if relative:
                    value = dict(value)
                    value.pop("data", None)
                    value["path"] = relative
            return {key: _rewrite(child) for key, child in value.items()}
        if isinstance(value, list):
            return [_rewrite(child) for child in value]
        if not isinstance(value, str):
            return value

        def _replace(match: re.Match[str]) -> str:
            return _store(match.group(1), match.group(2)) or match.group(0)

        return _DATA_IMAGE_RE.sub(_replace, value)

    changed = False
    output: list[str] = []
    for line in transcript_file.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            output.append(line)
            continue
        rewritten = _rewrite(event)
        serialized = json.dumps(
            rewritten,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        changed = changed or serialized != line
        output.append(serialized)
    if changed:
        transcript_file.write_text("\n".join(output) + "\n", encoding="utf-8")
    return written
