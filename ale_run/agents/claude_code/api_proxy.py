"""Local Anthropic-Messages proxy for gateways that only partially implement the API.

Runs inside the sandbox next to the Claude Code CLI (``ANTHROPIC_BASE_URL`` points
here) and forwards to an upstream Anthropic-compatible endpoint, adding fixed
headers (e.g. a Cloudflare Access service token, a session-affinity id). Three
fixes for self-hosted gateways that implement the Messages API only partially
(observed with an SGLang model gateway):

* ``document`` content blocks (the CLI's Read tool on a PDF) are rejected by the
  gateway -> replaced by a text block with the PDF's extracted text (pypdf when
  importable) or a short note.
* Requests are sent upstream NON-streaming and, when the client asked for a
  stream, re-emitted as the standard SSE event sequence. That makes the whole
  sample retryable: a "Failed to parse model output" (malformed tool JSON) or
  5xx/429 answer is re-sampled up to ``RETRIES`` times instead of killing the
  CLI session.
* Any other path (``count_tokens`` ...) is passed through unchanged.

Stdlib only. Usage: ``python -m ale_run.agents.claude_code.api_proxy --port N
--upstream https://host [--header 'Name: value' ...] [--ready-file PATH]``.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("api_proxy")

RETRIES = 4
RETRY_MARKERS = ("failed to parse model output", "json_syntax", "overloaded", "no worker available")
NORMALIZE_MARKERS = ("did not match any variant", "invalid json data", "failed to deserialize")
PDF_TEXT_CAP = 150_000
UPSTREAM_TIMEOUT_S = 1800

_HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}


def _pdf_text(data: bytes) -> str | None:
    try:
        from pypdf import PdfReader  # optional; installed by the deployer when possible
    except Exception:
        return None
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for i, page in enumerate(reader.pages):
            parts.append(f"--- page {i + 1} ---\n{page.extract_text() or ''}")
        return "\n".join(parts)
    except Exception as e:  # corrupt / encrypted
        return f"[PDF text extraction failed: {e}]"


def _convert_document(block: dict) -> dict:
    src = block.get("source") or {}
    media = src.get("media_type", "")
    title = block.get("title") or ""
    data = src.get("data")
    text = None
    if src.get("type") == "base64" and isinstance(data, str):
        try:
            raw = base64.b64decode(data)
        except Exception:
            raw = b""
        if media == "application/pdf":
            text = _pdf_text(raw)
        elif media.startswith("text/"):
            text = raw.decode("utf-8", "replace")
    elif src.get("type") == "text":
        text = src.get("data")
    if text is None:
        text = ("[The document could not be attached to the conversation (the model endpoint "
                "does not accept document blocks). Extract its text with a tool, e.g. Python "
                "(pypdf/pdfplumber) or `pdftotext`, and read the result.]")
    if len(text) > PDF_TEXT_CAP:
        text = text[:PDF_TEXT_CAP] + f"\n[... truncated at {PDF_TEXT_CAP} characters]"
    head = f"[Document {title}: {media or 'unknown type'}]\n" if title or media else "[Document]\n"
    return {"type": "text", "text": head + text}


def _strip_cache_control(obj):
    if isinstance(obj, dict):
        obj.pop("cache_control", None)
        for v in obj.values():
            _strip_cache_control(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_cache_control(v)
    return obj


def _normalize(body: dict) -> dict:
    """Conservative shape fallback for gateways whose request parser is stricter
    than Anthropic's: drop cache_control everywhere, flatten tool_result
    contents that are pure text lists into a string, and never send an empty
    content list."""
    body = _strip_cache_control(body)
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            if not content:
                msg["content"] = "(empty)"
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, list):
                        if not inner:
                            block["content"] = ""
                        elif all(isinstance(b, dict) and b.get("type") == "text" for b in inner):
                            block["content"] = "\n".join(b.get("text", "") for b in inner)
                elif block.get("type") == "text" and not block.get("text"):
                    block["text"] = " "
    return body


def _redacted(body: dict) -> dict:
    """Copy of the request with base64 payloads dropped (for diagnostics)."""
    def walk(o):
        if isinstance(o, dict):
            return {k: (f"<{len(v)} base64 chars>" if k == "data" and isinstance(v, str) and len(v) > 256 else walk(v))
                    for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    return walk(body)


def _watch_parent(pid: int) -> None:
    """Exit when the parent (the sandbox worker) is gone, so a leftover proxy
    never keeps the supervisor's process tree alive."""
    def alive() -> bool:
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            try:
                code = ctypes.c_ulong()
                if not k.GetExitCodeProcess(h, ctypes.byref(code)):
                    return False
                return code.value == 259  # STILL_ACTIVE
            finally:
                k.CloseHandle(h)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def loop():
        while True:
            time.sleep(5)
            if not alive():
                logger.info("parent %d gone; exiting", pid)
                os._exit(0)

    threading.Thread(target=loop, name="parent-watchdog", daemon=True).start()


def _sanitize(body: dict) -> dict:
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                block["content"] = [
                    _convert_document(b) if isinstance(b, dict) and b.get("type") == "document" else b
                    for b in block["content"]
                ]
            elif isinstance(block, dict) and block.get("type") == "document":
                content[i] = _convert_document(block)
    return body


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _to_sse(msg: dict) -> bytes:
    """Re-emit a complete Messages response as the standard streaming event sequence."""
    out = []
    usage = dict(msg.get("usage") or {})
    start = {k: v for k, v in msg.items() if k != "content"}
    start["content"] = []
    start["stop_reason"] = None
    start["stop_sequence"] = None
    start["usage"] = {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0,
                      **{k: v for k, v in usage.items() if k.startswith("cache_")}}
    out.append(_sse("message_start", {"type": "message_start", "message": start}))
    for idx, block in enumerate(msg.get("content") or []):
        btype = block.get("type")
        if btype == "text":
            out.append(_sse("content_block_start", {"type": "content_block_start", "index": idx,
                                                    "content_block": {"type": "text", "text": ""}}))
            out.append(_sse("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                    "delta": {"type": "text_delta", "text": block.get("text", "")}}))
        elif btype == "thinking":
            out.append(_sse("content_block_start", {"type": "content_block_start", "index": idx,
                                                    "content_block": {"type": "thinking", "thinking": ""}}))
            out.append(_sse("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                    "delta": {"type": "thinking_delta", "thinking": block.get("thinking", "")}}))
            if block.get("signature"):
                out.append(_sse("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                        "delta": {"type": "signature_delta", "signature": block["signature"]}}))
        elif btype == "redacted_thinking":
            out.append(_sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": block}))
        elif btype == "tool_use":
            out.append(_sse("content_block_start", {"type": "content_block_start", "index": idx,
                                                    "content_block": {"type": "tool_use", "id": block.get("id"),
                                                                      "name": block.get("name"), "input": {}}}))
            out.append(_sse("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                    "delta": {"type": "input_json_delta",
                                                              "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False)}}))
        else:
            out.append(_sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": block}))
        out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": idx}))
    out.append(_sse("message_delta", {"type": "message_delta",
                                      "delta": {"stop_reason": msg.get("stop_reason"), "stop_sequence": msg.get("stop_sequence")},
                                      "usage": {"output_tokens": usage.get("output_tokens", 0)}}))
    out.append(_sse("message_stop", {"type": "message_stop"}))
    return b"".join(out)


class Proxy(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, upstream: str, headers: dict[str, str], diag_dir: str | None = None):
        super().__init__(addr, Handler)
        self.upstream = upstream.rstrip("/")
        self.fixed_headers = headers
        self.diag_dir = diag_dir
        self.stats = {"requests": 0, "retries": 0, "documents": 0, "errors": 0, "normalized": 0}
        self._diag_lock = threading.Lock()

    def dump_diag(self, status: int, payload: dict, resp: bytes) -> None:
        if not self.diag_dir:
            return
        try:
            os.makedirs(self.diag_dir, exist_ok=True)
            with self._diag_lock:
                n = self.stats["errors"]
                path = os.path.join(self.diag_dir, f"bad_{n:03d}_{status}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"status": status, "response": resp.decode("utf-8", "replace")[:4000],
                               "request": _redacted(payload)}, f, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("diag dump failed: %s", e)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: Proxy

    def handle_error(self, request, client_address):
        pass

    def log_message(self, fmt, *args):  # quiet
        logger.debug(fmt, *args)

    def _upstream(self, method: str, path: str, body: bytes | None, headers: dict[str, str]):
        req = urllib.request.Request(self.server.upstream + path, data=body, method=method)
        for k, v in headers.items():
            if k.lower() not in _HOP_HEADERS:
                req.add_header(k, v)
        for k, v in self.server.fixed_headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def _reply(self, status: int, body: bytes, content_type: str = "application/json", extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        headers = {k: v for k, v in self.headers.items()}
        self.server.stats["requests"] += 1
        route = self.path.split("?", 1)[0].rstrip("/")   # the CLI calls /v1/messages?beta=true
        if not route.endswith("/messages"):
            status, h, body = self._upstream("POST", self.path, raw, headers)
            self._reply(status, body, h.get("Content-Type", "application/json"))
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            status, h, body = self._upstream("POST", self.path, raw, headers)
            self._reply(status, body, h.get("Content-Type", "application/json"))
            return
        n_docs = raw.count(b'"document"')
        payload = _sanitize(payload)
        if n_docs:
            self.server.stats["documents"] += n_docs
        want_stream = bool(payload.get("stream"))
        payload["stream"] = False
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        status = 500; h = {}; resp = b""
        normalized = False
        for attempt in range(RETRIES + 1):
            status, h, resp = self._upstream("POST", self.path, body, headers)
            if status == 200:
                break
            text = resp.decode("utf-8", "replace").lower()
            if status == 400 and not normalized and any(m in text for m in NORMALIZE_MARKERS):
                # Stricter-than-Anthropic request parser: retry once with the
                # conservative shape (no cache_control, string tool_result text).
                normalized = True
                self.server.stats["normalized"] += 1
                logger.warning("upstream 400 shape error, retrying normalized: %s", text[:200])
                payload = _normalize(payload)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                continue
            retryable = status in (429, 500, 502, 503, 504, 529) or any(m in text for m in RETRY_MARKERS)
            if not retryable or attempt == RETRIES:
                break
            self.server.stats["retries"] += 1
            logger.warning("upstream %s (attempt %d/%d): %s", status, attempt + 1, RETRIES, text[:200])
            time.sleep(min(2 ** attempt, 20))
        if status != 200:
            self.server.stats["errors"] += 1
            self.server.dump_diag(status, payload, resp)
            self._reply(status, resp, h.get("Content-Type", "application/json"))
            return
        if not want_stream:
            self._reply(200, resp, "application/json", {"request-id": h.get("request-id", "")})
            return
        try:
            msg = json.loads(resp.decode("utf-8"))
        except Exception:
            self._reply(502, json.dumps({"type": "error", "error": {"type": "api_error", "message": "proxy: upstream returned non-JSON"}}).encode())
            return
        sse = _to_sse(msg)
        self._reply(200, sse, "text/event-stream", {"Cache-Control": "no-cache", "request-id": h.get("request-id", "")})

    def do_GET(self):
        if self.path == "/__proxy/health":
            self._reply(200, json.dumps({"ok": True, **self.server.stats}).encode())
            return
        status, h, body = self._upstream("GET", self.path, None, {k: v for k, v in self.headers.items()})
        self._reply(status, body, h.get("Content-Type", "application/json"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--header", action="append", default=[], help="'Name: value' added to every upstream request")
    ap.add_argument("--ready-file")
    ap.add_argument("--parent-pid", type=int)
    ap.add_argument("--diag-dir")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s api_proxy: %(message)s", stream=sys.stderr)
    headers = {}
    for item in args.header:
        name, _, value = item.partition(":")
        headers[name.strip()] = value.strip()
    srv = Proxy(("127.0.0.1", args.port), args.upstream, headers, diag_dir=args.diag_dir)
    if args.parent_pid:
        _watch_parent(args.parent_pid)
    logger.info("listening on 127.0.0.1:%d -> %s (%d fixed headers)", args.port, srv.upstream, len(headers))
    if args.ready_file:
        with open(args.ready_file, "w") as f:
            f.write(str(args.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
