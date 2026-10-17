from __future__ import annotations

import http.client
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ale_run.agents.openclaw_cli import vision


@pytest.mark.parametrize("record_usage", [False, True])
def test_proxy_delivers_sse_before_upstream_finishes(tmp_path, monkeypatch, record_usage):
    release = threading.Event()
    first = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    last = b'data: {"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\ndata: [DONE]\n\n'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(first)
            self.wfile.flush()
            release.wait(timeout=5)
            self.wfile.write(last)
            self.wfile.flush()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    upstream.daemon_threads = True
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    monkeypatch.setattr(
        vision.http.client,
        "HTTPSConnection",
        lambda *args, **kwargs: http.client.HTTPConnection(
            "127.0.0.1", upstream.server_port, timeout=3
        ),
    )
    usage_log = tmp_path / "usage.jsonl"
    proxy = vision.VisionUsageProxy(
        upstream_url="https://example.invalid/v1",
        usage_log=usage_log if record_usage else None,
        provider="test",
        model="test",
        sanitize_empty_text_blocks=True,
    )
    proxy.start()
    try:
        request = urllib.request.Request(
            proxy.base_url + "/chat/completions",
            data=b'{"messages":[],"stream":true}',
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=1) as response:
            assert response.readline() == first.splitlines(keepends=True)[0]
            assert not release.is_set()
            release.set()
            assert response.read() == b"\n" + last
        if record_usage:
            assert json.loads(usage_log.read_text())["output_tokens"] == 2
        else:
            assert not usage_log.exists()
    finally:
        release.set()
        proxy.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=3)


@pytest.mark.parametrize("field", ["messages", "input"])
def test_sanitizer_changes_only_message_content(field):
    schema = {"type": "text", "text": ""}
    payload = {
        field: [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
                ],
            }
        ],
        "tools": [{"function": {"parameters": {"enum": [schema]}}}],
        "metadata": {"content": [], "items": [{"type": "text", "text": ""}]},
    }
    result = json.loads(vision.sanitize_openai_request_body(json.dumps(payload).encode()))
    assert result[field][0]["content"] == payload[field][0]["content"][1:]
    assert result["tools"] == payload["tools"]
    assert result["metadata"] == payload["metadata"]
