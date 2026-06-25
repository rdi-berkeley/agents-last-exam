"""Regression test for _read_first_sse_event.

The cua-server returns read_bytes/download_range results as a single multi-MB
SSE ``data:`` line. The old reader used ``requests.iter_lines`` (512-byte
default chunk) which is O(n^2) on such a line — a 5.6 MB reply took ~19s on a
live VM, blowing the gather budget and silently losing large transcripts +
screenshots. The reader must be O(n). Run: ``python tests/test_sse_read.py``.
"""
import base64
import json
import os
import sys
import time

# Import this worktree's ale_run (it's a PEP-420 namespace package merged with
# the editable-installed main checkout; put the worktree root first so the local
# copy under test wins regardless of cwd).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ale_run.base_interface import sandbox as S


class _MockResp:
    def __init__(self, body: bytes, net_chunk: int = 64 * 1024):
        self._body = body
        self._net = net_chunk

    def iter_content(self, chunk_size=1):  # noqa: ARG002 — emulate network slicing
        for i in range(0, len(self._body), self._net):
            yield self._body[i : i + self._net]


def _sse(d: dict) -> bytes:
    return b"data: " + json.dumps(d).encode() + b"\n\n"


def test_correctness_cases():
    assert S._read_first_sse_event(_MockResp(_sse({"success": True, "x": 1}))) == {"success": True, "x": 1}
    # leading comment + event lines before the data line
    assert S._read_first_sse_event(_MockResp(b": ka\nevent: message\n" + _sse({"n": 2}))) == {"n": 2}
    # CRLF endings
    crlf = b"event: message\r\ndata: " + json.dumps({"ok": 3}).encode() + b"\r\n\r\n"
    assert S._read_first_sse_event(_MockResp(crlf)) == {"ok": 3}
    # trailing unterminated data line (server closed right after)
    assert S._read_first_sse_event(_MockResp(b"data: " + json.dumps({"ok": 4}).encode())) == {"ok": 4}
    # first event only
    assert S._read_first_sse_event(_MockResp(_sse({"first": True}) + _sse({"second": True}))) == {"first": True}
    # no data line / malformed json -> None
    assert S._read_first_sse_event(_MockResp(b": ka\nevent: x\n\n")) is None
    assert S._read_first_sse_event(_MockResp(b"data: {nope\n\n")) is None


def test_large_payload_is_linear():
    # 16 MiB content -> ~22 MiB base64 on one data line, delivered in 64 KiB slices
    big = base64.b64encode(b"\0" * (16 * 1024 * 1024)).decode()
    body = b"data: " + json.dumps({"success": True, "content_b64": big}).encode() + b"\n\n"
    t = time.monotonic()
    r = S._read_first_sse_event(_MockResp(body))
    dt = time.monotonic() - t
    assert r["success"] is True
    assert len(r["content_b64"]) == len(big)
    # O(n): a 22 MB single line must parse well under a second (was ~35s via iter_lines)
    assert dt < 2.0, f"too slow: {dt:.2f}s (O(n^2) regression?)"


if __name__ == "__main__":
    test_correctness_cases()
    test_large_payload_is_linear()
    print("test_sse_read: ALL PASS")
