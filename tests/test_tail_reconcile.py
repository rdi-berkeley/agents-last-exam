"""Regression tests for tail_hot_artifacts reconcile robustness.

Before the fix, a failed pull returned the sentinel ``-2`` and the reconcile's
``if size == prev_size: break`` treated two repeated ``-2`` as a *stable size*
→ it returned ``None`` (success) and NO ``incremental_pull_final_failed`` event
was emitted, so a transcript that never reached the host was lost silently.
The reconcile must now report an error when pulls keep failing.

Run: ``python tests/test_tail_reconcile.py``.
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ale_run.executors import sandbox as S
from ale_run.base_interface.sandbox import RangeResult

S._TAIL_RECONCILE_DELAY_S = 0.0  # speed up the bounded reconcile retries


class _MockExec:
    def __init__(self, *, content=b"", fail=False, missing=False):
        self.content, self.fail, self.missing = content, fail, missing
        self.calls = 0

    async def download_range(self, *, src, start, max_bytes):
        self.calls += 1
        if self.fail:
            return RangeResult(success=False, error="simulated transport error")
        if self.missing:
            return RangeResult(success=True, new_size=-1)
        return RangeResult(success=True, new_size=len(self.content),
                           new_data=self.content[start:start + max_bytes])


def _run(content=b"", fail=False, missing=False):
    ex = _MockExec(content=content, fail=fail, missing=missing)
    d = tempfile.mkdtemp()
    dst = os.path.join(d, "transcript.jsonl")
    stop = asyncio.Event()
    stop.set()  # skip the live loop; exercise the reconcile directly
    err = asyncio.run(S.tail_hot_artifacts(
        executor=ex, targets=[("C:\\x\\transcript.jsonl", __import__("pathlib").Path(dst))],
        stop_event=stop,
    ))
    return err, dst, ex


def test_persistent_failure_is_reported():
    # THE BUG: repeated download_range failure must surface an error, not None.
    err, _, ex = _run(fail=True)
    assert err is not None, "persistent pull failure was silently swallowed (regression!)"
    assert "failed" in err.lower()
    assert ex.calls >= S._TAIL_RECONCILE_RETRIES + 1  # it actually retried, didn't break early


def test_success_mirrors_and_no_error():
    content = b'{"a":1}\n{"b":2}\n'
    err, dst, _ = _run(content=content)
    assert err is None, f"unexpected error: {err}"
    with open(dst, "rb") as f:
        got = f.read()
    assert got == content, f"mirror mismatch: {got!r}"


def test_missing_file_is_not_an_error():
    # remote file genuinely absent (-1) is a stable, non-error outcome.
    err, dst, _ = _run(missing=True)
    assert err is None, f"absent file wrongly reported as error: {err}"
    assert not os.path.exists(dst) or os.path.getsize(dst) == 0


def test_jsonl_boundary_safe():
    # a half-written trailing record (no newline) must NOT be committed.
    content = b'{"done":1}\n{"partial":'
    err, dst, _ = _run(content=content)
    assert err is None
    with open(dst, "rb") as f:
        got = f.read()
    assert got == b'{"done":1}\n', f"committed a partial line: {got!r}"


if __name__ == "__main__":
    test_persistent_failure_is_reported()
    test_success_mirrors_and_no_error()
    test_missing_file_is_not_an_error()
    test_jsonl_boundary_safe()
    print("test_tail_reconcile: ALL PASS")
