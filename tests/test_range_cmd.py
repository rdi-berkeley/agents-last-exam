"""Unit tests for the Windows download_range command + range stdout parsing.

- _build_range_cmd_windows must use -EncodedCommand (cmd.exe mangled the old
  -Command "<nested quotes>" form, so download_range always failed on Windows).
- _parse_range_stdout must map SIZE=-1 (file absent) to a non-error result
  (success=True, new_size=-1) so the tail can tell "no file yet" from a real
  transport failure; SIZE=-2 (remote exception) stays an error.

Run: ``python tests/test_range_cmd.py``.
"""
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ale_run.base_interface import sandbox as S


def test_build_cmd_is_encodedcommand_and_quote_free():
    cmd = S._build_range_cmd_windows("C:\\a b\\transcript.jsonl", 1234, 4 * 1024 * 1024)
    assert cmd.startswith("powershell -NoProfile -EncodedCommand ")
    assert '"' not in cmd and "'" not in cmd, "command line must be quote-free (cmd.exe-safe)"
    b64 = cmd.split()[-1]
    script = base64.b64decode(b64).decode("utf-16-le")
    # the real logic survived the encoding
    assert "ToBase64String" in script
    assert "1234" in script  # the offset
    assert "SIZE=" in script


def test_parse_file_absent_is_not_an_error():
    r = S._parse_range_stdout("SIZE=-1\r\nB64=\r\n", expected_start=0)
    assert r.success is True and r.new_size == -1, (r.success, r.new_size)


def test_parse_remote_exception_is_error():
    r = S._parse_range_stdout("SIZE=-2\r\nB64=\r\nERR=boom\r\n", expected_start=0)
    assert r.success is False and "boom" in (r.error or "")


def test_parse_no_new_bytes():
    r = S._parse_range_stdout("SIZE=500\nB64=\n", expected_start=500)
    assert r.success is True and r.new_size == 500 and r.new_data == b""


def test_parse_data_roundtrip():
    payload = b'{"x":1}\n'
    b64 = base64.b64encode(payload).decode()
    r = S._parse_range_stdout(f"SIZE=8\nB64={b64}\n", expected_start=0)
    assert r.success is True and r.new_size == 8 and r.new_data == payload


def test_parse_empty_and_malformed():
    assert S._parse_range_stdout("", expected_start=0).success is False
    assert S._parse_range_stdout("B64=abc\n", expected_start=0).success is False  # missing SIZE
    assert S._parse_range_stdout("SIZE=notanint\n", expected_start=0).success is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_range_cmd: ALL PASS")
