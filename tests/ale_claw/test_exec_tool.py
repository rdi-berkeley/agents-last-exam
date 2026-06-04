"""Tests for Remote-VM shell execution tool.

Covers ExecTool:
  - Registration & schema
  - Parameter validation (missing/empty command, cwd outside workspace)
  - Happy path (zero exit, non-zero exit, empty stdout)
  - cwd wrapping (Windows cd /d vs POSIX cd)
  - Output truncation (middle truncation with marker)
  - Timeout (client-side asyncio.wait_for; default, clamp, VM-still-running)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

from agent.tools.base import TOOL_REGISTRY
from ale_run.agents.ale_claw.harness.tools.tools_shell import (
    ExecTool,
    _DEFAULT_MAX_OUTPUT_CHARS,
    _DEFAULT_TIMEOUT_SECONDS,
    _MAX_TIMEOUT_SECONDS,
    _MIN_TIMEOUT_SECONDS,
    _resolve_timeout,
    _truncate_middle,
    _wrap_with_cwd,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeCommandResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def _make_interface(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    hang: bool = False,
    raise_exc: Exception | None = None,
) -> MagicMock:
    """Build a fake ``interface`` with ``run_command`` mocked.

    ``hang=True`` makes the coroutine never complete so ``asyncio.wait_for``
    must fire. ``raise_exc`` surfaces an RPC-layer failure.
    """
    iface = MagicMock()
    if hang:
        async def _never_returning(_cmd):
            await asyncio.Event().wait()  # never set
            raise AssertionError("unreachable")
        iface.run_command = _never_returning
    elif raise_exc is not None:
        iface.run_command = AsyncMock(side_effect=raise_exc)
    else:
        iface.run_command = AsyncMock(
            return_value=_FakeCommandResult(stdout=stdout, stderr=stderr, returncode=returncode)
        )
    return iface


# ---------------------------------------------------------------------------
# Registration & schema
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_exec_registered(self):
        assert "exec" in TOOL_REGISTRY and TOOL_REGISTRY["exec"] is ExecTool

    def test_exec_name(self):
        tool = ExecTool(_make_interface())
        assert tool.name == "exec"

    def test_exec_required_params(self):
        tool = ExecTool(_make_interface())
        assert tool.parameters["required"] == ["command"]

    def test_exec_schema_properties(self):
        tool = ExecTool(_make_interface())
        props = tool.parameters["properties"]
        for key in ("command", "cwd", "timeout"):
            assert key in props

    def test_exec_in_build_tools_default_list(self):
        from unittest.mock import patch

        from ale_run.agents.ale_claw.harness.tools.tools import build_tools

        session = MagicMock()
        session.interface = MagicMock()
        session._computer = object()

        with patch(
            "ale_run.agents.ale_claw.harness.tools.analyze_image.AnalyzeImageTool"
        ), patch("ale_run.agents.ale_claw.harness.tools.milestone.MilestoneTool"):
            tools = build_tools(
                session,
                MagicMock(),
                summary_model="anthropic/claude-sonnet-4-6-20260101",
            )
        assert any(isinstance(t, ExecTool) for t in tools)


# ---------------------------------------------------------------------------
# Helpers — _truncate_middle / _wrap_with_cwd / _resolve_timeout
# ---------------------------------------------------------------------------


class TestTruncateMiddle:
    def test_under_cap_untouched(self):
        s = "a" * 10
        out, truncated = _truncate_middle(s, 100)
        assert out == s
        assert truncated is False

    def test_equal_cap_untouched(self):
        s = "a" * 50
        out, truncated = _truncate_middle(s, 50)
        assert out == s
        assert truncated is False

    def test_over_cap_truncated_with_marker(self):
        s = "a" * 500 + "MID" + "b" * 500  # 1003 chars
        out, truncated = _truncate_middle(s, 100)
        assert truncated is True
        assert "[... output truncated:" in out
        # Head preserved
        assert out.startswith("a" * 50)
        # Tail preserved — final chars should be 'b's
        assert out.endswith("b" * 50)
        # Middle marker mentions how many chars dropped
        assert "903 chars omitted" in out

    def test_cap_zero_returns_untouched(self):
        # cap <= 0 is treated as "no cap" — defensive branch.
        out, truncated = _truncate_middle("hello", 0)
        assert out == "hello"
        assert truncated is False

    def test_empty_string(self):
        out, truncated = _truncate_middle("", 100)
        assert out == ""
        assert truncated is False


class TestWrapWithCwd:
    def test_no_cwd_unchanged(self):
        assert _wrap_with_cwd("dir", None) == "dir"
        assert _wrap_with_cwd("dir", "") == "dir"

    def test_windows_cwd_uses_cd_slash_d(self):
        wrapped = _wrap_with_cwd("dir", r"C:\Users\User\Desktop\tasks\foo")
        assert wrapped == r'cd /d "C:\Users\User\Desktop\tasks\foo" && dir'

    def test_windows_unc_path(self):
        wrapped = _wrap_with_cwd("dir", r"\\server\share\dir")
        assert wrapped.startswith('cd /d "\\\\server')

    def test_posix_cwd_uses_plain_cd(self):
        wrapped = _wrap_with_cwd("ls", "/home/user")
        assert wrapped == 'cd "/home/user" && ls'

    def test_cwd_with_double_quote_escaped(self):
        wrapped = _wrap_with_cwd("dir", r'C:\weird"path')
        # The embedded quote is backslash-escaped inside the cd prefix.
        assert r'\"' in wrapped


class TestResolveTimeout:
    def test_default_when_missing(self):
        assert _resolve_timeout(None, _DEFAULT_TIMEOUT_SECONDS) == _DEFAULT_TIMEOUT_SECONDS

    def test_default_when_non_numeric(self):
        assert _resolve_timeout("60", _DEFAULT_TIMEOUT_SECONDS) == _DEFAULT_TIMEOUT_SECONDS

    def test_default_when_bool(self):
        # bool is an int subclass; must be rejected explicitly.
        assert _resolve_timeout(True, _DEFAULT_TIMEOUT_SECONDS) == _DEFAULT_TIMEOUT_SECONDS

    def test_default_when_non_positive(self):
        assert _resolve_timeout(0, 60) == 60
        assert _resolve_timeout(-5, 60) == 60

    def test_clamped_to_min(self):
        assert _resolve_timeout(0.5, 60) == _MIN_TIMEOUT_SECONDS

    def test_clamped_to_max(self):
        assert _resolve_timeout(10_000, 60) == _MAX_TIMEOUT_SECONDS

    def test_in_range_value_kept(self):
        assert _resolve_timeout(30, 60) == 30


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestParamValidation:
    def test_missing_command(self):
        tool = ExecTool(_make_interface())
        result = tool.call({})
        assert result["success"] is False
        assert "command" in result["error"]

    def test_empty_command(self):
        tool = ExecTool(_make_interface())
        result = tool.call({"command": ""})
        assert result["success"] is False
        assert "command" in result["error"]

    def test_non_string_command(self):
        tool = ExecTool(_make_interface())
        result = tool.call({"command": 42})
        assert result["success"] is False

    def test_empty_cwd_rejected(self):
        tool = ExecTool(_make_interface(), workspace_root=r"C:\Users\User\Desktop\tasks\foo")
        result = tool.call({"command": "dir", "cwd": ""})
        assert result["success"] is False
        assert "cwd" in result["error"]

    def test_cwd_outside_workspace_rejected(self):
        tool = ExecTool(_make_interface(), workspace_root=r"C:\Users\User\Desktop\tasks\foo")
        result = tool.call({"command": "dir", "cwd": r"C:\Windows"})
        assert result["success"] is False
        assert "workspace" in result["error"].lower()

    def test_cwd_inside_workspace_accepted(self):
        tool = ExecTool(_make_interface(stdout="ok"), workspace_root=r"C:\Users\User\Desktop\tasks\foo")
        result = tool.call({
            "command": "dir",
            "cwd": r"C:\Users\User\Desktop\tasks\foo\sub",
        })
        assert result["success"] is True

    def test_sibling_prefix_rejected(self):
        # foobar must NOT be considered inside foo.
        tool = ExecTool(_make_interface(), workspace_root=r"C:\Users\User\Desktop\tasks\foo")
        result = tool.call({"command": "dir", "cwd": r"C:\Users\User\Desktop\tasks\foobar"})
        assert result["success"] is False

    def test_permissive_when_workspace_root_none(self):
        tool = ExecTool(_make_interface(stdout="ok"))  # workspace_root=None
        result = tool.call({"command": "dir", "cwd": r"C:\Windows"})
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_zero_exit_completed(self):
        iface = _make_interface(stdout="listing\n", stderr="", returncode=0)
        tool = ExecTool(iface)
        result = tool.call({"command": "dir"})
        assert result == {
            "success": True,
            "status": "completed",
            "exit_code": 0,
            "duration_ms": result["duration_ms"],
            "stdout": "listing\n",
            "stderr": "",
            "truncated": False,
            "cwd": None,
        }
        assert isinstance(result["duration_ms"], int) and result["duration_ms"] >= 0

    def test_non_zero_exit_failed(self):
        iface = _make_interface(stdout="", stderr="not found\n", returncode=1)
        tool = ExecTool(iface)
        result = tool.call({"command": "doesnotexist"})
        assert result["success"] is True
        assert result["status"] == "failed"
        assert result["exit_code"] == 1
        assert result["stderr"] == "not found\n"

    def test_empty_stdout_returns_empty_string(self):
        iface = _make_interface(stdout="", stderr="", returncode=0)
        tool = ExecTool(iface)
        result = tool.call({"command": "echo"})
        assert result["stdout"] == ""
        assert result["stderr"] == ""

    def test_cwd_forwarded_to_wrapped_command_windows(self):
        iface = _make_interface(stdout="ok", returncode=0)
        tool = ExecTool(iface, workspace_root=r"C:\Users\User\Desktop\tasks\foo")
        result = tool.call({
            "command": "dir",
            "cwd": r"C:\Users\User\Desktop\tasks\foo",
        })
        assert result["success"] is True
        # Inspect the wrapped command that reached the RPC.
        called_with = iface.run_command.call_args.args[0]
        assert called_with.startswith("cd /d ")
        assert called_with.endswith("&& dir")
        assert result["cwd"] == r"C:\Users\User\Desktop\tasks\foo"

    def test_cwd_forwarded_to_wrapped_command_posix(self):
        iface = _make_interface(stdout="ok", returncode=0)
        tool = ExecTool(iface, workspace_root="/tmp/workspace")
        result = tool.call({
            "command": "ls",
            "cwd": "/tmp/workspace/sub",
        })
        assert result["success"] is True
        called_with = iface.run_command.call_args.args[0]
        assert called_with == 'cd "/tmp/workspace/sub" && ls'

    def test_no_cwd_passes_command_through(self):
        iface = _make_interface(stdout="ok")
        tool = ExecTool(iface)
        tool.call({"command": "dir"})
        assert iface.run_command.call_args.args[0] == "dir"

    def test_rpc_exception_surfaces_as_error(self):
        iface = _make_interface(raise_exc=RuntimeError("transport closed"))
        tool = ExecTool(iface)
        result = tool.call({"command": "dir"})
        assert result["success"] is False
        assert "transport closed" in result["error"]


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_under_cap_not_truncated(self):
        iface = _make_interface(stdout="a" * 100)
        tool = ExecTool(iface, max_output_chars=1_000)
        result = tool.call({"command": "echo"})
        assert result["truncated"] is False
        assert result["stdout"] == "a" * 100

    def test_over_cap_truncated(self):
        iface = _make_interface(stdout="a" * 500 + "b" * 500)
        tool = ExecTool(iface, max_output_chars=100)
        result = tool.call({"command": "echo"})
        assert result["truncated"] is True
        assert "[... output truncated:" in result["stdout"]
        # Tail preserved — trailing chars should be 'b's, not 'a's.
        assert result["stdout"].endswith("b" * 50)

    def test_stderr_truncation(self):
        iface = _make_interface(stdout="", stderr="x" * 5_000, returncode=1)
        tool = ExecTool(iface, max_output_chars=200)
        result = tool.call({"command": "err"})
        assert result["truncated"] is True
        assert "[... output truncated:" in result["stderr"]

    def test_default_cap_is_200k(self):
        assert _DEFAULT_MAX_OUTPUT_CHARS == 200_000
        tool = ExecTool(_make_interface())
        assert tool.max_output_chars == 200_000

    def test_custom_cap_honored(self):
        tool = ExecTool(_make_interface(), max_output_chars=50)
        assert tool.max_output_chars == 50


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_fires_and_returns_timed_out(self):
        iface = _make_interface(hang=True)
        tool = ExecTool(iface, default_timeout=1)  # tight for test speed
        result = tool.call({"command": "ping 127.0.0.1"})
        assert result["success"] is False
        assert result["status"] == "failed"
        assert result["timed_out"] is True
        assert result["duration_ms"] >= 900  # ~1s elapsed
        assert "may still be running" in result["error"]

    def test_explicit_timeout_param_honored(self):
        iface = _make_interface(hang=True)
        tool = ExecTool(iface, default_timeout=_MAX_TIMEOUT_SECONDS)
        result = tool.call({"command": "ping 127.0.0.1", "timeout": 1})
        assert result["timed_out"] is True
        assert result["duration_ms"] < 5_000

    def test_explicit_timeout_clamped_to_min(self):
        # timeout=0.5 → clamped to _MIN_TIMEOUT_SECONDS=1.
        iface = _make_interface(hang=True)
        tool = ExecTool(iface)
        result = tool.call({"command": "ping", "timeout": 0.5})
        assert result["timed_out"] is True

    def test_default_timeout_applies_when_param_missing(self):
        tool = ExecTool(_make_interface(stdout="ok"), default_timeout=30)
        # Verify default_timeout is the one used when no param is given.
        # We can't easily assert the wait_for timeout, but we can assert the
        # resolved attribute matches expectation.
        assert tool.default_timeout == 30

    def test_default_timeout_clamped_at_construction(self):
        tool = ExecTool(_make_interface(), default_timeout=10_000)
        assert tool.default_timeout == _MAX_TIMEOUT_SECONDS

    def test_default_timeout_fallback_for_bad_value(self):
        tool = ExecTool(_make_interface(), default_timeout=-1)
        assert tool.default_timeout == _DEFAULT_TIMEOUT_SECONDS
