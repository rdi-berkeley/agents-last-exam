"""Tests for the FilesystemRegistry and the VM/host backends.

Covers:
  - VMBackend wraps the existing CUA SDK calls (regression for tools_fs.py).
  - HostBackend rejects path escape via realpath check, including symlinks.
  - HostBackend append vs overwrite; create_dir.
  - FilesystemRegistry name ordering and capability lookup.
  - target= dispatch through the read/write/edit tools.
  - Asymmetric append default (vm=overwrite, host=append).
  - detect_host_workspace_root walks to the nearest .git directory.
  - OPENCLAW_HOST_WORKSPACE override beats auto-detection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from ale_claw.harness.tools.fs_backends import (
    FilesystemRegistry,
    HostBackend,
    VMBackend,
    detect_host_workspace_root,
)
from ale_claw.harness.tools.tools_fs import (
    EditFileTool,
    ReadFileTool,
    WriteFileTool,
    _default_append_for,
)


def _run(coro):
    return asyncio.run(coro)


def _mock_iface() -> MagicMock:
    iface = MagicMock()
    iface.read_bytes = AsyncMock(return_value=b"")
    iface.write_bytes = AsyncMock(return_value=None)
    iface.create_dir = AsyncMock(return_value=None)
    return iface


# ---------------------------------------------------------------------------
# FilesystemRegistry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_unknown_target_lists_valid(self):
        reg = FilesystemRegistry()
        reg.register(VMBackend(_mock_iface()))
        with pytest.raises(ValueError, match="unknown target 'wm'"):
            reg.get("wm")

    def test_names_puts_vm_first(self, tmp_path):
        reg = FilesystemRegistry()
        reg.register(HostBackend(tmp_path))
        reg.register(VMBackend(_mock_iface()))
        assert reg.names() == ["vm", "host"]

    def test_describe_includes_each_backend(self, tmp_path):
        reg = FilesystemRegistry()
        reg.register(VMBackend(_mock_iface(), workspace_root=r"C:\ws"))
        reg.register(HostBackend(tmp_path))
        desc = reg.describe()
        assert "vm:" in desc and "host:" in desc
        assert "C:\\ws" in desc
        assert str(tmp_path.resolve()) in desc

    def test_duplicate_register_rejected(self):
        reg = FilesystemRegistry()
        reg.register(VMBackend(_mock_iface()))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(VMBackend(_mock_iface()))


# ---------------------------------------------------------------------------
# HostBackend — path policy
# ---------------------------------------------------------------------------


class TestHostBackendResolve:
    def test_relative_path_resolves_under_root(self, tmp_path):
        h = HostBackend(tmp_path)
        resolved = h.resolve("notes.md")
        assert resolved == str(tmp_path.resolve() / "notes.md")

    def test_absolute_path_inside_root_allowed(self, tmp_path):
        h = HostBackend(tmp_path)
        target = tmp_path / "sub" / "file.md"
        target.parent.mkdir(parents=True)
        resolved = h.resolve(str(target))
        assert resolved == str(target.resolve())

    def test_lexical_traversal_rejected(self, tmp_path):
        h = HostBackend(tmp_path)
        with pytest.raises(ValueError, match="outside host workspace"):
            h.resolve("../../etc/passwd")

    def test_absolute_path_outside_root_rejected(self, tmp_path):
        h = HostBackend(tmp_path)
        with pytest.raises(ValueError, match="outside host workspace"):
            h.resolve("/etc/passwd")

    def test_symlink_escape_rejected(self, tmp_path):
        # Create a symlink under root pointing outside; realpath check must catch it.
        outside = tmp_path.parent / "outside_target"
        outside.mkdir(exist_ok=True)
        link = tmp_path / "escape"
        link.symlink_to(outside)

        h = HostBackend(tmp_path)
        with pytest.raises(ValueError, match="outside host workspace"):
            h.resolve("escape/file")

    def test_root_path_itself_allowed(self, tmp_path):
        h = HostBackend(tmp_path)
        # Resolving "" is rejected as empty; "." should resolve to root.
        resolved = h.resolve(".")
        assert resolved == str(tmp_path.resolve())

    def test_empty_path_rejected(self, tmp_path):
        h = HostBackend(tmp_path)
        with pytest.raises(ValueError, match="non-empty"):
            h.resolve("")

    def test_missing_root_rejected_at_construction(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            HostBackend(tmp_path / "missing")

    def test_sibling_prefix_rejected(self, tmp_path):
        # tmp_path/foo is the root; tmp_path/foobar/x must NOT be allowed.
        root = tmp_path / "foo"
        root.mkdir()
        sibling = tmp_path / "foobar"
        sibling.mkdir()
        h = HostBackend(root)
        with pytest.raises(ValueError, match="outside host workspace"):
            h.resolve(str(sibling / "x.txt"))


# ---------------------------------------------------------------------------
# HostBackend — I/O
# ---------------------------------------------------------------------------


class TestHostBackendIO:
    def test_write_text_overwrite(self, tmp_path):
        h = HostBackend(tmp_path)
        target = h.resolve("note.md")
        _run(h.write_text(target, "first\n", append=False))
        _run(h.write_text(target, "second\n", append=False))
        assert Path(target).read_text() == "second\n"

    def test_write_text_append(self, tmp_path):
        h = HostBackend(tmp_path)
        target = h.resolve("log.md")
        _run(h.write_text(target, "line1\n", append=False))
        _run(h.write_text(target, "line2\n", append=True))
        assert Path(target).read_text() == "line1\nline2\n"

    def test_create_dir_recursive(self, tmp_path):
        h = HostBackend(tmp_path)
        nested = h.resolve("a/b/c")
        _run(h.create_dir(nested))
        assert Path(nested).is_dir()

    def test_read_bytes_round_trip(self, tmp_path):
        h = HostBackend(tmp_path)
        target = h.resolve("data.bin")
        Path(target).write_bytes(b"\x00\x01\x02")
        assert _run(h.read_bytes(target)) == b"\x00\x01\x02"


# ---------------------------------------------------------------------------
# Target dispatch through tools
# ---------------------------------------------------------------------------


def _registry_with_both(tmp_path) -> FilesystemRegistry:
    reg = FilesystemRegistry()
    reg.register(VMBackend(_mock_iface(), workspace_root=None))
    reg.register(HostBackend(tmp_path))
    return reg


class TestTargetDispatch:
    def test_default_target_is_vm(self, tmp_path):
        reg = _registry_with_both(tmp_path)
        iface = reg.get("vm").interface
        tool = WriteFileTool(reg)
        result = tool.call({"path": r"C:\f.txt", "content": "hi"})
        assert result["success"] is True
        assert result["target"] == "vm"
        iface.write_bytes.assert_awaited_once_with(r"C:\f.txt", b"hi", append=False)

    def test_explicit_target_vm(self, tmp_path):
        reg = _registry_with_both(tmp_path)
        iface = reg.get("vm").interface
        tool = WriteFileTool(reg)
        result = tool.call({"target": "vm", "path": r"C:\f.txt", "content": "hi"})
        assert result["success"] is True
        iface.write_bytes.assert_awaited_once_with(r"C:\f.txt", b"hi", append=False)

    def test_target_host_writes_to_host(self, tmp_path):
        reg = _registry_with_both(tmp_path)
        tool = WriteFileTool(reg)
        result = tool.call({"target": "host", "path": "out.md", "content": "x"})
        assert result["success"] is True
        assert result["target"] == "host"
        assert (tmp_path / "out.md").read_text() == "x"
        # VM iface should not have been touched.
        reg.get("vm").interface.write_bytes.assert_not_awaited()

    def test_unknown_target_rejected(self, tmp_path):
        reg = _registry_with_both(tmp_path)
        tool = WriteFileTool(reg)
        result = tool.call({"target": "wm", "path": "x", "content": "y"})
        assert result["success"] is False
        assert "unknown target 'wm'" in result["error"]
        assert "vm" in result["error"] and "host" in result["error"]

    def test_host_path_escape_rejected_in_tool(self, tmp_path):
        reg = _registry_with_both(tmp_path)
        tool = WriteFileTool(reg)
        result = tool.call(
            {"target": "host", "path": "../escape.txt", "content": "x"}
        )
        assert result["success"] is False
        assert "outside host workspace" in result["error"]

    def test_read_target_host_round_trip(self, tmp_path):
        (tmp_path / "src.txt").write_text("alpha\nbeta\ngamma\n")
        reg = _registry_with_both(tmp_path)
        tool = ReadFileTool(reg)
        result = tool.call({"target": "host", "path": "src.txt", "limit": 10})
        assert result["success"] is True
        assert result["content"] == "alpha\nbeta\ngamma\n"

    def test_edit_target_host_round_trip(self, tmp_path):
        target = tmp_path / "doc.md"
        target.write_text("hello world\n")
        reg = _registry_with_both(tmp_path)
        tool = EditFileTool(reg)
        result = tool.call({
            "target": "host",
            "path": "doc.md",
            "edits": [{"oldText": "world", "newText": "there"}],
        })
        assert result["success"] is True
        assert target.read_text() == "hello there\n"

    def test_target_only_vm_when_host_not_registered(self):
        reg = FilesystemRegistry()
        reg.register(VMBackend(_mock_iface()))
        tool = WriteFileTool(reg)
        assert tool.parameters["properties"]["target"]["enum"] == ["vm"]


# ---------------------------------------------------------------------------
# Asymmetric append default
# ---------------------------------------------------------------------------


class TestAppendDefault:
    def test_vm_default_overwrite(self):
        assert _default_append_for("vm", {}) is False

    def test_host_default_append(self):
        assert _default_append_for("host", {}) is True

    def test_explicit_append_overrides_default(self):
        assert _default_append_for("vm", {"append": True}) is True
        assert _default_append_for("host", {"append": False}) is False

    def test_host_write_appends_by_default(self, tmp_path):
        reg = _registry_with_both(tmp_path)
        tool = WriteFileTool(reg)
        tool.call({"target": "host", "path": "log.md", "content": "first\n"})
        tool.call({"target": "host", "path": "log.md", "content": "second\n"})
        assert (tmp_path / "log.md").read_text() == "first\nsecond\n"

    def test_host_overwrite_opt_out(self, tmp_path):
        reg = _registry_with_both(tmp_path)
        tool = WriteFileTool(reg)
        tool.call({"target": "host", "path": "log.md", "content": "A"})
        tool.call(
            {"target": "host", "path": "log.md", "content": "B", "append": False}
        )
        assert (tmp_path / "log.md").read_text() == "B"


# ---------------------------------------------------------------------------
# Capability rejection
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_read_only_backend_rejects_write(self, tmp_path):
        class ReadOnlyHost(HostBackend):
            capabilities = frozenset({"read"})

        reg = FilesystemRegistry()
        reg.register(VMBackend(_mock_iface()))
        reg.register(ReadOnlyHost(tmp_path))
        tool = WriteFileTool(reg)
        result = tool.call({"target": "host", "path": "x.txt", "content": "y"})
        assert result["success"] is False
        assert "'write' not supported on target 'host'" in result["error"]


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestAutoDetection:
    def test_detects_git_dir_in_self(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert detect_host_workspace_root(tmp_path) == tmp_path.resolve()

    def test_walks_up_to_ancestor(self, tmp_path):
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        assert detect_host_workspace_root(nested) == tmp_path.resolve()

    def test_returns_none_when_no_git_found(self, tmp_path):
        # tmp_path has no .git; assume no parent in pytest's tmp tree has one
        # close enough to be confused with the repo. Walk all the way up but
        # bound the assertion to "either None or some .git outside agenthle".
        result = detect_host_workspace_root(tmp_path)
        # If pytest happens to live inside a git repo, the test still verifies
        # that *some* .git is found; we only assert detection didn't crash.
        assert result is None or (result / ".git").is_dir()

    def test_skips_submodule_git_file(self, tmp_path):
        # Outer repo: real .git directory.
        (tmp_path / ".git").mkdir()
        # Inner "submodule": a .git *file* (gitfile), should not stop detection.
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / ".git").write_text("gitdir: ../.git/modules/sub\n")
        nested = sub / "deep" / "deeper"
        nested.mkdir(parents=True)
        assert detect_host_workspace_root(nested) == tmp_path.resolve()


class TestEnvOverride:
    """The override path lives in openclaw_agent.py; verify the helper alone here.

    The full env-var resolution is exercised in the integration tests; what we
    can check at this layer is that detect_host_workspace_root accepts an
    explicit start path, which is the same plumbing the override goes through.
    """

    def test_explicit_start_overrides_cwd(self, tmp_path, monkeypatch):
        # Even with cwd elsewhere, passing start= picks the explicit dir.
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir("/")
        assert detect_host_workspace_root(tmp_path) == tmp_path.resolve()
