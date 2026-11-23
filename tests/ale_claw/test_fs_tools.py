"""Tests for Remote-VM File I/O tools.

Covers ReadFileTool / WriteFileTool / EditFileTool:
  - Registration & schema
  - Read text (explicit limit, adaptive paging, errors)
  - Read image (happy path, MIME sniff, size cap)
  - Write (happy path, append, create_parents, errors)
  - Edit (happy path, multi-edit, mismatch hint, validation)
  - Path policy (workspace_root enforcement)
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

from agent.tools.base import TOOL_REGISTRY
from ale_claw.harness.tools._paths import (
    _assert_within_workspace,
    _is_windows_path,
)
from ale_claw.harness.tools.fs_backends import FilesystemRegistry, VMBackend
from ale_claw.harness.tools.tools_fs import (
    EditFileTool,
    ReadFileTool,
    WriteFileTool,
    _resolve_adaptive_read_max_bytes,
    _sniff_mime_from_bytes,
)


def _vm_registry(iface, workspace_root=None) -> FilesystemRegistry:
    reg = FilesystemRegistry()
    reg.register(VMBackend(iface, workspace_root=workspace_root))
    return reg


def _read_tool(iface, workspace_root=None, context_window_tokens=None):
    return ReadFileTool(
        _vm_registry(iface, workspace_root),
        context_window_tokens=context_window_tokens,
    )


def _write_tool(iface, workspace_root=None):
    return WriteFileTool(_vm_registry(iface, workspace_root))


def _edit_tool(iface, workspace_root=None):
    return EditFileTool(_vm_registry(iface, workspace_root))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _make_interface(
    *,
    read_bytes: bytes | Exception | None = None,
    write_text_ok: bool = True,
    create_dir_ok: bool = True,
) -> MagicMock:
    iface = MagicMock()
    if isinstance(read_bytes, Exception):
        iface.read_bytes = AsyncMock(side_effect=read_bytes)
    else:
        iface.read_bytes = AsyncMock(return_value=read_bytes or b"")
    iface.write_bytes = AsyncMock(return_value=None if write_text_ok else Exception("write failed"))
    iface.create_dir = AsyncMock(return_value=None if create_dir_ok else Exception("mkdir failed"))
    return iface


def _make_png(width: int = 4, height: int = 4, color=(255, 0, 0)) -> bytes:
    """Real PNG bytes — needed since the sanitizer decodes via Pillow."""
    import io as _io

    from PIL import Image as _PILImage

    buf = _io.BytesIO()
    _PILImage.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 4, height: int = 4, color=(255, 0, 0), quality: int = 85) -> bytes:
    """Real JPEG bytes."""
    import io as _io

    from PIL import Image as _PILImage

    buf = _io.BytesIO()
    _PILImage.new("RGB", (width, height), color).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


_PNG_BYTES = _make_png()
_JPEG_BYTES = _make_jpeg()
_PDF_BYTES = b"%PDF-1.4\n" + b"\x00" * 32


# ---------------------------------------------------------------------------
# Registration & schema
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_read_registered(self):
        assert "read" in TOOL_REGISTRY and TOOL_REGISTRY["read"] is ReadFileTool

    def test_write_registered(self):
        assert "write" in TOOL_REGISTRY and TOOL_REGISTRY["write"] is WriteFileTool

    def test_edit_registered(self):
        assert "edit" in TOOL_REGISTRY and TOOL_REGISTRY["edit"] is EditFileTool

    def test_read_required_params(self):
        tool = _read_tool(_make_interface())
        assert tool.parameters["required"] == ["path"]

    def test_write_required_params(self):
        tool = _write_tool(_make_interface())
        assert tool.parameters["required"] == ["path", "content"]

    def test_edit_required_params(self):
        tool = _edit_tool(_make_interface())
        assert tool.parameters["required"] == ["path", "edits"]

    def test_read_schema_properties(self):
        tool = _read_tool(_make_interface())
        props = tool.parameters["properties"]
        for key in ("path", "offset", "limit", "max_bytes", "encoding"):
            assert key in props


# ---------------------------------------------------------------------------
# Helpers — path + MIME
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_windows_path_drive(self):
        assert _is_windows_path(r"C:\Users\User\file.txt")
        assert _is_windows_path("C:/Users/User/file.txt")

    def test_is_windows_path_unc(self):
        assert _is_windows_path("\\\\server\\share\\file")

    def test_is_windows_path_posix(self):
        assert not _is_windows_path("/home/user/file.txt")

    def test_sniff_png(self):
        assert _sniff_mime_from_bytes(_PNG_BYTES) == "image/png"

    def test_sniff_jpeg(self):
        assert _sniff_mime_from_bytes(_JPEG_BYTES) == "image/jpeg"

    def test_sniff_pdf(self):
        assert _sniff_mime_from_bytes(_PDF_BYTES) == "application/pdf"

    def test_sniff_unknown(self):
        assert _sniff_mime_from_bytes(b"random bytes") is None

    def test_adaptive_cap_floor(self):
        assert _resolve_adaptive_read_max_bytes(1000) == 32 * 1024

    def test_adaptive_cap_ceiling(self):
        assert _resolve_adaptive_read_max_bytes(10_000_000) == 128 * 1024

    def test_adaptive_cap_midrange(self):
        # 200K tokens × 4 chars/tok × 0.10 = 80_000 bytes
        assert _resolve_adaptive_read_max_bytes(200_000) == 80_000

    def test_adaptive_cap_none(self):
        assert _resolve_adaptive_read_max_bytes(None) == 32 * 1024

    def test_workspace_policy_permissive(self):
        # workspace_root=None → no-op
        _assert_within_workspace(r"C:\anywhere", None)

    def test_workspace_policy_inside(self):
        _assert_within_workspace(
            r"C:\Users\User\Desktop\tasks\foo\file.txt",
            r"C:\Users\User\Desktop\tasks\foo",
        )

    def test_workspace_policy_case_insensitive(self):
        _assert_within_workspace(
            r"c:\users\user\desktop\tasks\foo\file.txt",
            r"C:\Users\User\Desktop\tasks\foo",
        )

    def test_workspace_policy_outside(self):
        import pytest

        with pytest.raises(ValueError, match="outside the task workspace"):
            _assert_within_workspace(
                r"C:\Users\User\Desktop\other\file.txt",
                r"C:\Users\User\Desktop\tasks\foo",
            )

    def test_workspace_policy_sibling_prefix_rejected(self):
        # "tasks\foobar" should not match workspace "tasks\foo"
        import pytest

        with pytest.raises(ValueError, match="outside the task workspace"):
            _assert_within_workspace(
                r"C:\Users\User\Desktop\tasks\foobar\file.txt",
                r"C:\Users\User\Desktop\tasks\foo",
            )


# ---------------------------------------------------------------------------
# ReadFileTool — text path
# ---------------------------------------------------------------------------


class TestReadText:
    def test_explicit_limit_slice(self):
        content = "\n".join(f"line{i}" for i in range(1, 21))  # 20 lines
        iface = _make_interface(read_bytes=content.encode("utf-8"))
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\f.txt", "offset": 5, "limit": 3})
        assert result["success"] is True
        assert result["content"] == "line5\nline6\nline7"
        assert result["truncated"] is True
        assert result["total_lines"] == 20
        assert result["next_offset"] == 8

    def test_explicit_limit_past_end(self):
        content = "one\ntwo\nthree"
        iface = _make_interface(read_bytes=content.encode("utf-8"))
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\f.txt", "offset": 1, "limit": 100})
        assert result["success"] is True
        assert result["content"] == content
        assert result["truncated"] is False
        assert result["total_lines"] == 3
        assert result["next_offset"] is None

    def test_adaptive_paging_fits_whole_file(self):
        content = "short\nfile\n"
        iface = _make_interface(read_bytes=content.encode("utf-8"))
        tool = _read_tool(iface, context_window_tokens=200_000)
        result = tool.call({"path": r"C:\f.txt"})
        assert result["success"] is True
        assert result["truncated"] is False
        assert result["next_offset"] is None
        assert "Read output capped" not in result["content"]

    def test_adaptive_paging_exceeds_cap(self):
        # Construct a file whose total bytes exceed a 32KB cap
        big_line = "x" * 1000
        lines = [big_line] * 100  # ≈ 100KB
        content = "\n".join(lines)
        iface = _make_interface(read_bytes=content.encode("utf-8"))
        tool = _read_tool(iface, context_window_tokens=None)  # 32KB cap
        result = tool.call({"path": r"C:\big.txt"})
        assert result["success"] is True
        assert result["truncated"] is True
        assert result["next_offset"] is not None and result["next_offset"] > 1
        assert "Read output capped at 32KB" in result["content"]
        assert f"Use offset={result['next_offset']}" in result["content"]

    def test_adaptive_paging_offset_honored(self):
        content = "\n".join(str(i) for i in range(1, 11))
        iface = _make_interface(read_bytes=content.encode("utf-8"))
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\f.txt", "offset": 5})
        assert result["success"] is True
        # starts from line 5 (index 4, value "5")
        assert result["content"].startswith("5")

    def test_missing_path_param(self):
        tool = _read_tool(_make_interface())
        result = tool.call({})
        assert result["success"] is False
        assert "path" in result["error"]

    def test_read_rpc_failure(self):
        iface = _make_interface(read_bytes=RuntimeError("file not found"))
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\missing.txt"})
        assert result["success"] is False
        assert "file not found" in result["error"] or "could not read" in result["error"]

    def test_binary_content_unicode_error(self):
        # Non-UTF-8 bytes, non-image extension
        iface = _make_interface(read_bytes=b"\xff\xfe\x00\x01binary garbage")
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\binary.dat"})
        assert result["success"] is False
        assert "not utf-8 text" in result["error"]
        assert "analyze_image" in result["error"]

    def test_read_uses_read_bytes_rpc(self):
        iface = _make_interface(read_bytes=b"hello")
        tool = _read_tool(iface)
        tool.call({"path": r"C:\f.txt", "offset": 1, "limit": 10})
        iface.read_bytes.assert_awaited_once_with(r"C:\f.txt")


# ---------------------------------------------------------------------------
# ReadFileTool — image path
# ---------------------------------------------------------------------------


class TestReadImage:
    def test_png_happy_path(self):
        iface = _make_interface(read_bytes=_PNG_BYTES)
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\icon.png"})
        assert result["success"] is True
        assert result["type"] == "image"
        assert result["mime_type"] == "image/png"
        assert result["text"] == "Read image file [image/png]"
        assert base64.b64decode(result["data"]) == _PNG_BYTES

    def test_jpeg_happy_path(self):
        iface = _make_interface(read_bytes=_JPEG_BYTES)
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\photo.jpg"})
        assert result["success"] is True
        assert result["mime_type"] == "image/jpeg"

    def test_sniff_overrides_extension(self):
        # .png extension but bytes are actually JPEG magic → sniff wins
        iface = _make_interface(read_bytes=_JPEG_BYTES)
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\lying.png"})
        assert result["success"] is True
        assert result["mime_type"] == "image/jpeg"
        assert result["text"] == "Read image file [image/jpeg]"

    def test_non_image_sniff_error(self):
        # .png extension but actually PDF → error
        iface = _make_interface(read_bytes=_PDF_BYTES)
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\lying.png"})
        assert result["success"] is False
        assert "file looks like application/pdf" in result["error"]
        assert "treated as image/png" in result["error"]

    def test_oversize_png_resized_to_jpeg(self):
        # a real >1200px PNG is resized & transcoded to JPEG
        # (was previously rejected with "image too large").
        big = _make_png(2400, 1800, color=(80, 160, 240))
        iface = _make_interface(read_bytes=big)
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\huge.png"})
        assert result["success"] is True
        # Format normalized to JPEG by the resize loop.
        assert result["mime_type"] == "image/jpeg"
        # Output sits under the 5 MB default cap.
        out = base64.b64decode(result["data"])
        assert len(out) <= 5 * 1024 * 1024

    def test_custom_max_bytes_unreachable_returns_placeholder(self):
        # Per-call max_bytes too small for any (side, quality) candidate to
        # fit — the sanitizer exhausts its grid and returns a placeholder
        # error, which the tool surfaces as success=False.
        data = _make_jpeg(800, 600, color=(200, 50, 50))
        iface = _make_interface(read_bytes=data)
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\mid.jpg", "max_bytes": 50})
        assert result["success"] is False
        assert "could not be reduced" in result["error"]

    def test_empty_file(self):
        iface = _make_interface(read_bytes=b"")
        tool = _read_tool(iface)
        result = tool.call({"path": r"C:\empty.png"})
        assert result["success"] is False
        assert "empty" in result["error"]

    def test_image_uses_read_bytes_not_read_text(self):
        iface = _make_interface(read_bytes=_PNG_BYTES)
        iface.read_text = AsyncMock(return_value="should not be called")
        tool = _read_tool(iface)
        tool.call({"path": r"C:\icon.png"})
        iface.read_text.assert_not_awaited()
        iface.read_bytes.assert_awaited_once_with(r"C:\icon.png")


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------


class TestWrite:
    def test_happy_path(self):
        iface = _make_interface()
        tool = _write_tool(iface)
        result = tool.call({"path": r"C:\out.txt", "content": "hello"})
        assert result["success"] is True
        assert result["bytes_written"] == 5
        assert result["append"] is False
        iface.write_bytes.assert_awaited_once_with(r"C:\out.txt", b"hello", append=False)

    def test_append_mode(self):
        iface = _make_interface()
        tool = _write_tool(iface)
        tool.call({"path": r"C:\log.txt", "content": "entry\n", "append": True})
        iface.write_bytes.assert_awaited_once_with(r"C:\log.txt", b"entry\n", append=True)

    def test_create_parents_default_true(self):
        iface = _make_interface()
        tool = _write_tool(iface)
        tool.call({"path": r"C:\a\b\c.txt", "content": "x"})
        iface.create_dir.assert_awaited_once_with(r"C:\a\b")

    def test_create_parents_false_skips_mkdir(self):
        iface = _make_interface()
        tool = _write_tool(iface)
        tool.call(
            {"path": r"C:\a\b\c.txt", "content": "x", "create_parents": False}
        )
        iface.create_dir.assert_not_awaited()

    def test_missing_path(self):
        tool = _write_tool(_make_interface())
        result = tool.call({"content": "x"})
        assert result["success"] is False
        assert "path" in result["error"]

    def test_missing_content(self):
        tool = _write_tool(_make_interface())
        result = tool.call({"path": r"C:\f.txt"})
        assert result["success"] is False
        assert "content" in result["error"]

    def test_workspace_policy_rejects_outside(self):
        iface = _make_interface()
        tool = _write_tool(
            iface, workspace_root=r"C:\Users\User\Desktop\tasks\foo"
        )
        result = tool.call(
            {"path": r"C:\Windows\System32\evil.txt", "content": "pwn"}
        )
        assert result["success"] is False
        assert "outside the task workspace" in result["error"]
        iface.write_bytes.assert_not_awaited()


# ---------------------------------------------------------------------------
# EditFileTool
# ---------------------------------------------------------------------------


class TestEdit:
    def test_single_edit_happy_path(self):
        original = "hello world\n"
        iface = _make_interface(read_bytes=original.encode("utf-8"))
        tool = _edit_tool(iface)
        result = tool.call(
            {
                "path": r"C:\f.txt",
                "edits": [{"oldText": "world", "newText": "there"}],
            }
        )
        assert result["success"] is True
        assert result["edits_applied"] == 1
        iface.write_bytes.assert_awaited_once_with(
            r"C:\f.txt", b"hello there\n", append=False
        )

    def test_multi_edit_sequential(self):
        original = "alpha\nbeta\ngamma\n"
        iface = _make_interface(read_bytes=original.encode("utf-8"))
        tool = _edit_tool(iface)
        result = tool.call(
            {
                "path": r"C:\f.txt",
                "edits": [
                    {"oldText": "alpha", "newText": "APPLE"},
                    {"oldText": "gamma", "newText": "GRAPE"},
                ],
            }
        )
        assert result["success"] is True
        assert result["edits_applied"] == 2
        final = iface.write_bytes.await_args.args[1]
        assert final == b"APPLE\nbeta\nGRAPE\n"

    def test_deletion_via_empty_new_text(self):
        original = "keep me DELETEME stay"
        iface = _make_interface(read_bytes=original.encode("utf-8"))
        tool = _edit_tool(iface)
        tool.call(
            {
                "path": r"C:\f.txt",
                "edits": [{"oldText": " DELETEME", "newText": ""}],
            }
        )
        final = iface.write_bytes.await_args.args[1]
        assert final == b"keep me stay"

    def test_mismatch_hint(self):
        original = "actual content here\n"
        iface = _make_interface(read_bytes=original.encode("utf-8"))
        tool = _edit_tool(iface)
        result = tool.call(
            {
                "path": r"C:\f.txt",
                "edits": [{"oldText": "nonexistent", "newText": "foo"}],
            }
        )
        assert result["success"] is False
        assert "could not find the exact text" in result["error"]
        assert "actual content here" in result["error"]
        iface.write_bytes.assert_not_awaited()

    def test_mismatch_hint_truncation(self):
        big = ("x" * 1000) + "\n"
        iface = _make_interface(read_bytes=big.encode("utf-8"))
        tool = _edit_tool(iface)
        result = tool.call(
            {
                "path": r"C:\f.txt",
                "edits": [{"oldText": "nope", "newText": "bar"}],
            }
        )
        assert result["success"] is False
        assert "(truncated)" in result["error"]
        # Only 800 chars of content plus truncation suffix
        assert "xxxx" in result["error"]

    def test_empty_edits_rejected(self):
        tool = _edit_tool(_make_interface())
        result = tool.call({"path": r"C:\f.txt", "edits": []})
        assert result["success"] is False
        assert "non-empty" in result["error"]

    def test_empty_old_text_rejected(self):
        tool = _edit_tool(_make_interface())
        result = tool.call(
            {
                "path": r"C:\f.txt",
                "edits": [{"oldText": "", "newText": "foo"}],
            }
        )
        assert result["success"] is False
        assert "non-empty string" in result["error"]

    def test_new_text_must_be_string(self):
        tool = _edit_tool(_make_interface())
        result = tool.call(
            {
                "path": r"C:\f.txt",
                "edits": [{"oldText": "a", "newText": 123}],
            }
        )
        assert result["success"] is False
        assert "newText" in result["error"]

    def test_workspace_policy_rejects_outside(self):
        iface = _make_interface()
        tool = _edit_tool(
            iface, workspace_root=r"C:\Users\User\Desktop\tasks\foo"
        )
        result = tool.call(
            {
                "path": r"C:\Windows\evil.txt",
                "edits": [{"oldText": "a", "newText": "b"}],
            }
        )
        assert result["success"] is False
        assert "outside the task workspace" in result["error"]
        iface.read_bytes.assert_not_awaited()

    def test_unchanged_when_oldText_equals_newText(self):
        original = "same\n"
        iface = _make_interface(read_bytes=original.encode("utf-8"))
        tool = _edit_tool(iface)
        result = tool.call(
            {
                "path": r"C:\f.txt",
                "edits": [{"oldText": "same", "newText": "same"}],
            }
        )
        assert result["success"] is True
        assert result.get("unchanged") is True
        iface.write_bytes.assert_not_awaited()


# ---------------------------------------------------------------------------
# Path policy integration (through ReadFileTool)
# ---------------------------------------------------------------------------


class TestPathPolicyIntegration:
    def test_read_permissive_no_workspace_root(self):
        iface = _make_interface(read_bytes=b"ok")
        tool = _read_tool(iface)
        result = tool.call({"path": r"D:\anywhere\file.txt", "limit": 1})
        assert result["success"] is True

    def test_read_outside_workspace_rejected(self):
        iface = _make_interface()
        tool = _read_tool(iface, workspace_root=r"C:\Users\User\Desktop\tasks\foo")
        result = tool.call({"path": r"C:\other\path\file.txt"})
        assert result["success"] is False
        assert "outside the task workspace" in result["error"]
        iface.read_bytes.assert_not_awaited()

    def test_read_inside_workspace_allowed(self):
        iface = _make_interface(read_bytes=b"content")
        tool = _read_tool(
            iface, workspace_root=r"C:\Users\User\Desktop\tasks\foo"
        )
        result = tool.call(
            {"path": r"C:\Users\User\Desktop\tasks\foo\sub\file.txt", "limit": 1}
        )
        assert result["success"] is True
