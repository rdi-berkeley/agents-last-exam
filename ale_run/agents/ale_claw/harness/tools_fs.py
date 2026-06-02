"""Remote-VM filesystem tools: read, write, edit (US-OC-055).

Three BaseTool subclasses that route all file I/O through
``session.interface`` (computer-server RPCs) — never the host ``open()``.
Ports OpenClaw's ``createHostWorkspace{Read,Write,Edit}Tool`` behavior to
CUA's sync-tool contract with async RPCs underneath.

Kept from OpenClaw (``openclaw/src/agents/pi-tools.read.ts``,
``pi-tools.host-edit.ts``, ``pi-tools.params.ts``):
  - Required-param groups: read=[path], write=[path, content],
    edit=[path, edits] — ``REQUIRED_PARAM_GROUPS`` at :75-85.
  - Adaptive byte-paging on read when no explicit ``limit``:
    ``cap = clamp(ctx_tokens * 4 * 0.10, 32KB, 128KB)`` —
    ``resolveAdaptiveReadMaxBytes`` at :69-82.
  - Image sanitization (MIME sniff vs extension, size cap, image
    content block return) — ``normalizeReadImageResult`` at :292-349.
  - Edit mismatch-hint recovery — ``wrapEditToolWithRecovery`` at
    :150-212 (``EDIT_MISMATCH_HINT_LIMIT = 800``).
  - Workspace-only path policy — ``wrapToolWorkspaceRootGuard``.

Dropped:
  - Sandbox bridge variants (docker-sandbox FS — no CUA analogue).
  - Post-write retroactive-success inference (``didEditLikelyApply``) —
    CUA writes over computer-server are single-shot RPCs with no
    writeFile-vs-stat race.
  - URL/file-URL/``@``-prefix handling — agents in our benchmark emit
    plain Windows/POSIX paths.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Optional, Union

from agent.tools.base import BaseTool, register_tool

from ._paths import _parent_dir
from ._tool_utils import _get_required_str, _run_tool_execute
from .fs_backends import FilesystemBackend, FilesystemRegistry
from .image_sanitization import (
    DEFAULT_LIMITS as _IMAGE_DEFAULT_LIMITS,
    sanitize_raw_image_bytes,
    sniff_mime_from_bytes as _sniff_mime_from_bytes,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-target defaults
# ---------------------------------------------------------------------------


def _default_append_for(target: str, params: dict) -> bool:
    """Resolve the effective ``append`` flag for ``write``.

    VM mirrors the historical default (overwrite). Host defaults to append
    so the agent's first instinct is journaling, not clobber. Either side
    can be opted out by setting ``append`` explicitly in ``params``.
    """
    if "append" in params:
        return bool(params["append"])
    return target == "host"


def _check_capability(backend: FilesystemBackend, op: str) -> Optional[str]:
    if op not in backend.capabilities:
        return (
            f"'{op}' not supported on target '{backend.name}'; "
            f"capabilities: {sorted(backend.capabilities)}"
        )
    return None

# ---------------------------------------------------------------------------
# Constants (match OpenClaw pi-tools.read.ts:45-49 / pi-tools.host-edit.ts:22-23)
# ---------------------------------------------------------------------------

_MAX_MISMATCH_HINT_CHARS = 800                         # matches EDIT_MISMATCH_HINT_LIMIT
_DEFAULT_READ_PAGE_MAX_BYTES = 32 * 1024               # matches DEFAULT_READ_PAGE_MAX_BYTES
_MAX_ADAPTIVE_READ_MAX_BYTES = 128 * 1024              # matches MAX_ADAPTIVE_READ_MAX_BYTES
_ADAPTIVE_READ_CONTEXT_SHARE = 0.10                    # matches ADAPTIVE_READ_CONTEXT_SHARE
_CHARS_PER_TOKEN_ESTIMATE = 4                          # matches CHARS_PER_TOKEN_ESTIMATE
_DEFAULT_READ_LIMIT_LINES = 2000                       # explicit-limit fallback

_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mime_from_extension(path: str) -> Optional[str]:
    lower = path.lower()
    for ext, mime in _MIME_MAP.items():
        if lower.endswith(ext):
            return mime
    return None


def _format_bytes(n: int) -> str:
    """Mirror OpenClaw ``formatBytes`` (pi-tools.read.ts:84-92)."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}MB"
    if n >= 1024:
        return f"{round(n / 1024)}KB"
    return f"{n}B"


def _resolve_adaptive_read_max_bytes(context_window_tokens: Optional[int]) -> int:
    """Port of ``resolveAdaptiveReadMaxBytes`` (pi-tools.read.ts:69-82)."""
    if context_window_tokens is None or not isinstance(context_window_tokens, (int, float)):
        return _DEFAULT_READ_PAGE_MAX_BYTES
    if context_window_tokens <= 0:
        return _DEFAULT_READ_PAGE_MAX_BYTES
    from_context = int(
        context_window_tokens * _CHARS_PER_TOKEN_ESTIMATE * _ADAPTIVE_READ_CONTEXT_SHARE
    )
    return max(_DEFAULT_READ_PAGE_MAX_BYTES, min(_MAX_ADAPTIVE_READ_MAX_BYTES, from_context))


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------


@register_tool("read")
class ReadFileTool(BaseTool):
    """Read a text or image file from the remote VM.

    Text files return line-paginated content with adaptive byte-paging
    when no explicit ``limit`` is supplied. Image files (PNG/JPEG/GIF/
    WEBP/BMP/TIFF by extension) return a structured image block with
    MIME-sniff correction and size cap (10 MB default).

    Adapted from OpenClaw ``createHostWorkspaceReadTool`` (pi-tools.read.ts).
    """

    def __init__(
        self,
        registry: FilesystemRegistry,
        context_window_tokens: Optional[int] = None,
        cfg: Optional[dict] = None,
    ):
        self.registry = registry
        self.context_window_tokens = context_window_tokens
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Read a file. Pick a filesystem via `target` (default 'vm'). "
            "Text files return line-paginated content; image files "
            "(PNG/JPEG/GIF/WEBP/BMP/TIFF) return the raw image for direct "
            "inspection. " + self.registry.describe()
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": self.registry.names(),
                    "description": (
                        "Which filesystem to read from. "
                        + self.registry.describe()
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Path on the chosen target. For 'vm', an absolute "
                        "Windows path. For 'host', a path under the host "
                        "workspace (absolute or relative to the host root)."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line offset to start reading from (default 1).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to return. Omit to enable adaptive byte-paging.",
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Max bytes (image reads only; default 10 MB).",
                },
                "encoding": {
                    "type": "string",
                    "description": "Text decode encoding (default utf-8).",
                },
            },
            "required": ["path"],
        }

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        try:
            parsed = self._verify_json_format_args(params)
            path = _get_required_str(parsed, "path", "read")
            target = parsed.get("target", "vm")
            backend = self.registry.get(target)
            cap_err = _check_capability(backend, "read")
            if cap_err:
                return {"success": False, "error": f"Error: {cap_err}"}
            resolved = backend.resolve(path)
        except ValueError as e:
            return {"success": False, "error": f"Error: {e}"}

        return _run_tool_execute(
            self._execute(parsed, backend, resolved),
            logger,
            f"read tool failure on {path}",
        )

    async def _execute(
        self,
        params: dict,
        backend: FilesystemBackend,
        path: str,
    ) -> dict:
        ext_mime = _mime_from_extension(path)
        if ext_mime is not None:
            return await self._read_image(params, backend, path, ext_mime)
        return await self._read_text(params, backend, path)

    # -- image branch --

    async def _read_image(
        self,
        params: dict,
        backend: FilesystemBackend,
        path: str,
        declared_mime: str,
    ) -> dict:
        max_bytes_raw = params.get("max_bytes")
        # Per-call override flows into ImageLimits; otherwise OpenClaw defaults
        # (5 MB / 1200 px / 25 MP) live in image_sanitization.py.
        if isinstance(max_bytes_raw, (int, float)) and max_bytes_raw > 0:
            from .image_sanitization import ImageLimits
            limits = ImageLimits(max_bytes=int(max_bytes_raw))
        else:
            limits = _IMAGE_DEFAULT_LIMITS

        data = await backend.read_bytes(path)
        if not data:
            return {"success": False, "error": f"Error: file is empty: {path}"}

        sniffed = _sniff_mime_from_bytes(data)
        effective_mime = declared_mime
        if sniffed is not None:
            if not sniffed.startswith("image/"):
                return {
                    "success": False,
                    "error": (
                        f"Error: file looks like {sniffed} but was treated as "
                        f"{declared_mime} ({path})"
                    ),
                }
            effective_mime = sniffed

        sanitized = sanitize_raw_image_bytes(
            data, effective_mime, label=f"read:{path}", limits=limits
        )
        if isinstance(sanitized, str):
            return {"success": False, "error": f"Error: {sanitized}"}
        out_bytes, out_mime = sanitized

        b64 = base64.b64encode(out_bytes).decode("utf-8")
        return {
            "success": True,
            "type": "image",
            "data": b64,
            "mime_type": out_mime,
            "text": f"Read image file [{out_mime}]",
        }

    # -- text branch --

    async def _read_text(
        self,
        params: dict,
        backend: FilesystemBackend,
        path: str,
    ) -> dict:
        encoding = params.get("encoding") or "utf-8"
        try:
            raw_bytes = await backend.read_bytes(path)
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": f"Error: could not read file: {path} — {e}"}

        try:
            text = raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            return {
                "success": False,
                "error": (
                    f"Error: file is not {encoding} text ({path}) — use analyze_image "
                    "for images or pass encoding='<other>'."
                ),
            }

        lines = text.split("\n")
        total_lines = len(lines)

        offset_raw = params.get("offset")
        offset = (
            int(offset_raw)
            if isinstance(offset_raw, (int, float)) and offset_raw > 0
            else 1
        )
        if offset > total_lines:
            return {
                "success": True,
                "content": "",
                "truncated": False,
                "total_lines": total_lines,
                "next_offset": None,
            }

        limit_raw = params.get("limit")
        has_explicit_limit = (
            isinstance(limit_raw, (int, float)) and limit_raw > 0
        )

        if has_explicit_limit:
            limit = int(limit_raw)
            start = offset - 1
            end = min(total_lines, start + limit)
            sliced = lines[start:end]
            truncated = end < total_lines
            content = "\n".join(sliced)
            return {
                "success": True,
                "content": content,
                "truncated": truncated,
                "total_lines": total_lines,
                "next_offset": (end + 1) if truncated else None,
            }

        # Adaptive byte-paging: accumulate lines from offset until cap reached
        cap = _resolve_adaptive_read_max_bytes(self.context_window_tokens)
        start = offset - 1
        accumulated: list[str] = []
        accumulated_bytes = 0
        next_index = start
        capped = False
        for i in range(start, total_lines):
            line = lines[i]
            # Byte cost for this line (include separator)
            line_bytes = len(line.encode(encoding)) + (1 if accumulated else 0)
            if accumulated and accumulated_bytes + line_bytes > cap:
                capped = True
                break
            accumulated.append(line)
            accumulated_bytes += line_bytes
            next_index = i + 1

        content = "\n".join(accumulated)
        truncated = next_index < total_lines
        next_offset = (next_index + 1) if truncated else None
        if capped and next_offset is not None:
            content = (
                f"{content}\n\n[Read output capped at {_format_bytes(cap)} "
                f"for this call. Use offset={next_offset} to continue.]"
            )
        return {
            "success": True,
            "content": content,
            "truncated": truncated,
            "total_lines": total_lines,
            "next_offset": next_offset,
        }


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------


@register_tool("write")
class WriteFileTool(BaseTool):
    """Write (or append) UTF-8 text to a file on the remote VM.

    Adapted from OpenClaw ``createHostWorkspaceWriteTool`` (pi-tools.read.ts:656).
    """

    def __init__(
        self,
        registry: FilesystemRegistry,
        cfg: Optional[dict] = None,
    ):
        self.registry = registry
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Create, overwrite, or append to a UTF-8 text file. Pick a "
            "filesystem via `target` (default 'vm'). Default `append` "
            "behavior is target-aware: vm overwrites, host appends. "
            + self.registry.describe()
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": self.registry.names(),
                    "description": (
                        "Which filesystem to write to. "
                        + self.registry.describe()
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Path on the chosen target. For 'vm', an absolute "
                        "Windows path. For 'host', a path under the host "
                        "workspace (absolute or relative to the host root)."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "UTF-8 text content to write.",
                },
                "append": {
                    "type": "boolean",
                    "description": (
                        "Append instead of overwrite. Default depends on "
                        "target: false for 'vm', true for 'host'."
                    ),
                },
                "create_parents": {
                    "type": "boolean",
                    "description": "Create parent directories if missing (default true).",
                },
            },
            "required": ["path", "content"],
        }

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        try:
            parsed = self._verify_json_format_args(params)
            path = _get_required_str(parsed, "path", "write")
            if "content" not in parsed or not isinstance(parsed["content"], str):
                raise ValueError('write: required parameter "content" is missing or not a string')
            target = parsed.get("target", "vm")
            backend = self.registry.get(target)
            cap_err = _check_capability(backend, "write")
            if cap_err:
                return {"success": False, "error": f"Error: {cap_err}"}
            resolved = backend.resolve(path)
        except ValueError as e:
            return {"success": False, "error": f"Error: {e}"}

        return _run_tool_execute(
            self._execute(parsed, backend, resolved, target),
            logger,
            f"write tool failure on {path}",
        )

    async def _execute(
        self,
        params: dict,
        backend: FilesystemBackend,
        path: str,
        target: str,
    ) -> dict:
        content: str = params["content"]
        append = _default_append_for(target, params)
        create_parents = bool(params.get("create_parents", True))

        if create_parents:
            parent = _parent_dir(path)
            if parent and parent not in (".", "/", ""):
                try:
                    await backend.create_dir(parent)
                except Exception as e:  # noqa: BLE001 — mkdir failure is recoverable
                    logger.debug("create_dir(%s) failed (may already exist): %s", parent, e)

        await backend.write_text(path, content, append=append)
        return {
            "success": True,
            "bytes_written": len(content.encode("utf-8")),
            "path": path,
            "target": target,
            "append": append,
        }


# ---------------------------------------------------------------------------
# EditFileTool
# ---------------------------------------------------------------------------


@register_tool("edit")
class EditFileTool(BaseTool):
    """Exact-match string replacement on a file in the remote VM.

    Adapted from OpenClaw ``createHostWorkspaceEditTool`` (pi-tools.read.ts:663)
    plus ``wrapEditToolWithRecovery`` (pi-tools.host-edit.ts). We keep the
    mismatch-hint recovery (include current file contents on failure) but
    drop the post-write retroactive-success inference — CUA's ``write_text``
    is a single-shot RPC with no write-vs-stat race.
    """

    def __init__(
        self,
        registry: FilesystemRegistry,
        cfg: Optional[dict] = None,
    ):
        self.registry = registry
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Make precise edits to a file. Pick a filesystem via `target` "
            "(default 'vm'). Each `{oldText, newText}` replacement must "
            "match exactly. " + self.registry.describe()
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": self.registry.names(),
                    "description": (
                        "Which filesystem to edit. " + self.registry.describe()
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Path on the chosen target. For 'vm', an absolute "
                        "Windows path. For 'host', a path under the host "
                        "workspace (absolute or relative to the host root)."
                    ),
                },
                "edits": {
                    "type": "array",
                    "description": (
                        "Non-empty array of {oldText, newText} replacements. "
                        "oldText must be non-empty; newText may be empty (deletion)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string"},
                            "newText": {"type": "string"},
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            "required": ["path", "edits"],
        }

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        try:
            parsed = self._verify_json_format_args(params)
            path = _get_required_str(parsed, "path", "edit")
            edits = self._validate_edits(parsed.get("edits"))
            target = parsed.get("target", "vm")
            backend = self.registry.get(target)
            cap_err = _check_capability(backend, "edit")
            if cap_err:
                return {"success": False, "error": f"Error: {cap_err}"}
            resolved = backend.resolve(path)
        except ValueError as e:
            return {"success": False, "error": f"Error: {e}"}

        return _run_tool_execute(
            self._execute(backend, resolved, edits, target),
            logger,
            f"edit tool failure on {path}",
        )

    @staticmethod
    def _validate_edits(raw: Any) -> list[tuple[str, str]]:
        if not isinstance(raw, list) or not raw:
            raise ValueError('edit: "edits" must be a non-empty array')
        normalized: list[tuple[str, str]] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"edit: edits[{i}] must be an object")
            old_text = entry.get("oldText")
            new_text = entry.get("newText")
            if not isinstance(old_text, str) or old_text == "":
                raise ValueError(
                    f"edit: edits[{i}].oldText must be a non-empty string"
                )
            if not isinstance(new_text, str):
                raise ValueError(f"edit: edits[{i}].newText must be a string")
            normalized.append((old_text, new_text))
        return normalized

    async def _execute(
        self,
        backend: FilesystemBackend,
        path: str,
        edits: list[tuple[str, str]],
        target: str,
    ) -> dict:
        try:
            original_bytes = await backend.read_bytes(path)
            original = original_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "success": False,
                "error": (
                    f"Error: file is not utf-8 text ({path}); edit only supports "
                    "utf-8 text files."
                ),
            }
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": f"Error: could not read file: {path} — {e}"}

        current = original
        for idx, (old_text, new_text) in enumerate(edits):
            pos = current.find(old_text)
            if pos < 0:
                snippet = current[:_MAX_MISMATCH_HINT_CHARS]
                truncated = len(current) > _MAX_MISMATCH_HINT_CHARS
                hint_suffix = "\n... (truncated)" if truncated else ""
                return {
                    "success": False,
                    "error": (
                        f"Error: could not find the exact text in {path} for "
                        f"edits[{idx}].oldText.\nCurrent file contents "
                        f"(first {_MAX_MISMATCH_HINT_CHARS} chars):\n"
                        f"{snippet}{hint_suffix}"
                    ),
                }
            current = current[:pos] + new_text + current[pos + len(old_text) :]

        if current == original:
            return {
                "success": True,
                "edits_applied": len(edits),
                "path": path,
                "target": target,
                "unchanged": True,
            }

        await backend.write_text(path, current, append=False)
        return {
            "success": True,
            "edits_applied": len(edits),
            "path": path,
            "target": target,
        }
