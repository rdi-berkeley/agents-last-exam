"""Shared path helpers for the filesystem/shell tools.

Extracted from ``tools_fs`` so ``fs_backends`` can reach
``_assert_within_workspace`` without importing ``tools_fs`` (which imports
``fs_backends`` back) — this module depends only on the stdlib, breaking the
former ``tools_fs`` <-> ``fs_backends`` import cycle.

Windows/POSIX paths are both supported because the remote VM may be either;
detection is heuristic (drive letter, UNC prefix, or any backslash).
"""

from __future__ import annotations

import ntpath
import posixpath
import re
from typing import Optional


def _is_windows_path(path: str) -> bool:
    """Detect Windows-style absolute paths."""
    return bool(
        re.match(r"^[A-Za-z]:[\\/]", path)
        or path.startswith("\\\\")
        or "\\" in path
    )


def _normalize_path(path: str) -> str:
    """Normalize a path using ntpath for Windows, posixpath otherwise."""
    if _is_windows_path(path):
        return ntpath.normpath(path)
    return posixpath.normpath(path)


def _parent_dir(path: str) -> str:
    if _is_windows_path(path):
        return ntpath.dirname(path)
    return posixpath.dirname(path)


def _assert_within_workspace(path: str, workspace_root: Optional[str]) -> None:
    """Raise ``ValueError`` if ``path`` is outside ``workspace_root``.

    Permissive no-op when ``workspace_root is None``.
    On Windows paths comparison is case-insensitive (drive-letter semantics).
    """
    if not workspace_root:
        return
    is_win = _is_windows_path(workspace_root) or _is_windows_path(path)
    normalized_path = _normalize_path(path)
    normalized_root = _normalize_path(workspace_root)
    sep = "\\" if is_win else "/"
    if is_win:
        candidate = normalized_path.lower()
        root = normalized_root.lower()
    else:
        candidate = normalized_path
        root = normalized_root
    if candidate == root or candidate.startswith(root + sep):
        return
    raise ValueError(
        f"path '{path}' is outside the task workspace ('{workspace_root}')."
    )
