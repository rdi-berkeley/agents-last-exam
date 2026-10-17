"""Content identity for a task folder, independent of repository history."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


_GENERATED_NAMES = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "node_modules", ".venv", ".DS_Store",
})


def task_revision(task_directory: Path) -> str:
    """Hash file paths, bytes, executable bits and links; exclude runtime caches."""
    task_directory = Path(task_directory).resolve(strict=True)
    if not task_directory.is_dir():
        raise NotADirectoryError(task_directory)
    entries: list[Path] = []
    pending = [task_directory]
    while pending:
        directory = pending.pop()
        for entry in directory.iterdir():
            if entry.name in _GENERATED_NAMES or entry.suffix in {".pyc", ".pyo"}:
                continue
            if entry.is_symlink():
                entries.append(entry)
            elif entry.is_dir():
                pending.append(entry)
            else:
                entries.append(entry)

    digest = hashlib.sha256(b"ALE task tree v1\0")
    for entry in sorted(entries, key=lambda item: item.relative_to(task_directory).as_posix()):
        relative = entry.relative_to(task_directory).as_posix()
        before = entry.lstat()
        if stat.S_ISLNK(before.st_mode):
            kind = "symlink"
            content = os.readlink(entry)
        elif stat.S_ISREG(before.st_mode):
            kind = "executable" if before.st_mode & 0o111 else "file"
            file_digest = hashlib.sha256()
            with entry.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            content = file_digest.hexdigest()
        else:
            raise ValueError(f"Unsupported task file type: {entry}")
        after = entry.lstat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise RuntimeError(f"Task file changed while computing its revision: {entry}")
        record = json.dumps([relative, kind, content], ensure_ascii=True, separators=(",", ":"))
        digest.update(record.encode("utf-8") + b"\0")
    return "sha256:" + digest.hexdigest()
