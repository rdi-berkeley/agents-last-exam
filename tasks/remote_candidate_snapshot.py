"""Authenticated, race-detecting snapshots of untrusted remote output trees.

Task evaluators must never mix an inventory taken at one instant with bytes or
executables reopened later from the candidate-owned path.  This module keeps
that boundary in one place.  A literal controller on the task VM inventories
the whole tree, rejects links/special files/hard links, copies through
``O_NOFOLLOW`` descriptors (or Windows reparse-point handles), seals the copy,
and reports a content- and inode-bound manifest.  The host transfers only that
copy, verifies every byte, materializes a read-only local tree, and rechecks
both copies after the consumer has finished.

The controller is embedded in the command.  It does not import candidate or
staged evaluator code, and a missing candidate root becomes an authenticated
empty snapshot so evaluator prerequisites can still be checked first.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from tasks.command_results import extract_return_code, is_exact_zero_return_code


PROTOCOL = "agenthle.remote-candidate-snapshot.v1"
SNAPSHOT_PREFIX = "agenthle-candidate-snapshot-"
SNAPSHOT_PARENT_PREFIX = "agenthle-candidate-snapshot-parent-"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NONCE = re.compile(r"^[0-9a-f]{48}$")


class RemoteSnapshotError(RuntimeError):
    """The evaluator could not produce or authenticate a candidate snapshot."""


class CandidateSnapshotRejected(ValueError):
    """The candidate tree has a safely classifiable structural defect."""


@dataclass(frozen=True)
class SnapshotLimits:
    max_entries: int = 4096
    max_file_bytes: int = 512 * 1024 * 1024
    max_total_bytes: int = 2 * 1024 * 1024 * 1024
    max_depth: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("max_entries", self.max_entries),
            ("max_file_bytes", self.max_file_bytes),
            ("max_total_bytes", self.max_total_bytes),
            ("max_depth", self.max_depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    size: int
    sha256: str


_REMOTE_CONTROLLER = r"""
import base64, hashlib, json, os, re, shutil, stat, sys, tempfile

PROTOCOL = "agenthle.remote-candidate-snapshot.v1"
PREFIX = "agenthle-candidate-snapshot-"
PARENT_PREFIX = "agenthle-candidate-snapshot-parent-"
NONCE = re.compile(r"^[0-9a-f]{48}$")

class CandidateProblem(Exception):
    pass

class EnvironmentProblem(Exception):
    pass

def emit(status, **values):
    print(json.dumps({"protocol": PROTOCOL, "status": status, **values}, sort_keys=True, separators=(",", ":")))

def kind(mode):
    if stat.S_ISREG(mode): return "file"
    if stat.S_ISDIR(mode): return "directory"
    if stat.S_ISLNK(mode): return "symlink"
    return "special"

def metadata(value):
    return {
        "type": kind(value.st_mode),
        "mode": stat.S_IMODE(value.st_mode),
        "dev": int(value.st_dev),
        "ino": int(value.st_ino),
        "nlink": int(value.st_nlink),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "uid": int(value.st_uid) if hasattr(value, "st_uid") else None,
        "gid": int(value.st_gid) if hasattr(value, "st_gid") else None,
    }

def same_identity(left, right):
    keys = ("type", "mode", "dev", "ino", "nlink", "size", "mtime_ns", "ctime_ns", "uid", "gid")
    return all(left[key] == right[key] for key in keys)

def expected_paths(nonce):
    if not isinstance(nonce, str) or NONCE.fullmatch(nonce) is None:
        raise EnvironmentProblem("snapshot authorization nonce is malformed")
    temporary = os.path.abspath(tempfile.gettempdir())
    parent = os.path.join(temporary, PARENT_PREFIX + nonce)
    root = os.path.join(parent, PREFIX + nonce)
    if os.path.dirname(parent) != temporary or os.path.dirname(root) != parent:
        raise EnvironmentProblem("snapshot authorization paths are unsafe")
    return parent, root

def effective_uid():
    return int(os.geteuid()) if hasattr(os, "geteuid") else None

def require_owned_directory(path, mode, label):
    value = os.lstat(path)
    item = metadata(value)
    if item["type"] != "directory" or item["mode"] != mode:
        raise EnvironmentProblem(label + " has invalid type or mode")
    owner = effective_uid()
    if owner is not None and item["uid"] != owner:
        raise EnvironmentProblem(label + " has invalid owner")
    return item

def authorization_state(nonce):
    parent, root = expected_paths(nonce)
    body = {
        "nonce": nonce,
        "parent_path": parent,
        "root_path": root,
        "owner_uid": effective_uid(),
        "parent": require_owned_directory(parent, 0o500, "snapshot parent"),
        "root": require_owned_directory(root, 0o500, "snapshot root"),
    }
    if body["parent"]["dev"] == body["root"]["dev"] and body["parent"]["ino"] == body["root"]["ino"]:
        raise EnvironmentProblem("snapshot parent and root share an inode")
    body["fingerprint"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body

def require_authorization(spec):
    expected = spec.get("authorization")
    if not isinstance(expected, dict):
        raise EnvironmentProblem("snapshot authorization is missing")
    observed = authorization_state(spec.get("nonce"))
    if expected != observed:
        raise EnvironmentProblem("candidate snapshot inode, inventory, or hash changed (authorization)")
    return observed

def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise CandidateProblem("unsafe candidate entry name")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise CandidateProblem("unsafe candidate entry path")
    return parts

def linux_root(path):
    if not hasattr(os, "O_NOFOLLOW"):
        raise EnvironmentProblem("runtime lacks O_NOFOLLOW")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CandidateProblem("candidate root is not a safe directory: %s" % exc)
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        os.close(descriptor)
        raise CandidateProblem("candidate root is not a directory")
    return descriptor, metadata(observed)

def linux_scan(root_fd, limits=None):
    entries = []
    total_bytes = 0
    def record(item):
        nonlocal total_bytes
        entries.append(item)
        if limits is None:
            return
        if len(entries) > limits["max_entries"]:
            raise CandidateProblem("candidate tree exceeds the entry limit")
        if item["type"] == "file":
            if item["size"] < 0 or item["size"] > limits["max_file_bytes"]:
                raise CandidateProblem("candidate file exceeds the per-file limit: " + item["path"])
            total_bytes += item["size"]
            if total_bytes > limits["max_total_bytes"]:
                raise CandidateProblem("candidate tree exceeds the total byte limit")
    def visit(directory_fd, prefix, depth):
        if limits is not None and depth > limits["max_depth"]:
            raise CandidateProblem("candidate tree exceeds the depth limit")
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise CandidateProblem("candidate directory cannot be inventoried: %s" % exc)
        for name in names:
            if not name or name in (".", "..") or "/" in name or "\\" in name:
                raise CandidateProblem("unsafe candidate entry name")
            relative = prefix + name
            try:
                before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise CandidateProblem("candidate entry disappeared during inventory: %s" % exc)
            item = metadata(before)
            if item["type"] not in ("file", "directory"):
                raise CandidateProblem("candidate tree contains a link or special file: " + relative)
            if item["type"] == "file" and item["nlink"] != 1:
                raise CandidateProblem("candidate tree contains a hard-linked file: " + relative)
            record({"path": relative, **item})
            if item["type"] == "directory":
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
                try:
                    child = os.open(name, flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise CandidateProblem("candidate directory changed during inventory: %s" % exc)
                try:
                    if not same_identity(item, metadata(os.fstat(child))):
                        raise CandidateProblem("candidate directory changed during inventory: " + relative)
                    visit(child, relative + "/", depth + 1)
                finally:
                    os.close(child)
    visit(root_fd, "", 1)
    return entries

def linux_open_parent(root_fd, parts, inventory):
    current = os.dup(root_fd)
    prefix = []
    try:
        for part in parts:
            prefix.append(part)
            relative = "/".join(prefix)
            expected = inventory.get(relative)
            if expected is None or expected["type"] != "directory":
                raise CandidateProblem("candidate parent inventory mismatch: " + relative)
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
            child = os.open(part, flags, dir_fd=current)
            observed = metadata(os.fstat(child))
            os.close(current)
            current = child
            if not same_identity(expected, observed):
                raise CandidateProblem("candidate directory changed before transfer: " + relative)
        return current
    except BaseException:
        os.close(current)
        raise

def windows_api():
    import ctypes, msvcrt
    from ctypes import wintypes
    class Info(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD), ("creation_low", wintypes.DWORD),
            ("creation_high", wintypes.DWORD), ("access_low", wintypes.DWORD),
            ("access_high", wintypes.DWORD), ("write_low", wintypes.DWORD),
            ("write_high", wintypes.DWORD), ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD), ("size_low", wintypes.DWORD),
            ("nlink", wintypes.DWORD), ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        ]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    get_info = kernel.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Info)]
    get_info.restype = wintypes.BOOL
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    invalid = wintypes.HANDLE(-1).value
    return ctypes, msvcrt, Info, create, get_info, close, invalid

def windows_open(path, directory=False, keep=False):
    ctypes, msvcrt, Info, create, get_info, close, invalid = windows_api()
    GENERIC_READ = 0x80000000
    SHARE_ALL = 0x00000001 | 0x00000002 | 0x00000004
    OPEN_EXISTING = 3
    OPEN_REPARSE = 0x00200000
    BACKUP = 0x02000000
    REPARSE_ATTRIBUTE = 0x00000400
    handle = create(path, GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, OPEN_REPARSE | (BACKUP if directory else 0), None)
    if handle == invalid:
        raise OSError(ctypes.get_last_error(), "CreateFileW failed", path)
    info = Info()
    if not get_info(handle, ctypes.byref(info)):
        error = ctypes.get_last_error()
        close(handle)
        raise OSError(error, "GetFileInformationByHandle failed", path)
    if info.attributes & REPARSE_ATTRIBUTE:
        close(handle)
        raise CandidateProblem("candidate tree contains a reparse point: " + path)
    value = os.lstat(path)
    item = metadata(value)
    item.update({
        "dev": int(info.volume),
        "ino": (int(info.index_high) << 32) | int(info.index_low),
        "nlink": int(info.nlink),
        "size": (int(info.size_high) << 32) | int(info.size_low),
    })
    if keep:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        return msvcrt.open_osfhandle(handle, flags), item
    close(handle)
    return None, item

def windows_scan(root, limits=None):
    try:
        _, root_meta = windows_open(root, directory=True)
    except CandidateProblem:
        raise
    except OSError as exc:
        raise CandidateProblem("candidate root is not a safe directory: %s" % exc)
    if root_meta["type"] != "directory":
        raise CandidateProblem("candidate root is not a directory")
    entries = []
    total_bytes = 0
    def record(item):
        nonlocal total_bytes
        entries.append(item)
        if limits is None:
            return
        if len(entries) > limits["max_entries"]:
            raise CandidateProblem("candidate tree exceeds the entry limit")
        if item["type"] == "file":
            if item["size"] < 0 or item["size"] > limits["max_file_bytes"]:
                raise CandidateProblem("candidate file exceeds the per-file limit: " + item["path"])
            total_bytes += item["size"]
            if total_bytes > limits["max_total_bytes"]:
                raise CandidateProblem("candidate tree exceeds the total byte limit")
    def visit(directory, prefix, depth):
        if limits is not None and depth > limits["max_depth"]:
            raise CandidateProblem("candidate tree exceeds the depth limit")
        try:
            values = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise CandidateProblem("candidate directory cannot be inventoried: %s" % exc)
        for value in values:
            name = value.name
            if not name or name in (".", "..") or "/" in name or "\\" in name:
                raise CandidateProblem("unsafe candidate entry name")
            relative = prefix + name
            path = os.path.join(directory, name)
            _, item = windows_open(path, directory=value.is_dir(follow_symlinks=False))
            if item["type"] not in ("file", "directory"):
                raise CandidateProblem("candidate tree contains a link or special file: " + relative)
            if item["type"] == "file" and item["nlink"] != 1:
                raise CandidateProblem("candidate tree contains a hard-linked file: " + relative)
            record({"path": relative, **item})
            if item["type"] == "directory":
                visit(path, relative + "/", depth + 1)
    visit(root, "", 1)
    return root_meta, entries

def scan_source(root, limits=None):
    if os.name == "nt":
        return windows_scan(root, limits)
    descriptor, root_meta = linux_root(root)
    try:
        return root_meta, linux_scan(descriptor, limits)
    finally:
        os.close(descriptor)

def enforce_limits(entries, limits):
    if len(entries) > limits["max_entries"]:
        raise CandidateProblem("candidate tree exceeds the entry limit")
    total = 0
    for item in entries:
        if item["type"] != "file":
            continue
        if item["size"] < 0 or item["size"] > limits["max_file_bytes"]:
            raise CandidateProblem("candidate file exceeds the per-file limit: " + item["path"])
        total += item["size"]
        if total > limits["max_total_bytes"]:
            raise CandidateProblem("candidate tree exceeds the total byte limit")

def enforce_source_binding(expected, root_meta, entries, digests):
    if expected is None:
        return
    if not isinstance(expected, dict):
        raise EnvironmentProblem("expected source binding is malformed")
    bound_root = expected.get("root")
    bound_entries = expected.get("entries")
    if not isinstance(bound_root, dict) or not isinstance(bound_entries, list):
        raise EnvironmentProblem("expected source binding is malformed")
    observed = []
    for item in entries:
        value = dict(item)
        if item["type"] == "file":
            digest = digests.get(item["path"])
            if digest is None:
                raise EnvironmentProblem("source binding omitted a file digest")
            value["sha256"] = digest
        observed.append(value)
    if bound_root != root_meta or bound_entries != observed:
        raise CandidateProblem("candidate source inode, inventory, or digest changed")

def copy_stream(source_fd, destination_fd):
    digest = hashlib.sha256()
    total = 0
    while True:
        block = os.read(source_fd, 1024 * 1024)
        if not block:
            break
        digest.update(block)
        total += len(block)
        view = memoryview(block)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:
                raise EnvironmentProblem("short write while creating candidate snapshot")
            view = view[written:]
    return digest.hexdigest(), total

def transfer_linux(source, destination, entries):
    root_fd, root_meta = linux_root(source)
    inventory = {item["path"]: item for item in entries}
    digests = {}
    try:
        for item in entries:
            parts = safe_relative(item["path"])
            target = os.path.join(destination, *parts)
            if item["type"] == "directory":
                os.mkdir(target, 0o700)
                continue
            parent = linux_open_parent(root_fd, parts[:-1], inventory)
            try:
                before_path = metadata(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False))
                if not same_identity(item, before_path):
                    raise CandidateProblem("candidate file changed before transfer: " + item["path"])
                source_fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW, dir_fd=parent)
                try:
                    if not same_identity(item, metadata(os.fstat(source_fd))):
                        raise CandidateProblem("candidate file changed before transfer: " + item["path"])
                    destination_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW, 0o600)
                    try:
                        digest, size = copy_stream(source_fd, destination_fd)
                        os.fsync(destination_fd)
                        os.fchmod(destination_fd, 0o400)
                    finally:
                        os.close(destination_fd)
                    after_fd = metadata(os.fstat(source_fd))
                    after_path = metadata(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False))
                    if not same_identity(item, after_fd) or not same_identity(item, after_path):
                        raise CandidateProblem("candidate file changed during transfer: " + item["path"])
                    if size != item["size"]:
                        raise CandidateProblem("candidate file size changed during transfer: " + item["path"])
                    digests[item["path"]] = digest
                finally:
                    os.close(source_fd)
            finally:
                os.close(parent)
        if not same_identity(root_meta, metadata(os.fstat(root_fd))):
            raise CandidateProblem("candidate root changed during transfer")
    finally:
        os.close(root_fd)
    return digests

def transfer_windows(source, destination, entries):
    digests = {}
    for item in entries:
        parts = safe_relative(item["path"])
        source_path = os.path.join(source, *parts)
        target = os.path.join(destination, *parts)
        if item["type"] == "directory":
            _, observed = windows_open(source_path, directory=True)
            if not same_identity(item, observed):
                raise CandidateProblem("candidate directory changed before transfer: " + item["path"])
            os.mkdir(target)
            continue
        source_fd, observed = windows_open(source_path, keep=True)
        try:
            if not same_identity(item, observed):
                raise CandidateProblem("candidate file changed before transfer: " + item["path"])
            destination_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
            try:
                digest, size = copy_stream(source_fd, destination_fd)
                os.fsync(destination_fd)
            finally:
                os.close(destination_fd)
            _, after_path = windows_open(source_path)
            if not same_identity(item, after_path) or size != item["size"]:
                raise CandidateProblem("candidate file changed during transfer: " + item["path"])
            os.chmod(target, stat.S_IREAD)
            digests[item["path"]] = digest
        finally:
            os.close(source_fd)
    return digests

def hash_file(path):
    if os.name == "nt":
        descriptor, observed = windows_open(path, keep=True)
    else:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW)
        observed = metadata(os.fstat(descriptor))
    try:
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block: break
            digest.update(block); total += len(block)
        after = metadata(os.fstat(descriptor)) if os.name != "nt" else windows_open(path)[1]
        if not same_identity(observed, after) or total != observed["size"]:
            raise EnvironmentProblem("snapshot file changed while hashing: " + path)
        return digest.hexdigest(), observed
    finally:
        os.close(descriptor)

def snapshot_state(root):
    root_meta, entries = scan_source(root)
    files = []
    normalized = []
    for item in entries:
        relative = item["path"]
        observed = dict(item)
        if item["type"] == "file":
            digest, file_meta = hash_file(os.path.join(root, *safe_relative(relative)))
            if not same_identity(item, file_meta):
                raise EnvironmentProblem("snapshot metadata changed while hashing: " + relative)
            files.append({"path": relative, "size": item["size"], "sha256": digest})
            observed["sha256"] = digest
        normalized.append(observed)
    body = {"root": root_meta, "entries": normalized}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    directories = [item["path"] for item in entries if item["type"] == "directory"]
    return digest, files, directories, body

def seal_tree(root, entries):
    for item in entries:
        if item["type"] == "file":
            os.chmod(os.path.join(root, *safe_relative(item["path"])), stat.S_IREAD)
    directories = [item for item in entries if item["type"] == "directory"]
    for item in reversed(directories):
        os.chmod(os.path.join(root, *safe_relative(item["path"])), stat.S_IREAD | stat.S_IEXEC)
    os.chmod(root, stat.S_IREAD | stat.S_IEXEC)

def remove_tree(root):
    try:
        value = os.lstat(root)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(value.st_mode):
        os.unlink(root)
        return
    if not stat.S_ISDIR(value.st_mode):
        raise EnvironmentProblem("refusing to recursively clean a non-directory snapshot")
    os.chmod(root, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        for name in [*names, *files]:
            path = os.path.join(directory, name)
            value = os.lstat(path)
            if stat.S_ISLNK(value.st_mode):
                os.unlink(path)
                if name in names:
                    names.remove(name)
            elif stat.S_ISDIR(value.st_mode):
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            else:
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    def repair(function, path, _exc):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        function(path)
    shutil.rmtree(root, onerror=repair)

def cleanup_created(nonce, authorization=None):
    parent, root = expected_paths(nonce)
    if authorization is not None:
        observed = authorization_state(nonce)
        if authorization != observed:
            raise EnvironmentProblem("refusing to clean a changed snapshot authorization")
    try:
        parent_value = os.lstat(parent)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(parent_value.st_mode):
        raise EnvironmentProblem("refusing to clean a linked snapshot parent")
    if not stat.S_ISDIR(parent_value.st_mode):
        raise EnvironmentProblem("refusing to clean a non-directory snapshot parent")
    os.chmod(parent, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    remove_tree(root)
    try:
        parent_value = os.lstat(parent)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(parent_value.st_mode):
        raise EnvironmentProblem("refusing to clean a linked snapshot parent")
    if not stat.S_ISDIR(parent_value.st_mode):
        raise EnvironmentProblem("refusing to clean a non-directory snapshot parent")
    os.rmdir(parent)

def create(spec):
    source = spec["source_root"]
    limits = spec["limits"]
    nonce = spec.get("nonce")
    parent, snapshot = expected_paths(nonce)
    os.mkdir(parent, 0o700)
    try:
        parent_meta = metadata(os.lstat(parent))
        if parent_meta["type"] != "directory" or parent_meta["mode"] != 0o700:
            raise EnvironmentProblem("exclusive snapshot parent was not created safely")
        if effective_uid() is not None and parent_meta["uid"] != effective_uid():
            raise EnvironmentProblem("exclusive snapshot parent has an invalid owner")
        os.mkdir(snapshot, 0o700)
    except BaseException:
        cleanup_created(nonce)
        raise
    source_missing = False
    try:
        try:
            os.lstat(source)
        except FileNotFoundError:
            source_missing = True
            entries = []
            root_meta = None
            if spec.get("source_binding") is not None:
                raise CandidateProblem("bound candidate source disappeared")
        else:
            root_meta, entries = scan_source(source, limits)
            enforce_limits(entries, limits)
            digests = transfer_windows(source, snapshot, entries) if os.name == "nt" else transfer_linux(source, snapshot, entries)
            after_root, after_entries = scan_source(source, limits)
            if not same_identity(root_meta, after_root) or entries != after_entries:
                raise CandidateProblem("candidate tree changed during transfer")
            for item in entries:
                if item["type"] == "file" and item["path"] not in digests:
                    raise EnvironmentProblem("snapshot transfer omitted a candidate file")
            enforce_source_binding(spec.get("source_binding"), root_meta, entries, digests)
        seal_tree(snapshot, entries)
        os.chmod(parent, stat.S_IREAD | stat.S_IEXEC)
        digest, files, directories, source_binding = snapshot_state(snapshot)
        authorization = authorization_state(nonce)
        emit("ok", source_missing=source_missing, snapshot_parent=parent, snapshot_root=snapshot, snapshot_digest=digest, authorization=authorization, source_binding=source_binding, files=files, directories=directories)
        return
    except BaseException:
        cleanup_created(nonce)
        raise

def verify(spec):
    authorization = require_authorization(spec)
    digest, _files, _directories, _source_binding = snapshot_state(authorization["root_path"])
    if digest != spec["snapshot_digest"]:
        raise EnvironmentProblem("candidate snapshot inode, inventory, or hash changed")
    if authorization_state(spec["nonce"]) != authorization:
        raise EnvironmentProblem("candidate snapshot authorization changed during verification")
    emit("verified", snapshot_digest=digest, authorization_fingerprint=authorization["fingerprint"])

def cleanup(spec):
    authorization = require_authorization(spec)
    cleanup_created(spec["nonce"], authorization)
    emit("cleaned")

spec = json.loads(base64.urlsafe_b64decode(sys.argv[1].encode()).decode())
try:
    operation = spec.get("operation")
    if operation == "create": create(spec)
    elif operation == "verify": verify(spec)
    elif operation == "cleanup": cleanup(spec)
    else: raise EnvironmentProblem("unknown snapshot operation")
except CandidateProblem as exc:
    emit("candidate_invalid", reason=str(exc))
except EnvironmentProblem as exc:
    emit("environment_error", reason=str(exc))
except Exception as exc:
    emit("environment_error", reason="%s: %s" % (type(exc).__name__, exc))
"""


def _encoded_command(spec: Mapping[str, Any], *, windows: bool) -> str:
    script = base64.b64encode(
        zlib.compress(_REMOTE_CONTROLLER.encode(), level=9)
    ).decode()
    payload = base64.urlsafe_b64encode(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    python = "python" if windows else "/usr/bin/python3"
    # Both POSIX shells and cmd.exe preserve these base64-only single arguments.
    return (
        f'{python} -I -c "import base64,zlib;'
        f"exec(zlib.decompress(base64.b64decode('{script}')))\" {payload}"
        " # O_NOFOLLOW descriptor snapshot"
    )


def _command_value(result: object, name: str, default: object = None) -> object:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


async def _run_controller(
    session: object,
    spec: Mapping[str, Any],
    *,
    windows: bool,
    timeout: float,
) -> dict[str, Any]:
    command = _encoded_command(spec, windows=windows)
    try:
        result = await session.run_command(command, check=False, timeout=timeout)
    except TypeError:
        try:
            result = await session.run_command(command, check=False)
        except TypeError:
            result = await session.run_command(command)
    stdout = str(_command_value(result, "stdout", "") or "")
    stderr = str(_command_value(result, "stderr", "") or "")
    if not is_exact_zero_return_code(extract_return_code(result)):
        raise RemoteSnapshotError(
            "snapshot controller failed: " + (stderr[-2000:] or stdout[-2000:])
        )
    report: object | None = None
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        break
    if not isinstance(report, dict) or report.get("protocol") != PROTOCOL:
        raise RemoteSnapshotError(
            "snapshot controller returned no valid protocol report"
        )
    status = report.get("status")
    if status == "candidate_invalid":
        raise CandidateSnapshotRejected(
            str(report.get("reason") or "unsafe candidate tree")
        )
    if status == "environment_error":
        raise RemoteSnapshotError(
            str(report.get("reason") or "snapshot controller failed")
        )
    return report


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RemoteSnapshotError("snapshot manifest contains an unsafe path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise RemoteSnapshotError("snapshot manifest contains an unsafe path")
    if parsed.as_posix() != value:
        raise RemoteSnapshotError("snapshot manifest path is not canonical")
    return value


def _safe_snapshot_paths(
    parent_value: object,
    root_value: object,
    *,
    nonce: str,
    windows: bool,
) -> tuple[str, str]:
    if not isinstance(parent_value, str) or not parent_value:
        raise RemoteSnapshotError("snapshot controller omitted its parent")
    if not isinstance(root_value, str) or not root_value:
        raise RemoteSnapshotError("snapshot controller omitted its root")
    path_type = PureWindowsPath if windows else PurePosixPath
    parent = path_type(parent_value)
    root = path_type(root_value)
    if (
        not parent.is_absolute()
        or not root.is_absolute()
        or parent.name != SNAPSHOT_PARENT_PREFIX + nonce
        or root.name != SNAPSHOT_PREFIX + nonce
        or root.parent != parent
        or str(parent) != parent_value.rstrip("/\\")
        or str(root) != root_value.rstrip("/\\")
    ):
        raise RemoteSnapshotError("snapshot controller returned an unauthorized path")
    return parent_value, root_value


_REMOTE_METADATA_KEYS = {
    "type",
    "mode",
    "dev",
    "ino",
    "nlink",
    "size",
    "mtime_ns",
    "ctime_ns",
    "uid",
    "gid",
}


def _validated_authorization(
    value: object,
    *,
    nonce: str,
    parent_path: str,
    root_path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RemoteSnapshotError("snapshot controller omitted its authorization")
    expected_keys = {
        "nonce",
        "parent_path",
        "root_path",
        "owner_uid",
        "parent",
        "root",
        "fingerprint",
    }
    if set(value) != expected_keys:
        raise RemoteSnapshotError("snapshot authorization is malformed")
    owner = value.get("owner_uid")
    if owner is not None and (isinstance(owner, bool) or not isinstance(owner, int)):
        raise RemoteSnapshotError("snapshot authorization owner is malformed")
    if (
        value.get("nonce") != nonce
        or value.get("parent_path") != parent_path
        or value.get("root_path") != root_path
    ):
        raise RemoteSnapshotError("snapshot authorization path or nonce mismatch")
    metadata_values: list[dict[str, Any]] = []
    for name in ("parent", "root"):
        item = value.get(name)
        if not isinstance(item, dict) or set(item) != _REMOTE_METADATA_KEYS:
            raise RemoteSnapshotError("snapshot authorization metadata is malformed")
        if item.get("type") != "directory" or item.get("mode") != 0o500:
            raise RemoteSnapshotError("snapshot authorization type or mode is invalid")
        for field in _REMOTE_METADATA_KEYS - {"type", "uid", "gid"}:
            field_value = item.get(field)
            if isinstance(field_value, bool) or not isinstance(field_value, int):
                raise RemoteSnapshotError(
                    "snapshot authorization metadata is malformed"
                )
        for field in ("uid", "gid"):
            field_value = item.get(field)
            if field_value is not None and (
                isinstance(field_value, bool) or not isinstance(field_value, int)
            ):
                raise RemoteSnapshotError(
                    "snapshot authorization metadata is malformed"
                )
        if owner is not None and item.get("uid") != owner:
            raise RemoteSnapshotError("snapshot authorization owner mismatch")
        metadata_values.append(item)
    if (metadata_values[0]["dev"], metadata_values[0]["ino"]) == (
        metadata_values[1]["dev"],
        metadata_values[1]["ino"],
    ):
        raise RemoteSnapshotError("snapshot parent and root share an inode")
    fingerprint = value.get("fingerprint")
    unsigned = {key: item for key, item in value.items() if key != "fingerprint"}
    expected_fingerprint = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        not isinstance(fingerprint, str)
        or _SHA256.fullmatch(fingerprint) is None
        or not secrets.compare_digest(fingerprint, expected_fingerprint)
    ):
        raise RemoteSnapshotError("snapshot authorization fingerprint mismatch")
    return dict(value)


def _remote_join(root: str, relative: str, *, windows: bool) -> str:
    separator = "\\" if windows else "/"
    return root.rstrip("/\\") + separator + relative.replace("/", separator)


def _local_state(root: Path) -> str:
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RemoteSnapshotError("local candidate snapshot root changed type")
    entries: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        before = path.lstat()
        item = {
            "path": relative,
            "type": "directory"
            if stat.S_ISDIR(before.st_mode)
            else "file"
            if stat.S_ISREG(before.st_mode)
            else "unsafe",
            "mode": stat.S_IMODE(before.st_mode),
            "dev": before.st_dev,
            "ino": before.st_ino,
            "nlink": before.st_nlink,
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }
        if item["type"] == "unsafe" or (item["type"] == "file" and item["nlink"] != 1):
            raise RemoteSnapshotError(
                "local candidate snapshot contains an unsafe entry"
            )
        if item["type"] == "file":
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            if not hasattr(os, "O_NOFOLLOW"):
                raise RemoteSnapshotError("local runtime lacks O_NOFOLLOW")
            descriptor = os.open(path, flags | os.O_NOFOLLOW)
            try:
                opened = os.fstat(descriptor)
                digest = hashlib.sha256()
                total = 0
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    total += len(block)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            identity = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if (
                identity(before) != identity(opened)
                or identity(before) != identity(after)
                or total != before.st_size
            ):
                raise RemoteSnapshotError(
                    "local candidate snapshot changed while hashing"
                )
            item["sha256"] = digest.hexdigest()
        entries.append(item)
    root_item = {
        "type": "directory",
        "mode": stat.S_IMODE(root_stat.st_mode),
        "dev": root_stat.st_dev,
        "ino": root_stat.st_ino,
        "nlink": root_stat.st_nlink,
        "size": root_stat.st_size,
        "mtime_ns": root_stat.st_mtime_ns,
        "ctime_ns": root_stat.st_ctime_ns,
    }
    return hashlib.sha256(
        json.dumps(
            {"root": root_item, "entries": entries},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_local_snapshot_file(
    root_descriptor: int,
    relative: str,
    expected: SnapshotFile,
    identities: Mapping[str, tuple[int, ...]],
) -> bytes:
    """Read one sealed file through descriptors rooted at the authenticated tree."""

    parts = PurePosixPath(_relative_path(relative)).parts
    directory_descriptor = os.dup(root_descriptor)
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | os.O_NOFOLLOW
        )
        walked: list[str] = []
        for part in parts[:-1]:
            walked.append(part)
            child = os.open(part, directory_flags, dir_fd=directory_descriptor)
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode) or _stat_identity(
                observed
            ) != identities.get("/".join(walked)):
                os.close(child)
                raise RemoteSnapshotError(
                    "local candidate snapshot parent changed type"
                )
            os.close(directory_descriptor)
            directory_descriptor = child

        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected.size
                or _stat_identity(before) != identities.get(relative)
            ):
                raise RemoteSnapshotError(
                    "local candidate snapshot file changed before reading"
                )
            digest = hashlib.sha256()
            payload = bytearray()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                payload.extend(block)
            after = os.fstat(descriptor)
            if (
                _stat_identity(before) != _stat_identity(after)
                or len(payload) != expected.size
                or digest.hexdigest() != expected.sha256
            ):
                raise RemoteSnapshotError(
                    "local candidate snapshot changed while reading"
                )
            return bytes(payload)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RemoteSnapshotError(
            "local candidate snapshot path changed while reading"
        ) from exc
    finally:
        os.close(directory_descriptor)


async def _read_remote_snapshot_file(
    session: object,
    path: str,
    expected_size: int,
) -> bytes:
    """Read one authenticated manifest file through the production CUA API.

    ``RemoteDesktopSession.read_bytes`` accepts only ``path``.  Size and digest
    validation intentionally remain local and are performed by the caller
    against the controller-authenticated manifest.
    """

    del expected_size
    payload = await session.read_bytes(path)
    if not isinstance(payload, bytes):
        raise RemoteSnapshotError("snapshot transfer returned a non-bytes payload")
    return payload


class CandidateSnapshot:
    """One authenticated candidate view used by every evaluator consumer."""

    def __init__(
        self,
        *,
        source_missing: bool,
        remote_root: str,
        local_root: Path,
        files: tuple[SnapshotFile, ...],
        directories: tuple[str, ...],
        remote_digest: str,
        local_digest: str,
        local_root_identity: tuple[int, ...],
        local_identities: Mapping[str, tuple[int, ...]],
        source_binding: Mapping[str, Any],
        windows: bool,
    ) -> None:
        self.source_missing = source_missing
        self.remote_root = remote_root
        self.local_root = local_root
        self.files = files
        self.directories = directories
        self.remote_digest = remote_digest
        self._local_digest = local_digest
        self._local_root_identity = local_root_identity
        self._local_identities = dict(local_identities)
        self.source_binding = dict(source_binding)
        self._windows = windows

    @property
    def exists(self) -> bool:
        return not self.source_missing

    @property
    def relative_files(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files)

    @property
    def top_level_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    path.split("/", 1)[0]
                    for path in (*self.relative_files, *self.directories)
                }
            )
        )

    def remote_path(self, relative: str) -> str:
        return _remote_join(
            self.remote_root, _relative_path(relative), windows=self._windows
        )

    def local_path(self, relative: str) -> Path:
        return self.local_root.joinpath(*PurePosixPath(_relative_path(relative)).parts)

    def payloads(self, names: Mapping[str, str] | None = None) -> dict[str, bytes]:
        selected = names or {path: path for path in self.relative_files}
        result: dict[str, bytes] = {}
        available = {item.path: item for item in self.files}
        self.verify_local()
        if not hasattr(os, "O_NOFOLLOW"):
            raise RemoteSnapshotError("local runtime lacks O_NOFOLLOW")
        try:
            root_descriptor = os.open(
                self.local_root,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | os.O_NOFOLLOW,
            )
        except OSError as exc:
            raise RemoteSnapshotError(
                "local candidate snapshot root changed while reading"
            ) from exc
        try:
            if _stat_identity(os.fstat(root_descriptor)) != self._local_root_identity:
                raise RemoteSnapshotError("local candidate snapshot root changed")
            for name, relative in selected.items():
                canonical = _relative_path(relative)
                expected = available.get(canonical)
                if expected is not None:
                    result[name] = _read_local_snapshot_file(
                        root_descriptor,
                        canonical,
                        expected,
                        self._local_identities,
                    )
        finally:
            os.close(root_descriptor)
        self.verify_local()
        return result

    def verify_local(self) -> None:
        if _local_state(self.local_root) != self._local_digest:
            raise RemoteSnapshotError(
                "local candidate snapshot inode, inventory, or hash changed"
            )


class _SnapshotContext:
    def __init__(
        self,
        session: object,
        source_root: str,
        *,
        limits: SnapshotLimits,
        windows: bool,
        timeout: float,
        source_binding: Mapping[str, Any] | None,
    ) -> None:
        if not isinstance(source_root, str) or not source_root:
            raise ValueError("source_root must be a nonempty string")
        self._session = session
        self._source_root = source_root
        self._limits = limits
        self._windows = windows
        self._timeout = timeout
        self._source_binding = (
            dict(source_binding) if source_binding is not None else None
        )
        self._nonce = secrets.token_hex(24)
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._snapshot: CandidateSnapshot | None = None
        self._authorization: dict[str, Any] | None = None
        self._remote_parent: str | None = None
        self._remote_root: str | None = None
        self._remote_digest: str | None = None

    async def __aenter__(self) -> CandidateSnapshot:
        report = await _run_controller(
            self._session,
            {
                "operation": "create",
                "nonce": self._nonce,
                "source_root": self._source_root,
                "limits": {
                    "max_entries": self._limits.max_entries,
                    "max_file_bytes": self._limits.max_file_bytes,
                    "max_total_bytes": self._limits.max_total_bytes,
                    "max_depth": self._limits.max_depth,
                },
                "source_binding": self._source_binding,
            },
            windows=self._windows,
            timeout=self._timeout,
        )
        if report.get("status") != "ok":
            raise RemoteSnapshotError("snapshot controller did not create a snapshot")
        remote_parent, remote_root = _safe_snapshot_paths(
            report.get("snapshot_parent"),
            report.get("snapshot_root"),
            nonce=self._nonce,
            windows=self._windows,
        )
        authorization = _validated_authorization(
            report.get("authorization"),
            nonce=self._nonce,
            parent_path=remote_parent,
            root_path=remote_root,
        )
        self._authorization = authorization
        self._remote_parent = remote_parent
        self._remote_root = remote_root
        digest = report.get("snapshot_digest")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise RemoteSnapshotError("snapshot controller returned an invalid digest")
        files_value = report.get("files")
        directories_value = report.get("directories")
        source_binding = report.get("source_binding")
        if not isinstance(files_value, list) or not isinstance(directories_value, list):
            raise RemoteSnapshotError(
                "snapshot controller returned an invalid inventory"
            )
        if not isinstance(source_binding, dict):
            raise RemoteSnapshotError("snapshot controller omitted its source binding")
        files: list[SnapshotFile] = []
        seen: set[str] = set()
        for value in files_value:
            if not isinstance(value, dict):
                raise RemoteSnapshotError("snapshot file manifest is malformed")
            path = _relative_path(value.get("path"))
            size = value.get("size")
            sha256 = value.get("sha256")
            if (
                path in seen
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(sha256, str)
                or _SHA256.fullmatch(sha256) is None
            ):
                raise RemoteSnapshotError("snapshot file manifest is malformed")
            seen.add(path)
            files.append(SnapshotFile(path, size, sha256))
        directories = tuple(_relative_path(value) for value in directories_value)
        if len(set(directories)) != len(directories) or seen.intersection(directories):
            raise RemoteSnapshotError("snapshot inventory contains duplicate paths")
        all_paths = seen.union(directories)
        if len(all_paths) > self._limits.max_entries:
            raise RemoteSnapshotError("snapshot manifest exceeds the entry limit")
        total_bytes = 0
        for item in files:
            if item.size > self._limits.max_file_bytes:
                raise RemoteSnapshotError(
                    "snapshot manifest exceeds the per-file limit"
                )
            total_bytes += item.size
            if total_bytes > self._limits.max_total_bytes:
                raise RemoteSnapshotError(
                    "snapshot manifest exceeds the total byte limit"
                )
        for value in all_paths:
            parts = PurePosixPath(value).parts
            if len(parts) > self._limits.max_depth:
                raise RemoteSnapshotError("snapshot manifest exceeds the depth limit")
            for index in range(1, len(parts)):
                if "/".join(parts[:index]) not in directories:
                    raise RemoteSnapshotError(
                        "snapshot inventory omitted a parent directory"
                    )

        temporary = tempfile.TemporaryDirectory(prefix=SNAPSHOT_PREFIX)
        local_root = Path(temporary.name)
        try:
            for relative in sorted(
                directories, key=lambda value: (value.count("/"), value)
            ):
                local_root.joinpath(*PurePosixPath(relative).parts).mkdir(mode=0o700)
            for item in sorted(files, key=lambda value: value.path):
                payload = await _read_remote_snapshot_file(
                    self._session,
                    _remote_join(remote_root, item.path, windows=self._windows),
                    item.size,
                )
                if (
                    len(payload) != item.size
                    or hashlib.sha256(payload).hexdigest() != item.sha256
                ):
                    raise RemoteSnapshotError(
                        "candidate snapshot transfer size or SHA-256 mismatch"
                    )
                target = local_root.joinpath(*PurePosixPath(item.path).parts)
                flags = (
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
                )
                if not hasattr(os, "O_NOFOLLOW"):
                    raise RemoteSnapshotError("local runtime lacks O_NOFOLLOW")
                descriptor = os.open(target, flags | os.O_NOFOLLOW, 0o600)
                try:
                    view = memoryview(payload)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise RemoteSnapshotError("short local snapshot write")
                        view = view[written:]
                    os.fsync(descriptor)
                    os.fchmod(descriptor, 0o400)
                finally:
                    os.close(descriptor)
            for relative in sorted(
                directories, key=lambda value: value.count("/"), reverse=True
            ):
                os.chmod(local_root.joinpath(*PurePosixPath(relative).parts), 0o500)
            os.chmod(local_root, 0o500)
            local_digest = _local_state(local_root)
            local_root_identity = _stat_identity(local_root.lstat())
            local_identities = {
                value: _stat_identity(
                    local_root.joinpath(*PurePosixPath(value).parts).lstat()
                )
                for value in all_paths
            }
            self._temporary = temporary
            self._remote_digest = digest
            self._snapshot = CandidateSnapshot(
                source_missing=report.get("source_missing") is True,
                remote_root=remote_root,
                local_root=local_root,
                files=tuple(sorted(files, key=lambda value: value.path)),
                directories=tuple(sorted(directories)),
                remote_digest=digest,
                local_digest=local_digest,
                local_root_identity=local_root_identity,
                local_identities=local_identities,
                source_binding=source_binding,
                windows=self._windows,
            )
            await self._verify_remote()
            return self._snapshot
        except BaseException as original:
            temporary.cleanup()
            try:
                await self._cleanup_remote(remote_root)
            except BaseException:
                pass
            raise original

    async def _verify_remote(self) -> None:
        if self._remote_root is None or self._remote_digest is None:
            return
        report = await _run_controller(
            self._session,
            {
                "operation": "verify",
                "nonce": self._nonce,
                "snapshot_root": self._remote_root,
                "snapshot_digest": self._remote_digest,
                "authorization": self._authorization,
            },
            windows=self._windows,
            timeout=self._timeout,
        )
        if (
            report.get("status") != "verified"
            or report.get("snapshot_digest") != self._remote_digest
            or not isinstance(self._authorization, dict)
            or report.get("authorization_fingerprint")
            != self._authorization.get("fingerprint")
        ):
            raise RemoteSnapshotError(
                "candidate snapshot post-transfer verification failed"
            )

    async def _cleanup_remote(self, root: str | None = None) -> None:
        target = root or self._remote_root
        if target is None:
            return
        try:
            report = await _run_controller(
                self._session,
                {
                    "operation": "cleanup",
                    "nonce": self._nonce,
                    "snapshot_root": target,
                    "authorization": self._authorization,
                },
                windows=self._windows,
                timeout=self._timeout,
            )
            if report.get("status") != "cleaned":
                raise RemoteSnapshotError(
                    "candidate snapshot cleanup was not confirmed"
                )
        finally:
            if target == self._remote_root:
                self._remote_root = None
                self._remote_parent = None
                self._authorization = None

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        verification_error: BaseException | None = None
        try:
            if self._snapshot is not None:
                self._snapshot.verify_local()
                await self._verify_remote()
        except BaseException as error:
            verification_error = error
        cleanup_error: BaseException | None = None
        try:
            await self._cleanup_remote()
        except BaseException as error:
            cleanup_error = error
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        if exc is None:
            if verification_error is not None:
                raise verification_error
            if cleanup_error is not None:
                raise cleanup_error
        return False


def snapshot_remote_tree(
    session: object,
    source_root: str,
    *,
    limits: SnapshotLimits | None = None,
    windows: bool = False,
    timeout: float = 600.0,
    source_binding: Mapping[str, Any] | None = None,
) -> _SnapshotContext:
    """Return an async context manager for one authenticated candidate tree."""

    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        raise ValueError("timeout must be positive")
    return _SnapshotContext(
        session,
        source_root,
        limits=limits or SnapshotLimits(),
        windows=windows,
        timeout=float(timeout),
        source_binding=source_binding,
    )
