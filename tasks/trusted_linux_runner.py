"""Execute authenticated evaluator bytes without reopening their staged paths.

The returned command runs a small, literal controller under the system Python.
That controller opens each staged asset once with ``O_NOFOLLOW``, streams the
authenticated bytes into a sealed memfd, verifies the sealed copy, and only
then replaces path arguments with inherited ``/proc/self/fd`` paths.  An
unlinked, read-only inode is the fail-closed fallback on kernels without
sealable memfd support; the mutable staged path is never an execution fallback.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
from collections.abc import Mapping, Sequence

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Keep this controller self-contained: it is passed literally to a root-owned
# interpreter and therefore does not depend on another mutable staged file.
_SEALED_EXEC_CONTROLLER = r'''
import base64, errno, fcntl, hashlib, json, os, stat, sys, tempfile

def fail(message):
    raise SystemExit("authenticated evaluator launch refused: " + message)

def copy_and_hash(source_fd, destination_fd):
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
                fail("short write while sealing asset")
            view = view[written:]
    return digest.hexdigest(), total

def verify_copy(fd, expected, expected_size):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        digest.update(block)
        total += len(block)
    os.lseek(fd, 0, os.SEEK_SET)
    if digest.hexdigest() != expected or total != expected_size:
        fail("sealed asset verification failed")

def unlinked_fallback(source_fd, expected, source_size):
    directory = tempfile.mkdtemp(prefix="agenthle-sealed-")
    try:
        path = os.path.join(directory, "asset")
        write_fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        os.unlink(path)
    finally:
        os.rmdir(directory)
    try:
        digest, copied = copy_and_hash(source_fd, write_fd)
        if digest != expected or copied != source_size:
            fail("authenticated asset digest mismatch")
        os.fsync(write_fd)
        os.fchmod(write_fd, 0o500)
        read_fd = os.open("/proc/self/fd/%d" % write_fd, os.O_RDONLY | os.O_CLOEXEC)
    finally:
        os.close(write_fd)
    verify_copy(read_fd, expected, source_size)
    os.set_inheritable(read_fd, True)
    return read_fd

def seal(path, expected, label):
    if not hasattr(os, "O_NOFOLLOW"):
        fail("runtime lacks O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= os.O_NOFOLLOW
    try:
        before_path = os.lstat(path)
        source_fd = os.open(path, flags)
    except OSError as exc:
        fail("cannot open %s: %s" % (label, exc))
    try:
        before = os.fstat(source_fd)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns,
        )
        if not stat.S_ISREG(before.st_mode) or identity(before_path) != identity(before):
            fail("%s is not one stable regular file" % label)
        if before.st_nlink != 1:
            fail("%s has multiple hard links" % label)
        try:
            sealed_fd = os.memfd_create(
                "agenthle-" + label,
                getattr(os, "MFD_ALLOW_SEALING", 0x0002) | getattr(os, "MFD_CLOEXEC", 0x0001),
            )
            try:
                digest, copied = copy_and_hash(source_fd, sealed_fd)
                after = os.fstat(source_fd)
                try:
                    after_path = os.lstat(path)
                except OSError:
                    fail("%s path changed while being authenticated" % label)
                if identity(before) != identity(after) or identity(before) != identity(after_path):
                    fail("%s changed while being authenticated" % label)
                if digest != expected or copied != before.st_size:
                    fail("%s digest mismatch" % label)
                seals = (
                    getattr(fcntl, "F_SEAL_WRITE", 0x0008)
                    | getattr(fcntl, "F_SEAL_GROW", 0x0004)
                    | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
                    | getattr(fcntl, "F_SEAL_SEAL", 0x0001)
                )
                fcntl.fcntl(sealed_fd, getattr(fcntl, "F_ADD_SEALS", 1033), seals)
                observed = fcntl.fcntl(sealed_fd, getattr(fcntl, "F_GET_SEALS", 1034))
                if observed & seals != seals:
                    fail("%s memfd is not fully sealed" % label)
                verify_copy(sealed_fd, expected, before.st_size)
                os.set_inheritable(sealed_fd, True)
                return sealed_fd
            except BaseException:
                os.close(sealed_fd)
                raise
        except (AttributeError, OSError) as exc:
            if isinstance(exc, OSError) and exc.errno not in {
                errno.ENOSYS, errno.EINVAL, errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP
            }:
                raise
            os.lseek(source_fd, 0, os.SEEK_SET)
            fallback_fd = unlinked_fallback(source_fd, expected, before.st_size)
            try:
                after = os.fstat(source_fd)
                after_path = os.lstat(path)
                if identity(before) != identity(after) or identity(before) != identity(after_path):
                    fail("%s changed while being authenticated" % label)
                return fallback_fd
            except BaseException:
                os.close(fallback_fd)
                raise
    finally:
        os.close(source_fd)

spec = json.loads(base64.urlsafe_b64decode(sys.argv[1].encode()).decode())
command = sys.argv[2:]
if not command:
    fail("empty evaluator command")
fds = {}
for index, item in enumerate(spec["assets"]):
    path = item["path"]
    if path in fds:
        fail("duplicate authenticated asset path")
    fds[path] = seal(path, item["sha256"], "asset-%d" % index)

command = ["/proc/self/fd/%d" % fds.get(value, -1) if value in fds else value for value in command]
modules = spec.get("python_modules", {})
if modules:
    launcher = 'import importlib.util,json,runpy,sys\nfrom importlib.machinery import SourceFileLoader\nrunner=sys.argv[1]\nmodules=json.loads(sys.argv[2])\nsys.argv=[runner,*sys.argv[3:]]\nfor name,path in modules.items():\n loader=SourceFileLoader(name,path)\n spec=importlib.util.spec_from_loader(name,loader)\n if spec is None: raise RuntimeError("sealed module load failed: "+name)\n module=importlib.util.module_from_spec(spec)\n sys.modules[name]=module\n loader.exec_module(module)\nrunpy.run_path(runner,run_name="__main__")'
    mapped = {name: "/proc/self/fd/%d" % fds[path] for name, path in modules.items()}
    runner_index = spec["runner_index"]
    runner = command[runner_index]
    command = command[:runner_index] + ["-c", launcher, runner, json.dumps(mapped, separators=(",", ":"))] + command[runner_index + 1:]

os.execvp(command[0], command)
'''


def sealed_exec_command(
    argv: Sequence[str],
    *,
    authenticated_assets: Mapping[str, str],
    runner_index: int,
    python_modules: Mapping[str, str] | None = None,
    controller_python: str = "/usr/bin/python3",
) -> str:
    """Return a shell-safe command that executes only authenticated copies.

    ``runner_index`` is the index of the Python script (or executable) in
    ``argv``.  Each authenticated path must appear as one complete argv token;
    it is replaced with the corresponding inherited descriptor path.
    """

    command = [str(value) for value in argv]
    if not command or not 0 <= runner_index < len(command):
        raise ValueError("runner_index is outside argv")
    if not authenticated_assets:
        raise ValueError("at least one authenticated asset is required")
    if command[runner_index] not in authenticated_assets:
        raise ValueError("runner path is not authenticated")
    modules = dict(python_modules or {})
    normalized: dict[str, str] = {}
    for path, digest in authenticated_assets.items():
        if not path or (path not in command and path not in modules.values()):
            raise ValueError(f"authenticated asset is unused by the sealed command: {path!r}")
        if _SHA256.fullmatch(digest) is None:
            raise ValueError(f"invalid SHA-256 for authenticated asset: {path!r}")
        normalized[path] = digest
    if any(path not in normalized for path in modules.values()):
        raise ValueError("every preloaded Python module must be authenticated")
    spec = {
        "assets": [
            {"path": path, "sha256": normalized[path]} for path in sorted(normalized)
        ],
        "python_modules": modules,
        "runner_index": runner_index,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    return " ".join(
        shlex.quote(part)
        for part in (
            controller_python,
            "-I",
            "-c",
            _SEALED_EXEC_CONTROLLER,
            encoded,
            *command,
        )
    )
