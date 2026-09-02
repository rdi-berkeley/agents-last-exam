"""SandboxHandle — the cua-server target the framework hosts a task in.

``SandboxHandle`` IS:

* The identity of a live cua-server endpoint (id, endpoint URL, OS).
* The framework-level path conventions baked into the image
  (work_dir_base / task_data_root / node / python / mcp_server_dir).
* The I/O surface other ALE code uses to act on that endpoint
  (``run_command`` / ``write_file`` / ``read_file`` / ``exists`` /
  ``mkdir`` / ``rm`` / ``list_dir`` / ``upload_local_file`` /
  ``download_to_local`` / ``download_range`` / ``check_reachable``).

It replaces what used to be split between :class:`EnvHandle` (data
dataclass) and :mod:`environments.remote` (free functions taking that
dataclass). One class, one place.

Image conventions are **flat fields** on the handle (not nested under
an ``image`` sub-object) so deployer code reads them as
``sandbox.node_exe`` rather than ``sandbox.image.node_exe``. The
Provider populates these from OS defaults at ``acquire`` time; new
image families with non-default layouts override per-field.

Agent-specific binary locations (the claude CLI, the codex CLI, ...)
are NOT in here. Each deployer discovers its own binary at install
time (``which claude`` on linux / ``Get-Command claude`` on windows).
"""
from __future__ import annotations

import abc
import asyncio
import base64
import json
import logging
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import requests


OS = Literal["linux", "windows"]
ReleaseMode = Literal["delete", "stop", "keep"]


logger = logging.getLogger(__name__)


# ============================================================================
# Network policy — per-task egress restriction (from task_card ``vm.network``)
# ============================================================================

NetworkMode = Literal["open", "allowlist", "off"]
"""``open`` — no restriction (default, current behaviour).
``allowlist`` — block all egress except the model API host (auto-derived from
the agent's ``base_url``) plus any hosts in ``allow``.
``off`` — full air-gap; nothing reaches the network (special case of
``allowlist`` with an empty allow set; suited to host-side agents)."""


@dataclass(frozen=True)
class NetworkPolicy:
    """Per-task egress policy, declared under ``task_card.json`` ``vm.network``.

    The policy only carries what the *task* declares. The model API host is
    NOT listed here: it is unioned in by the provider at enforcement time from
    the agent's resolved ``base_url`` (see :meth:`effective_allow`), so a task
    card never has to name the endpoint and stays valid across experiments that
    swap the ``base_url``.
    """

    mode: NetworkMode = "open"
    allow: tuple[str, ...] = ()
    """Extra hostnames permitted on top of the auto-derived model host.
    Only meaningful when ``mode == "allowlist"``."""

    @classmethod
    def from_card(cls, raw: Any) -> "NetworkPolicy":
        """Parse+validate a task-card ``vm.network`` object. ``None``/absent →
        ``open`` (backward compatible; existing cards need no changes)."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(
                f"task_card vm.network must be an object, got {type(raw).__name__}"
            )
        mode = raw.get("mode", "open")
        if mode not in ("open", "allowlist", "off"):
            raise ValueError(
                f"task_card vm.network.mode={mode!r}; expected one of "
                "'open', 'allowlist', 'off'"
            )
        raw_allow = raw.get("allow", ())
        if isinstance(raw_allow, str) or not isinstance(raw_allow, (list, tuple)):
            raise ValueError(
                "task_card vm.network.allow must be a list of hostname strings"
            )
        allow: list[str] = []
        for h in raw_allow:
            if not isinstance(h, str) or not h.strip():
                raise ValueError(
                    f"task_card vm.network.allow entries must be non-empty "
                    f"strings, got {h!r}"
                )
            allow.append(h.strip())
        return cls(mode=mode, allow=tuple(allow))

    @staticmethod
    def model_host_from_env(env: dict[str, str]) -> str | None:
        """Best-effort extract the model API hostname from the agent's injected
        env, so the enforcer can allowlist exactly the endpoint the framework
        already routes to (no per-task config of the endpoint). Checks the
        base-URL vars ALE injects, in precedence order."""
        from urllib.parse import urlparse
        for key in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENROUTER_BASE_URL"):
            val = env.get(key)
            if val:
                host = urlparse(val).hostname
                if host:
                    return host
        return None

    def effective_allow(self, model_host: str | None) -> frozenset[str]:
        """Hosts egress is permitted to reach under this policy.

        ``open`` → sentinel empty set is not used; callers gate on ``mode``
        first. For ``allowlist`` the model API host (from the agent's
        ``base_url``) is unioned with the task-declared ``allow``. For ``off``
        the set is empty (true air-gap) — the model host is deliberately NOT
        added, since host-side agents don't call it from inside the VM."""
        hosts = set(self.allow)
        if self.mode == "allowlist" and model_host:
            hosts.add(model_host)
        return frozenset(hosts)


# ============================================================================
# Spec — pre-provision request shape
# ============================================================================

@dataclass(frozen=True)
class SandboxSpec:
    """Pre-provision: what kind of sandbox the task wants.

    Built from ``task_card.json``. Handed to ``Provider.acquire``.
    """

    snapshot: str
    os: OS = "linux"
    machine_type: str | None = None
    """Task-card ``vm.machineType`` override. None → provider uses its yaml
    fallback list. Boot disk size always comes from the image (no override)."""
    vcpus: int | None = None
    memory_gb: int | None = None
    gpu: str | None = None
    task_id: str = ""
    harness: str = ""
    model_tag: str = ""
    network: NetworkPolicy = field(default_factory=NetworkPolicy)
    """Per-task egress policy from task_card ``vm.network``. Defaults to
    ``open`` (no restriction). Enforced by the Provider at ``acquire`` time."""


# ============================================================================
# Range result — surfaced by SandboxHandle.download_range
# ============================================================================

@dataclass
class RangeResult:
    """Outcome of an incremental file fetch (see
    :meth:`SandboxHandle.download_range`).

    Cleanly distinguishes "remote file shrank" / "got data" / "no new
    bytes" / "error" for the caller (see
    :func:`ale_run.executors.sandbox.tail_hot_artifacts`).
    """

    success: bool
    new_data: bytes = b""
    new_size: int = 0
    error: str | None = None


class SandboxUnreachableError(RuntimeError):
    """Raised when cua-server doesn't answer ``/status`` healthily."""


# ============================================================================
# SandboxHandle — data + API
# ============================================================================

@dataclass
class SandboxHandle:
    """A live cua-server target + the API to act on it.

    Returned by ``Provider.acquire``. Carried through the whole
    framework. Deployer code reads identity / path fields directly, and
    calls the I/O methods to make things happen on the sandbox.
    """

    # ─── identity ───
    id: str
    endpoint: str
    os: OS

    # ─── baked path conventions (Provider populates from default_paths_for) ───
    work_dir_base: str
    """Per-run scratch root. e.g. ``/home/user/.ale`` (linux) /
    ``C:\\Users\\User\\.ale`` (windows). Deployers compose
    ``<work_dir_base>/<agent>/<run_id>/`` as their work dir."""

    task_data_root: str
    """Sandbox-side root for staged task data (input/, reference/,
    output/, ...). ``/media/user/data/ale-data`` (linux) /
    ``E:\\ale-data`` (windows). data_staging composes
    ``<task_data_root>/<domain>/<task>/<variant>/<subdir>``."""

    node: str
    """Absolute path to the ``node`` binary. Used by MCP-server-driven agents."""

    python: str
    """Absolute path to the ``python`` binary."""

    mcp_server_dir: str
    """Where the cua MCP server is installed on this image."""

    cua_server_port: int = 5000
    """Port the cua-server listens on inside the sandbox (image-specific:
    8000 on ale-kasm, 5000 on GCE families). The cua MCP bridge is told this
    via ``CUA_SERVER_URL`` so it doesn't fall back to its built-in default."""

    # ─── provider extras (rarely-used, free-form) ───
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_linux(self) -> bool:
        return self.os == "linux"

    @property
    def cmd_url(self) -> str:
        return f"{self.endpoint.rstrip('/')}/cmd"

    # ========================================================================
    # I/O — async API, thin wrappers around the sync wire impl below
    # ========================================================================

    async def run_command(
        self, command: str, *, timeout: float = 60,
    ) -> subprocess.CompletedProcess:
        return await asyncio.to_thread(_run_remote_sync, self, command, timeout)

    async def write_file(self, path: str, content: str | bytes) -> None:
        if isinstance(content, bytes):
            await asyncio.to_thread(_write_binary_sync, self, path, content)
        else:
            await asyncio.to_thread(_write_text_sync, self, path, content)

    async def read_file(self, path: str) -> bytes:
        """Read ``path`` as bytes. Raises :class:`FileNotFoundError` if the
        path does not exist; :class:`RuntimeError` on a transport-level
        failure (cua-server unreachable, decode error). Empty file → ``b""``."""
        return await asyncio.to_thread(_read_bytes_sync, self, path)

    async def read_text(self, path: str) -> str:
        """UTF-8 decode of :meth:`read_file`. Raises the same exceptions."""
        return (await self.read_file(path)).decode("utf-8", errors="replace")

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(_exists_sync, self, path)

    async def mkdir(self, path: str) -> None:
        await asyncio.to_thread(_mkdir_sync, self, path)

    async def rm(self, paths: Iterable[str]) -> None:
        await asyncio.to_thread(_rm_sync, self, list(paths))

    async def list_dir(self, path: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(_list_dir_sync, self, path)

    async def upload_local_file(self, local_path: str, remote_path: str) -> None:
        """Upload a HOST-side file into the sandbox."""
        await asyncio.to_thread(_upload_local_file_sync, self, local_path, remote_path)

    async def download_to_local(
        self, remote_path: str, local_path: str, *, timeout: float = 60,
    ) -> bool:
        """Download a sandbox-side file to HOST disk. Returns True on
        success. Used by gather + incremental puller."""
        return await asyncio.to_thread(
            _download_to_local_sync, self, remote_path, local_path, timeout,
        )

    async def download_range(
        self, remote_path: str, *, start: int, max_chunk_bytes: int,
        timeout: float = 60,
    ) -> RangeResult:
        """Incremental fetch of a sandbox-side file. See
        :class:`RangeResult`."""
        return await asyncio.to_thread(
            _download_range_sync, self, remote_path, start, max_chunk_bytes, timeout,
        )

    async def check_reachable(self, *, label: str = "sandbox") -> None:
        """Ping ``/status``. Raise :class:`SandboxUnreachableError` on
        any non-healthy response."""
        await asyncio.to_thread(_check_reachable_sync, self, label)


# ============================================================================
# Provider ABC
# ============================================================================

class NetworkPolicyUnsupportedError(RuntimeError):
    """A task requested egress restriction (``vm.network.mode != "open"``) on a
    provider that does not enforce it yet. Raised **before** the VM is used so a
    task never runs unisolated while believing it is — fail closed, never fail
    open."""


class Provider(abc.ABC):
    """ABC for sandbox lifecycle. The framework consumes Provider; nothing
    else."""

    enforces_network_policy: bool = False
    """Set True on a provider once it actually applies ``spec.network`` (egress
    cut-off + proxy routing). Until then, :meth:`assert_network_supported`
    fail-closes any non-``open`` policy on that provider."""

    @abc.abstractmethod
    async def acquire(self, spec: SandboxSpec) -> SandboxHandle: ...

    @abc.abstractmethod
    async def release(
        self, sandbox: SandboxHandle, *, mode: ReleaseMode = "delete",
    ) -> None: ...

    @abc.abstractmethod
    def open_session(self, sandbox: SandboxHandle) -> Any:
        """Return a cua-bench DesktopSession talking to ``sandbox``."""

    def assert_network_supported(self, spec: SandboxSpec) -> None:
        """Fail closed if the task asks for isolation this provider can't apply.

        Called by the framework before the sandbox is put to work. An ``open``
        policy is always fine (it's today's behaviour). Any other mode on a
        provider that hasn't set :attr:`enforces_network_policy` raises, so we
        never silently run a task that expected a firewalled sandbox on one
        that gives it full internet."""
        if spec.network.mode == "open":
            return
        if not self.enforces_network_policy:
            raise NetworkPolicyUnsupportedError(
                f"task requested vm.network.mode={spec.network.mode!r} but "
                f"provider {type(self).__name__} does not enforce network "
                "policy yet (enforcement is staged; see docs/network-isolation.md)"
            )

    async def heartbeat(self, sandbox: SandboxHandle) -> None:
        return None

    async def cancel_external(self, sandbox: SandboxHandle) -> None:
        return None


# ============================================================================
# Wire impl (private — sync; SandboxHandle's async methods to_thread these)
# ============================================================================

_MAX_RUN_REMOTE_RETRIES = 3
_MAX_UPLOAD_RETRIES = 3
_DEFAULT_MAX_CHUNK_BYTES = 16 * 1024 * 1024


def _strip_bom(raw: bytes) -> bytes:
    return raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw


# SSE body read chunk. Large on purpose: ``requests.iter_lines`` defaults to a
# 512-byte chunk and rebuilds a growing ``pending`` buffer on every chunk, which
# is O(n^2) on the single multi-MB ``data:`` line cua-server returns for
# read_bytes / download_range (a 5.6 MB reply measured ~19s via iter_lines vs
# ~1s reading the body in MB chunks). We read big chunks into a bytearray
# (amortized O(1) append) and scan for the first complete line ourselves.
_SSE_READ_CHUNK = 1 << 20  # 1 MiB


def _read_first_sse_event(
    resp: requests.Response, read_timeout: float = 30,
) -> dict[str, Any] | None:
    """Read until the first SSE ``data:`` line, parse + return its JSON.

    cua-server replies as SSE — the first data event is the result; any later
    events are progress/keepalives we don't consume here, so we stop and return
    as soon as the first ``data:`` line is complete.

    Reads via :meth:`requests.Response.iter_content` with a large chunk into a
    ``bytearray`` (O(n)) rather than :meth:`iter_lines` (512-byte default chunk
    → O(n^2) on a multi-MB single line; see ``_SSE_READ_CHUNK``). The socket
    read timeout is enforced by the originating ``requests.post(timeout=...)``;
    ``read_timeout`` is accepted for call-site symmetry."""
    buf = bytearray()
    line_start = 0   # index of the current (in-progress) line's first byte
    scan_pos = 0     # index to resume the newline search from
    for chunk in resp.iter_content(chunk_size=_SSE_READ_CHUNK):
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n", scan_pos)
            if nl == -1:
                scan_pos = len(buf)
                break
            line = bytes(buf[line_start:nl]).rstrip(b"\r")
            line_start = scan_pos = nl + 1
            if line.startswith(b"data:"):
                payload = _strip_bom(line[len(b"data:"):].strip())
                try:
                    return json.loads(payload)
                except json.JSONDecodeError as e:
                    logger.debug("SSE parse failed: %s -- raw=%s", e, payload[:200])
                    return None
    # Stream ended without a newline-terminated data line — accept a trailing
    # unterminated ``data:`` line if present (server closed right after it).
    tail = bytes(buf[line_start:]).rstrip(b"\r\n")
    if tail.startswith(b"data:"):
        payload = _strip_bom(tail[len(b"data:"):].strip())
        try:
            return json.loads(payload)
        except json.JSONDecodeError as e:
            logger.debug("SSE parse failed (trailing): %s -- raw=%s", e, payload[:200])
            return None
    return None


def _post_cmd(sandbox: SandboxHandle, body: dict, *, timeout: float) -> dict[str, Any] | None:
    try:
        with requests.post(
            sandbox.cmd_url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            stream=True,
        ) as resp:
            return _read_first_sse_event(resp, read_timeout=timeout)
    except requests.RequestException as e:
        logger.debug("POST %s failed: %s", sandbox.cmd_url, e)
        return None


def _run_remote_sync(
    sandbox: SandboxHandle, command: str, timeout: float,
) -> subprocess.CompletedProcess:
    last_err = "no response"
    for attempt in range(_MAX_RUN_REMOTE_RETRIES):
        data = _post_cmd(
            sandbox,
            {"command": "run_command", "params": {"command": command}},
            timeout=timeout,
        )
        if data is None:
            last_err = "transport error"
            continue
        if data.get("success"):
            return subprocess.CompletedProcess(
                args=command,
                returncode=int(data.get("return_code", 0)),
                stdout=data.get("stdout", "") or "",
                stderr=data.get("stderr", "") or "",
            )
        last_err = data.get("error") or "command failed"
        # Don't retry on cmd failure — only on transport.
        return subprocess.CompletedProcess(
            args=command, returncode=1, stdout="", stderr=last_err,
        )
    return subprocess.CompletedProcess(
        args=command, returncode=-1, stdout="", stderr=last_err,
    )


def _write_text_sync(sandbox: SandboxHandle, remote_path: str, content: str) -> None:
    last_err = "no response"
    for attempt in range(_MAX_UPLOAD_RETRIES):
        data = _post_cmd(
            sandbox,
            {"command": "write_text", "params": {"path": remote_path, "content": content}},
            timeout=120,
        )
        if data and data.get("success"):
            return
        last_err = (data or {}).get("error", "transport error")
    raise RuntimeError(f"write_file({remote_path}) failed after retries: {last_err}")


def _write_binary_sync(sandbox: SandboxHandle, remote_path: str, content: bytes) -> None:
    """base64-stage then decode on sandbox. Linux uses ``base64 -d``,
    Windows uses ``[Convert]::FromBase64String``."""
    encoded = base64.b64encode(content).decode("ascii")
    b64_path = f"{remote_path}.b64"
    _write_text_sync(sandbox, b64_path, encoded)
    if sandbox.is_linux:
        decode = (
            f"base64 -d {shlex.quote(b64_path)} > {shlex.quote(remote_path)} && "
            f"rm -f {shlex.quote(b64_path)}"
        )
    else:
        decode = (
            'powershell -NoProfile -Command "'
            f"$b=[IO.File]::ReadAllText('{b64_path}').Trim();"
            f"[IO.File]::WriteAllBytes('{remote_path}',[Convert]::FromBase64String($b));"
            f"Remove-Item -Path '{b64_path}' -Force"
            '"'
        )
    result = _run_remote_sync(sandbox, decode, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"write_file binary decode failed for {remote_path}: "
            f"rc={result.returncode} stderr={(result.stderr or '')[:200]}"
        )


def _read_bytes_sync(sandbox: SandboxHandle, remote_path: str) -> bytes:
    # Pre-check existence so callers get a clean FileNotFoundError instead of
    # a vague transport failure. One extra RPC, but the distinction matters
    # for diagnose paths where the file may legitimately be absent.
    if not _exists_sync(sandbox, remote_path):
        raise FileNotFoundError(remote_path)
    fd, tmp = tempfile.mkstemp(prefix="ale_dl_")
    os.close(fd)
    try:
        ok = _download_to_local_sync(sandbox, remote_path, tmp, 60)
        if not ok:
            raise RuntimeError(
                f"sandbox read_file: transport failure for {remote_path}"
            )
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _exists_sync(sandbox: SandboxHandle, path: str) -> bool:
    # cua-server's ``file_exists`` returns ``false`` for *directories* (it
    # tests regular files only), so we use a shell test that handles both.
    if sandbox.is_linux:
        cmd = f"test -e {shlex.quote(path)}"
    else:
        safe = path.replace("'", "''")
        cmd = (
            'powershell -NoProfile -Command "'
            f"if (Test-Path -LiteralPath '{safe}') {{ exit 0 }} else {{ exit 1 }}"
            '"'
        )
    r = _run_remote_sync(sandbox, cmd, timeout=15)
    return r.returncode == 0


def _mkdir_sync(sandbox: SandboxHandle, path: str) -> None:
    if sandbox.is_linux:
        cmd = f"mkdir -p {shlex.quote(path)}"
    else:
        cmd = (
            'powershell -NoProfile -Command "'
            f"New-Item -ItemType Directory -Force -Path '{path}' | Out-Null"
            '"'
        )
    result = _run_remote_sync(sandbox, cmd, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"mkdir {path} failed rc={result.returncode}: "
            f"{(result.stderr or '')[:200]}"
        )


def _rm_sync(sandbox: SandboxHandle, paths: list[str]) -> None:
    if not paths:
        return
    if sandbox.is_linux:
        quoted = " ".join(shlex.quote(p) for p in paths)
        cmd = f"rm -rf {quoted}"
    else:
        inner = "; ".join(
            f"Remove-Item -Recurse -Force -ErrorAction SilentlyContinue '{p}'"
            for p in paths
        )
        cmd = f'powershell -NoProfile -Command "{inner}"'
    _run_remote_sync(sandbox, cmd, timeout=30)


def _list_dir_sync(sandbox: SandboxHandle, remote_dir: str) -> list[dict[str, Any]]:
    """Recursive walk: emits {"relpath", "is_dir", "size"} per entry.
    Returns [] on missing dir or transport error."""
    if sandbox.is_linux:
        safe = remote_dir.replace("'", "'\\''")
        cmd = (
            f"if [ ! -d '{safe}' ]; then echo '[]'; exit 0; fi; "
            f"cd '{safe}' && find . -mindepth 1 -printf '%P\\t%y\\t%s\\n'"
        )
    else:
        safe = remote_dir.replace("'", "''")
        cmd = (
            'powershell -NoProfile -Command "'
            f"if (-not (Test-Path -LiteralPath '{safe}' -PathType Container)) {{ '[]'; exit 0 }}; "
            f"Get-ChildItem -LiteralPath '{safe}' -Recurse | ForEach-Object {{ "
            f"  $type = if ($_.PSIsContainer) {{ 'd' }} else {{ 'f' }}; "
            f"  $size = if ($_.PSIsContainer) {{ 0 }} else {{ $_.Length }}; "
            f"  Write-Output ($_.FullName + [char]9 + $type + [char]9 + $size) "
            f"}}"
            '"'
        )
    result = _run_remote_sync(sandbox, cmd, timeout=30)
    if result.returncode != 0:
        return []
    out = (result.stdout or "").strip()
    if not out or out == "[]":
        return []
    prefix_variants: list[str] = []
    if not sandbox.is_linux:
        norm = remote_dir.rstrip("/\\")
        prefix_variants = [norm + "\\", norm + "/"]
    entries: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rel, kind, size = parts[0], parts[1], parts[2]
        for p in prefix_variants:
            if rel.startswith(p):
                rel = rel[len(p):]
                break
        try:
            size_int = int(size)
        except ValueError:
            size_int = 0
        is_dir = kind == "d"
        entries.append({"relpath": rel, "is_dir": is_dir, "size": size_int})
    return entries


def _upload_local_file_sync(sandbox: SandboxHandle, local_path: str, remote_path: str) -> None:
    with open(local_path, "rb") as f:
        content = f.read()
    _write_binary_sync(sandbox, remote_path, content)


# Per-RPC read_bytes chunk for whole-file downloads. Larger is faster here: with
# the O(n) SSE reader (see _read_first_sse_event) throughput RISES with chunk
# size (measured on a GCE win10 VM: 4MiB≈4.3, 8MiB≈6.9, 16MiB≈12 MB/s) because
# the per-RPC overhead is amortized. 8 MiB halves the round-trips vs 4 MiB and
# gives headroom for very large transcripts under the gather deadline, while
# keeping the in-flight base64 payload (~11 MiB) modest.
_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


def _download_to_local_sync(
    sandbox: SandboxHandle, remote_path: str, local_path: str, timeout: float,
) -> bool:
    """Chunked, binary-safe download with a hard wall-clock deadline.

    Streams the file in ``_DOWNLOAD_CHUNK_BYTES`` slices via the cua-server's
    native ``read_bytes`` command (offset/length) so each RPC carries a bounded
    payload, for BOTH linux and windows. ``timeout`` is the TOTAL budget for the
    file, not a per-read bound.

    Previously the linux path read the WHOLE file in one ``read_text`` SSE event:
    on large outputs (a 96 MB CSV) that ran far past the per-op timeout — which
    only bounds a single read, not the total — and effectively hung the unit
    until the task wall-clock fired; on binary outputs (PNG, .nc, ...) the UTF-8
    round-trip corrupted or failed them (the source of spurious download errors).
    Windows already streamed read_bytes; this unifies both and adds the deadline.
    """
    import time as _time

    deadline = _time.monotonic() + timeout
    try:
        offset = 0
        parts: list[bytes] = []
        while True:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                logger.debug(
                    "download_to_local timed out for %s at offset %d (budget=%.0fs)",
                    remote_path, offset, timeout,
                )
                return False
            data = _post_cmd(
                sandbox,
                {"command": "read_bytes",
                 "params": {"path": remote_path, "offset": offset,
                            "length": _DOWNLOAD_CHUNK_BYTES}},
                timeout=max(10.0, min(timeout, remaining)),
            )
            if data is None:
                logger.debug("read_bytes transport error for %s at offset %d", remote_path, offset)
                return False
            if not data.get("success"):
                logger.debug("read_bytes failed for %s: %s", remote_path, data.get("error"))
                return False
            raw = base64.b64decode(data.get("content_b64", "") or "")
            parts.append(raw)
            offset += len(raw)
            if len(raw) < _DOWNLOAD_CHUNK_BYTES:
                break
        Path(local_path).write_bytes(b"".join(parts))
        return True
    except Exception as e:
        logger.debug("download_to_local failed for %s: %s", remote_path, e)
        return False


def _build_range_cmd_linux(remote_path: str, start: int, max_chunk_bytes: int) -> str:
    safe_path = remote_path.replace("'", "'\\''")
    return (
        f"S=$(stat -c%s '{safe_path}' 2>/dev/null || echo -1); "
        f"if [ \"$S\" -lt 0 ]; then printf 'SIZE=-1\\nB64=\\n'; exit 0; fi; "
        f"B=\"\"; "
        f"if [ \"$S\" -gt {start} ]; then "
        f"  B=$(tail -c +$(( {start} + 1 )) '{safe_path}' "
        f"     | head -c {max_chunk_bytes} | base64 -w0); "
        f"fi; "
        f"printf 'SIZE=%s\\nB64=%s\\n' \"$S\" \"$B\""
    )


def _build_range_cmd_windows(remote_path: str, start: int, max_chunk_bytes: int) -> str:
    safe_path = remote_path.replace("'", "''")
    ps = (
        "try{"
        f"$fi=[IO.FileInfo]::new('{safe_path}');"
        "if(-not $fi.Exists){\"SIZE=-1\";\"B64=\";exit};"
        "$len=$fi.Length;$b64='';"
        f"if($len -gt {start}){{"
        f"$fs=[IO.File]::Open('{safe_path}','Open','Read','ReadWrite');"
        "try{"
        f"$null=$fs.Seek({start},'Begin');"
        f"$rem=[Math]::Min($len-{start},{max_chunk_bytes});"
        "$buf=New-Object byte[] $rem;"
        "$null=$fs.Read($buf,0,$rem);"
        "$b64=[Convert]::ToBase64String($buf)"
        "}finally{$fs.Close()}"
        "};"
        "\"SIZE=$len\";\"B64=$b64\""
        "}catch{\"SIZE=-2\";\"B64=\";\"ERR=$($_.Exception.Message)\"}"
    )
    # Pass via -EncodedCommand (base64 of UTF-16LE) rather than -Command "<ps>":
    # the cua-server runs commands through cmd.exe, which mangles this script's
    # many nested double-quotes — the quotes broke, PowerShell then tried to
    # execute the bare "SIZE=-2" output token ("not recognized as a cmdlet"),
    # so download_range ALWAYS failed on Windows (the incremental tail was dead;
    # transcripts only ever arrived via the slower post-run gather). base64 is
    # cmd-safe (only [A-Za-z0-9+/=]), so the script reaches PowerShell intact.
    encoded = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -EncodedCommand {encoded}"


def _parse_range_stdout(stdout: str, *, expected_start: int) -> RangeResult:
    if not stdout:
        return RangeResult(success=False, error="empty stdout")
    size: int | None = None
    b64_text: str | None = None
    err_text: str | None = None
    for line in stdout.splitlines():
        if line.startswith("SIZE="):
            try:
                size = int(line[5:].strip())
            except ValueError:
                return RangeResult(success=False, error=f"bad SIZE: {line!r}")
        elif line.startswith("B64="):
            b64_text = line[4:].strip()
        elif line.startswith("ERR="):
            err_text = line[4:].strip()
    if size is None:
        return RangeResult(success=False, error="missing SIZE line")
    if size == -1:
        # Remote file does not exist (yet). This is a valid, non-error outcome:
        # the tail's _tick_one keys off ``new_size == -1`` to reset its offset
        # and await creation. Collapsing it into success=False would make the
        # reconcile (post-fix) report a spurious failure for a never-written
        # transcript. Only the -2 (exception) sentinel is a real error.
        return RangeResult(success=True, new_size=-1)
    if size < 0:
        return RangeResult(
            success=False, new_size=0,
            error=err_text or f"size sentinel {size}",
        )
    if size <= expected_start:
        return RangeResult(success=True, new_size=size, new_data=b"")
    try:
        new_data = base64.b64decode(b64_text or "")
    except Exception as e:
        return RangeResult(success=False, error=f"base64 decode failed: {e}")
    return RangeResult(success=True, new_size=size, new_data=new_data)


def _download_range_sync(
    sandbox: SandboxHandle, remote_path: str, start: int,
    max_chunk_bytes: int, timeout: float,
) -> RangeResult:
    if sandbox.is_linux:
        cmd = _build_range_cmd_linux(remote_path, start, max_chunk_bytes)
    else:
        cmd = _build_range_cmd_windows(remote_path, start, max_chunk_bytes)
    result = _run_remote_sync(sandbox, cmd, timeout=timeout)
    if result.returncode != 0:
        return RangeResult(
            success=False,
            error=f"rc={result.returncode} stderr={(result.stderr or '')[:200]}",
        )
    return _parse_range_stdout(result.stdout, expected_start=start)


def _check_reachable_sync(sandbox: SandboxHandle, label: str) -> None:
    try:
        resp = requests.get(
            f"{sandbox.endpoint.rstrip('/')}/status", timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "ok":
            raise SandboxUnreachableError(
                f"{label} {sandbox.endpoint} unhealthy: {body}"
            )
    except requests.RequestException as e:
        raise SandboxUnreachableError(
            f"{label} {sandbox.endpoint} unreachable: {e}"
        ) from e
