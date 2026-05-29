"""SandboxExecutor — deployer runs INSIDE the cua-server sandbox VM.

The framework ships the ``ale_run/`` source tree to the sandbox (one-time
per run, size-skipped on repeats), writes a ``_spec.json`` into the
deployer's work_dir, fires a small launcher that ``setsid``-spawns
``python -m ale_run.executors._sandbox_entry <spec>``, then polls until
the in-sandbox process drops a ``_done.marker``.

Mirrors :class:`simprun.deployers.claude_code` 's setsid + poll pattern:
**no long-running HTTP connection is held for the agent run**. The cua
``/cmd`` endpoint sees only short calls (launcher / poll / kill).

From the deployer's point of view, "where am I running" is invisible:
its ``self.executor`` is a fresh :class:`LocalExecutor` reconstructed
inside the sandbox by :mod:`_sandbox_entry`, with the sandbox-native
``work_dir`` and the same config + env it would have seen on the host.

What this class owns
--------------------

* :meth:`run_deployer` — ship code + write spec + launcher + poll
* :meth:`gather_dir`   — recursive cua HTTP pull of remote dir → host
* :meth:`download_range` — forward to :meth:`SandboxHandle.download_range`

Hot-artifact incremental tail (called by lifecycle when the deployer
declares ``hot_artifacts``) is exposed as a module-level function,
:func:`tail_hot_artifacts`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..base_interface import (
    BaseExecutor,
    GatherReport,
    RangeResult,
    SandboxHandle,
)
from ._secrets import SECRET_GATHER_EXCLUDES, SECRETS_FILE

if TYPE_CHECKING:
    from ..base_interface import AgentRunResult, BaseAgentDeployer

logger = logging.getLogger(__name__)


# Convention: ale_run source ships to ``<home>/.ale-src/`` on the sandbox,
# where <home> is the parent of work_dir_base (e.g. /home/user/.ale → /home/user).
def _ale_src_root_for(sandbox: SandboxHandle) -> str:
    if sandbox.is_linux:
        home = sandbox.work_dir_base.rstrip("/").rsplit("/", 1)[0]
        return f"{home}/.ale-src"
    home = sandbox.work_dir_base.rstrip("\\").rsplit("\\", 1)[0]
    return rf"{home}\.ale-src"


# Gather retries
_GATHER_RETRIES = 3
_GATHER_BACKOFFS_S = (1.0, 3.0, 9.0)

# Poll loop tuning (matches simprun)
_POLL_INTERVAL_S = 10.0
_PID_WAIT_S = 4.5            # how long to wait for the launcher to write the PID file
_PID_WAIT_TICK_S = 0.3

# Incremental tail tuning
_TAIL_INTERVAL_S = 60.0
_TAIL_RECONCILE_TIMEOUT_S = 60.0
_TAIL_RECONCILE_RETRIES = 3
_TAIL_RECONCILE_DELAY_S = 1.0
_TAIL_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass
class SandboxExecutor(BaseExecutor):
    """In-sandbox substrate. Code is shipped + run inside the cua VM via
    a detached subprocess; host polls until done.marker."""

    type: ClassVar[str] = "sandbox"

    async def run_deployer(
        self,
        *,
        deployer_cls: type["BaseAgentDeployer"],
        prompt: str,
        timeout_s: float,
    ) -> "AgentRunResult":
        from ..base_interface import AgentRunResult

        sb = self.sandbox
        sep = "/" if sb.is_linux else "\\"
        wd = self.work_dir.rstrip(sep)
        ale_src_root = _ale_src_root_for(sb)

        spec_path = f"{wd}{sep}_spec.json"
        secrets_path = f"{wd}{sep}{SECRETS_FILE}"
        pid_file = f"{wd}{sep}_pid"
        result_path = f"{wd}{sep}_result.json"
        done_marker = f"{wd}{sep}_done.marker"
        entry_log = f"{wd}{sep}_entry.log"
        launcher_path = (
            f"{wd}{sep}_launcher.sh" if sb.is_linux
            else f"{wd}\\_launcher.ps1"
        )

        # 1. Ship ale_run/ to sandbox (size-skip on repeats)
        try:
            await self._ship_ale_subtree(ale_src_root)
        except Exception as e:                                      # noqa: BLE001
            logger.exception("ship_ale_subtree failed")
            return AgentRunResult(
                status="failed",
                error=f"ship_ale_subtree: {type(e).__name__}: {e}",
            )

        # 2. Make sure work_dir exists on sandbox
        await sb.mkdir(self.work_dir)

        # 3. Reset stale state from any prior attempt (best-effort)
        await sb.rm([pid_file, result_path, done_marker, entry_log, secrets_path])

        # 4. Write spec.json into the sandbox's work_dir.
        #    Secrets (api keys etc.) are deliberately KEPT OUT of the spec —
        #    _spec.json is gathered back to host .logs and must stay keyless.
        #    The env goes in a separate _secrets.json that the entry reads
        #    once and deletes (see _secrets.py).
        spec = {
            "deployer_module": deployer_cls.__module__,
            "deployer_class": deployer_cls.__name__,
            "config_module": self.config.__class__.__module__,
            "config_class": self.config.__class__.__name__,
            "config_kwargs": _config_to_kwargs(self.config),
            "sandbox_kwargs": _sandbox_to_kwargs(self.sandbox),
            "work_dir": self.work_dir,
            "secrets_file": SECRETS_FILE,
            "prompt": prompt,
            "timeout_s": float(timeout_s),
        }
        await sb.write_file(spec_path, json.dumps(spec, indent=2))

        # 4b. Write the transient secrets sidecar (read-once + self-deleted
        #     by the entry). Never gathered to host logs.
        await sb.write_file(secrets_path, json.dumps(dict(self.env or {})))

        # 5. Write launcher script + fire it (short RPC: returns in seconds)
        launcher_body = _build_launcher(
            sandbox=sb,
            python=sb.python,
            ale_src_root=ale_src_root,
            spec_path=spec_path,
            pid_file=pid_file,
            entry_log=entry_log,
        )
        await sb.write_file(launcher_path, launcher_body)

        if sb.is_linux:
            spawn_cmd = (
                f"chmod +x {shlex.quote(launcher_path)} && "
                f"bash {shlex.quote(launcher_path)}"
            )
        else:
            spawn_cmd = (
                f'powershell -NoProfile -ExecutionPolicy Bypass -File '
                f'"{launcher_path}"'
            )
        spawn_res = await sb.run_command(spawn_cmd, timeout=60)
        # The launcher backgrounds the entry via setsid+disown and returns in
        # milliseconds, so a slow cua-server can drop the spawn RPC's SSE
        # response (rc=-1 "transport error") even though the command actually
        # ran server-side. Don't treat a transport-level failure as fatal:
        # fall through to the PID check, which authoritatively tells us whether
        # the entry started. Only a clean command failure (rc>0) is fatal here.
        if spawn_res.returncode > 0:
            return AgentRunResult(
                status="failed",
                error=f"launcher spawn rc={spawn_res.returncode}: "
                      f"{(spawn_res.stderr or '').strip()[:300]}",
            )
        if spawn_res.returncode != 0:
            logger.warning(
                "sandbox: launcher spawn RPC returned rc=%s (%s); "
                "verifying via PID file",
                spawn_res.returncode, (spawn_res.stderr or "").strip()[:120],
            )

        # 6. Read PID (launcher writes it synchronously; tolerate tiny flush gap)
        pid = await self._read_pid(pid_file)
        if pid is None:
            entry_tail = await self._tail_log(entry_log)
            spawn_note = (
                f"spawn rc={spawn_res.returncode}: "
                f"{(spawn_res.stderr or '').strip()[:120]}; "
                if spawn_res.returncode != 0 else ""
            )
            return AgentRunResult(
                status="failed",
                error=f"launcher did not write usable PID; {spawn_note}"
                      f"entry log tail: {entry_tail}",
            )

        logger.info(
            "sandbox: spawned pid=%s (deployer=%s, work_dir=%s)",
            pid, deployer_cls.__name__, self.work_dir,
        )

        # 7. Poll loop — short RPCs every _POLL_INTERVAL_S
        t0 = time.monotonic()
        deadline = t0 + timeout_s
        marker_hit = False
        while time.monotonic() < deadline:
            try:
                if await sb.exists(done_marker):
                    marker_hit = True
                    break
            except Exception as e:                                  # noqa: BLE001
                logger.debug("done.marker probe failed: %s", e)

            alive_cmd = (
                f"kill -0 {pid}" if sb.is_linux
                else (
                    'powershell -NoProfile -Command "'
                    f"Get-Process -Id {pid} -ErrorAction Stop | Out-Null"
                    '"'
                )
            )
            alive = await sb.run_command(alive_cmd, timeout=60)
            if alive.returncode != 0:
                # Process gone — give marker one more chance (race with disk flush)
                await asyncio.sleep(2)
                if await sb.exists(done_marker):
                    marker_hit = True
                    break
                entry_tail = await self._tail_log(entry_log)
                return AgentRunResult(
                    status="failed",
                    pid=pid,
                    duration_s=time.monotonic() - t0,
                    error=f"sandbox process disappeared before done.marker; "
                          f"entry log tail: {entry_tail}",
                )
            await asyncio.sleep(_POLL_INTERVAL_S)

        duration_s = time.monotonic() - t0

        # 8. Handle timeout — kill the in-sandbox process
        if not marker_hit:
            logger.warning(
                "sandbox: wall budget %.0fs exceeded — killing pid=%s", timeout_s, pid,
            )
            await self._kill(pid)
            entry_tail = await self._tail_log(entry_log)
            return AgentRunResult(
                status="timeout",
                pid=pid,
                duration_s=duration_s,
                error=f"agent wall budget {timeout_s:.0f}s exceeded on sandbox; "
                      f"entry log tail: {entry_tail}",
            )

        # 9. Read result.json
        try:
            raw = await sb.read_text(result_path)
            out = json.loads(raw)
        except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as e:
            entry_tail = await self._tail_log(entry_log)
            return AgentRunResult(
                status="failed",
                pid=pid,
                duration_s=duration_s,
                error=f"cannot read _result.json ({e}); "
                      f"entry log tail: {entry_tail}",
            )

        if not out.get("ok", False):
            tb = out.get("traceback") or ""
            err = out.get("error") or "sandbox bootstrap failed"
            return AgentRunResult(
                status=out.get("status", "failed"),
                pid=pid,
                duration_s=out.get("duration_s") or duration_s,
                error=f"{err}\n{tb}" if tb else err,
            )
        return AgentRunResult(
            status=out.get("status", "failed"),
            error=out.get("error"),
            transcript_path=out.get("transcript_path"),
            stderr_path=out.get("stderr_path"),
            pid=out.get("pid") or pid,
            exit_code=out.get("exit_code"),
            duration_s=out.get("duration_s") or duration_s,
        )

    async def gather_dir(
        self, *, src: str, dst: Path,
    ) -> GatherReport:
        dst.mkdir(parents=True, exist_ok=True)
        try:
            entries = await self.sandbox.list_dir(src)
        except Exception as e:                                      # noqa: BLE001
            logger.warning("list_dir failed for %s: %s", src, e)
            return GatherReport(transport="cua", error=str(e))

        if not entries:
            return GatherReport(transport="cua")

        file_count = 0
        total_bytes = 0
        last_error: str | None = None
        sep = "/" if self.sandbox.is_linux else "\\"

        for entry in entries:
            rel = entry["relpath"]
            # Never pull secret-bearing control files to host logs.
            if Path(rel.replace("\\", "/")).name in SECRET_GATHER_EXCLUDES:
                continue
            local = dst / rel.replace("\\", "/")
            if entry["is_dir"]:
                local.mkdir(parents=True, exist_ok=True)
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            remote_path = f"{src.rstrip(sep)}{sep}{rel.replace('/', sep)}"
            ok = await _download_with_retry(self.sandbox, remote_path, local)
            if ok:
                file_count += 1
                try:
                    total_bytes += local.stat().st_size
                except OSError:
                    pass
            else:
                last_error = f"download failed: {rel}"
                logger.warning("gather_dir: %s", last_error)

        return GatherReport(
            transport="cua",
            files=file_count,
            bytes=total_bytes,
            error=last_error,
        )

    async def download_range(
        self, *, src: str, start: int, max_bytes: int,
    ) -> RangeResult:
        return await self.sandbox.download_range(
            src, start=start, max_chunk_bytes=max_bytes,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _ship_ale_subtree(self, ale_src_root: str) -> None:
        """scp every ``ale_run/**/*.py`` into the sandbox at ``ale_src_root``.

        Idempotent: per-file size compare against the remote first;
        skip when bytes match.

        Vendored upstream agent sources (``ale_run/agents/*/upstream/``) are
        skipped: they carry no ``__init__.py``, are never imported by the
        in-sandbox ``_sandbox_entry`` runner, and deployers that need them
        re-fetch from their own git remote inside the container.  Shipping
        them is pure waste -- the hermes fork alone is 1337 .py files, each a
        base64 round-trip over the CUA endpoint, which under host load can
        stretch the ship phase into tens of minutes.
        """
        host_root = _host_ale_root()
        sandbox = self.sandbox
        sep = "/" if sandbox.is_linux else "\\"

        await sandbox.mkdir(ale_src_root)

        files: list[tuple[Path, str]] = []
        patterns = (
            "*.py",
            "agents/*/pyproject.toml",
            # cua MCP bridge source (package.json + package-lock.json + src/*.js).
            # Shipped so the in-sandbox ensure-step can install the bridge into
            # mcp_server_dir; node_modules is never shipped (rebuilt on-VM by
            # npm install). Scoped to the bridge dir so we don't sweep stray
            # json across the tree. See ensure_cua_mcp_server.
            "agents/_assets/cua_mcp_server/**/*.js",
            "agents/_assets/cua_mcp_server/**/*.json",
        )
        for pattern in patterns:
            for src_path in sorted(host_root.rglob(pattern)):
                rel = src_path.relative_to(host_root)
                if "upstream" in rel.parts or "node_modules" in rel.parts:
                    continue
                sandbox_rel = "ale_run" + sep + rel.as_posix().replace("/", sep)
                files.append((src_path, sandbox_rel))

        for src_path, rel in files:
            remote_path = f"{ale_src_root.rstrip(sep)}{sep}{rel}"
            data = src_path.read_bytes()
            try:
                remote_bytes = await sandbox.read_file(remote_path)
                if remote_bytes == data:
                    continue
            except FileNotFoundError:
                pass
            except RuntimeError:
                pass  # transport miss → treat as cache miss; overwrite below
            parent = remote_path.rsplit(sep, 1)[0]
            await sandbox.mkdir(parent)
            await sandbox.write_file(remote_path, data)

    async def _read_pid(self, pid_file: str) -> int | None:
        """Poll for the launcher's pid file. Returns None on timeout."""
        deadline = time.monotonic() + _PID_WAIT_S
        while time.monotonic() < deadline:
            try:
                raw = (await self.sandbox.read_text(pid_file)).strip()
                if raw:
                    try:
                        return int(raw)
                    except ValueError:
                        return None
            except FileNotFoundError:
                pass
            except Exception:                                       # noqa: BLE001
                pass
            await asyncio.sleep(_PID_WAIT_TICK_S)
        return None

    async def _kill(self, pid: int) -> None:
        """TERM + KILL the in-sandbox pid; idempotent."""
        sb = self.sandbox
        try:
            if sb.is_linux:
                await sb.run_command(
                    f"kill -TERM {pid} 2>/dev/null || true", timeout=30,
                )
                await asyncio.sleep(2)
                await sb.run_command(
                    f"kill -KILL {pid} 2>/dev/null || true", timeout=30,
                )
            else:
                await sb.run_command(
                    'powershell -NoProfile -Command "'
                    f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"
                    '"',
                    timeout=30,
                )
        except Exception as e:                                      # noqa: BLE001
            logger.debug("_kill pid=%s failed: %s", pid, e)

    async def _tail_log(self, entry_log: str, max_bytes: int = 1500) -> str:
        """Tail the in-sandbox entry log for diagnostic messages.
        Returns ``"(unavailable)"`` if read fails."""
        try:
            text = await self.sandbox.read_text(entry_log)
            return text[-max_bytes:] if text else "(empty)"
        except Exception:                                           # noqa: BLE001
            return "(unavailable)"


def _build_launcher(
    *,
    sandbox: SandboxHandle,
    python: str,
    ale_src_root: str,
    spec_path: str,
    pid_file: str,
    entry_log: str,
) -> str:
    """Compose the per-OS launcher script.

    The launcher fires the entry as a fully detached subprocess and
    immediately writes its PID. After it returns (within seconds), the
    host's ``run_command`` RPC for the launcher returns — no long
    connection is held.
    """
    if sandbox.is_linux:
        # Idempotent: a dropped spawn-RPC response triggers a host-side retry
        # that re-runs this launcher. Guard against spawning a second entry by
        # bailing if the recorded PID is still alive.
        return (
            "#!/bin/bash\n"
            "set -u\n"
            f"PIDF={shlex.quote(pid_file)}\n"
            "if [ -s \"$PIDF\" ] && kill -0 \"$(cat \"$PIDF\")\" 2>/dev/null; then\n"
            "  exit 0\n"
            "fi\n"
            f"export PYTHONPATH={shlex.quote(ale_src_root)}:${{PYTHONPATH:-}}\n"
            f"setsid {shlex.quote(python)} -m ale_run.executors._sandbox_entry "
            f"{shlex.quote(spec_path)} "
            f"</dev/null >{shlex.quote(entry_log)} 2>&1 &\n"
            "CHILD=$!\n"
            "echo \"$CHILD\" > \"$PIDF\"\n"
            "disown $CHILD 2>/dev/null || true\n"
        )
    # Windows: PowerShell launcher
    # We add ale_src_root to PYTHONPATH for the spawned python, then
    # Start-Process with -PassThru to capture the PID.
    py_quoted = python.replace("'", "''")
    src_quoted = ale_src_root.replace("'", "''")
    spec_quoted = spec_path.replace("'", "''")
    pid_quoted = pid_file.replace("'", "''")
    log_quoted = entry_log.replace("'", "''")
    # Idempotent guard (mirrors the Linux launcher): a dropped spawn-RPC
    # response triggers a host-side retry that re-runs this launcher. Without
    # the guard a second entry would spawn, and since the first entry reads +
    # deletes _secrets.json, the second comes up with no API keys. Bail if the
    # recorded PID is still alive.
    return (
        "$ErrorActionPreference = 'Continue'\n"
        f"$pidFile = '{pid_quoted}'\n"
        "if (Test-Path $pidFile) {\n"
        "  $oldPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)\n"
        "  if ($oldPid) {\n"
        "    $running = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue\n"
        "    if ($running) { exit 0 }\n"
        "  }\n"
        "}\n"
        f"$env:PYTHONPATH = '{src_quoted};' + $env:PYTHONPATH\n"
        f"$proc = Start-Process -FilePath '{py_quoted}' "
        f"-ArgumentList '-m','ale_run.executors._sandbox_entry','{spec_quoted}' "
        f"-WindowStyle Hidden -PassThru "
        f"-RedirectStandardOutput '{log_quoted}' "
        f"-RedirectStandardError '{log_quoted}.err'\n"
        f"$proc.Id | Out-File -FilePath $pidFile -Encoding ascii -NoNewline\n"
    )


# ======================================================================
# Hot-artifact incremental tail — function, not class
# ======================================================================


async def tail_hot_artifacts(
    *,
    executor: BaseExecutor,
    targets: list[tuple[str, Path]],   # [(sandbox_src, host_dst), ...]
    stop_event: asyncio.Event,
    interval_s: float = _TAIL_INTERVAL_S,
) -> str | None:
    """Background tail: append new bytes from sandbox files into host
    mirrors at ``interval_s`` cadence. Returns first reconcile error or
    ``None`` on clean stop.

    JSONL boundary-safe: only commit bytes up to the last ``\\n`` so a
    half-written record isn't appended; the same bytes are re-fetched
    next tick once the writer flushes a newline.
    """
    offsets: dict[str, int] = {src: _local_offset(dst) for src, dst in targets}
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            break  # stop signalled → fall through to reconcile
        except asyncio.TimeoutError:
            pass
        for src, dst in targets:
            try:
                await _tick_one(executor, src, dst, offsets)
            except Exception as e:                                  # noqa: BLE001
                logger.debug("tail tick failed for %s: %s", src, e)

    # Final reconcile
    deadline = time.monotonic() + _TAIL_RECONCILE_TIMEOUT_S
    last_err: str | None = None
    for src, dst in targets:
        prev_size = -1
        for _ in range(_TAIL_RECONCILE_RETRIES + 1):
            if time.monotonic() > deadline:
                last_err = f"reconcile timeout after {_TAIL_RECONCILE_TIMEOUT_S}s"
                break
            try:
                size = await _tick_one(executor, src, dst, offsets)
            except Exception as e:                                  # noqa: BLE001
                last_err = str(e)
                await asyncio.sleep(_TAIL_RECONCILE_DELAY_S)
                continue
            if size == prev_size:
                break
            prev_size = size
            await asyncio.sleep(_TAIL_RECONCILE_DELAY_S)
    return last_err


async def _tick_one(
    executor: BaseExecutor,
    src: str,
    dst: Path,
    offsets: dict[str, int],
) -> int:
    """Pull a chunk starting at ``offsets[src]``, commit jsonl-safe to
    ``dst``, return the remote size."""
    start = offsets[src]
    rr = await executor.download_range(
        src=src, start=start, max_bytes=_TAIL_CHUNK_BYTES,
    )
    if not rr.success:
        return -2
    size = rr.new_size
    if size == -1:
        offsets[src] = 0
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        return -1
    if size < start:
        # rotation/truncation
        offsets[src] = 0
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        return size
    delta = rr.new_data
    if not delta:
        return size
    last_nl = delta.rfind(b"\n")
    if last_nl == -1:
        return size
    safe = delta[: last_nl + 1]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "ab") as f:
        f.write(safe)
        f.flush()
    offsets[src] += len(safe)
    return size


def _local_offset(dst: Path) -> int:
    """Byte offset after the last ``\\n`` in ``dst``, so a re-launched
    tail picks up where the previous one left off."""
    if not dst.exists():
        return 0
    try:
        data = dst.read_bytes()
    except OSError:
        return 0
    last_nl = data.rfind(b"\n")
    return 0 if last_nl == -1 else last_nl + 1


# ======================================================================
# helpers
# ======================================================================


def _host_ale_root() -> Path:
    """Host's ``ale_run/`` package root."""
    return Path(__file__).resolve().parents[1]


def _config_to_kwargs(cfg: Any) -> dict[str, Any]:
    import dataclasses

    out: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if isinstance(val, (str, int, float, bool, type(None), list, dict, tuple)):
            out[f.name] = val
    return out


def _sandbox_to_kwargs(sb: SandboxHandle) -> dict[str, Any]:
    return {
        "id": sb.id,
        "endpoint": sb.endpoint,
        "os": sb.os,
        "work_dir_base": sb.work_dir_base,
        "task_data_root": sb.task_data_root,
        "node": sb.node,
        "python": sb.python,
        "mcp_server_dir": sb.mcp_server_dir,
        "cua_server_port": sb.cua_server_port,
        "metadata": dict(sb.metadata or {}),
    }


async def _download_with_retry(
    sandbox: SandboxHandle, remote_path: str, local: Path,
) -> bool:
    for attempt in range(_GATHER_RETRIES):
        try:
            ok = await sandbox.download_to_local(
                remote_path, str(local), timeout=120,
            )
        except Exception as e:                                      # noqa: BLE001
            logger.debug("download_to_local raised %s (attempt %d): %s",
                         remote_path, attempt + 1, e)
            ok = False
        if ok:
            return True
        if attempt < _GATHER_RETRIES - 1:
            await asyncio.sleep(_GATHER_BACKOFFS_S[attempt])
    return False
