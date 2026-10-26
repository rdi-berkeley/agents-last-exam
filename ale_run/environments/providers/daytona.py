"""DaytonaProvider — one Daytona Windows (or Linux VM) sandbox per run unit.

Daytona sandboxes are reached through Daytona's control plane, not by IP, so
this provider does three things the GCE path does not need:

1. ``acquire`` creates a sandbox from a *Daytona snapshot* that already carries
   ALE's runtime (see ``env/daytona/build_snapshot.py``:
   Python 3.12 + ``C:\\ale-run\\.venv``, Node, 7-Zip, the cua computer-server
   registered as the interactive scheduled task ``CuaServer-ale`` on port 5000,
   the per-task system packages and the baked task data).
2. It makes sure the cua-server is up inside the box (the scheduled task runs
   at logon; a fresh sandbox boots into the autologon desktop, so normally it
   already is) using Daytona's toolbox ``process.exec`` — the only control path
   that exists before the cua-server answers.
3. It opens an SSH local port-forward (``ssh -N -L <local>:localhost:5000
   <token>@ssh.app.daytona.io``) and hands the framework a plain
   ``http://127.0.0.1:<local>`` endpoint. Every other ALE component (the
   SandboxHandle wire impl, cua-bench's RemoteDesktopSession, the deployers)
   then works unchanged, exactly as against a GCE VM's ``http://<ip>:5000``.
   Inside the sandbox the agent's MCP bridge keeps using ``http://127.0.0.1:5000``
   (``_sandbox_entry`` rewrites the endpoint), so no traffic goes through the
   tunnel during the agent run except the framework's own staging / grading.

Environment yaml (per-snapshot form)::

    snapshots:
      cpu-free:
        provider: daytona
        image: ale-win10            # ALE path conventions the snapshot reproduces
        daytona:
          snapshot: ale-win-20261024   # Daytona snapshot name (sets the sandbox class + size)
          api_key: ${env:DAYTONA_API_KEY}
          target: us                   # optional region
          ttl_minutes: 600             # optional wall-clock kill switch (default 10h)
    task_data_source: baked_in_sandbox
    output_path: local

``api_key`` / ``api_url`` / ``target`` / ``ssh_gateway`` / ``ttl_minutes`` /
``cua_ready_timeout_s`` are provider-wide (last snapshot wins); ``snapshot``,
``cpu`` and ``memory`` are per logical snapshot. ``cpu`` / ``memory`` (GiB), when
given, hot-resize the sandbox upward after creation (Daytona only allows
increases while running); a Daytona snapshot otherwise carries the shape of
the sandbox it was taken from.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from typing import Any

import requests

from ...base_interface import Provider, ReleaseMode, SandboxHandle, SandboxSpec

logger = logging.getLogger(__name__)

_DEFAULT_SSH_GATEWAY = "ssh.app.daytona.io"
_DEFAULT_TTL_MINUTES = 10 * 60
_DEFAULT_CUA_READY_TIMEOUT_S = 600
_CREATE_TIMEOUT_S = 900
_CUA_TASK_NAME = "CuaServer-ale"
_TUNNEL_READY_TIMEOUT_S = 90
_SSH_EXPIRY_MARGIN_MIN = 60


# ======================================================================
# Config
# ======================================================================


@dataclasses.dataclass(frozen=True)
class DaytonaSnapshotRoute:
    """One logical snapshot tag (``cpu-free`` …) → Daytona snapshot."""

    image: str
    """ALE image family whose path conventions the Daytona snapshot reproduces
    (``ale-win10`` for the Windows runtime built by ``env/daytona/build_snapshot.py``)."""

    snapshot: str
    """Daytona snapshot name to create sandboxes from."""

    cpu: int | None = None
    memory: int | None = None
    disk: int | None = None
    """Optional hot-resize targets (vCPU / GiB / GiB disk), applied after creation.
    A grown disk is also extended into the Windows C: partition in ``_post_create``."""
    python: str | None = None
    """Override of the image's in-box interpreter, e.g. ``C:\\ale-py\\alepy.exe`` on the
    hardened snapshots (a renamed copy that agents' python-wide kills miss)."""


# Org quota / runner capacity: how long acquire() waits for a free slot (the
# Daytona tier caps total vCPUs org-wide), polling every _CAPACITY_POLL_S.
_CAPACITY_WAIT_S = int(os.environ.get("ALE_DAYTONA_CAPACITY_WAIT_S", "14400"))
_CAPACITY_POLL_S = 30
_CAPACITY_MARKERS = ("limit exceeded", "capacity", "quota", "insufficient resources", "too many")


def _is_capacity_error(e: Exception) -> bool:
    text = str(e).lower()
    return any(m in text for m in _CAPACITY_MARKERS)


@dataclasses.dataclass(frozen=True)
class DaytonaProviderConfig:
    snapshots: dict[str, DaytonaSnapshotRoute]
    api_key: str | None = None
    """Daytona API key. Falls back to ``$DAYTONA_API_KEY``."""
    api_url: str | None = None
    target: str | None = None
    ssh_gateway: str = _DEFAULT_SSH_GATEWAY
    ttl_minutes: int = _DEFAULT_TTL_MINUTES
    cua_ready_timeout_s: int = _DEFAULT_CUA_READY_TIMEOUT_S
    labels: dict[str, str] = dataclasses.field(default_factory=dict)
    disable_windows_update: bool = True
    """Stop + disable Windows Update inside each fresh sandbox (an update reboot
    kills the cua-server transport mid-run)."""
    isolate_task_data: bool = False
    """Delete every other task's directory under ``data_root`` right after
    creation, so an agent only sees its own task's inputs (the snapshot carries
    the data of every task it was built for)."""
    data_root: str = "E:\\agenthle"
    hide_cua_console: bool = True
    """Re-point the ``CuaServer-ale`` scheduled task at a hidden launcher
    (wscript) and restart it: the stock task runs a visible cmd loop that an
    agent's Alt+F4 / SendKeys experiments can close, killing the transport."""
    post_create_ps: tuple[str, ...] = ()
    """Extra PowerShell commands run through the toolbox after the cua-server
    is up (failures are logged, not fatal)."""


_PROVIDER_WIDE_KEYS = ("api_key", "api_url", "target", "ssh_gateway", "ttl_minutes", "cua_ready_timeout_s", "labels",
                       "disable_windows_update", "isolate_task_data", "data_root", "post_create_ps", "hide_cua_console")
_WATCHDOG_PERIOD_S = 30
_WATCHDOG_FAILS_BEFORE_REPAIR = 2

_HIDE_CUA_CONSOLE_PS = (
    "$q=[char]34; $vbs='C:\\cua-server\\run_hidden.vbs'; "
    "Set-Content -Path $vbs -Value ('CreateObject(' + $q + 'WScript.Shell' + $q + ').Run ' + $q + "
    "'cmd /c C:\\cua-server\\run_cua_server.cmd' + $q + ', 0, False'); "
    "$a = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument ($q + $vbs + $q); "
    f"Set-ScheduledTask -TaskName {_CUA_TASK_NAME} -Action $a | Out-Null; "
    f"Stop-ScheduledTask -TaskName {_CUA_TASK_NAME} -ErrorAction SilentlyContinue; "
    "$me = $PID; $parent = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $PID)).ParentProcessId; "
    "Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $me -and $_.ProcessId -ne $parent -and "
    "$_.Name -match '^(python|pythonw|cmd)\\.exe$' -and $_.CommandLine -match 'computer_server|run_cua_server' } | "
    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; "
    f"Start-Sleep 2; Start-ScheduledTask -TaskName {_CUA_TASK_NAME}; 'cua console hidden'"
)

_DISABLE_WINDOWS_UPDATE_PS = (
    "Stop-Service wuauserv -Force -ErrorAction SilentlyContinue; "
    "Set-Service wuauserv -StartupType Disabled -ErrorAction SilentlyContinue; "
    "Stop-Service UsoSvc -Force -ErrorAction SilentlyContinue; "
    "Set-Service UsoSvc -StartupType Disabled -ErrorAction SilentlyContinue; "
    "New-Item -Path 'HKLM:\\SOFTWARE\\Policies\\Microsoft\\Windows\\WindowsUpdate\\AU' -Force | Out-Null; "
    "Set-ItemProperty -Path 'HKLM:\\SOFTWARE\\Policies\\Microsoft\\Windows\\WindowsUpdate\\AU' -Name NoAutoUpdate -Value 1 -Type DWord; "
    "'windows update disabled'"
)


def _build_provider_config(raw: dict[str, Any]) -> DaytonaProviderConfig:
    snaps_raw = raw.get("snapshots") or {}
    if not snaps_raw:
        raise KeyError("daytona provider config needs a non-empty `snapshots` mapping")
    routes: dict[str, DaytonaSnapshotRoute] = {}
    for tag, entry in snaps_raw.items():
        if not entry.get("snapshot"):
            raise KeyError(f"daytona snapshot {tag!r} is missing `daytona.snapshot` (the Daytona snapshot name)")
        routes[str(tag)] = DaytonaSnapshotRoute(
            image=str(entry.get("image") or "ale-win10"),
            snapshot=str(entry["snapshot"]),
            cpu=int(entry["cpu"]) if entry.get("cpu") is not None else None,
            memory=int(entry["memory"]) if entry.get("memory") is not None else None,
            disk=int(entry["disk"]) if entry.get("disk") is not None else None,
            python=str(entry["python"]) if entry.get("python") else None,
        )
    kwargs: dict[str, Any] = {k: raw[k] for k in _PROVIDER_WIDE_KEYS if raw.get(k) is not None}
    if "ttl_minutes" in kwargs:
        kwargs["ttl_minutes"] = int(kwargs["ttl_minutes"])
    if "cua_ready_timeout_s" in kwargs:
        kwargs["cua_ready_timeout_s"] = int(kwargs["cua_ready_timeout_s"])
    if "labels" in kwargs:
        kwargs["labels"] = {str(k): str(v) for k, v in dict(kwargs["labels"]).items()}
    for flag in ("disable_windows_update", "isolate_task_data", "hide_cua_console"):
        if flag in kwargs:
            kwargs[flag] = bool(kwargs[flag])
    if "post_create_ps" in kwargs:
        v = kwargs["post_create_ps"]
        kwargs["post_create_ps"] = tuple([v] if isinstance(v, str) else [str(x) for x in v])
    return DaytonaProviderConfig(snapshots=routes, **kwargs)


# ======================================================================
# Helpers
# ======================================================================


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _cua_status_ok(url: str, timeout: float = 5) -> bool:
    try:
        resp = requests.get(f"{url.rstrip('/')}/status", timeout=timeout)
        return resp.status_code == 200 and resp.json().get("status") == "ok"
    except Exception:
        return False


def _sandbox_label(spec: SandboxSpec) -> dict[str, str]:
    out = {"ale": "run", "ale-snapshot": spec.snapshot}
    if spec.task_id:
        out["ale-task"] = spec.task_id.replace("/", "_")[:60]
    if spec.harness:
        out["ale-harness"] = spec.harness[:60]
    return out


# ======================================================================
# Provider
# ======================================================================


class DaytonaProvider(Provider):
    """Provider impl over the Daytona Python SDK (``pip install daytona``)."""

    def __init__(self, config: DaytonaProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config
        self._client: Any | None = None
        self._tunnels: dict[str, subprocess.Popen] = {}
        self._watchdogs: dict[str, threading.Event] = {}
        if shutil.which("ssh") is None:
            raise RuntimeError("DaytonaProvider needs an `ssh` client on the host for the cua-server port-forward")

    @property
    def config(self) -> DaytonaProviderConfig:
        return self._cfg

    # ---------------------------------------------------------------- client

    def _daytona(self) -> Any:
        if self._client is None:
            from daytona import Daytona, DaytonaConfig

            api_key = self._cfg.api_key or os.environ.get("DAYTONA_API_KEY")
            if not api_key:
                raise RuntimeError("daytona provider: set `api_key` in the environment yaml or $DAYTONA_API_KEY")
            kwargs: dict[str, Any] = {"api_key": api_key}
            if self._cfg.api_url:
                kwargs["api_url"] = self._cfg.api_url
            if self._cfg.target:
                kwargs["target"] = self._cfg.target
            self._client = Daytona(DaytonaConfig(**kwargs))
        return self._client

    # --------------------------------------------------------------- acquire

    async def acquire(self, spec: SandboxSpec) -> SandboxHandle:
        from ..images import get as get_image

        route = self._cfg.snapshots.get(spec.snapshot)
        if route is None:
            raise KeyError(
                f"daytona provider: snapshot {spec.snapshot!r} is not routed "
                f"(configured: {sorted(self._cfg.snapshots)})"
            )
        image = get_image(route.image)
        sandbox = await asyncio.to_thread(self._create_sandbox, spec, route)
        try:
            await asyncio.to_thread(self._resize_if_requested, sandbox, route)
            await asyncio.to_thread(self._ensure_cua_server, sandbox, image.cua_server_port)
            await asyncio.to_thread(self._post_create, sandbox, spec, image.cua_server_port, route)
            local_port, proc = await asyncio.to_thread(self._open_tunnel, sandbox, image.cua_server_port)
        except BaseException:
            try:
                await asyncio.to_thread(sandbox.delete)
            except Exception:
                logger.exception("daytona: failed to delete sandbox %s after acquire failure", sandbox.id)
            raise
        self._tunnels[sandbox.id] = proc
        endpoint = f"http://127.0.0.1:{local_port}"
        self._start_watchdog(sandbox, endpoint, local_port, image.cua_server_port)
        logger.info(
            "daytona: sandbox %s (%s, %s) ready; cua-server tunnelled at %s",
            sandbox.id, route.snapshot, getattr(sandbox, "target", None), endpoint,
        )
        paths = dict(image.sandbox_paths())
        if route.python:
            paths["python"] = route.python
        return SandboxHandle(
            id=sandbox.id,
            endpoint=endpoint,
            os=image.os,
            **paths,
            metadata={
                "provider": "daytona",
                "sandbox_id": sandbox.id,
                "daytona_snapshot": route.snapshot,
                "daytona_target": getattr(sandbox, "target", None),
                "image": route.image,
                "image_provenance": {"daytona_snapshot": route.snapshot},
                "snapshot": spec.snapshot,
                "machine_type": spec.machine_type,
                "cua_port": local_port,
                "tunnel_pid": proc.pid,
                "resource_limits": {
                    "cpu": getattr(sandbox, "cpu", None),
                    "memory_gb": getattr(sandbox, "memory", None),
                    "disk_gb": getattr(sandbox, "disk", None),
                },
            },
        )

    def _create_sandbox(self, spec: SandboxSpec, route: DaytonaSnapshotRoute) -> Any:
        from daytona import CreateSandboxFromSnapshotParams

        params = CreateSandboxFromSnapshotParams(
            snapshot=route.snapshot,
            labels={**_sandbox_label(spec), **self._cfg.labels},
            auto_stop_interval=0,           # never auto-stop mid-run
            auto_delete_interval=0,         # a stopped sandbox is deleted right away
            ttl_minutes=self._cfg.ttl_minutes,
            target=self._cfg.target,
        )
        t0 = time.time()
        deadline = t0 + _CAPACITY_WAIT_S
        waited = False
        while True:
            try:
                sandbox = self._daytona().create(params, timeout=_CREATE_TIMEOUT_S)
                break
            except Exception as e:
                # Org-wide quota ("Total CPU limit exceeded. Maximum allowed: N") or
                # runner capacity: wait for a slot instead of failing the run unit.
                if not _is_capacity_error(e) or time.time() + _CAPACITY_POLL_S > deadline:
                    raise
                if not waited:
                    logger.warning("daytona: no capacity for %s (%s); waiting up to %ds for a slot",
                                   route.snapshot, str(e).splitlines()[0][:160], _CAPACITY_WAIT_S)
                    waited = True
                time.sleep(_CAPACITY_POLL_S)
        logger.info("daytona: created sandbox %s from %s in %.0fs%s", sandbox.id, route.snapshot,
                    time.time() - t0, " (after waiting for capacity)" if waited else "")
        return sandbox

    @staticmethod
    def _resize_if_requested(sandbox: Any, route: DaytonaSnapshotRoute) -> None:
        if route.cpu is None and route.memory is None and route.disk is None:
            return
        from daytona import Resources

        try:
            sandbox.resize(Resources(cpu=route.cpu, memory=route.memory, disk=route.disk), timeout=600)
            logger.info("daytona: resized %s to cpu=%s memory=%s disk=%s", sandbox.id, route.cpu, route.memory, route.disk)
        except Exception as e:  # increases only; a no-op shape raises
            logger.warning("daytona: resize of %s to cpu=%s memory=%s disk=%s skipped: %s",
                           sandbox.id, route.cpu, route.memory, route.disk, e)

    def _ensure_cua_server(self, sandbox: Any, port: int) -> None:
        """Start the ``CuaServer-ale`` scheduled task if the server is not answering
        yet, then wait for ``/status`` inside the box (through Daytona's toolbox)."""
        deadline = time.time() + self._cfg.cua_ready_timeout_s
        probe = f'curl.exe -s -m 5 http://127.0.0.1:{port}/status'
        kicked = False
        while time.time() < deadline:
            try:
                r = sandbox.process.exec(probe, timeout=30)
                if r.exit_code == 0 and '"status":"ok"' in (r.result or "").replace(" ", ""):
                    return
            except Exception as e:
                logger.debug("daytona: toolbox probe on %s failed: %s", sandbox.id, e)
            if not kicked:
                try:
                    sandbox.process.exec(
                        f'powershell -NoProfile -Command "Start-ScheduledTask -TaskName {_CUA_TASK_NAME}"',
                        timeout=60,
                    )
                except Exception as e:
                    logger.warning("daytona: could not start %s on %s: %s", _CUA_TASK_NAME, sandbox.id, e)
                kicked = True
            time.sleep(5)
        raise RuntimeError(
            f"daytona: cua-server on sandbox {sandbox.id} port {port} did not answer /status "
            f"within {self._cfg.cua_ready_timeout_s}s (is `{_CUA_TASK_NAME}` baked into the snapshot?)"
        )

    def _ps(self, sandbox: Any, script: str, timeout: int = 300) -> str:
        r = sandbox.process.exec(f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{script}"', timeout=timeout)
        out = (r.result or "").strip()
        if r.exit_code != 0:
            raise RuntimeError(f"exit {r.exit_code}: {out[-400:]}")
        return out

    def _post_create(self, sandbox: Any, spec: SandboxSpec, port: int, route: DaytonaSnapshotRoute | None = None) -> None:
        """One-time in-box setup after the cua-server answers: hidden cua-server
        console, no Windows Update reboots, only this task's data visible,
        operator extras."""
        steps: list[tuple[str, str]] = []
        if route is not None and route.disk is not None:
            # Daytona grows the virtual disk; Windows must grow the C: partition into it.
            steps.append(("extend C: partition", (
                "$s = Get-PartitionSupportedSize -DriveLetter C; $p = Get-Partition -DriveLetter C; "
                "if ($s.SizeMax - $p.Size -gt 1GB) { Resize-Partition -DriveLetter C -Size $s.SizeMax }; "
                "$d = Get-PSDrive C; 'C: free ' + [math]::Round($d.Free/1GB,1) + ' GB of ' + [math]::Round(($d.Used+$d.Free)/1GB,1)"
            )))
        if self._cfg.hide_cua_console:
            steps.append(("hide cua console", _HIDE_CUA_CONSOLE_PS))
        if self._cfg.disable_windows_update:
            steps.append(("disable windows update", _DISABLE_WINDOWS_UPDATE_PS))
        if self._cfg.isolate_task_data and spec.task_id:
            keep = spec.task_id.strip("/").replace("/", "\\")
            root = self._cfg.data_root
            steps.append(("isolate task data", (
                f"$root='{root}'; $keep='{keep}'; $n=0; "
                "Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue | ForEach-Object { $dom=$_; "
                "Get-ChildItem -LiteralPath $dom.FullName -Directory | ForEach-Object { "
                "$rel = $dom.Name + '\\' + $_.Name; if ($rel -ne $keep) { "
                "Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue; $n++ } } }; "
                "if (-not (Test-Path -LiteralPath ($root + '\\' + $keep))) { throw ('task data missing: ' + $keep) }; "
                "'removed ' + $n + ' other task dirs'"
            )))
        for i, cmd in enumerate(self._cfg.post_create_ps):
            steps.append((f"post_create_ps[{i}]", cmd))
        for name, script in steps:
            t0 = time.time()
            try:
                out = self._ps(sandbox, script, timeout=600)
                logger.info("daytona: %s on %s: %s (%.0fs)", name, sandbox.id, out[-120:], time.time() - t0)
            except Exception as e:
                if name == "isolate task data":
                    raise RuntimeError(f"daytona: {name} failed on {sandbox.id}: {e}") from e
                logger.warning("daytona: %s on %s failed: %s", name, sandbox.id, e)
            if name == "hide cua console":
                self._ensure_cua_server(sandbox, port)   # wait for the restarted server

    def _start_watchdog(self, sandbox: Any, endpoint: str, local_port: int, port: int) -> None:
        stop = threading.Event()
        self._watchdogs[sandbox.id] = stop
        threading.Thread(target=self._watchdog, name=f"daytona-watchdog-{sandbox.id[:8]}",
                         args=(sandbox, endpoint, local_port, port, stop), daemon=True).start()

    def _watchdog(self, sandbox: Any, endpoint: str, local_port: int, port: int, stop: threading.Event) -> None:
        """Keep the cua transport alive for the run: reopen a dead SSH tunnel on
        the same local port; restart the cua-server task when the box stops
        answering; keep trying through a reboot."""
        fails = 0
        repairs = 0
        while not stop.wait(_WATCHDOG_PERIOD_S):
            if _cua_status_ok(f"{endpoint}", timeout=10):
                if fails:
                    logger.info("daytona: cua transport to %s recovered after %d probe failure(s)", sandbox.id, fails)
                fails = 0
                continue
            fails += 1
            if fails < _WATCHDOG_FAILS_BEFORE_REPAIR:
                continue
            repairs += 1
            proc = self._tunnels.get(sandbox.id)
            try:
                if proc is None:
                    return
                if proc.poll() is not None:
                    logger.warning("daytona: tunnel to %s died (rc=%s); reopening on %d (repair %d)",
                                   sandbox.id, proc.returncode, local_port, repairs)
                    _, new_proc = self._open_tunnel(sandbox, port, local_port=local_port)
                    self._tunnels[sandbox.id] = new_proc
                    continue
                # Tunnel alive but the box does not answer: probe from inside.
                inside_ok = False
                try:
                    r = sandbox.process.exec(f"curl.exe -s -m 5 http://127.0.0.1:{port}/status", timeout=30)
                    inside_ok = r.exit_code == 0 and '"status":"ok"' in (r.result or "").replace(" ", "")
                except Exception as e:
                    logger.warning("daytona: %s unreachable through the toolbox (%s); waiting (repair %d)",
                                   sandbox.id, str(e)[:120], repairs)
                    continue
                if inside_ok:
                    # Server fine, forward broken: rebuild the tunnel.
                    logger.warning("daytona: tunnel to %s forwards nothing; rebuilding (repair %d)", sandbox.id, repairs)
                    with contextlib.suppress(Exception):
                        proc.terminate()
                    _, new_proc = self._open_tunnel(sandbox, port, local_port=local_port)
                    self._tunnels[sandbox.id] = new_proc
                else:
                    logger.warning("daytona: cua-server on %s not answering; restarting %s (repair %d)",
                                   sandbox.id, _CUA_TASK_NAME, repairs)
                    sandbox.process.exec(
                        f'powershell -NoProfile -Command "Start-ScheduledTask -TaskName {_CUA_TASK_NAME}"', timeout=60)
            except Exception as e:
                logger.warning("daytona: watchdog repair of %s failed: %s", sandbox.id, str(e)[:200])

    def _open_tunnel(self, sandbox: Any, port: int, local_port: int | None = None) -> tuple[int, subprocess.Popen]:
        access = sandbox.create_ssh_access(expires_in_minutes=self._cfg.ttl_minutes + _SSH_EXPIRY_MARGIN_MIN)
        last_err = ""
        fixed_port = local_port
        for attempt in range(3):
            local_port = fixed_port if fixed_port is not None else _free_local_port()
            cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=6",
                "-N",
                "-L", f"127.0.0.1:{local_port}:localhost:{port}",
                f"{access.token}@{self._cfg.ssh_gateway}",
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
            url = f"http://127.0.0.1:{local_port}"
            deadline = time.time() + _TUNNEL_READY_TIMEOUT_S
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                if _cua_status_ok(url):
                    return local_port, proc
                time.sleep(1)
            err = ""
            if proc.poll() is None:
                proc.terminate()
                err = "tunnel up but /status never answered"
            else:
                err = (proc.stderr.read().decode(errors="replace") if proc.stderr else "")[-300:]
            last_err = f"attempt {attempt + 1}: {err or 'ssh exited'}"
            logger.warning("daytona: tunnel to %s failed (%s)", sandbox.id, last_err)
        raise RuntimeError(f"daytona: could not tunnel to cua-server on {sandbox.id}: {last_err}")

    # --------------------------------------------------------------- release

    async def release(self, sandbox: SandboxHandle, *, mode: ReleaseMode = "delete") -> None:
        stop = self._watchdogs.pop(sandbox.id, None)
        if stop is not None:
            stop.set()
        proc = self._tunnels.pop(sandbox.id, None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if mode == "keep":
            logger.info("daytona: sandbox %s kept (mode=keep); tunnel closed", sandbox.id)
            return
        try:
            sb = await asyncio.to_thread(self._daytona().get, sandbox.id)
            if mode == "delete":
                try:
                    await asyncio.to_thread(sb.delete)
                    logger.info("daytona: deleted sandbox %s", sandbox.id)
                except Exception as e:
                    # An API key without the delete scope gets 403 here; the sandbox was created with
                    # auto_delete_interval=0, so stopping it makes the platform destroy it instead.
                    logger.info("daytona: delete of %s refused (%s); stopping so auto-delete removes it", sandbox.id, e)
                    await asyncio.to_thread(sb.stop)
            elif mode == "stop":
                await asyncio.to_thread(sb.stop)
                logger.info("daytona: stopped sandbox %s", sandbox.id)
            else:
                raise ValueError(f"unknown release mode: {mode!r}")
        except ValueError:
            raise
        except Exception as e:
            logger.error("daytona: release(%s) of %s failed: %s", mode, sandbox.id, e)

    async def heartbeat(self, sandbox: SandboxHandle) -> None:
        proc = self._tunnels.get(sandbox.id)
        if proc is not None and proc.poll() is not None:
            logger.warning("daytona: tunnel to %s died (rc=%s); reopening", sandbox.id, proc.returncode)
            sb = await asyncio.to_thread(self._daytona().get, sandbox.id)
            port = int(sandbox.metadata.get("cua_port") or 0)
            local_port, new_proc = await asyncio.to_thread(self._open_tunnel, sb, sandbox.cua_server_port)
            if local_port != port:
                # The handle's endpoint is immutable for callers; re-bind the same local port.
                new_proc.terminate()
                raise RuntimeError(f"daytona: tunnel for {sandbox.id} could not be re-bound to 127.0.0.1:{port}")
            self._tunnels[sandbox.id] = new_proc
        try:
            sb = await asyncio.to_thread(self._daytona().get, sandbox.id)
            await asyncio.to_thread(sb.refresh_activity)
        except Exception as e:
            logger.debug("daytona: heartbeat refresh for %s failed: %s", sandbox.id, e)

    # ---------------------------------------------------------------- session

    def open_session(self, sandbox: SandboxHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession

        from .gcloud import _init_computer_skip_wait

        session = RemoteDesktopSession(api_url=sandbox.endpoint, os_type=sandbox.os)
        _init_computer_skip_wait(session)
        return session
