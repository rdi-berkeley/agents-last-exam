"""Sandbox-side entry. Invoked as a normal Python module:

    python -m ale_run.executors._sandbox_entry <spec_path>

inside the sandbox VM after :class:`SandboxExecutor` has scp'd the
``ale_run/`` source tree to the sandbox's ``ale_src_root`` and exported
that directory on ``PYTHONPATH``.

Reads ``<spec_path>`` (a JSON file the host wrote into the sandbox's
work_dir), reconstructs config + sandbox handle + a :class:`LocalExecutor`
in-sandbox, supervises the deployer in a contained worker, and writes
``_result.json`` + ``_done.marker`` only after stopping all descendants.

Symmetric with :mod:`_docker_entry`. No cua ``python_exec`` involved —
this is a normal Python process spawned via ``setsid`` (linux) or
``Start-Process`` (windows) by the host-side ``SandboxExecutor``.
"""
from __future__ import annotations

import asyncio
import ctypes
import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("sandbox_entry")
_PROCESS_STOP_TIMEOUT_S = 20.0


class _LinuxProcessTree:
    def __init__(self) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
        enabled = ctypes.c_int()
        if libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1:
            raise RuntimeError("child subreaper unavailable")

    def stop(self) -> None:
        deadline = time.monotonic() + _PROCESS_STOP_TIMEOUT_S
        children_path = Path(f"/proc/self/task/{os.getpid()}/children")
        while time.monotonic() < deadline:
            while True:
                try:
                    child_pid, _status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    return
                if child_pid == 0:
                    break
            for child_pid in map(int, children_path.read_text().split()):
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.02)
        raise RuntimeError("solver descendants remain after SIGKILL")


class _JobBasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JobLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobBasicLimits),
        ("IoInfo", ctypes.c_uint64 * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JobAccounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


class _WindowsJob:
    def __init__(self) -> None:
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, ctypes.c_wchar_p], ctypes.c_void_p),
            "SetInformationJobObject": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int,
            ),
            "OpenProcess": ([ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], ctypes.c_void_p),
            "AssignProcessToJobObject": ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
            "TerminateJobObject": ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
            "QueryInformationJobObject": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                 ctypes.c_uint32, ctypes.c_void_p], ctypes.c_int,
            ),
            "CloseHandle": ([ctypes.c_void_p], ctypes.c_int),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes = argtypes
            function.restype = restype
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError()
        limits = _JobLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits),
        ):
            error = ctypes.WinError()
            self.kernel.CloseHandle(self.handle)
            raise error

    def assign(self, pid: int) -> None:
        process = self.kernel.OpenProcess(0x0101, False, pid)
        if not process:
            raise ctypes.WinError()
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError()
        finally:
            self.kernel.CloseHandle(process)

    def stop(self) -> None:
        try:
            if not self.kernel.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError()
            deadline = time.monotonic() + _PROCESS_STOP_TIMEOUT_S
            while time.monotonic() < deadline:
                accounting = _JobAccounting()
                if not self.kernel.QueryInformationJobObject(
                    self.handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None,
                ):
                    raise ctypes.WinError()
                if accounting.ActiveProcesses == 0:
                    return
                time.sleep(0.02)
            raise RuntimeError("solver processes remain in Windows Job Object")
        finally:
            self.kernel.CloseHandle(self.handle)


def supervise(spec_path: Path, spec: dict) -> dict:
    """Run a gated worker; publish stop evidence only after draining descendants."""
    work_dir = Path(spec["work_dir"])
    token = spec["run_token"]
    worker = None
    process_tree = None
    stopped = False
    out = {"ok": False, "status": "failed", "error": "solver did not start"}
    try:
        if sys.platform == "linux":
            process_tree = _LinuxProcessTree()
        elif sys.platform == "win32":
            process_tree = _WindowsJob()
        else:
            raise RuntimeError(f"unsupported process containment: {sys.platform}")

        stop_path = work_dir / "_stop.request"
        if stop_path.exists() and stop_path.read_text() == token:
            out = {"ok": False, "status": "failed", "error": "solver stopped before launch"}
        else:
            worker = subprocess.Popen(
                [sys.executable, "-m", "ale_run.executors._sandbox_entry",
                 str(spec_path), "--worker"],
                stdin=subprocess.PIPE,
            )
            if isinstance(process_tree, _WindowsJob):
                process_tree.assign(worker.pid)
            worker.stdin.write(b"1")
            worker.stdin.close()
            deadline = time.monotonic() + float(spec["timeout_s"])
            while worker.poll() is None:
                if stop_path.exists() and stop_path.read_text() == token:
                    out = {"ok": False, "status": "failed", "error": "solver stop requested"}
                    break
                if time.monotonic() >= deadline:
                    out = {"ok": False, "status": "timeout", "error": "solver budget exceeded"}
                    break
                time.sleep(0.05)
            else:
                out = json.loads((work_dir / "_worker_result.json").read_text())
    except BaseException as exc:
        logger.exception("sandbox supervisor failed")
        out = {"ok": False, "status": "failed", "error": str(exc),
               "traceback": traceback.format_exc()}
    finally:
        try:
            if worker is not None:
                if worker.stdin and not worker.stdin.closed:
                    worker.stdin.close()
                if worker.poll() is None:
                    worker.kill()
                worker.wait(timeout=_PROCESS_STOP_TIMEOUT_S)
            if process_tree is not None:
                process_tree.stop()
            stopped = True
        except BaseException as exc:
            logger.exception("sandbox process cleanup failed")
            out = {"ok": False, "status": "failed", "error": f"process cleanup: {exc}"}
        report = {"run_token": token, "stopped": stopped, "error": out.get("error")}
        report_path = work_dir / "_stopped.json"
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report))
        temporary.replace(report_path)
    return out


def run(spec: dict, env: dict | None = None) -> dict:
    """Drive ``install() + launch()`` of the deployer. Return result dict.

    Pure data in / out. Caught exceptions become ``ok=False`` with a
    full traceback so the host poller can surface them.

    ``env`` holds the framework-supplied secrets (api keys, base URLs)
    read from the read-once ``_secrets.json`` sidecar; it is never part
    of ``spec`` (which is gathered to host logs and must stay keyless).
    """
    # Inject framework env vars so the deployer's spawned subprocess
    # inherits them. Fall back to a legacy in-spec env for forward/back
    # compatibility, but the writer no longer puts secrets in the spec.
    if env is None:
        env = spec.get("env") or {}
    from ale_run.executors._secrets import restore_config_secrets
    config_kwargs, env = restore_config_secrets(spec["config_kwargs"], env)
    for k, v in env.items():
        os.environ[str(k)] = str(v)

    try:
        # Install agent-declared Python deps before importing the deployer
        # so top-level imports (e.g. `import yaml`) don't crash.
        from ale_run.base_interface import BaseExecutor
        if spec.get("install_agent_deps", True):
            BaseExecutor.install_agent_deps(spec["deployer_module"])

        from ale_run.base_interface import SandboxHandle
        from ale_run.executors.local import LocalExecutor

        cfg_mod = importlib.import_module(spec["config_module"])
        dep_mod = importlib.import_module(spec["deployer_module"])
        cfg_cls = getattr(cfg_mod, spec["config_class"])
        dep_cls = getattr(dep_mod, spec["deployer_class"])

        cfg = cfg_cls(**config_kwargs)
        sandbox = SandboxHandle(**spec["sandbox_kwargs"])
        # We are running INSIDE the sandbox: cua-server is co-located on
        # loopback at the image's declared port. The handle's ``endpoint`` was
        # the host-side URL (e.g. a socat sidecar port) which is NOT reachable
        # from in here, so point it at loopback. This is what the cua MCP bridge
        # (via executor.cua_bridge_url) and any in-sandbox run_command must use.
        if spec.get("sandbox_local", True):
            sandbox.endpoint = f"http://127.0.0.1:{sandbox.cua_server_port}"
        executor = LocalExecutor(
            config=cfg,
            work_dir=spec["work_dir"],
            sandbox=sandbox,
            env=env,
        )
        deployer = dep_cls(executor)

        loop = asyncio.new_event_loop()
        try:
            logger.info(
                "sandbox_entry: %s.install (work_dir=%s)",
                dep_cls.__name__, executor.work_dir,
            )
            loop.run_until_complete(deployer.install())
            timeout_s = float(spec.get("timeout_s") or 1800.0)
            logger.info(
                "sandbox_entry: %s.launch (timeout_s=%.0f)",
                dep_cls.__name__, timeout_s,
            )
            result = loop.run_until_complete(
                asyncio.wait_for(
                    deployer.launch(spec["prompt"]),
                    timeout=timeout_s,
                )
            )
        finally:
            loop.close()

        return {
            "ok": True,
            "status": result.status,
            "error": result.error,
            "transcript_path": result.transcript_path,
            "stderr_path": result.stderr_path,
            "pid": result.pid,
            "exit_code": result.exit_code,
            "duration_s": result.duration_s,
        }
    except Exception as exc:                                       # noqa: BLE001
        logger.exception("sandbox_entry crashed")
        return {
            "ok": False,
            "status": (
                "timeout" if isinstance(exc, asyncio.TimeoutError) else "failed"
            ),
            "error": f"sandbox_entry: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    """Read ``spec_path`` from argv, run, write ``_result.json`` +
    ``_done.marker`` into the work_dir specified in the spec."""
    if len(sys.argv) < 2:
        print(
            "usage: python -m ale_run.executors._sandbox_entry <spec_path>",
            file=sys.stderr,
        )
        return 2
    spec_path = Path(sys.argv[1])
    try:
        spec = json.loads(spec_path.read_text())
    except Exception as e:                                          # noqa: BLE001
        print(f"sandbox_entry: cannot read spec {spec_path}: {e}", file=sys.stderr)
        return 2

    work_dir = Path(spec["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    if sys.argv[2:] == ["--worker"]:
        if sys.stdin.buffer.read(1) != b"1":
            return 2
        from ale_run.executors._secrets import read_and_delete_secrets
        env = read_and_delete_secrets(spec_path.parent)
        out = run(spec, env)
        (work_dir / "_worker_result.json").write_text(json.dumps(out, indent=2))
        return 0

    try:
        with (work_dir / f"_supervisor_{spec['run_token']}.claim").open("x"):
            pass
    except FileExistsError:
        return 0
    out = supervise(spec_path, spec)
    (work_dir / "_result.json").write_text(json.dumps(out, indent=2))
    # done.marker last — the host poller treats its presence as "result is
    # ready to read".
    (work_dir / "_done.marker").write_text("0\n" if out.get("ok") else "1\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
