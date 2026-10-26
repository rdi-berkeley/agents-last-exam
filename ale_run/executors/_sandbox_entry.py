"""Sandbox-side entry. Invoked as a normal Python module:

    python -m ale_run.executors._sandbox_entry <spec_path>

inside the sandbox VM after :class:`SandboxExecutor` has scp'd the
``ale_run/`` source tree to the sandbox's ``ale_src_root`` and exported
that directory on ``PYTHONPATH``.

Reads ``<spec_path>`` (a JSON file the host wrote into the sandbox's
work_dir), reconstructs config + sandbox handle + a :class:`LocalExecutor`
in-sandbox, supervises the deployer in a contained worker, and writes
``_result.json`` + ``_done.marker`` only after all descendants have exited.

Symmetric with :mod:`_docker_entry`. No cua ``python_exec`` involved —
this is a normal Python process detached via systemd/setsid (linux) or
``Start-Process`` (windows) by the host-side ``SandboxExecutor``.
"""
from __future__ import annotations

import asyncio
import ctypes
import importlib
import json
import logging
import os
import shutil
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
# After the worker has reported, how long leftover descendants may linger
# before the supervisor terminates them and reports anyway.
_DRAIN_GRACE_S = 60.0
_SYSTEMD_RUNTIME = Path("/run/systemd/system")


def launch(spec_path: Path, spec: dict) -> int:
    """Detach one supervisor per run, isolated from the CUA service on systemd."""
    import fcntl

    work_dir = Path(spec["work_dir"])
    token = spec["run_token"]
    if not token or any(character not in "0123456789abcdefghijklmnopqrstuvwxyz-" for character in token):
        raise ValueError("invalid supervisor run token")
    claim = work_dir / f"_supervisor_{token}.claim"
    with (work_dir / "_launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _SYSTEMD_RUNTIME.is_dir():
            systemctl = shutil.which("systemctl")
            systemd_run = shutil.which("systemd-run")
            if not systemctl or not systemd_run:
                raise RuntimeError("systemd is running but systemctl/systemd-run is unavailable")
            privilege = []
            if os.geteuid() != 0:
                sudo = shutil.which("sudo")
                if not sudo:
                    raise RuntimeError("systemd supervisor requires noninteractive sudo")
                privilege = [sudo, "-n", "--"]
            unit = f"ale-solver-{token}.service"
            state = subprocess.run(
                [*privilege, systemctl, "show", unit,
                 "--property=LoadState,ActiveState,MainPID"],
                capture_output=True, text=True, timeout=15,
            )
            properties = dict(
                line.split("=", 1) for line in state.stdout.splitlines() if "=" in line
            )
            if state.returncode not in (0, 4) or not properties.get("LoadState"):
                raise RuntimeError(f"cannot inspect supervisor unit {unit}: {state.stderr.strip()}")
            if properties["LoadState"] != "not-found":
                if properties.get("ActiveState") in ("active", "activating"):
                    return 0
                if claim.exists():
                    return 0
                raise RuntimeError(f"supervisor unit {unit} is not running: {properties}")
            if claim.exists():
                return 0

            environment_path = work_dir / "_env"
            descriptor = os.open(
                environment_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600,
            )
            with os.fdopen(descriptor, "w") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(dict(os.environ), stream)
            bootstrap = (
                "import json,os,sys; "
                "environment=json.load(open(sys.argv[1])); os.unlink(sys.argv[1]); "
                "os.execve(sys.executable, [sys.executable, '-m', "
                "'ale_run.executors._sandbox_entry', sys.argv[2], '--systemd-supervisor'], environment)"
            )
            command = [
                *privilege, systemd_run, "--quiet", f"--unit={unit}",
                "--service-type=exec", "--property=OOMPolicy=continue",
                "--property=OOMScoreAdjust=-1000", "--property=Restart=no",
                "--property=KillMode=control-group",
                f"--property=User={os.geteuid()}", f"--property=Group={os.getegid()}",
                "--property=SupplementaryGroups=" + " ".join(map(str, os.getgroups())),
                f"--working-directory={os.getcwd()}",
                "--property=StandardInput=null",
                f"--property=StandardOutput=append:{work_dir / '_entry.log'}",
                "--property=StandardError=inherit",
                "--", sys.executable, "-c", bootstrap, str(environment_path), str(spec_path),
            ]
            try:
                started = subprocess.run(command, capture_output=True, text=True, timeout=30)
                if started.returncode != 0:
                    raise RuntimeError(
                        f"systemd supervisor launch failed ({started.returncode}): {started.stderr.strip()}",
                    )
            except BaseException as exc:
                try:
                    environment_path.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    exc.add_note(f"launcher environment cleanup failed: {cleanup_exc}")
                raise
        else:
            if claim.exists():
                return 0
            with (work_dir / "_entry.log").open("ab") as log:
                subprocess.Popen(
                    [sys.executable, "-m", "ale_run.executors._sandbox_entry", str(spec_path)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                )
    return 0


class _LinuxProcessTree:
    def __init__(self) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
        enabled = ctypes.c_int()
        if libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1:
            raise RuntimeError("child subreaper unavailable")

    def is_empty(self) -> bool:
        while True:
            try:
                child_pid, _status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return True
            if child_pid == 0:
                return False

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

    def is_empty(self) -> bool:
        accounting = _JobAccounting()
        if not self.kernel.QueryInformationJobObject(
            self.handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None,
        ):
            raise ctypes.WinError()
        return accounting.ActiveProcesses == 0

    def stop(self) -> None:
        try:
            if not self.kernel.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError()
            deadline = time.monotonic() + _PROCESS_STOP_TIMEOUT_S
            while time.monotonic() < deadline:
                if self.is_empty():
                    return
                time.sleep(0.02)
            raise RuntimeError("solver processes remain in Windows Job Object")
        finally:
            self.kernel.CloseHandle(self.handle)


def supervise(spec_path: Path, spec: dict, *, protected: bool = False) -> dict:
    """Wait for the worker and its descendants within one shared time budget."""
    work_dir = Path(spec["work_dir"])
    token = spec["run_token"]
    worker = None
    process_tree = None
    stopped = False
    drained = False
    started = time.monotonic()
    deadline = started + float(spec["timeout_s"])
    out = {"ok": False, "status": "failed", "error": "solver did not start"}
    try:
        if sys.platform == "linux":
            if protected and Path("/proc/self/oom_score_adj").read_text().strip() != "-1000":
                raise RuntimeError("systemd supervisor OOM protection unavailable")
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
            worker_finished = False
            while True:
                if not worker_finished and worker.poll() is not None:
                    try:
                        out = json.loads((work_dir / "_worker_result.json").read_text())
                    except FileNotFoundError:
                        out = {
                            "ok": False, "status": "failed", "pid": worker.pid,
                            "exit_code": worker.returncode,
                            "error": f"solver worker exited with code {worker.returncode} "
                                     "without _worker_result.json",
                        }
                        logger.error("%s", out["error"])
                    worker_finished = True
                    worker_done_at = time.monotonic()
                    if out.get("status") == "timeout" or out.get("phase") == "install":
                        break
                if stop_path.exists() and stop_path.read_text() == token:
                    prior_error = out.get("error") if worker_finished else None
                    out.update(ok=False, status="failed", error="solver stop requested" + (
                        f"; prior error: {prior_error}" if prior_error else ""
                    ))
                    break
                if time.monotonic() >= deadline:
                    prior_error = out.get("error") if worker_finished else None
                    out.update(ok=False, status="timeout", error="solver budget exceeded" + (
                        f"; prior error: {prior_error}" if prior_error else ""
                    ))
                    break
                if worker_finished and process_tree.is_empty():
                    drained = True
                    break
                if worker_finished and time.monotonic() - worker_done_at >= _DRAIN_GRACE_S:
                    # The agent left descendants behind (GUI apps it opened, a
                    # helper it did not close). Its result is already in hand;
                    # terminate the leftovers instead of idling until the
                    # episode deadline.
                    logger.warning("solver finished %.0fs ago but descendants remain; terminating them",
                                   time.monotonic() - worker_done_at)
                    try:
                        process_tree.stop()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("terminating leftover descendants failed: %s", exc)
                    process_tree = None
                    drained = True
                    break
                time.sleep(0.05)
    except BaseException as exc:
        logger.exception("sandbox supervisor failed")
        prior_error = out.get("error")
        out.update(ok=False, status="failed", error=str(exc) + (
            f"; prior error: {prior_error}" if prior_error else ""
        ), traceback=traceback.format_exc())
    finally:
        try:
            if worker is not None:
                if worker.stdin and not worker.stdin.closed:
                    worker.stdin.close()
                if worker.poll() is None:
                    worker.kill()
                worker.wait(timeout=_PROCESS_STOP_TIMEOUT_S)
            if process_tree is not None:
                if not drained:
                    process_tree.stop()
                elif isinstance(process_tree, _WindowsJob):
                    process_tree.kernel.CloseHandle(process_tree.handle)
            stopped = True
        except BaseException as exc:
            logger.exception("sandbox process cleanup failed")
            prior_error = out.get("error")
            out.update(ok=False, status="failed", error=f"process cleanup: {exc}" + (
                f"; prior error: {prior_error}" if prior_error else ""
            ))
            out["traceback"] = (
                (out.get("traceback") or "") + "\nProcess cleanup:\n" + traceback.format_exc()
            )
        report = {"run_token": token, "stopped": stopped, "error": out.get("error")}
        report_path = work_dir / "_stopped.json"
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report))
        temporary.replace(report_path)
    out["duration_s"] = time.monotonic() - started
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

    phase = "install"
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
            version = getattr(deployer, "actual_cli_version", None)
            source = getattr(deployer, "cli_source", None)
            identity = {
                "cli_version": version if isinstance(version, str) else None,
                "source": source if isinstance(source, str) and source in {"official", "fork", "custom"} else None,
            }
            (Path(executor.work_dir) / "_cli_identity.json").write_text(
                json.dumps(identity), encoding="utf-8",
            )
            timeout_s = float(spec.get("timeout_s") or 1800.0)
            logger.info(
                "sandbox_entry: %s.launch (timeout_s=%.0f)",
                dep_cls.__name__, timeout_s,
            )
            (Path(executor.work_dir) / "_agent_started").write_text(
                str(spec.get("run_token", "")), encoding="utf-8",
            )
            phase = "agent_run"
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
                "timeout" if phase == "agent_run" and isinstance(exc, asyncio.TimeoutError)
                else "failed"
            ),
            "phase": phase,
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
    if sys.argv[2:] == ["--launch"]:
        return launch(spec_path, spec)
    if sys.argv[2:] == ["--worker"]:
        # Deployer phases at INFO into the work dir (the supervisor only keeps
        # stderr, which stays empty unless something raises).
        try:
            handler = logging.FileHandler(work_dir / "_worker.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logging.getLogger().addHandler(handler)
            logging.getLogger().setLevel(logging.INFO)
        except Exception:  # noqa: BLE001
            pass
        if sys.platform == "linux":
            try:
                score = Path("/proc/self/oom_score_adj")
                try:
                    score.write_text("0")
                    lowered = True
                except PermissionError:
                    # Container hosts (Kubernetes burstable pods pin oom_score_adj > 0) forbid lowering
                    # it without CAP_SYS_RESOURCE; keep the inherited value rather than refusing to run.
                    lowered = False
                    logger.warning("worker OOM score reset not permitted; keeping oom_score_adj=%s", score.read_text().strip())
                if lowered and score.read_text().strip() != "0":
                    raise RuntimeError("worker OOM score reset did not take effect")
            except Exception as exc:
                logger.exception("worker OOM score reset failed")
                (work_dir / "_worker_result.json").write_text(json.dumps({
                    "ok": False, "status": "failed", "pid": os.getpid(),
                    "error": f"worker OOM score reset failed: {exc}",
                    "traceback": traceback.format_exc(),
                }))
                return 1
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
    if sys.platform == "linux":
        temporary = work_dir / "_pid.tmp"
        temporary.write_text(str(os.getpid()))
        temporary.replace(work_dir / "_pid")
    out = supervise(spec_path, spec, protected=sys.argv[2:] == ["--systemd-supervisor"])
    (work_dir / "_result.json").write_text(json.dumps(out, indent=2))
    # done.marker last — the host poller treats its presence as "result is
    # ready to read".
    (work_dir / "_done.marker").write_text("0\n" if out.get("ok") else "1\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
