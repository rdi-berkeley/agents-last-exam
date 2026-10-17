from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import subprocess
import time


logger = logging.getLogger(__name__)


class SimulationError(RuntimeError):
    pass


def powershell_command(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand {encoded}"


def quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def validate_rollout_results(result):
    metrics = ("lin_vel_rmse_xy", "ang_vel_mae_z", "fall_rate")
    if not isinstance(result, dict):
        raise SimulationError("Isaac evaluator wrote incomplete rollout results")
    per_seed = result.get("per_seed")
    if not isinstance(per_seed, dict) or per_seed.get("seeds") != list(range(10)):
        raise SimulationError("Isaac evaluator wrote incomplete rollout results: expected seeds 0..9")
    for metric in metrics:
        values = per_seed.get(metric)
        if not isinstance(values, list) or len(values) != 10:
            raise SimulationError(f"Isaac evaluator wrote incomplete rollout results: {metric}")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in [result.get(metric), *values]):
            raise SimulationError(f"Isaac evaluator wrote invalid rollout metrics: {metric}")


async def run_simulation(session, *, executable, arguments, working_directory, results_path, timeout, poll_interval):
    status_path = results_path + ".status.json"
    stdout_path = results_path + ".stdout.log"
    stderr_path = results_path + ".stderr.log"
    script_path = results_path + ".supervisor.ps1"
    for path in (executable, arguments[1]):
        if not await session.file_exists(path):
            raise SimulationError(f"Isaac evaluator prerequisite is missing: {path}")
    stale = ",".join(quote(path) for path in (results_path, status_path, stdout_path, stderr_path))
    cleared = await session.run_command(powershell_command(
        f"$ErrorActionPreference = 'Stop'; foreach ($path in @({stale})) {{ "
        "if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force } }"
    ), check=False)
    if cleared.get("return_code", 1) != 0:
        raise SimulationError(f"Could not clear previous Isaac evaluation: {cleared}")
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        "$env:OMNI_KIT_ACCEPT_EULA = 'YES'\n"
        "$env:PYTHONDONTWRITEBYTECODE = '1'\n"
        "try {\n"
        f"$process = Start-Process -FilePath {quote(executable)} "
        f"-ArgumentList {quote(subprocess.list2cmdline(arguments))} "
        f"-WorkingDirectory {quote(working_directory)} "
        f"-RedirectStandardOutput {quote(stdout_path)} -RedirectStandardError {quote(stderr_path)} "
        "-WindowStyle Hidden -Wait -PassThru\n"
        "$status = @{ exit_code = $process.ExitCode; pid = $process.Id }\n"
        "} catch { $status = @{ exit_code = -1; error = $_.Exception.ToString() } }\n"
        f"$status | ConvertTo-Json -Compress | Set-Content -LiteralPath {quote(status_path + '.tmp')} -Encoding UTF8\n"
        f"Move-Item -LiteralPath {quote(status_path + '.tmp')} -Destination {quote(status_path)} -Force\n"
    )
    await session.write_file(script_path, script)
    launch_args = subprocess.list2cmdline(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", script_path])
    launch = await session.run_command(powershell_command(
        "$ErrorActionPreference = 'Stop'; "
        f"$process = Start-Process powershell.exe -ArgumentList {quote(launch_args)} -WindowStyle Hidden -PassThru; "
        "$process.Id"
    ), check=False)
    try:
        if launch.get("return_code", 1) != 0:
            raise ValueError("nonzero launcher exit")
        supervisor_pid = int(str(launch.get("stdout", "")).strip())
    except (ValueError, TypeError) as exc:
        raise SimulationError(f"Isaac evaluator launch failed: {launch}") from exc
    logger.info("Isaac evaluator started: supervisor=%s stdout=%s stderr=%s", supervisor_pid, stdout_path, stderr_path)
    deadline = time.monotonic() + timeout
    completed = False
    try:
        while time.monotonic() < deadline:
            if await session.file_exists(status_path):
                status = json.loads((await session.read_bytes(status_path)).decode("utf-8-sig"))
                completed = True
                if status.get("exit_code") != 0:
                    raise SimulationError(f"Isaac evaluator exited unsuccessfully: {status}")
                if not await session.file_exists(results_path):
                    raise SimulationError("Isaac evaluator exited 0 without a results file")
                result = json.loads((await session.read_bytes(results_path)).decode("utf-8-sig"))
                validate_rollout_results(result)
                logger.info("Isaac evaluator completed: %s", status)
                return True
            alive = await session.run_command(powershell_command(
                f"if (Get-Process -Id {supervisor_pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 3 }}"
            ), check=False)
            if alive.get("return_code", 1) != 0:
                if await session.file_exists(status_path):
                    continue
                raise SimulationError(f"Isaac supervisor {supervisor_pid} exited without a completion record")
            await asyncio.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
        raise SimulationError(f"Isaac evaluator exceeded {timeout}s; incomplete evaluation, no score")
    except BaseException:
        if not completed:
            await session.run_command(f"taskkill /PID {supervisor_pid} /T /F", check=False)
        diagnostics = await session.run_command(powershell_command(
            f"Get-Content -LiteralPath {quote(stdout_path)},{quote(stderr_path)} -Tail 30 -ErrorAction SilentlyContinue"
        ), check=False)
        logger.error("Isaac evaluator diagnostic tail: %s", diagnostics.get("stdout", ""))
        raise
