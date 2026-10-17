"""Process and artifact safety for sandbox-resident CLI adapters."""

from __future__ import annotations

import asyncio
import json
import locale
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def secret_values(config: Any, env: dict[str, str] | None = None) -> tuple[str, ...]:
    values = {getattr(config, "api_key", None)}
    for name, value in {**os.environ, **(env or {}), **getattr(config, "extra_envs", {})}.items():
        if any(part in name.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            values.add(value)
    return tuple(sorted((value for value in values if value), key=len, reverse=True))


def redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            name: "[REDACTED]"
            if name.lower()
            in {
                "apikey",
                "api_key",
                "authorization",
                "access_token",
                "password",
            }
            else redact(item, secrets)
            for name, item in value.items()
        }
    return value


def write_private_json(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, indent=2)


async def _stop_process(proc: subprocess.Popen, grace_s: float) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.02)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    while proc.poll() is None:
        await asyncio.sleep(0.02)


async def wait_process(
    proc: subprocess.Popen,
    *,
    timeout: float | None = None,
    grace_s: float = 2.0,
    cleanup_on_success: bool = False,
) -> int:
    """Reap the CLI; preserve successful background work unless explicitly owned."""
    deadline = None if timeout is None else time.monotonic() + timeout
    succeeded = False
    try:
        while proc.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("CLI command timed out")
            await asyncio.sleep(0.05)
        succeeded = proc.returncode == 0
        return proc.returncode
    finally:
        if not succeeded or cleanup_on_success:
            cleanup = asyncio.create_task(_stop_process(proc, grace_s))
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError


async def run_command(
    argv: list[str],
    *,
    capture_output: bool = True,
    text: bool = True,
    timeout: float = 300,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """Run an installer or probe, owning its descendants and captured output."""
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        await wait_process(proc, timeout=timeout, cleanup_on_success=True)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout, stderr = stdout_file.read(), stderr_file.read()
    if text:
        stdout = stdout.decode(locale.getencoding())
        stderr = stderr.decode(locale.getencoding())
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


async def _node_version(node: str) -> tuple[int, ...]:
    probe = await run_command([node, "--version"], timeout=15)
    try:
        installed = tuple(int(part) for part in probe.stdout.strip().lstrip("v").split("."))
    except ValueError:
        installed = ()
    return installed if probe.returncode == 0 else ()


async def ensure_cli_node(node: str, npm: str, minimum: tuple[int, int, int]) -> str:
    """Upgrade only these CLI adapters to a pinned, user-writable Node runtime."""
    if await _node_version(node) >= minimum:
        return node
    pinned = (24, 14, 0)
    version = ".".join(str(part) for part in pinned)
    prefix = Path.home() / ".ale-node" / version
    runtime = prefix / "node_modules" / ".bin" / "node"
    cached = runtime.is_file() and await _node_version(str(runtime)) == pinned
    if not cached:
        prefix.mkdir(parents=True, exist_ok=True)
        result = await run_command(
            [
                npm,
                "install",
                "--prefix",
                str(prefix),
                "--no-save",
                "--package-lock=false",
                "--no-audit",
                "--no-fund",
                f"node@{version}",
            ],
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CLI could not install its private Node {version} runtime")
        if not runtime.is_file() or await _node_version(str(runtime)) != pinned:
            raise RuntimeError(f"CLI private Node runtime does not match {version}")
    os.environ["PATH"] = str(runtime.parent) + os.pathsep + os.environ.get("PATH", "")
    return str(runtime)
