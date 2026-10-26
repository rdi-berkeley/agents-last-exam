"""Check host evaluation dependencies before starting any task sandboxes."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable
from contextlib import redirect_stdout
from pathlib import Path

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]


class HostDependencyError(RuntimeError):
    """The host cannot load or execute a required evaluator dependency."""


def check_host_dependencies(task_paths: Iterable[str]) -> None:
    """Run imports in a separate process so checks cannot pollute task state."""
    selected = sorted(set(task_paths))
    if not selected:
        return
    command = [sys.executable, "-m", __name__, "--json"]
    for path in selected:
        command.extend(["--task", path])
    try:
        result = subprocess.run(
            command,
            check=False,
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("ALE_PREFLIGHT_TIMEOUT_S", "240")),
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HostDependencyError(
            f"Host dependency preflight could not finish ({type(error).__name__}); "
            "no sandboxes were started. Run bash env/verify_host_env.sh."
        ) from error
    if result.returncode:
        raise HostDependencyError(
            "Host dependency preflight failed; no sandboxes were started.\n"
            + (result.stdout or result.stderr)[-16000:]
            + "\nRun bash env/install_host_deps.sh, uv sync --all-packages --locked, "
            "then bash env/verify_host_env.sh."
        )
    logger.info("Host dependency preflight passed for %d selected task(s)", len(selected))


def inspect_host(task_paths: Iterable[str]) -> dict:
    """Check versions, native tools, deferred imports, and task entry points."""
    manifest = json.loads((_ROOT / "env/host-deps.json").read_text())
    dependencies = tomllib.loads((_ROOT / "tasks/pyproject.toml").read_text())
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    issues: list[str] = []
    versions: dict[str, str] = {}
    tools: dict[str, str] = {}
    loaded: list[str] = []

    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name

        locked: dict[str, set[str]] = {}
        for package in lock["package"]:
            locked.setdefault(canonicalize_name(package["name"]), set()).add(package["version"])
        for declaration in dependencies["project"]["dependencies"]:
            requirement = Requirement(declaration)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            try:
                installed = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                issues.append(f"Missing Python dependency: {declaration}")
                continue
            versions[requirement.name] = installed
            if installed not in requirement.specifier:
                issues.append(
                    f"{requirement.name}: installed {installed}, requires {requirement.specifier}"
                )
            elif installed not in locked.get(canonicalize_name(requirement.name), set()):
                issues.append(f"{requirement.name}: installed {installed} differs from uv.lock")
    except ImportError:
        issues.append("Missing packaging; run uv sync --all-packages --locked")

    for modules in manifest["python_imports"].values():
        for module in modules:
            try:
                importlib.import_module(module)
            except Exception as error:  # noqa: BLE001 - aggregate third-party import failures
                issues.append(f"Cannot import {module}: {type(error).__name__}: {error}")

    for command, arguments in manifest["commands"].items():
        try:
            result = subprocess.run(
                [command, *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            tools[command] = (result.stdout or result.stderr).splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError) as error:
            issues.append(f"Cannot execute {command}: {type(error).__name__}: {error}")

    if "ffmpeg" in tools and "ffprobe" in tools:
        try:
            with tempfile.TemporaryDirectory(prefix="ale-host-media-") as directory:
                video = str(Path(directory) / "probe.mp4")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=red:s=32x32:r=1",
                        "-frames:v",
                        "1",
                        "-threads",
                        "1",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        video,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name",
                        "-of",
                        "json",
                        video,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if json.loads(result.stdout)["streams"][0]["codec_name"] != "h264":
                    raise ValueError("H.264 encode/probe did not round-trip")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        video,
                        "-frames:v",
                        "1",
                        "-threads",
                        "1",
                        str(Path(directory) / "frame.png"),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as error:
            issues.append(f"FFmpeg encode/probe/decode failed: {type(error).__name__}: {error}")

    from ale_run.tasks.loader import TaskLoader

    for task_path in sorted(set(task_paths)):
        try:
            directory = (_ROOT / "tasks" / task_path).resolve()
            if not directory.is_relative_to(_ROOT / "tasks"):
                raise ValueError("Task path must be inside tasks/")
            TaskLoader(str(directory))._load_module()
            loaded.append(task_path)
        except Exception as error:  # noqa: BLE001 - report every selected task's import failure
            issues.append(f"Cannot load {task_path}: {type(error).__name__}: {error}")

    return {"ok": not issues, "python": versions, "tools": tools, "tasks": loaded, "errors": issues}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", action="append", help="Task path; repeat to select several.")
    selection.add_argument(
        "--task-list", type=Path, help="Task list; defaults to selected_tasks/full.txt."
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    args = parser.parse_args(argv)
    paths = args.task
    if paths is None:
        listing = args.task_list or _ROOT / "selected_tasks/full.txt"
        paths = [line.split("#", 1)[0].strip() for line in listing.read_text().splitlines()]
        paths = [line for line in paths if line]
    if not paths:
        parser.error("The task selection is empty")
    os.environ.setdefault("MPLBACKEND", "Agg")
    with redirect_stdout(sys.stderr):
        report = inspect_host(paths)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for error in report["errors"]:
            print(f"FAIL: {error}")
        for tool, version in report["tools"].items():
            print(f"{tool}: {version}")
        print(
            f"{'PASS' if report['ok'] else 'FAIL'}: {len(report['tasks'])} task imports; "
            f"{len(report['python'])} Python dependencies checked against uv.lock"
        )
        if not report["ok"]:
            print("Run bash env/install_host_deps.sh and uv sync --all-packages --locked.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
