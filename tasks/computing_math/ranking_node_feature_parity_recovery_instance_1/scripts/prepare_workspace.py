"""Prepare a writable /workspace tree from the staged input."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _sudo(*args: str) -> None:
    subprocess.run(["sudo", "-n", *args], check=True)


def _sudo_bash(script: str) -> None:
    subprocess.run(["sudo", "-n", "bash", "-lc", script], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-workspace", required=True)
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--runtime-env-dir", required=True)
    parser.add_argument("--workspace-root", default="/workspace")
    parser.add_argument("--protected-root", default="/protected")
    return parser.parse_args()


def _needs_sudo(path: Path) -> bool:
    text = str(path)
    return text == "/workspace" or text.startswith("/workspace/") or text == "/protected" or text.startswith("/protected/")


def _remove_tree(path: Path) -> None:
    if _needs_sudo(path):
        _sudo("rm", "-rf", str(path))
    else:
        shutil.rmtree(path, ignore_errors=True)


def _mkdir(path: Path) -> None:
    if _needs_sudo(path):
        _sudo("mkdir", "-p", str(path))
    else:
        path.mkdir(parents=True, exist_ok=True)


def _copy_tree(src: Path, dst: Path) -> None:
    if _needs_sudo(dst):
        _sudo("cp", "-a", f"{src}/.", str(dst))
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=True)


def _copy_file(src: Path, dst: Path) -> None:
    if _needs_sudo(dst):
        _sudo("cp", str(src), str(dst))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _extract_archive(src: Path, dst: Path) -> None:
    if _needs_sudo(dst):
        _sudo("tar", "-xzf", str(src), "-C", str(dst))
    else:
        subprocess.run(["tar", "-xzf", str(src), "-C", str(dst)], check=True)


def main() -> int:
    args = parse_args()
    input_workspace = Path(args.input_workspace)
    instruction_file = Path(args.instruction_file)
    runtime_env_dir = Path(args.runtime_env_dir)
    workspace_root = Path(args.workspace_root)
    protected_root = Path(args.protected_root)
    workspace_venv = workspace_root / ".venv"
    user = os.environ.get("USER", "user")
    group = subprocess.run(["id", "-gn"], check=True, capture_output=True, text=True).stdout.strip() or user

    if not input_workspace.exists():
        raise SystemExit(f"missing input workspace archive: {input_workspace}")
    if not instruction_file.exists():
        raise SystemExit(f"missing instruction file: {instruction_file}")
    if not runtime_env_dir.exists():
        raise SystemExit(f"missing runtime env dir: {runtime_env_dir}")

    _remove_tree(workspace_root)
    _mkdir(workspace_root)
    _extract_archive(input_workspace, workspace_root)
    _copy_file(instruction_file, workspace_root / "instruction.md")
    if _needs_sudo(workspace_root):
        _sudo("chown", "-R", f"{user}:{group}", str(workspace_root))

    subprocess.run(
        [
            "uv",
            "sync",
            "--project",
            str(runtime_env_dir),
            "--frozen",
        ],
        check=True,
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(workspace_venv)},
    )

    _remove_tree(protected_root)
    _mkdir(protected_root)
    if _needs_sudo(protected_root):
        _sudo_bash(f"printf 'sentinel\\n' > {protected_root / 'sentinel.txt'}")
        _sudo("chmod", "700", str(protected_root))
    else:
        (protected_root / "sentinel.txt").write_text("sentinel\n")
        os.chmod(protected_root, 0o700)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
