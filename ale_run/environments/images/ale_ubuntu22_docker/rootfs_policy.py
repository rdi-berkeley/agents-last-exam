"""Shared pre-import export policy and bounded streaming archive validation."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import posixpath
import re
import stat
import sys
import tarfile
from pathlib import Path

REQUIRED = (
    "/usr/local/bin/node",
    "/opt/cua-server/.venv/bin/python",
    "/opt/ale-run/.venv/bin/python",
    "/home/user/cua_mcp_server",
    "/usr/local/bin/codex",
    "/usr/local/bin/openclaw",
    "/home/user/.local/bin/claude",
    "/home/user/.local/bin/gemini",
    "/home/user/.local/bin/hermes",
)
VOLATILE = ("/proc", "/sys", "/dev", "/run", "/tmp", "/var/tmp")
EXCLUDED = VOLATILE + (
    "/boot",
    "/snap",
    "/var/snap",
    "/var/lib/snapd",
    "/var/lib/flatpak",
    "/var/log",
    "/var/cache",
    "/swapfile",
    "/cdrom",
    "/lost+found",
    "/srv",
    "/workspace",
    "/reference",
    "/protected",
    "/output",
    "/output_test_pos",
    "/output_test_neg",
    "/.cache",
    "/media/user/data",
    "/media/floppy",
    "/media/floppy0",
    "/mnt",
    "/opt/ale-docker-images",
    "/var/lib/docker",
    "/var/lib/containerd",
    "/var/lib/dind",
    "/var/lib/cloud",
    "/etc/agenthle",
    "/etc/boto.cfg",
    "/etc/ssh/ssh_host_*",
    "/home/user/ale-test",
    "/home/user/terminus2",
    "/home/user/codex-build",
    "/home/user/reference.frc",
    "/home/user/.config/Code",
    "/home/user/.config/Sabaki",
    "/home/user/.config/google-chrome",
    "/home/user/.mozilla",
    "/root/.config/google-chrome",
    "/root/.mozilla",
)
COMPONENTS = (
    ".ssh",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    ".netrc",
    ".boto",
    ".env",
    ".env.*",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    ".bash_history",
    ".zsh_history",
    ".python_history",
    ".node_repl_history",
    ".ale",
    ".ale-src",
    ".agenthle_hidden_eval_assets",
    "agenthle-artifacts",
    "gcloud",
    "gcloud-agenthle-artifacts",
    ".audit*",
    "gcs-reader.json",
    "gcp_key.json",
    "reference.7z",
    "task-data*",
    "ale_export_*",
    "ale-export-manifest.*",
)
COMPONENT_PATTERN = re.compile("|".join(fnmatch.translate(pattern) for pattern in COMPONENTS))
HOME_STATE = (
    ".codex/auth.json",
    ".codex/config.toml",
    ".codex/state*.sqlite*",
    ".codex/sessions",
    ".codex/archived_sessions",
    ".codex/history.jsonl",
    ".codex/log",
    ".claude.json",
    ".claude/projects",
    ".claude/.credentials.json",
    ".claude/settings*.json",
    ".claude/statsig",
    ".claude/todos",
    ".claude/history.jsonl",
    ".claude/debug",
    ".kimi",
    ".hermes/auth.json",
    ".hermes/sessions",
    ".hermes/state.db*",
    ".hermes/config.yaml",
    ".hermes/MEMORY.md",
    ".hermes/logs",
    ".hermes/memories",
    ".openclaw/agents",
    ".openclaw/credentials",
    ".openclaw/workspace",
    ".openclaw/openclaw.json",
    ".openclaw/openclaw.json.*",
    ".openhands/sessions",
    ".openhands/conversations",
    ".grok/user-settings.json",
    ".grok/sessions",
    ".grok/history*",
    ".gemini/settings.json",
    ".gemini/oauth_creds.json",
    ".gemini/google_accounts.json",
    ".gemini/tmp",
    ".local/share/docker",
    ".local/share/containers",
    ".npm",
    ".cache",
)
CACHE_PREFIXES = (
    "/home/user/.cache/uv",
    "/root/.cache/uv",
    "/home/user/.cache/huggingface",
    "/root/.cache/huggingface",
)


def beneath(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def canonical(name: str) -> str:
    if "\x00" in name or "\n" in name or ".." in name.split("/"):
        raise ValueError("unsafe archive path")
    while name.startswith("./"):
        name = name[2:]
    return "/" + name.lstrip("/") if name not in (".", "", "/") else "/"


def validate_contract(contract: dict) -> None:
    required = contract.get("required_paths")
    probes = contract.get("probes")
    if not isinstance(required, list) or not required:
        raise ValueError("runtime manifest needs required_paths")
    if not isinstance(probes, list) or not probes:
        raise ValueError("runtime manifest needs nonempty probes (Python and R)")
    source_probes = contract.get("source_probes", [])
    if not isinstance(source_probes, list):
        raise ValueError("source_probes must be an argv list")
    for command in source_probes + probes:
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(arg, str) and arg for arg in command)
            or not command[0].startswith("/")
        ):
            raise ValueError("runtime probes must be absolute argv arrays")
    if not any(re.fullmatch(r"python[0-9.]*", Path(command[0]).name) for command in probes):
        raise ValueError("runtime manifest must probe Python")
    if not any(Path(command[0]).name in {"R", "Rscript"} for command in probes):
        raise ValueError("runtime manifest must probe R")
    optional_paths = []
    for field in ("preserve_cache_paths", "preserve_runtime_paths", "exclude_paths"):
        paths = contract.get(field, [])
        if not isinstance(paths, list):
            raise ValueError(f"{field} must be a path list")
        optional_paths.extend(paths)
    for path in required + optional_paths:
        if not isinstance(path, str) or not path.startswith("/") or canonical(path) != path:
            raise ValueError("runtime paths must be canonical absolute paths")
    for path in contract.get("preserve_cache_paths", []):
        if not any(beneath(path, prefix) for prefix in CACHE_PREFIXES):
            raise ValueError("only audited uv/Hugging Face runtime cache paths may be preserved")
    for path in contract.get("preserve_runtime_paths", []):
        if not re.fullmatch(r"/media/user/data/(opt|toolchains)/[A-Za-z0-9][A-Za-z0-9_.+-]*", path):
            raise ValueError("data runtime allowlist needs an individual opt/toolchains root")
        if path not in required:
            raise ValueError("preserved data runtime must also be a required path")
    baseline = contract.get("baseline_missing_links", {})
    if not isinstance(baseline, dict):
        raise ValueError("baseline_missing_links must map paths to resolved targets")
    for path, target in baseline.items():
        for value in (path, target):
            if not isinstance(value, str) or not value.startswith("/") or canonical(value) != value:
                raise ValueError("baseline links must use canonical absolute paths")
            if forbidden(value, contract):
                raise ValueError("baseline links cannot permit excluded content")
        if path in required or path in REQUIRED:
            raise ValueError("explicitly required runtime cannot be a missing baseline link")


def forbidden(path: str, contract: dict) -> bool:
    path = canonical(path)
    parts = path.strip("/").split("/")
    parents = [path, *[str(item) for item in Path(path).parents]]
    if path.startswith("/home/") and len(parts) > 1 and parts[1] != "user":
        return True
    if any(COMPONENT_PATTERN.fullmatch(part) for part in parts):
        return True
    if any(beneath(path, prefix) for prefix in contract.get("exclude_paths", [])):
        return True
    for pattern in EXCLUDED:
        matches = (
            beneath(path, pattern)
            if "*" not in pattern
            else any(fnmatch.fnmatchcase(parent, pattern) for parent in parents)
        )
        if matches:
            if pattern == "/media/user/data" and any(
                beneath(path, keep) or beneath(keep, path)
                for keep in contract.get("preserve_runtime_paths", [])
            ):
                continue
            return True
    for home in ("/home/user", "/root"):
        if not beneath(path, home):
            continue
        for state in HOME_STATE:
            prefix = home + "/" + state
            matches = (
                beneath(path, prefix)
                if "*" not in state
                else any(fnmatch.fnmatchcase(parent, prefix) for parent in parents)
            )
            if matches:
                if state == ".cache" and any(
                    beneath(path, keep) or beneath(keep, path)
                    for keep in contract.get("preserve_cache_paths", [])
                ):
                    continue
                return True
    return False


def resolve_guest(root: Path, path: str) -> str:
    pending = path.strip("/").split("/")
    resolved: list[str] = []
    follows = 0
    while pending:
        part = pending.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            if not resolved:
                raise ValueError("runtime link escapes root")
            resolved.pop()
            continue
        candidate = root.joinpath(*resolved, part)
        if candidate.is_symlink():
            follows += 1
            if follows > 40:
                raise ValueError("runtime symlink cycle")
            target = os.readlink(candidate)
            if target.startswith("/"):
                resolved = []
            pending = target.split("/") + pending
        else:
            resolved.append(part)
    return "/" + "/".join(resolved)


def export_manifest(root: Path, contract: dict, output) -> int:
    validate_contract(contract)
    required = set(REQUIRED) | set(contract["required_paths"])
    for path in required:
        target = resolve_guest(root, path)
        if forbidden(path, contract) or forbidden(target, contract):
            raise ValueError(f"required runtime excluded: {path}")
        if not root.joinpath(target.lstrip("/")).exists():
            raise ValueError(f"missing runtime: {path}")
    count = 0
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=sys.exit):
        for name in list(dirs) + files:
            actual = Path(directory) / name
            path = "/" + actual.relative_to(root).as_posix()
            if forbidden(path, contract):
                if name in dirs:
                    dirs.remove(name)
                continue
            metadata = actual.lstat()
            if not (
                stat.S_ISREG(metadata.st_mode)
                or stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
            ):
                continue
            if actual.is_symlink():
                target = resolve_guest(root, path)
                is_runtime = any(beneath(path, runtime) for runtime in required)
                if (
                    forbidden(target, contract)
                    and not any(beneath(target, prefix) for prefix in VOLATILE)
                ) or (
                    is_runtime
                    and (
                        forbidden(target, contract)
                        or (
                            not root.joinpath(target.lstrip("/")).exists()
                            and contract.get("baseline_missing_links", {}).get(path) != target
                        )
                    )
                ):
                    raise ValueError(f"excluded or missing symlink target: {path} -> {target}")
            if metadata.st_uid > 65535 or metadata.st_gid > 65535:
                os.chown(actual, 0, 0, follow_symlinks=False)
            output.write(("." + path).encode() + b"\0")
            count += 1
    return count


def inspect_archive(stream, contract: dict) -> dict:
    validate_contract(contract)
    members: dict[str, tuple[str, str]] = {}
    total = 0
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            path = canonical(member.name)
            if forbidden(path, contract) or path in members:
                raise ValueError(f"forbidden or duplicate archive member: {path}")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError(f"unsupported archive member: {path}")
            if member.uid > 65535 or member.gid > 65535 or member.uid < 0 or member.gid < 0:
                raise ValueError("unmapped archive ownership")
            target = ""
            if member.issym() or member.islnk():
                target = posixpath.normpath(
                    posixpath.join(
                        "/" if member.islnk() else posixpath.dirname(path), member.linkname
                    )
                )
                if forbidden(target, contract) and not any(
                    beneath(target, prefix) for prefix in VOLATILE
                ):
                    raise ValueError(f"archive link targets excluded content: {path}")
            members[path] = ("link" if target else "file", target)
            total += member.size
    for path in set(REQUIRED) | set(contract["required_paths"]):
        pending = [path]
        pending.extend(item for item in members if beneath(item, path))
        for candidate in pending:
            original = candidate
            for _ in range(40):
                prefixes = [candidate, *[str(item) for item in Path(candidate).parents]]
                link = next(
                    (
                        prefix
                        for prefix in reversed(prefixes)
                        if members.get(prefix, (None,))[0] == "link"
                    ),
                    None,
                )
                if link is None:
                    break
                candidate = posixpath.normpath(members[link][1] + candidate[len(link) :])
            else:
                raise ValueError("archive runtime link cycle")
            baseline = (
                members.get(original, (None,))[0] == "link"
                and contract.get("baseline_missing_links", {}).get(original) == candidate
            )
            if (candidate not in members and not baseline) or forbidden(candidate, contract):
                raise ValueError(f"archive missing runtime or link target: {path}")
    if not members:
        raise ValueError("empty export")
    return {"members": len(members), "logical_bytes": total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/"))
    args = parser.parse_args()
    export_manifest(args.root, json.loads(args.contract.read_text()), sys.stdout.buffer)


if __name__ == "__main__":
    main()
