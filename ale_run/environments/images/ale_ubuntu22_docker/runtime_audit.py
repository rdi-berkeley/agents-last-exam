"""Bounded metadata-only export-policy inspection; never changes the guest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path

import rootfs_policy as policy


def inspect_tree(
    root: Path,
    contract: dict,
    *,
    max_entries: int = 2_000_000,
    max_seconds: float = 180,
    max_findings: int = 100,
) -> dict:
    policy.validate_contract(contract)
    if min(max_entries, max_seconds, max_findings) <= 0:
        raise ValueError("audit bounds must be positive")
    started = time.monotonic()
    result = {
        "read_only": True,
        "complete": False,
        "export_ready": False,
        "visited": 0,
        "retained": 0,
        "excluded_roots_or_files": 0,
        "logical_bytes_upper_bound": 0,
        "allocated_bytes_upper_bound": 0,
        "oversized_owners": 0,
        "symlinks": 0,
        "findings_count": 0,
        "findings": [],
        "required": [],
        "groups": {},
    }
    required = set(policy.REQUIRED) | set(contract["required_paths"])
    digest = hashlib.sha256()

    def finding(path, reason):
        result["findings_count"] += 1
        if len(result["findings"]) < max_findings:
            result["findings"].append({"path": path, "reason": reason})

    for path in sorted(required):
        try:
            target = policy.resolve_guest(root, path)
            exists = root.joinpath(target.lstrip("/")).exists()
            excluded = policy.forbidden(path, contract) or policy.forbidden(target, contract)
            result["required"].append(
                {"path": path, "target": target, "exists": exists, "excluded": excluded}
            )
            if not exists or excluded:
                finding(path, "required runtime missing or excluded")
        except (OSError, ValueError) as error:
            finding(path, str(error))

    def walk_error(error):
        finding(str(error.filename), str(error))

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        dirs.sort()
        for name in list(dirs) + sorted(files):
            if result["visited"] >= max_entries or time.monotonic() - started >= max_seconds:
                result["limit_reached"] = True
                result["elapsed_seconds"] = time.monotonic() - started
                return result
            result["visited"] += 1
            actual = Path(directory) / name
            path = "/" + actual.relative_to(root).as_posix()
            try:
                if policy.forbidden(path, contract):
                    result["excluded_roots_or_files"] += 1
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
                result["retained"] += 1
                digest.update(("." + path).encode() + b"\0")
                if metadata.st_uid > 65535 or metadata.st_gid > 65535:
                    result["oversized_owners"] += 1
                if stat.S_ISREG(metadata.st_mode):
                    logical = metadata.st_size
                    allocated = metadata.st_blocks * 512
                    result["logical_bytes_upper_bound"] += logical
                    result["allocated_bytes_upper_bound"] += allocated
                    parts = path.split("/")
                    group = "/".join(parts[:4] if path.startswith("/home/") else parts[:3])
                    result["groups"][group] = result["groups"].get(group, 0) + logical
                if stat.S_ISLNK(metadata.st_mode):
                    result["symlinks"] += 1
                    target = policy.resolve_guest(root, path)
                    excluded = policy.forbidden(target, contract)
                    runtime = any(policy.beneath(path, prefix) for prefix in required)
                    if (
                        excluded
                        and not any(policy.beneath(target, prefix) for prefix in policy.VOLATILE)
                    ) or (
                        runtime
                        and (
                            excluded
                            or (
                                not root.joinpath(target.lstrip("/")).exists()
                                and contract.get("baseline_missing_links", {}).get(path) != target
                            )
                        )
                    ):
                        finding(path, "excluded or missing symlink target: " + target)
            except (OSError, ValueError) as error:
                finding(path, str(error))
                if name in dirs:
                    dirs.remove(name)
    result["complete"] = True
    result["export_ready"] = result["findings_count"] == 0
    result["manifest_sha256"] = digest.hexdigest()
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/"))
    parser.add_argument("--max-entries", type=int, default=2_000_000)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--max-findings", type=int, default=100)
    args = parser.parse_args()
    result = inspect_tree(
        args.root,
        json.loads(args.contract.read_text()),
        max_entries=args.max_entries,
        max_seconds=args.max_seconds,
        max_findings=args.max_findings,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["export_ready"] else 1)


if __name__ == "__main__":
    main()
