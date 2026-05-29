#!/usr/bin/env python
"""Score OS log permission guard outputs.

The task uses CSV/JSON metadata as the ownership source of truth. Kernel-level
ownership is intentionally not scored because the staged snapshot is a sandbox
owned by the VM user.
"""

from __future__ import annotations

import argparse
import csv
import json
import stat
import tarfile
from pathlib import Path
from typing import Any


def _read_csv_map(path: Path, value_fields: tuple[str, ...]) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        data: dict[str, dict[str, str]] = {}
        for row in rows:
            item = {field: row[field].strip() for field in value_fields}
            data[row["path"].strip()] = item
        return data


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.lstat().st_mode):03o}"


def _load_final_state(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("files"), dict):
        return {
            str(item_path): {
                "type": str(meta.get("type", "regular")),
                "owner": str(meta.get("owner", "")),
                "group": str(meta.get("group", "")),
                "mode": str(meta.get("mode", "")),
            }
            for item_path, meta in data["files"].items()
        }
    if isinstance(data, dict) and isinstance(data.get("files"), list):
        result: dict[str, dict[str, str]] = {}
        for item in data["files"]:
            result[str(item["path"])] = {
                "type": str(item.get("type", "regular")),
                "owner": str(item.get("owner", "")),
                "group": str(item.get("group", "")),
                "mode": str(item.get("mode", "")),
            }
        return result
    raise ValueError("final_state.json must contain files as a dict or list")


def _expected_modes(
    permissions: dict[str, dict[str, str]],
    ownership: dict[str, dict[str, str]],
    active_writers: set[str],
    protected_user: str,
) -> dict[str, str]:
    expected: dict[str, str] = {}
    for item_path, mode_meta in permissions.items():
        owner = ownership[item_path]["owner"]
        if (
            item_path.endswith(".log")
            and owner != protected_user
            and item_path not in active_writers
        ):
            expected[item_path] = "444"
        else:
            expected[item_path] = mode_meta["mode"]
    return expected


def _snapshot_contents(snapshot: Path, item_paths: set[str]) -> dict[str, bytes]:
    contents: dict[str, bytes] = {}
    with tarfile.open(snapshot, "r:gz") as archive:
        for item_path in item_paths:
            member_name = item_path.lstrip("/")
            try:
                extracted = archive.extractfile(member_name)
            except KeyError:
                continue
            if extracted is not None:
                contents[item_path] = extracted.read()
    return contents


def score(input_dir: Path, output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    ownership = _read_csv_map(input_dir / "ownership.csv", ("owner", "group"))
    permissions = _read_csv_map(input_dir / "permissions.csv", ("mode",))
    active_data = json.loads((input_dir / "active_writers.json").read_text(encoding="utf-8"))
    active_writers = set(active_data.get("active_writers", []))
    reference = json.loads((reference_dir / "reference_state.json").read_text(encoding="utf-8"))
    protected_user = reference.get("protected_system_user", "syslog")

    expected = _expected_modes(permissions, ownership, active_writers, protected_user)
    reference_modes = reference["final_modes"]
    expected_paths = set(permissions)
    original_contents = _snapshot_contents(input_dir / "fs_snapshot.tar.gz", expected_paths)

    errors: list[str] = []
    if expected != reference_modes:
        errors.append("internal reference_state.json does not match input-derived rules")

    final_path = output_dir / "final_state.json"
    if not final_path.exists():
        return {"score": 0.0, "pass": False, "errors": ["missing output/final_state.json"]}

    try:
        final_state = _load_final_state(final_path)
    except Exception as exc:
        return {"score": 0.0, "pass": False, "errors": [f"invalid final_state.json: {exc}"]}

    actual_paths = set(final_state)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    for item_path in missing:
        errors.append(f"final_state.json missing {item_path}")
    for item_path in extra:
        errors.append(f"final_state.json includes unexpected path {item_path}")

    workspace = output_dir / "sandbox_fs" / "var" / "logs"
    if not workspace.exists():
        errors.append("missing output/sandbox_fs/var/logs workspace")

    for item_path in sorted(expected_paths):
        meta = final_state.get(item_path)
        if meta is None:
            continue
        expected_owner = ownership[item_path]["owner"]
        expected_group = ownership[item_path]["group"]
        expected_mode = reference_modes[item_path]

        if meta.get("type") not in {"regular", "file"}:
            errors.append(f"{item_path}: final_state type must be regular/file")
        if meta.get("owner") != expected_owner:
            errors.append(f"{item_path}: owner changed in final_state")
        if meta.get("group") != expected_group:
            errors.append(f"{item_path}: group changed in final_state")
        if meta.get("mode") != expected_mode:
            errors.append(
                f"{item_path}: final_state mode {meta.get('mode')} != expected {expected_mode}"
            )

        fs_path = workspace / Path(item_path).name
        if not fs_path.exists():
            errors.append(f"{item_path}: missing from sandbox workspace")
            continue
        actual_mode = _mode(fs_path)
        if actual_mode != expected_mode:
            errors.append(f"{item_path}: filesystem mode {actual_mode} != expected {expected_mode}")
        if item_path not in original_contents:
            errors.append(f"{item_path}: missing from input fs_snapshot.tar.gz")
            continue
        if fs_path.read_bytes() != original_contents[item_path]:
            errors.append(f"{item_path}: file contents changed")

    return {
        "score": 1.0 if not errors else 0.0,
        "pass": not errors,
        "errors": errors,
        "expected_modes": reference_modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.input, args.output, args.reference), indent=2))


if __name__ == "__main__":
    main()
