#!/usr/bin/env python
"""Score os_log_hardening_rotation_audit_v1 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import stat
import tarfile
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "final_state.json",
    "audit_report.csv",
    "rotation_exceptions.json",
    "rollback_plan.sh",
)


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _read_snapshot_metadata(snapshot: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with tarfile.open(snapshot, "r:gz") as archive:
        for member in archive.getmembers():
            item_path = "/" + member.name.lstrip("/")
            entry: dict[str, Any] = {
                "is_symlink": member.issym(),
                "mode": f"{stat.S_IMODE(member.mode):03o}",
            }
            if member.issym():
                entry["linkname"] = member.linkname
            elif member.isfile():
                extracted = archive.extractfile(member)
                entry["content"] = extracted.read() if extracted is not None else b""
            metadata[item_path] = entry
    return metadata


def _lstat_mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.lstat().st_mode):03o}"


def _extract_chmod_pairs(script: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for line in script.splitlines():
        line = line.strip()
        m = re.match(r'chmod\s+(\d{3,4})\s+["\']?(.*?)["\']?\s*$', line)
        if not m:
            continue
        mode = m.group(1)
        path = m.group(2)
        path = re.sub(r'\$\{[^}]+\}', '', path)
        path = re.sub(r'\$\w+', '', path)
        if not path.startswith('/'):
            path = '/' + path
        pairs.add((path, mode))
    return pairs


def _build_csv_index(
    rows: list[list[str]], header: list[str],
) -> dict[str, dict[str, str]]:
    path_idx = header.index("path")
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        if len(row) <= path_idx:
            continue
        entry = {}
        for i, col in enumerate(header):
            if i < len(row):
                entry[col] = row[i]
        index[entry["path"]] = entry
    return index


def _read_original_modes(input_dir: Path) -> dict[str, str]:
    modes: dict[str, str] = {}
    perms = input_dir / "permissions.csv"
    if not perms.exists():
        return modes
    with perms.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0] != "path":
                modes[row[0]] = row[1]
    return modes


def _normalize_action(value: str) -> str:
    v = value.strip().lower()
    if v in ("modified", "hardened_444", "hardened", "changed"):
        return "modified"
    if v in ("skipped", "unchanged", "no_change"):
        return "skipped"
    return v


def score(input_dir: Path, output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    errors: list[str] = []

    for filename in REQUIRED_FILES:
        if not (output_dir / filename).exists():
            errors.append(f"missing output/{filename}")
    if errors:
        return {"score": 0.0, "pass": False, "errors": errors}

    submitted_state = json.loads((output_dir / "final_state.json").read_text(encoding="utf-8"))
    reference_state = json.loads((reference_dir / "reference_state.json").read_text(encoding="utf-8"))
    if submitted_state != reference_state:
        errors.append("final_state.json does not match reference_state.json")

    submitted_rotation = json.loads(
        (output_dir / "rotation_exceptions.json").read_text(encoding="utf-8")
    )
    reference_rotation = json.loads(
        (reference_dir / "reference_rotation_exceptions.json").read_text(encoding="utf-8")
    )
    ref_exempt_set = set(reference_rotation.get("exempted_paths", []))
    sub_exempt_set = set(submitted_rotation.get("exempted_paths", []))
    if ref_exempt_set != sub_exempt_set:
        missing = ref_exempt_set - sub_exempt_set
        extra = sub_exempt_set - ref_exempt_set
        detail = []
        if missing:
            detail.append(f"missing: {sorted(missing)}")
        if extra:
            detail.append(f"extra: {sorted(extra)}")
        errors.append(f"rotation_exceptions.json exempted_paths mismatch: {'; '.join(detail)}")
    ref_reason_keys = set(reference_rotation.get("reasons", {}).keys())
    sub_reason_keys = set(submitted_rotation.get("reasons", {}).keys())
    if ref_reason_keys != sub_reason_keys:
        errors.append("rotation_exceptions.json reasons keys mismatch")

    original_modes = _read_original_modes(input_dir)
    final_modes = reference_state.get("final_modes", {})

    ref_rollback_pairs = _extract_chmod_pairs(
        (reference_dir / "reference_rollback_plan.sh").read_text(encoding="utf-8")
    )
    sub_rollback_pairs = _extract_chmod_pairs(
        (output_dir / "rollback_plan.sh").read_text(encoding="utf-8")
    )
    required_rollback = {
        (p, m) for p, m in ref_rollback_pairs
        if original_modes.get(p) != final_modes.get(p)
    }
    if not required_rollback.issubset(sub_rollback_pairs):
        missing_rb = required_rollback - sub_rollback_pairs
        errors.append(
            f"rollback_plan.sh missing required chmod commands: {sorted(missing_rb)}"
        )

    ref_audit_rows = _read_csv_rows(reference_dir / "reference_audit_report.csv")
    sub_audit_rows = _read_csv_rows(output_dir / "audit_report.csv")
    if not sub_audit_rows:
        errors.append("audit_report.csv is empty")
    else:
        sub_header = sub_audit_rows[0]
        ref_header = ref_audit_rows[0]
        if sub_header != ref_header:
            errors.append(f"audit_report.csv header mismatch: {sub_header} != {ref_header}")
        else:
            ref_index = _build_csv_index(ref_audit_rows[1:], ref_header)
            sub_index = _build_csv_index(sub_audit_rows[1:], sub_header)
            for path, ref_entry in sorted(ref_index.items()):
                if path not in sub_index:
                    errors.append(f"audit_report.csv missing row for {path}")
                    continue
                sub_entry = sub_index[path]
                if sub_entry.get("final_mode") != ref_entry.get("final_mode"):
                    errors.append(
                        f"audit_report.csv {path}: final_mode "
                        f"{sub_entry.get('final_mode')} != {ref_entry.get('final_mode')}"
                    )
                noop = original_modes.get(path) == final_modes.get(path)
                if not noop and _normalize_action(
                    sub_entry.get("action_taken", "")
                ) != _normalize_action(ref_entry.get("action_taken", "")):
                    errors.append(
                        f"audit_report.csv {path}: action_taken "
                        f"{sub_entry.get('action_taken')} != {ref_entry.get('action_taken')}"
                    )

    sandbox_root = output_dir / "sandbox_fs"
    if not sandbox_root.exists():
        errors.append("missing output/sandbox_fs")
        return {"score": 0.0, "pass": False, "errors": errors}

    snapshot_meta = _read_snapshot_metadata(input_dir / "fs_snapshot.tar.gz")
    final_modes = reference_state.get("final_modes", {})
    for item_path, expected_mode in sorted(final_modes.items()):
        target = sandbox_root / item_path.lstrip("/")
        if not target.exists() and not target.is_symlink():
            errors.append(f"{item_path}: missing from sandbox_fs")
            continue
        actual_mode = _lstat_mode(target)
        if actual_mode != expected_mode:
            errors.append(f"{item_path}: filesystem mode {actual_mode} != expected {expected_mode}")

        snapshot_entry = snapshot_meta.get(item_path)
        if snapshot_entry is None:
            errors.append(f"{item_path}: missing from input fs_snapshot.tar.gz")
            continue
        if snapshot_entry.get("is_symlink"):
            if not target.is_symlink():
                errors.append(f"{item_path}: expected symlink in sandbox_fs")
            elif target.readlink().as_posix() != snapshot_entry.get("linkname"):
                errors.append(f"{item_path}: symlink target changed")
            continue

        if target.is_symlink():
            errors.append(f"{item_path}: unexpected symlink in sandbox_fs")
            continue
        if target.read_bytes() != snapshot_entry.get("content", b""):
            errors.append(f"{item_path}: file contents changed")

    return {
        "score": 0.0 if errors else 1.0,
        "pass": not errors,
        "errors": errors,
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
