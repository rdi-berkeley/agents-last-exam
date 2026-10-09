#!/usr/bin/env python3
"""Preview or apply the release task map to completed v1 run logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "releases/v1.1/upgrade.json"
TASK_ID = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+")
REVISION = re.compile(r"sha256:[0-9a-f]{64}")
VARIANT = re.compile(r"v(0|[1-9][0-9]*)")
TIMESTAMP = re.compile(r"[0-9]{8}_[0-9]{6}(?:_[0-9]+)?")
TERMINAL = {"completed", "timeout", "failed", "cancelled", "not_executed"}
OPAQUE = {"origin", "origin_log", "output", ".git", "__pycache__", "node_modules", ".venv"}
MAX_METADATA = 8 * 1024 * 1024


class UnsafeLogs(ValueError):
    """The requested migration cannot safely proceed."""


@dataclass(frozen=True)
class Operation:
    directory: Path
    task: str
    action: str
    reason: str
    original: bytes | None
    updated: dict | None = None
    prune: tuple[Path, ...] = ()


def checked_path(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise UnsafeLogs(f"Symlink in migration path: {component}")
    return path


def read_json(path: Path) -> tuple[dict, bytes]:
    checked_path(path)
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > MAX_METADATA:
        raise UnsafeLogs(f"Not a bounded regular metadata file: {path}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise UnsafeLogs(f"Duplicate metadata key {key!r}: {path}")
            result[key] = value
        return result

    raw = path.read_bytes()
    try:
        metadata = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as error:
        raise UnsafeLogs(f"Malformed JSON: {path}: {error}") from error
    if not isinstance(metadata, dict):
        raise UnsafeLogs(f"Expected a metadata object: {path}")
    return metadata, raw


def load_manifest(path: Path) -> dict:
    manifest, _ = read_json(path)
    if manifest.get("schema_version") != 1 or manifest.get("release") != "v1.1":
        raise UnsafeLogs("Unsupported upgrade manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("baseline_commit", ""))):
        raise UnsafeLogs("Missing manifest baseline commit")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise UnsafeLogs("Missing manifest task map")
    for task_id, entry in tasks.items():
        if not TASK_ID.fullmatch(task_id) or not isinstance(entry, dict):
            raise UnsafeLogs(f"Invalid manifest task: {task_id}")
        if entry.get("action") not in ("retain", "rerun", "retire"):
            raise UnsafeLogs(f"Invalid manifest action: {task_id}")
        if entry["action"] != "retire" and not REVISION.fullmatch(str(entry.get("revision", ""))):
            raise UnsafeLogs(f"Missing clean release revision: {task_id}")
    return manifest


def discover_tasks(root: Path, tasks: dict) -> list[tuple[Path, str]]:
    """Find task slugs at most four containers below root; artifacts stay opaque."""
    slugs = {task_id.replace("/", "__"): task_id for task_id in tasks}
    pending = [(root, 0)]
    found = []
    while pending:
        directory, depth = pending.pop()
        checked_path(directory)
        if directory.name in slugs:
            found.append((directory, slugs[directory.name]))
            continue
        for child in sorted(directory.iterdir()):
            if child.name in OPAQUE or child.name.startswith("."):
                continue
            if "__" in child.name and child.name not in slugs:
                continue
            if child.is_symlink():
                raise UnsafeLogs(f"Symlink in log layout: {child}")
            if not child.is_dir():
                continue
            if child.name in slugs:
                found.append((child, slugs[child.name]))
            elif depth < 4 and not VARIANT.fullmatch(child.name):
                pending.append((child, depth + 1))
    return sorted(found)


def read_completed(
    directory: Path, task_id: str, variant: int, *, allow_corrupt: bool = False
) -> tuple[dict, bytes | None]:
    path = checked_path(directory / "run.json")
    corrupt = False
    try:
        metadata, raw = read_json(path)
    except (UnsafeLogs, FileNotFoundError):
        if not allow_corrupt:
            raise
        if path.exists() and (
            not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > MAX_METADATA
        ):
            raise
        raw = path.read_bytes() if path.exists() else None
        metadata, corrupt = {"status": "completed"}, True
    if type(metadata.get("schema_version", 1)) is not int or metadata.get(
        "schema_version", 1
    ) not in (1, 2):
        raise UnsafeLogs(f"Unsupported run schema: {directory}")
    if not isinstance(metadata.get("status"), str) or metadata["status"] not in TERMINAL:
        raise UnsafeLogs(f"Running or incomplete run: {directory}")
    task = metadata.get("task")
    identities = [metadata.get("task_path")]
    if isinstance(task, dict):
        identities.append(task.get("path"))
        if "slug" in task and task["slug"] != task_id.replace("/", "__"):
            raise UnsafeLogs(f"Task slug disagrees with layout: {directory}")
        if "variant_index" in task and (
            type(task["variant_index"]) is not int or task["variant_index"] != variant
        ):
            raise UnsafeLogs(f"Variant disagrees with layout: {directory}")
        if task.get("revision") is not None and not REVISION.fullmatch(str(task["revision"])):
            raise UnsafeLogs(f"Malformed task revision: {directory}")
    elif isinstance(task, str):
        identities.append(task)
    elif task is not None:
        raise UnsafeLogs(f"Malformed task metadata: {directory}")
    for identity in identities:
        if identity is not None and (
            not isinstance(identity, str) or identity.removeprefix("tasks/") != task_id
        ):
            raise UnsafeLogs(f"Task identity disagrees with layout: {directory}")
    finished = False
    for name in ("eval_result.json", "events.jsonl"):
        path = directory / name
        checked_path(path)
        if not path.exists():
            continue
        if name == "eval_result.json":
            evaluation, _ = read_json(path)
            if not isinstance(evaluation.get("eval_status"), str) or evaluation[
                "eval_status"
            ] not in TERMINAL | {"success", "skipped"}:
                raise UnsafeLogs(f"Running or incomplete evaluation: {directory}")
        else:
            if not stat.S_ISREG(path.stat().st_mode):
                raise UnsafeLogs(f"Not a regular event log: {path}")
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - MAX_METADATA))
                lines = stream.read(MAX_METADATA).splitlines()
            try:
                event = json.loads(lines[-1])
                finished = isinstance(event, dict) and event.get("type") == "run_completed"
                if finished and isinstance(event.get("data"), dict) and "status" in event["data"]:
                    status = event["data"]["status"]
                    finished = isinstance(status, str) and status in TERMINAL
            except (ValueError, IndexError, UnicodeError):
                finished = False
            if not finished:
                raise UnsafeLogs(f"Missing final run_completed event: {directory}")
    if corrupt and not finished:
        raise UnsafeLogs(f"Corrupt run metadata without a final run_completed event: {directory}")
    return metadata, raw


def additional_changes(repository: Path, commit: str, manifest: dict) -> set[str]:
    try:
        baseline = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{commit}^{{commit}}"],
            cwd=repository,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        changed = (
            subprocess.check_output(
                [
                    "git",
                    "diff",
                    "--name-only",
                    "--no-renames",
                    "-z",
                    baseline,
                    manifest["release_commit"],
                    "--",
                    "tasks",
                ],
                cwd=repository,
                stderr=subprocess.PIPE,
            )
            .decode()
            .split("\0")
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise UnsafeLogs(f"Cannot compare --from-commit {commit!r}: {error}") from error
    tasks = manifest["tasks"]
    extra = set()
    for filename in filter(None, changed):
        parts = filename.split("/")
        task_id = "/".join(parts[1:3])
        if task_id in tasks:
            extra.add(task_id)
        elif "_shared" in parts or filename in {
            "tasks/utils/evaluation.py",
            "tasks/linux_runtime.py",
        }:
            extra.update(tasks)
    return extra


def build_plan(
    root: Path,
    *,
    repository: Path = ROOT,
    manifest_path: Path = MANIFEST,
    from_commit: str | None = None,
) -> list[Operation]:
    from ale_run.tasks.revision import task_revision

    root = checked_path(root)
    if not root.is_dir():
        raise UnsafeLogs(f"Not a log directory: {root}")
    manifest = load_manifest(manifest_path)
    extra = additional_changes(repository, from_commit, manifest) if from_commit else set()
    revisions = {}
    operations = []
    task_directories = discover_tasks(root, manifest["tasks"])
    for task_directory, task_id in task_directories:
        entry = manifest["tasks"][task_id]
        if task_id not in revisions and entry["action"] != "retire":
            source = checked_path(repository / "tasks" / task_id)
            revisions[task_id] = task_revision(source) if source.is_dir() else None
        current = revisions.get(task_id)
        for variant_directory in sorted(task_directory.iterdir()):
            match = VARIANT.fullmatch(variant_directory.name)
            if not match:
                continue
            checked_path(variant_directory)
            if not variant_directory.is_dir():
                raise UnsafeLogs(f"Not a variant directory: {variant_directory}")
            for directory in sorted(variant_directory.iterdir()):
                checked_path(directory)
                if not directory.is_dir() or not TIMESTAMP.fullmatch(directory.name):
                    raise UnsafeLogs(f"Unexpected entry in variant layout: {directory}")
                metadata, raw = read_completed(
                    directory,
                    task_id,
                    int(match[1]),
                    allow_corrupt=entry["action"] in {"rerun", "retire"} or task_id in extra,
                )
                task = metadata.get("task")
                previous = task.get("revision") if isinstance(task, dict) else None
                action, reason, updated = "clear", entry["action"], None
                if entry["action"] != "retire":
                    if current != entry["revision"]:
                        reason = "task source differs from clean release (or is missing)"
                    elif (
                        task_id in extra
                        and isinstance(task, dict)
                        and task.get("revision_migration")
                    ):
                        reason = "changed since --from-commit; prior revision was a v1 stamp"
                    elif previous == current:
                        action, reason = "retain", "already has the clean release revision"
                    elif previous is not None:
                        reason = "recorded revision differs from release"
                    elif entry["action"] == "retain" and task_id not in extra:
                        action, reason = "retain", "unchanged in release map"
                        updated = json.loads(raw)
                        migrated = (
                            dict(task)
                            if isinstance(task, dict)
                            else {"path": f"tasks/{task_id}", "variant_index": int(match[1])}
                        )
                        if "revision_migration" in migrated:
                            raise UnsafeLogs(f"Unversioned migration marker: {directory}")
                        migrated["revision"] = current
                        migrated["revision_migration"] = {
                            "migrated_from": "v1",
                            "release": manifest["release"],
                            "original_revision": None,
                        }
                        if not isinstance(task, dict):
                            migrated["revision_migration"]["original_task"] = task
                        updated["task"] = migrated
                    elif task_id in extra:
                        reason = "changed since --from-commit"
                prune = tuple(path for path in (variant_directory, task_directory) if path != root)
                operations.append(
                    Operation(directory, task_id, action, reason, raw, updated, prune)
                )
    return operations


def apply_plan(operations: list[Operation]) -> None:
    if not shutil.rmtree.avoids_symlink_attacks:
        raise UnsafeLogs("This platform cannot safely delete run trees without following symlinks")
    for operation in operations:
        variant = int(operation.directory.parent.name[1:])
        _, raw = read_completed(
            operation.directory, operation.task, variant, allow_corrupt=operation.action == "clear"
        )
        if raw != operation.original:
            raise UnsafeLogs(f"Metadata changed after preview: {operation.directory}")
    for operation in operations:
        if operation.updated is None:
            continue
        path = checked_path(operation.directory / "run.json")
        descriptor, temporary = tempfile.mkstemp(prefix=".run-upgrade-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
                json.dump(operation.updated, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    for operation in operations:
        if operation.action == "clear":
            checked_path(operation.directory)
            shutil.rmtree(operation.directory)
    directories = {
        path for operation in operations if operation.action == "clear" for path in operation.prune
    }
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        checked_path(directory)
        try:
            directory.rmdir()
        except OSError as error:
            if error.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade completed v1 logs for v1.1. Preview by default; no artifacts are copied.",
        epilog="Stop the experiment first. ROOT is the output root, or a collection with up to "
        "two enclosing folders. Only known task/vN/timestamp runs are considered; origin/output "
        "trees and unknown tasks are preserved. --apply deletes changed/retired runs and stamps "
        "retained run.json with an explicit v1 migration marker, preserving scores/timings. "
        "Then resume normally. The map includes task repairs and open-source conversions from "
        "v1 commit 2d4d220; use --from-commit for an older snapshot. "
        "Default use needs no Git. Back up valuable logs before applying.",
    )
    parser.add_argument("root", type=Path, help="Completed log output root or collection")
    parser.add_argument("--apply", action="store_true", help="Apply the displayed changes")
    parser.add_argument(
        "--from-commit", help="Additionally rerun source changes from this v1 commit"
    )
    arguments = parser.parse_args(argv)
    try:
        operations = build_plan(arguments.root, from_commit=arguments.from_commit)
        for operation in operations:
            print(f"{operation.action.upper():6} {operation.directory} ({operation.reason})")
        retained = sum(operation.action == "retain" for operation in operations)
        print(
            f"{'Apply' if arguments.apply else 'Preview'}: {retained} retained, "
            f"{len(operations) - retained} cleared runs."
        )
        if arguments.apply:
            apply_plan(operations)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Upgrade refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
