#!/usr/bin/env python3
"""Clean and validate an Ubuntu root filesystem before QEMU image export."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


TASK_DATA_REL = Path("media/user/data/agenthle")
USER_HOME_REL = Path("home/user")
REFERENCE_PASSWORD_ENV = "ALE_REFERENCE_ARCHIVE_PASSWORD"

GCE_SERVICE_PREFIXES = (
    "gce-",
    "google-cloud-ops-agent",
    "google-guest-agent",
    "google-osconfig-agent",
    "google-oslogin-cache",
    "google-shutdown-scripts",
    "google-startup-scripts",
)

REMOVE_USER_PATHS = (
    ".agenthle_hidden_eval_assets",
    ".ale",
    ".ale-src",
    ".claude",
    ".codex",
    ".cursor",
    ".factory",
    ".forge",
    ".gemini",
    ".gsutil",
    ".netrc",
    ".openhands",
    ".ssh",
    ".config/agenthle-artifacts",
    ".config/cursor",
    ".config/gcloud",
    ".config/google-chrome",
    ".config/Code/logs",
    ".ipython",
    ".local/share/Trash",
    ".local/share/gvfs-metadata",
    ".local/share/uv/credentials",
    ".local/share/xorg",
    ".mozilla",
    ".npm-ale/_logs",
    ".playwright-firefox",
    "stage1_base_new",
    "stage1_base_repair",
    "stage1_base_repair2",
    "stage1_base_sync",
    "stage4_install",
    "temp",
    "test_area",
)

REMOVE_ROOT_PATHS = (
    "root/.bash_history",
    "root/.config/gcloud",
    "root/.gsutil",
    "root/.npm/_logs",
    "root/.python_history",
    "root/.ssh",
    "root/.wget-hsts",
    "etc/boto.cfg",
    "var/lib/cloud",
    "var/lib/dhcp",
    "var/lib/google",
    "var/lib/systemd/random-seed",
    "var/log/google-cloud-ops-agent",
)

REMOVE_HOME_FILE_PATTERNS = (
    "*.log",
    "*_exit.txt",
    "*_http_code.txt",
    "*_launcher.sh",
    "*_output.jsonl",
    "*_pid.txt",
    "*_prompt.txt",
    "*_rc.txt",
    "*_request.json",
    "*_response.json",
    "*_runner.sh",
    "*_stderr.log",
    "*_stdout.log",
    "difficulty_element_log.xlsx",
    "input_tmp.in",
    "mbmd_install_software.sh",
    "nxf-tmp.*",
    "phonon_disc.py",
    "probe*.pid",
    "provision_n8n_runtime.sh",
    "regrade_*.py",
    "reference.frc",
    "requirement.txt",
    "screenshot.png",
    "serve.sh",
    "sse_patch.js",
    "test_*.txt",
    ".wget-hsts",
)

PRESERVE_AGENT_CHILDREN = {
    ".grok": frozenset({"bin", "install.json"}),
    ".hermes": frozenset({"hermes-agent"}),
    ".openclaw": frozenset({"extensions", "plugin-runtime-deps"}),
}

ALLOWED_REFERENCE_EXTRA_PATTERNS = {
    "engineering/sumo_urban_am_peak_calibration/base": (
        "evaluator_env/activate_evaluator_env.sh",
        "evaluator_env/.venv/bin/activate*",
        "evaluator_env/.venv/lib/python3.10/site-packages/__pycache__/*.pyc",
        "evaluator_env/.venv/lib/python3.10/site-packages/*/__pycache__/*.pyc",
    ),
}


def _read_manifest(path: Path) -> list[str]:
    values = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            values.append(value)
    return values


def build_manifests(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    selected_tasks: set[str] = set()
    extra_variants = set(args.extra_variant)
    for raw in args.task_list.read_text(encoding="utf-8").splitlines():
        task = raw.split("#", 1)[0].strip()
        if not task:
            continue
        task_card = repo_root / "tasks" / task / "task_card.json"
        if not task_card.is_file():
            raise FileNotFoundError(f"task card not found: {task_card}")
        card = json.loads(task_card.read_text(encoding="utf-8"))
        if card.get("vm", {}).get("snapshot") == args.snapshot:
            selected_tasks.add(task)

    bucket_prefix = args.bucket.rstrip("/") + "/"
    variants: set[str] = set()
    visible_files: set[str] = set()
    reference_variants: set[str] = set()
    reference_files: set[str] = set()
    for raw in args.gcs_listing.read_text(encoding="utf-8").splitlines():
        uri = raw.strip()
        if not uri.startswith(bucket_prefix) or uri.endswith("/") or uri.endswith("/:"):
            continue
        relative = uri.removeprefix(bucket_prefix)
        parts = relative.split("/")
        if len(parts) < 5:
            continue
        variant = "/".join(parts[:3])
        if "/".join(parts[:2]) not in selected_tasks and variant not in extra_variants:
            continue
        phase = parts[3]
        variants.add(variant)
        if phase in {"input", "software"}:
            visible_files.add(relative)
        elif phase == "reference":
            reference_variants.add(variant)
            reference_files.add(relative)

    variants.update(extra_variants)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        "expected-ubuntu-all-variants.txt": variants,
        "expected-ubuntu-visible-files.txt": visible_files,
        "expected-reference-variants.txt": reference_variants,
        "expected-ubuntu-reference-files.txt": reference_files,
    }
    for filename, values in manifests.items():
        (output_dir / filename).write_text(
            "".join(f"{value}\n" for value in sorted(values)),
            encoding="utf-8",
        )
    print(
        "manifests written: "
        f"{len(variants)} variants, "
        f"{len(visible_files)} visible files, "
        f"{len(reference_variants)} reference variants, "
        f"{len(reference_files)} reference files"
    )


def _remove(path: Path, *, dry_run: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    print(f"remove {path}")
    if dry_run:
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _validate_root(root: Path) -> None:
    required = (
        root / "etc/os-release",
        root / USER_HOME_REL,
        root / TASK_DATA_REL,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"not an ALE Ubuntu root filesystem: missing {missing}")


def _iter_variants(task_root: Path) -> set[str]:
    variants: set[str] = set()
    for domain in task_root.iterdir():
        if not domain.is_dir():
            continue
        for task in domain.iterdir():
            if not task.is_dir():
                continue
            for variant in task.iterdir():
                if variant.is_dir():
                    variants.add(variant.relative_to(task_root).as_posix())
    return variants


def _clean_task_data(
    *,
    task_root: Path,
    expected_variants: set[str],
    hg002_reference: Path,
    dry_run: bool,
) -> None:
    for variant in sorted(_iter_variants(task_root) - expected_variants):
        _remove(task_root / variant, dry_run=dry_run)

    for variant in sorted(expected_variants):
        base = task_root / variant
        if not base.is_dir():
            continue
        for residue in ("output", "reference", ".reference_extract"):
            _remove(base / residue, dry_run=dry_run)

    nested_archive = (
        task_root
        / "life_sciences/hg002_chr22_germline_variant_pipeline/base"
        / "input/starter_project/reference.7z"
    )
    nested_target = nested_archive.with_suffix("")
    _remove(nested_archive, dry_run=dry_run)
    _remove(nested_target, dry_run=dry_run)
    print(f"restore {hg002_reference} -> {nested_target}")
    if not dry_run:
        shutil.copytree(hg002_reference, nested_target)
        user_stat = (task_root.parents[3] / USER_HOME_REL).stat()
        for path in [nested_target, *nested_target.rglob("*")]:
            os.chown(path, user_stat.st_uid, user_stat.st_gid)


def _clean_agent_state(user_home: Path, *, dry_run: bool) -> None:
    for relative in REMOVE_USER_PATHS:
        _remove(user_home / relative, dry_run=dry_run)

    for directory, preserved in PRESERVE_AGENT_CHILDREN.items():
        root = user_home / directory
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.name not in preserved:
                _remove(child, dry_run=dry_run)

    for pattern in REMOVE_HOME_FILE_PATTERNS:
        for path in user_home.glob(pattern):
            _remove(path, dry_run=dry_run)

    for history in (
        ".bash_history",
        ".python_history",
        ".zsh_history",
        ".octave_hist",
    ):
        _remove(user_home / history, dry_run=dry_run)


def _clean_system_state(root: Path, *, dry_run: bool) -> None:
    for relative in REMOVE_ROOT_PATHS:
        _remove(root / relative, dry_run=dry_run)

    for path in (root / "tmp", root / "var/tmp"):
        if not path.is_dir():
            continue
        for child in path.iterdir():
            _remove(child, dry_run=dry_run)

    for journal_root in (root / "var/log/journal", root / "run/log/journal"):
        _remove(journal_root, dry_run=dry_run)

    log_root = root / "var/log"
    if log_root.is_dir():
        for log_file in log_root.rglob("*"):
            if not log_file.is_file() or log_file.is_symlink():
                continue
            print(f"truncate {log_file}")
            if not dry_run:
                log_file.write_bytes(b"")

    for host_key in (root / "etc/ssh").glob("ssh_host_*"):
        _remove(host_key, dry_run=dry_run)
    if not dry_run:
        subprocess.run(
            ["chroot", str(root), "ssh-keygen", "-A"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    for identity in (root / "etc/machine-id", root / "var/lib/dbus/machine-id"):
        _remove(identity, dry_run=dry_run)
    if not dry_run:
        (root / "etc/machine-id").touch(mode=0o444)


def _mask_gce_services(root: Path, *, dry_run: bool) -> None:
    unit_names: set[str] = set()
    for unit_root in (root / "lib/systemd/system", root / "usr/lib/systemd/system"):
        if not unit_root.is_dir():
            continue
        for unit in unit_root.iterdir():
            if unit.name.startswith(GCE_SERVICE_PREFIXES):
                unit_names.add(unit.name)

    mask_root = root / "etc/systemd/system"
    for unit_name in sorted(unit_names):
        mask = mask_root / unit_name
        _remove(mask, dry_run=dry_run)
        print(f"mask {mask}")
        if not dry_run:
            mask.symlink_to("/dev/null")


def clean(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    _validate_root(root)
    expected_variants = set(_read_manifest(args.expected_variants))
    _clean_task_data(
        task_root=root / TASK_DATA_REL,
        expected_variants=expected_variants,
        hg002_reference=args.hg002_reference.resolve(),
        dry_run=args.dry_run,
    )
    _clean_agent_state(root / USER_HOME_REL, dry_run=args.dry_run)
    _clean_system_state(root, dry_run=args.dry_run)
    if args.disable_gce_services:
        _mask_gce_services(root, dry_run=args.dry_run)


def _archive_inventory(root: Path, archive: Path, password: str) -> set[str]:
    relative_archive = "/" + archive.relative_to(root).as_posix()
    command = [
        "chroot",
        str(root),
        "7z",
        "l",
        "-slt",
        f"-p{password}",
        relative_archive,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise RuntimeError(f"cannot list {archive}: {detail}")

    files: set[str] = set()
    for block in result.stdout.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, separator, value = line.partition(" = ")
            if separator:
                fields[key] = value
        path = fields.get("Path")
        attributes = fields.get("Attributes", "")
        if path and "Size" in fields and not attributes.startswith("D"):
            files.add(path.replace("\\", "/"))
    return files


def _validate_reference_archives(
    *,
    root: Path,
    task_root: Path,
    expected_reference_files: list[str],
) -> list[str]:
    password = os.environ.get(REFERENCE_PASSWORD_ENV, "")
    if not password:
        return [f"{REFERENCE_PASSWORD_ENV} is required to validate reference archives"]

    expected_by_variant: dict[str, set[str]] = defaultdict(set)
    for relative in expected_reference_files:
        parts = Path(relative).parts
        variant = "/".join(parts[:3])
        expected_by_variant[variant].add("/".join(parts[4:]))

    errors: list[str] = []
    for variant, expected_files in sorted(expected_by_variant.items()):
        archive = task_root / variant / "reference.7z"
        try:
            actual_files = _archive_inventory(root, archive, password)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
            continue
        missing = sorted(expected_files - actual_files)
        if missing:
            errors.append(f"{variant}/reference.7z inventory mismatch: missing={missing[:10]}")
        extra = sorted(actual_files - expected_files)
        allowed_patterns = ALLOWED_REFERENCE_EXTRA_PATTERNS.get(variant, ())
        unexpected_extra = [
            path
            for path in extra
            if not any(fnmatch.fnmatchcase(path, pattern) for pattern in allowed_patterns)
        ]
        if unexpected_extra:
            errors.append(
                f"{variant}/reference.7z inventory mismatch: unexpected={unexpected_extra[:10]}"
            )
        elif extra:
            print(
                f"WARNING: {variant}/reference.7z contains "
                f"{len(extra)} allowed generated evaluator entries",
                file=sys.stderr,
            )
    return errors


def validate(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    _validate_root(root)
    task_root = root / TASK_DATA_REL
    user_home = root / USER_HOME_REL
    expected_variants = set(_read_manifest(args.expected_variants))
    expected_visible_files = _read_manifest(args.expected_visible_files)
    expected_reference_variants = set(_read_manifest(args.expected_reference_variants))
    expected_reference_files = _read_manifest(args.expected_reference_files)

    errors: list[str] = []
    actual_variants = _iter_variants(task_root)
    if actual_variants != expected_variants:
        errors.append(
            "variant mismatch: "
            f"missing={sorted(expected_variants - actual_variants)}, "
            f"extra={sorted(actual_variants - expected_variants)}"
        )

    missing_visible = [
        relative for relative in expected_visible_files if not (task_root / relative).is_file()
    ]
    if missing_visible:
        errors.append(f"missing canonical input/software files: {missing_visible[:20]}")

    nested_archives = sorted(
        path.relative_to(task_root).as_posix()
        for path in task_root.glob("*/*/*/input/**/reference.7z")
    )
    if nested_archives:
        errors.append(f"nested input reference archives remain: {nested_archives}")

    phase_residue = []
    for variant in sorted(actual_variants):
        base = task_root / variant
        for residue in ("output", "reference", ".reference_extract"):
            if (base / residue).exists():
                phase_residue.append(f"{variant}/{residue}")
    if phase_residue:
        errors.append(f"task phase residue remains: {phase_residue}")

    actual_reference_variants = {
        archive.parent.relative_to(task_root).as_posix()
        for archive in task_root.glob("*/*/*/reference.7z")
    }
    if actual_reference_variants != expected_reference_variants:
        errors.append(
            "reference archive mismatch: "
            f"missing={sorted(expected_reference_variants - actual_reference_variants)}, "
            f"extra={sorted(actual_reference_variants - expected_reference_variants)}"
        )

    residue_paths = [relative for relative in REMOVE_USER_PATHS if (user_home / relative).exists()]
    residue_paths.extend(relative for relative in REMOVE_ROOT_PATHS if (root / relative).exists())
    if residue_paths:
        errors.append(f"sensitive or stale paths remain: {residue_paths}")

    nonempty_logs = [
        path.relative_to(root).as_posix()
        for path in (root / "var/log").rglob("*")
        if path.is_file() and not path.is_symlink() and path.stat().st_size
    ]
    if nonempty_logs:
        errors.append(f"non-empty system logs remain: {nonempty_logs[:20]}")

    if args.expect_gce_services_disabled:
        unmasked = []
        for unit_root in (root / "lib/systemd/system", root / "usr/lib/systemd/system"):
            if not unit_root.is_dir():
                continue
            for unit in unit_root.iterdir():
                if not unit.name.startswith(GCE_SERVICE_PREFIXES):
                    continue
                mask = root / "etc/systemd/system" / unit.name
                if not mask.is_symlink() or os.readlink(mask) != "/dev/null":
                    unmasked.append(unit.name)
        if unmasked:
            errors.append(f"GCE services are not masked: {sorted(set(unmasked))}")

    if args.verify_reference_archives:
        errors.extend(
            _validate_reference_archives(
                root=root,
                task_root=task_root,
                expected_reference_files=expected_reference_files,
            )
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(
        "validation passed: "
        f"{len(actual_variants)} variants, "
        f"{len(expected_visible_files)} canonical visible files, "
        f"{len(actual_reference_variants)} reference archives"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifests")
    manifest_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    manifest_parser.add_argument("--task-list", type=Path, required=True)
    manifest_parser.add_argument("--gcs-listing", type=Path, required=True)
    manifest_parser.add_argument("--bucket", default="gs://ale-data-public")
    manifest_parser.add_argument("--snapshot", default="cpu-free-ubuntu")
    manifest_parser.add_argument("--extra-variant", action="append", default=[])
    manifest_parser.add_argument("--output-dir", type=Path, required=True)
    manifest_parser.set_defaults(func=build_manifests)

    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--root", type=Path, required=True)
    clean_parser.add_argument("--expected-variants", type=Path, required=True)
    clean_parser.add_argument("--hg002-reference", type=Path, required=True)
    clean_parser.add_argument("--disable-gce-services", action="store_true")
    clean_parser.add_argument("--dry-run", action="store_true")
    clean_parser.set_defaults(func=clean)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--root", type=Path, required=True)
    validate_parser.add_argument("--expected-variants", type=Path, required=True)
    validate_parser.add_argument("--expected-visible-files", type=Path, required=True)
    validate_parser.add_argument("--expected-reference-variants", type=Path, required=True)
    validate_parser.add_argument("--expected-reference-files", type=Path, required=True)
    validate_parser.add_argument("--verify-reference-archives", action="store_true")
    validate_parser.add_argument("--expect-gce-services-disabled", action="store_true")
    validate_parser.set_defaults(func=validate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
