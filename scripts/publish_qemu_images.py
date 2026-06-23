"""Publish ALE QEMU guest disks to a Hugging Face dataset repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "agents-last-exam/ale-images-qcow2"
DEFAULT_FILENAMES = ("ale-win10.qcow2", "ale-ubuntu22.qcow2")
DEFAULT_PART_SIZE_GB = 10
MANIFEST_FORMAT = "ale-qemu-disk-parts-v1"
STATE_FILENAME = ".ale-publish-state.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split and upload ALE qcow2 guest disks with resumable transfers.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("~/.cache/ale/qemu/images").expanduser(),
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("~/.cache/ale/qemu/hf-upload").expanduser(),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--filename",
        action="append",
        dest="filenames",
        help="qcow2 filename to publish; repeat to select multiple files",
    )
    parser.add_argument("--part-size-gb", type=int, default=DEFAULT_PART_SIZE_GB)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--dataset-card",
        type=Path,
        default=Path("ale_run/environments/images/ale_qemu/README.md"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _staging_is_current(
    *,
    source: Path,
    staging_dir: Path,
    part_size: int,
) -> bool:
    state = _load_json(staging_dir / STATE_FILENAME)
    manifest = _load_json(staging_dir / f"{source.name}.manifest.json")
    if (
        state.get("source") != str(source)
        or state.get("source_size") != source.stat().st_size
        or state.get("source_mtime_ns") != source.stat().st_mtime_ns
        or state.get("part_size") != part_size
        or manifest.get("format") != MANIFEST_FORMAT
    ):
        return False
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        return False
    return all(
        isinstance(part, dict)
        and (staging_dir / str(part.get("filename") or "")).is_file()
        and (staging_dir / str(part["filename"])).stat().st_size == part.get("size")
        for part in parts
    )


def _stage_disk(source: Path, staging_root: Path, part_size: int) -> Path:
    staging_dir = staging_root / source.name
    if _staging_is_current(
        source=source,
        staging_dir=staging_dir,
        part_size=part_size,
    ):
        print(f"reusing staged parts for {source.name}")
        return staging_dir

    shutil.rmtree(staging_dir, ignore_errors=True)
    parts_dir = staging_dir / f"{source.name}.parts"
    parts_dir.mkdir(parents=True)
    full_digest = hashlib.sha256()
    parts: list[dict[str, Any]] = []

    with source.open("rb") as source_file:
        index = 0
        while True:
            part_path = parts_dir / f"part-{index:05d}"
            part_digest = hashlib.sha256()
            part_bytes = 0
            with part_path.open("wb") as part_file:
                while part_bytes < part_size:
                    chunk = source_file.read(min(8 * 1024 * 1024, part_size - part_bytes))
                    if not chunk:
                        break
                    part_file.write(chunk)
                    part_digest.update(chunk)
                    full_digest.update(chunk)
                    part_bytes += len(chunk)
            if part_bytes == 0:
                part_path.unlink()
                break
            relative_path = part_path.relative_to(staging_dir).as_posix()
            parts.append(
                {
                    "filename": relative_path,
                    "size": part_bytes,
                    "sha256": part_digest.hexdigest(),
                }
            )
            print(f"staged {relative_path}: {part_bytes / 1_000_000_000:.2f} GB")
            index += 1

    manifest = {
        "format": MANIFEST_FORMAT,
        "filename": source.name,
        "size": source.stat().st_size,
        "sha256": full_digest.hexdigest(),
        "parts": parts,
    }
    manifest_path = staging_dir / f"{source.name}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging_dir / STATE_FILENAME).write_text(
        json.dumps(
            {
                "source": str(source),
                "source_size": source.stat().st_size,
                "source_mtime_ns": source.stat().st_mtime_ns,
                "part_size": part_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return staging_dir


def _delete_stale_remote_parts(
    *,
    api: HfApi,
    repo_id: str,
    source: Path,
    staging_dir: Path,
) -> None:
    manifest = _load_json(staging_dir / f"{source.name}.manifest.json")
    expected_parts = {
        str(part["filename"])
        for part in manifest.get("parts", [])
        if isinstance(part, dict) and part.get("filename")
    }
    parts_prefix = f"{source.name}.parts/"
    stale_parts = sorted(
        path
        for path in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if path.startswith(parts_prefix) and path not in expected_parts
    )
    if not stale_parts:
        return
    api.delete_files(
        repo_id=repo_id,
        repo_type="dataset",
        delete_patterns=stale_parts,
        commit_message=f"remove stale {source.name} disk parts",
    )
    print(f"deleted {len(stale_parts)} stale remote parts for {source.name}")


def main() -> None:
    args = _parse_args()
    if not 1 <= args.part_size_gb <= 20:
        raise ValueError("--part-size-gb must be between 1 and 20")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")

    image_dir = args.image_dir.expanduser().resolve()
    staging_root = args.staging_dir.expanduser().resolve()
    filenames = tuple(args.filenames or DEFAULT_FILENAMES)
    sources: list[Path] = []
    for filename in filenames:
        source = image_dir / filename
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"QEMU guest disk not found: {source}")
        sources.append(source)

    api = HfApi()
    dataset_card = args.dataset_card.resolve()
    if dataset_card.is_file():
        api.upload_file(
            path_or_fileobj=dataset_card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="document ALE QEMU guest images",
        )

    part_size = args.part_size_gb * 1_000_000_000
    for source in sources:
        staging_dir = _stage_disk(source, staging_root, part_size)
        print(f"uploading staged {source.name} to {args.repo_id}")
        api.upload_large_folder(
            repo_id=args.repo_id,
            folder_path=staging_dir,
            repo_type="dataset",
            allow_patterns=[
                f"{source.name}.manifest.json",
                f"{source.name}.parts/*",
            ],
            num_workers=args.num_workers,
        )
        _delete_stale_remote_parts(
            api=api,
            repo_id=args.repo_id,
            source=source,
            staging_dir=staging_dir,
        )


if __name__ == "__main__":
    main()
