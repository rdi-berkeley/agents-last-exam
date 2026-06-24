from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
from pathlib import Path

import pytest


def _load_script(name: str):
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_manifests_filters_snapshot_variants_and_directory_entries(
    tmp_path: Path,
) -> None:
    prepare = _load_script("prepare_qemu_ubuntu22_image.py")
    repo_root = tmp_path / "repo"
    for task, snapshot in (
        ("domain/linux_task", "cpu-free-ubuntu"),
        ("domain/windows_task", "cpu-free"),
    ):
        card = repo_root / "tasks" / task / "task_card.json"
        card.parent.mkdir(parents=True)
        card.write_text(json.dumps({"vm": {"snapshot": snapshot}}), encoding="utf-8")

    task_list = tmp_path / "tasks.txt"
    task_list.write_text(
        "domain/linux_task\ndomain/windows_task  # excluded by snapshot\n",
        encoding="utf-8",
    )
    listing = tmp_path / "gcs.txt"
    listing.write_text(
        "gs://bucket/domain/linux_task/base/input/:\n"
        "gs://bucket/domain/linux_task/base/input/data.txt\n"
        "gs://bucket/domain/linux_task/base/reference/answer.txt\n"
        "gs://bucket/domain/windows_task/base/input/data.txt\n"
        "gs://bucket/demo/hello/base/input/order.json\n"
        "gs://bucket/demo/hello/other/input/order.json\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "manifests"

    prepare.build_manifests(
        argparse.Namespace(
            repo_root=repo_root,
            task_list=task_list,
            gcs_listing=listing,
            bucket="gs://bucket",
            snapshot="cpu-free-ubuntu",
            extra_variant=["demo/hello/base"],
            output_dir=output_dir,
        )
    )

    assert (output_dir / "expected-ubuntu-all-variants.txt").read_text() == (
        "demo/hello/base\ndomain/linux_task/base\n"
    )
    assert (output_dir / "expected-ubuntu-visible-files.txt").read_text() == (
        "demo/hello/base/input/order.json\ndomain/linux_task/base/input/data.txt\n"
    )
    assert (output_dir / "expected-reference-variants.txt").read_text() == (
        "domain/linux_task/base\n"
    )


def test_stage_disk_refuses_symlinked_staging_directory(tmp_path: Path) -> None:
    publish = _load_script("publish_qemu_images.py")
    source = tmp_path / "disk.qcow2"
    source.write_bytes(b"disk")
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (staging_root / source.name).symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlinked staging directory"):
        publish._stage_disk(source, staging_root, part_size=2)


def test_reference_extra_allowlist_is_narrow() -> None:
    prepare = _load_script("prepare_qemu_ubuntu22_image.py")
    patterns = prepare.ALLOWED_REFERENCE_EXTRA_PATTERNS[
        "engineering/sumo_urban_am_peak_calibration/base"
    ]
    allowed = (
        "evaluator_env/activate_evaluator_env.sh",
        "evaluator_env/.venv/bin/activate",
        "evaluator_env/.venv/bin/activate.ps1",
        "evaluator_env/.venv/lib/python3.10/site-packages/__pycache__/_virtualenv.cpython-310.pyc",
        "evaluator_env/.venv/lib/python3.10/site-packages/jsonschema/__pycache__/"
        "validators.cpython-310.pyc",
    )

    assert all(any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns) for path in allowed)
    assert not any(
        fnmatch.fnmatchcase("evaluator_env/unexpected_secret.txt", pattern) for pattern in patterns
    )
