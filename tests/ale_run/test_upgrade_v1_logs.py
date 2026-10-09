from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

from ale_run.tasks.revision import task_revision
from scripts import upgrade_v1_logs as upgrade


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release(tmp_path):
    repository = tmp_path / "repository"
    entries = {}
    for name, action in (("keep", "retain"), ("changed", "rerun"), ("data_only", "rerun")):
        directory = repository / "tasks" / "domain" / name
        directory.mkdir(parents=True)
        (directory / "main.py").write_text("value = 1\n")
        entries[f"domain/{name}"] = {"action": action, "revision": task_revision(directory)}
    entries["domain/retired"] = {"action": "retire"}
    manifest = repository / "upgrade.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release": "v1.1",
                "baseline_commit": "a" * 40,
                "release_commit": "b" * 40,
                "tasks": entries,
            }
        )
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    return repository, manifest, logs


def write_run(
    logs,
    name="keep",
    *,
    status="completed",
    timestamp="20260727_120000",
    metadata=None,
    task_id=None,
):
    task_id = task_id or f"domain/{name}"
    directory = logs / "agent" / "model" / task_id.replace("/", "__") / "v0" / timestamp
    directory.mkdir(parents=True)
    if metadata is None:
        metadata = {
            "schema_version": 2,
            "task": {"path": f"tasks/{task_id}", "variant_index": 0},
            "status": status,
            "score": 0.625,
            "timings": {"duration_s": 321.09},
            "usage": {"output_tokens": 321},
        }
    (directory / "run.json").write_text(json.dumps(metadata))
    (directory / "trajectory.json").write_text('{"original": "trajectory"}')
    (directory / "eval_result.json").write_text('{"eval_status": "success", "score": 0.625}')
    (directory / "events.jsonl").write_text('{"type":"run_completed"}\n')
    return directory


def plan(release, **kwargs):
    repository, manifest, logs = release
    return upgrade.build_plan(logs, repository=repository, manifest_path=manifest, **kwargs)


def test_preview_apply_and_resume_preserve_original_metrics(release):
    repository, _, logs = release
    retained = write_run(logs)
    changed = write_run(logs, "changed")
    data_only = write_run(logs, "data_only")
    retired = write_run(logs, "retired")
    unknown = write_run(logs, "unknown")
    originals = {path: path.read_bytes() for path in logs.rglob("*") if path.is_file()}
    operations = plan(release)
    assert {operation.task: operation.action for operation in operations} == {
        "domain/keep": "retain",
        "domain/changed": "clear",
        "domain/data_only": "clear",
        "domain/retired": "clear",
    }
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    upgrade.apply_plan(operations)
    assert retained.is_dir() and unknown.is_dir()
    assert not any(directory.exists() for directory in (changed, data_only, retired))
    metadata = json.loads((retained / "run.json").read_text())
    assert metadata["task"]["revision"] == task_revision(repository / "tasks/domain/keep")
    assert metadata["task"]["revision_migration"]["migrated_from"] == "v1"
    assert metadata["task"]["revision_migration"]["original_revision"] is None
    original_metadata = json.loads(originals[retained / "run.json"])
    assert {key: value for key, value in metadata.items() if key != "task"} == {
        key: value for key, value in original_metadata.items() if key != "task"
    }
    for path, raw in originals.items():
        if path.parent in (retained, unknown) and path != retained / "run.json":
            assert path.read_bytes() == raw
    migrated = (retained / "run.json").read_bytes()
    upgrade.apply_plan(plan(release))
    assert (retained / "run.json").read_bytes() == migrated


@pytest.mark.parametrize(
    "task_fields",
    [
        {"task": "domain/keep"},
        {"task": "tasks/domain/keep"},
        {"task_path": "domain/keep"},
        {},
    ],
)
def test_legacy_schemas_keep_scores_and_original_task(release, task_fields):
    _, _, logs = release
    directory = write_run(
        logs,
        metadata={
            "schema_version": 1,
            "status": "timeout",
            "score": 0,
            "duration_s": 18000,
            **task_fields,
        },
    )
    (directory / "events.jsonl").unlink()
    (directory / "eval_result.json").unlink()
    upgrade.apply_plan(plan(release))
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["score"] == 0 and metadata["duration_s"] == 18000
    assert metadata["schema_version"] == 1
    assert metadata["task"]["revision_migration"]["original_task"] == task_fields.get("task")


def test_nested_collections_and_artifacts_are_not_discovered(release, monkeypatch):
    _, _, logs = release
    first = write_run(logs / "collection" / "experiment")
    second = write_run(logs / "other-experiment", "changed")
    hidden = write_run(first / "origin_log", "changed")
    unknown = write_run(logs, "unknown")
    write_run(unknown / "output", "changed")
    original_iterdir = Path.iterdir

    def bounded_iterdir(path):
        assert "origin_log" not in path.parts and "output" not in path.parts
        assert "domain__unknown" not in path.parts
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", bounded_iterdir)
    operations = plan(release)
    assert {operation.directory for operation in operations} == {first, second}
    upgrade.apply_plan(operations)
    assert hidden.exists() and unknown.exists()


def test_discovery_has_a_fixed_depth_bound(release):
    _, _, logs = release
    directory = write_run(logs / "one" / "two" / "three", "changed")
    assert plan(release) == []
    assert directory.exists()


@pytest.mark.parametrize(
    "part", ["root", "agent", "model", "task", "variant", "run", "metadata", "events", "eval"]
)
def test_structural_symlinks_abort_before_any_mutation(release, part):
    _, _, logs = release
    changed = write_run(logs, "changed")
    retained = write_run(logs)
    paths = {
        "root": logs,
        "agent": retained.parents[3],
        "model": retained.parents[2],
        "task": retained.parents[1],
        "variant": retained.parent,
        "run": retained,
        "metadata": retained / "run.json",
        "events": retained / "events.jsonl",
        "eval": retained / "eval_result.json",
    }
    target = paths[part]
    outside = logs.parent / "outside"
    target.rename(outside)
    target.symlink_to(outside, target_is_directory=outside.is_dir())
    with pytest.raises(upgrade.UnsafeLogs, match="Symlink"):
        plan(release)
    assert changed.exists()
    assert outside.exists()


def test_artifact_symlinks_never_delete_outside_targets(release):
    _, _, logs = release
    directory = write_run(logs, "changed")
    outside = logs.parent / "valuable"
    outside.mkdir()
    (outside / "evidence").write_text("keep")
    (directory / "output").symlink_to(outside, target_is_directory=True)
    upgrade.apply_plan(plan(release))
    assert not directory.exists()
    assert (outside / "evidence").read_text() == "keep"


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "broken",
        "array",
        "duplicate",
        "running",
        "missing_status",
        "schema",
        "wrong_task",
        "wrong_variant",
        "bad_task",
        "bad_revision",
        "unfinished_events",
        "bad_eval",
    ],
)
def test_incomplete_or_malformed_runs_abort_entire_plan(release, problem):
    _, _, logs = release
    changed = write_run(logs, "changed")
    bad = write_run(logs)
    path = bad / "run.json"
    metadata = json.loads(path.read_text())
    if problem == "missing":
        path.unlink()
    elif problem == "broken":
        path.write_text("{")
    elif problem == "array":
        path.write_text("[]")
    elif problem == "duplicate":
        path.write_text('{"status":"running","status":"completed"}')
    elif problem == "unfinished_events":
        (bad / "events.jsonl").write_text('{"type":"agent_started"}\n')
    elif problem == "bad_eval":
        (bad / "eval_result.json").write_text("null")
    else:
        if problem == "running":
            metadata["status"] = "running"
        if problem == "missing_status":
            metadata.pop("status")
        if problem == "schema":
            metadata["schema_version"] = 999
        if problem == "wrong_task":
            metadata["task"]["path"] = "tasks/domain/unknown"
        if problem == "wrong_variant":
            metadata["task"]["variant_index"] = 1
        if problem == "bad_task":
            metadata["task"] = []
        if problem == "bad_revision":
            metadata["task"]["revision"] = "sha256:garbage"
        path.write_text(json.dumps(metadata))
    with pytest.raises((upgrade.UnsafeLogs, OSError)):
        plan(release)
    assert changed.exists()


def test_failed_attempts_are_terminal_but_missing_attempt_metadata_is_not(release):
    _, _, logs = release
    write_run(logs, "changed", status="failed")
    write_run(logs, "changed", timestamp="20260727_120000_01")
    assert len(plan(release)) == 2
    interrupted = logs / "agent/model/domain__changed/v1/20260727_120001"
    interrupted.mkdir(parents=True)
    with pytest.raises(upgrade.UnsafeLogs, match="run_completed"):
        plan(release)


@pytest.mark.parametrize("modification", ["content", "untracked", "mode", "removed"])
def test_dirty_or_missing_current_task_is_never_stamped(release, modification):
    repository, _, logs = release
    write_run(logs)
    source = repository / "tasks/domain/keep/main.py"
    if modification == "content":
        source.write_text("value = 2\n")
    if modification == "untracked":
        (source.parent / "audit.json").write_text("{}")
    if modification == "mode":
        source.chmod(0o755)
    if modification == "removed":
        source.unlink()
    (operation,) = plan(release)
    assert operation.action == "clear" and operation.updated is None


def test_fresh_release_runs_survive_repeated_migration(release):
    _, manifest_path, logs = release
    entry = json.loads(manifest_path.read_text())["tasks"]["domain/changed"]
    directory = write_run(
        logs,
        "changed",
        metadata={
            "status": "completed",
            "task": {"path": "domain/changed", "revision": entry["revision"]},
        },
    )
    original = (directory / "run.json").read_bytes()
    upgrade.apply_plan(plan(release))
    assert (directory / "run.json").read_bytes() == original


def test_changed_only_migration_is_idempotent(release):
    _, _, logs = release
    write_run(logs, "changed")
    upgrade.apply_plan(plan(release))
    assert plan(release) == []


def test_apply_rechecks_every_run_before_writing_or_deleting(release):
    _, _, logs = release
    changed = write_run(logs, "changed")
    retained = write_run(logs)
    operations = plan(release)
    metadata = json.loads((retained / "run.json").read_text())
    metadata["score"] = 1
    (retained / "run.json").write_text(json.dumps(metadata))
    with pytest.raises(upgrade.UnsafeLogs, match="changed after preview"):
        upgrade.apply_plan(operations)
    assert changed.exists()
    assert "revision" not in json.loads((retained / "run.json").read_text())["task"]


def test_metadata_replace_is_atomic_and_failure_preserves_logs(release, monkeypatch):
    _, _, logs = release
    changed = write_run(logs, "changed")
    retained = write_run(logs)
    original = (retained / "run.json").read_bytes()
    operations = plan(release)

    def fail_replace(source, destination):
        assert Path(source).parent == destination.parent
        assert destination.read_bytes() == original
        assert json.loads(Path(source).read_text())["task"]["revision"].startswith("sha256:")
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        upgrade.apply_plan(operations)
    assert (retained / "run.json").read_bytes() == original
    assert changed.exists()
    assert not list(retained.glob(".run-upgrade-*"))


def test_cli_defaults_to_preview_and_requires_apply(release, monkeypatch, capsys):
    _, _, logs = release
    directory = write_run(logs, "changed")
    real_plan = upgrade.build_plan
    monkeypatch.setattr(
        upgrade,
        "build_plan",
        lambda root, **kwargs: real_plan(
            root, repository=release[0], manifest_path=release[1], **kwargs
        ),
    )
    assert upgrade.main([str(logs)]) == 0
    assert directory.exists() and "Preview:" in capsys.readouterr().out
    assert upgrade.main([str(logs), "--apply"]) == 0
    assert not directory.exists() and "Apply:" in capsys.readouterr().out


def test_from_commit_is_additive_and_cannot_reuse_an_earlier_stamp(release):
    repository, manifest_path, logs = release

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repository, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("add", "tasks")
    git("commit", "-qm", "old v1 snapshot")
    old = git("rev-parse", "HEAD")
    source = repository / "tasks/domain/keep/main.py"
    source.write_text("value = 2\n")
    git("add", "tasks")
    git("commit", "-qm", "release")
    manifest = json.loads(manifest_path.read_text())
    manifest["release_commit"] = git("rev-parse", "HEAD")
    manifest["tasks"]["domain/keep"]["revision"] = task_revision(source.parent)
    manifest_path.write_text(json.dumps(manifest))
    write_run(logs)
    write_run(logs, "data_only")
    upgrade.apply_plan(plan(release))
    (operation,) = plan(release, from_commit=old)
    assert operation.action == "clear" and "prior revision was a v1 stamp" in operation.reason
    write_run(logs, "data_only", timestamp="20260727_120000_01")
    assert all(operation.action == "clear" for operation in plan(release, from_commit=old))
    with pytest.raises(upgrade.UnsafeLogs, match="Cannot compare"):
        plan(release, from_commit="--help")


def test_release_manifest_covers_current_selection_repairs_and_retirement():
    manifest = upgrade.load_manifest(upgrade.MANIFEST)
    tasks = manifest["tasks"]
    assert manifest["release"] == "v1.1"
    assert "release_revision" not in manifest
    assert sum(entry["action"] == "rerun" for entry in tasks.values()) == 62
    assert sum(entry["action"] == "retain" for entry in tasks.values()) == 89
    categories = manifest["category_definitions"]
    counts = {category: 0 for category in categories}
    for entry in tasks.values():
        changes = entry.get("changes", [])
        assert len(changes) == len(set(changes))
        assert set(changes) <= categories.keys()
        assert bool(changes) == (entry["action"] == "rerun")
        for category in changes:
            counts[category] += 1
    assert manifest["change_summary"] == {"updated_tasks": 62, **counts}
    assets = json.loads((ROOT / "releases/v1.1/assets.json").read_text())
    assert assets["scope"]["updated_tasks"] == 62
    assert assets["scope"]["task_change_categories"] == counts
    for task in (
        "engineering/2d_drawings_to_3d_building_model",
        "engineering/cailian_road_highway_alignment_2",
        "engineering/gcode",
        "engineering/inner_support_elevation_optimization",
        "visual_media/music_transcription",
        "visual_media/project_migration",
        "engineering/chisel_verilog_alignment_seq_1",
        "computing_math/ising_post_measurement_1",
        "computing_math/mp_checkpoint_consolidation_v2",
        "computing_math/os_log_permission_guard_v1",
        "health_medicine/causal_ihdp_ite_estimation_6a_v1",
        "health_medicine/flusight_offline_hosp_forecast_2024_12_14",
        "life_sciences/zdock_hiv_dimer_interface_scoring_v1",
    ):
        assert tasks[task]["action"] == "rerun"
    selected = {
        line
        for line in (ROOT / "selected_tasks/full.txt").read_text().splitlines()
        if line and not line.startswith("#")
    }
    assert set(tasks) == selected | {"engineering/mold-flow"}
    assert tasks["engineering/mold-flow"]["action"] == "retire"
    assert tasks["visual_media/atlas_outpost_graybox_navigation"]["action"] == "retain"
    for task in (
        "engineering/abb_irb6700_asset_to_urdf_instance_1",
        "health_medicine/wsi_tumor_localization_1",
        "life_sciences/tms_marrow_cell_type_annotation_instance_1",
        "physical_sciences/lenacapavir_sar_table2_extraction",
    ):
        assert tasks[task]["action"] == "retain"
        assert tasks[task]["changes"] == []
        assert tasks[task]["retention_note"]


@pytest.fixture
def tracked_task_files():
    listing = subprocess.check_output(["git", "ls-files", "-z", "--", "tasks"], cwd=ROOT)
    return [ROOT / os.fsdecode(filename) for filename in listing.split(b"\0") if filename]


def test_all_release_hashes_match_current_tracked_projection(tracked_task_files, monkeypatch):
    manifest = upgrade.load_manifest(upgrade.MANIFEST)
    allowed = set(tracked_task_files)
    for path in tracked_task_files:
        allowed.update(path.parents)
    original_iterdir = Path.iterdir

    def tracked_iterdir(directory):
        return (entry for entry in original_iterdir(directory) if entry in allowed)

    monkeypatch.setattr(Path, "iterdir", tracked_iterdir)
    for task, entry in manifest["tasks"].items():
        if entry["action"] != "retire":
            assert task_revision(ROOT / "tasks" / task) == entry["revision"], task


@pytest.fixture
def clean_retained_tasks(tmp_path, tracked_task_files):
    manifest = upgrade.load_manifest(upgrade.MANIFEST)
    repository = tmp_path / "repository"
    for source in tracked_task_files:
        relative = source.relative_to(ROOT)
        task = "/".join(relative.parts[1:3])
        if manifest["tasks"].get(task, {}).get("action") != "retain":
            continue
        if not source.exists() and not source.is_symlink():
            continue
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            target.write_bytes(source.read_bytes())
            target.chmod(stat.S_IMODE(source.stat().st_mode))
    for task, entry in manifest["tasks"].items():
        if entry["action"] == "retain":
            assert task_revision(repository / "tasks" / task) == entry["revision"], task
    return repository


def test_full_v1_fixture_migrates_152_logs_and_resumes_only_62(clean_retained_tasks, tmp_path):
    from ale_run.cli import _filter_resume
    from ale_run.orchestration.experiment_spec import AgentSpec, RunUnit

    manifest = upgrade.load_manifest(upgrade.MANIFEST)
    logs = tmp_path / "logs"
    originals = {}
    units = []
    for task_id, entry in manifest["tasks"].items():
        directory = write_run(logs, task_id=task_id)
        originals[directory / "trajectory.json"] = (directory / "trajectory.json").read_bytes()
        if entry["action"] != "retire":
            units.append(
                RunUnit(
                    agent_id="agent",
                    agent_spec=AgentSpec(id="agent", class_="dummy", config={"model": "model"}),
                    task_path=task_id,
                    variant_index=0,
                )
            )
    operations = upgrade.build_plan(logs, repository=clean_retained_tasks)
    assert len(operations) == 152
    assert sum(operation.action == "retain" for operation in operations) == 89
    assert all(path.exists() for path in originals)
    upgrade.apply_plan(operations)
    assert len(list(logs.glob("*/*/*/v*/*/run.json"))) == 89
    for task_id, entry in manifest["tasks"].items():
        directory = logs / "agent/model" / task_id.replace("/", "__")
        assert directory.exists() == (entry["action"] == "retain")
    for path, raw in originals.items():
        if path.exists():
            assert path.read_bytes() == raw
    previous_cwd = Path.cwd()
    try:
        os.chdir(clean_retained_tasks)
        remaining = _filter_resume(units, logs)
    finally:
        os.chdir(previous_cwd)
    assert len(remaining) == 62
    assert {unit.task_path for unit in remaining} == {
        task_id for task_id, entry in manifest["tasks"].items() if entry["action"] == "rerun"
    }


@pytest.mark.parametrize("broken", ["missing", "malformed"])
def test_changed_corrupt_metadata_requires_final_completion_event(release, broken):
    _, _, logs = release
    directory = write_run(logs, "changed")
    if broken == "missing":
        (directory / "run.json").unlink()
    else:
        (directory / "run.json").write_text("{")
    operations = plan(release)
    assert operations[0].action == "clear"
    (directory / "events.jsonl").write_text('{"type":"run_started"}\n')
    with pytest.raises(upgrade.UnsafeLogs, match="run_completed"):
        upgrade.apply_plan(operations)
    (directory / "events.jsonl").write_text('{"type":"run_completed"}\n')
    upgrade.apply_plan(operations)
    assert not directory.parents[1].exists()


def test_null_revision_is_legacy_and_supplied_task_root_is_preserved(release):
    repository, manifest, logs = release
    directory = write_run(
        logs,
        metadata={
            "status": "failed",
            "task": {"path": "domain/keep", "revision": None},
        },
    )
    upgrade.apply_plan(plan(release))
    assert json.loads((directory / "run.json").read_text())["task"]["revision"].startswith(
        "sha256:"
    )
    changed = write_run(logs, "changed")
    task_root = changed.parents[1]
    upgrade.apply_plan(upgrade.build_plan(task_root, repository=repository, manifest_path=manifest))
    assert task_root.is_dir() and not list(task_root.iterdir())


def test_default_migration_needs_no_git(release, monkeypatch):
    _, _, logs = release
    write_run(logs)

    def no_git(*args, **kwargs):
        raise AssertionError("Default migration must not invoke Git")

    monkeypatch.setattr(subprocess, "check_output", no_git)
    upgrade.apply_plan(plan(release))


@pytest.mark.parametrize("field,value", [("status", []), ("status", {}), ("schema_version", True)])
def test_invalid_metadata_types_fail_cleanly(release, field, value):
    _, _, logs = release
    metadata = {"status": "completed", "task": "domain/keep", field: value}
    write_run(logs, metadata=metadata)
    with pytest.raises(upgrade.UnsafeLogs):
        plan(release)
