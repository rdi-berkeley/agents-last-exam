from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tasks.computing_math.ranking_node_feature_parity_recovery_instance_1.scripts import (
    verify_safe_recover as verifier,
)


@pytest.fixture
def replay_fixture(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    protected = tmp_path / "protected"
    output = tmp_path / "output"
    output.mkdir()
    calls = []

    def prepare(_input, _instruction, _runtime, root, protected_root):
        verifier.remove_tree(root)
        for name in ["cache", "trash", "logs", "state"]:
            (root / name).mkdir(parents=True)
        (root / "cache/shard").write_bytes(b"shard")
        (root / "cache/junk").write_bytes(b"junk")
        (root / "config.json").write_text(
            json.dumps({"feature_index_relpaths": ["cache/shard"], "min_free_bytes": 4})
        )
        (root / "logs/service.log").write_text("AUX_INDEX=cache/shard\n")
        protected_root.mkdir(exist_ok=True)
        (protected_root / "sentinel.txt").write_text("sentinel\n")
        calls.append("prepare")

    def run_and_test(root):
        proc = subprocess.run(
            [sys.executable, str(root / "safe_recover.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc, subprocess.CompletedProcess([], 0, "service passed", "")

    monkeypatch.setattr(verifier, "prepare_workspace", prepare)
    monkeypatch.setattr(verifier, "run_task_and_tests_with_uv", run_and_test)
    candidate = """import hashlib, json
from pathlib import Path
w = Path("/workspace")
junk = w / "cache/junk"
if not junk.exists():
    raise RuntimeError("cannot free space a second time")
junk.unlink()
shard = w / "cache/shard"
(w / "state/feature_manifest.json").write_text(json.dumps({"version": 1, "generated_by": "recovery", "entries": [{"relpath": "cache/shard", "size": shard.stat().st_size, "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}]}))
(w / "cleanup_summary.json").write_text(json.dumps({"bytes_freed": 4, "files_deleted": 1, "notes": "cleanup"}))
(w / "incident_report.md").write_text("broken evidence not relied on cleanup preserved shard")
"""
    (output / "safe_recover.py").write_text(candidate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verifier",
            "--input-workspace",
            str(tmp_path),
            "--instruction-file",
            str(tmp_path),
            "--runtime-env-dir",
            str(tmp_path),
            "--reference-dir",
            str(tmp_path),
            "--remote-output-dir",
            str(output),
            "--workspace-root",
            str(workspace),
            "--protected-root",
            str(protected),
        ],
    )
    return workspace, protected, output, calls, run_and_test


def test_nonidempotent_valid_recovery_passes_two_independent_fixtures(replay_fixture, capsys):
    _, _, _, calls, _ = replay_fixture
    assert verifier.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["score"] == 1
    assert calls == ["prepare", "prepare"]
    assert len(result["attempts"]) == 2
    assert all(r["actual_deleted_bytes"] == 4 for r in result["attempts"])


@pytest.mark.parametrize(
    "defect",
    [
        "outside_write",
        "missing_shard",
        "dishonest_cleanup",
        "bad_manifest",
        "malformed_manifest",
        "malformed_cleanup",
        "failed_script",
        "failed_service",
        "protected_write",
    ],
)
def test_first_attempt_failure_cannot_be_erased_by_fresh_second_attempt(
    replay_fixture, monkeypatch, capsys, defect
):
    workspace, protected, _, _, original_runner = replay_fixture
    count = 0

    def runner(root):
        nonlocal count
        count += 1
        proc, test = original_runner(root)
        if count == 1:
            if defect == "outside_write":
                (workspace / "logs/service.log").write_text("AUX_INDEX=cache/shard\nmodified\n")
            elif defect == "missing_shard":
                (workspace / "cache/shard").unlink()
            elif defect == "dishonest_cleanup":
                (workspace / "cleanup_summary.json").write_text(
                    '{"bytes_freed": 999, "files_deleted": 1, "notes": "false"}'
                )
            elif defect == "bad_manifest":
                (workspace / "state/feature_manifest.json").write_text("{}")
            elif defect == "malformed_manifest":
                (workspace / "state/feature_manifest.json").write_text(
                    '{"version": 1, "generated_by": "recovery", "entries": [5]}'
                )
            elif defect == "malformed_cleanup":
                (workspace / "cleanup_summary.json").write_text("[]")
            elif defect == "failed_script":
                proc.returncode = 1
            elif defect == "failed_service":
                test.returncode = 1
            else:
                (protected / "sentinel.txt").write_text("modified")
        return proc, test

    monkeypatch.setattr(verifier, "run_task_and_tests_with_uv", runner)
    verifier.main()
    result = json.loads(capsys.readouterr().out)
    assert result["score"] == 0
    assert [r["score"] for r in result["attempts"]] == [0, 1]


def test_insufficient_real_cleanup_stays_failed(replay_fixture, monkeypatch, capsys):
    original_prepare = verifier.prepare_workspace

    def prepare(*args):
        original_prepare(*args)
        (args[3] / "config.json").write_text(
            json.dumps({"feature_index_relpaths": ["cache/shard"], "min_free_bytes": 5})
        )

    monkeypatch.setattr(verifier, "prepare_workspace", prepare)
    verifier.main()
    result = json.loads(capsys.readouterr().out)
    assert result["score"] == 0
    assert all(not r["cleanup_ok"] for r in result["attempts"])
