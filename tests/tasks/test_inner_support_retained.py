import hashlib
import json
import shlex
import subprocess

import pytest

from tasks.engineering.inner_support_elevation_optimization import native_task
from tasks.engineering.inner_support_elevation_optimization.verification import (
    CASES,
    EvaluationUnavailableError,
)


class LocalSession:
    async def read_bytes(self, path):
        from pathlib import Path

        return Path(path).read_bytes()

    async def read_file(self, path):
        from pathlib import Path

        return Path(path).read_text()

    async def run_command(self, command, *, check=True):
        arguments = shlex.split(command)
        assert arguments[:2] == ["python3", "-c"]
        assert "systemd" not in command and "native_family_runner" not in command
        result = subprocess.run(arguments, text=True, capture_output=True)
        return {"return_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


@pytest.fixture
def retained(tmp_path):
    evidence = tmp_path / "review"
    evidence.mkdir()
    (tmp_path / "family").mkdir()
    (tmp_path / "family/family.json").write_text(json.dumps({"engineering_choices": {}}))
    remote = tmp_path / "remote"
    (remote / "replay").mkdir(parents=True)
    review = {"input_hashes": {"model.mdpa": "a"}, "monitors": {}, "wall_range_z_m": [-26.5, 7.0]}
    for path in (evidence / "native-review.json", remote / "review.json"):
        path.write_text(json.dumps(review))
    replay = {
        "completed": True,
        "input_hashes": review["input_hashes"],
        "cases": {},
        "cumulative_native_compute_s": 7200.0,
    }
    descriptor = {
        "format": "support-evaluator-retained-replay-1",
        "remote": str(remote),
        "input_hashes": review["input_hashes"],
        "observation_sha256": {},
    }
    for case in CASES:
        directory = remote / case
        directory.mkdir()
        fields = directory / "field.tsv"
        fields.write_text("native test fixture; no physical accuracy claimed")
        observation = directory / "independent_monitors.json"
        observation.write_text(json.dumps({"case": case}))
        descriptor["observation_sha256"][case] = hashlib.sha256(
            observation.read_bytes()
        ).hexdigest()
        terminal = {
            "case": case,
            "native_exit_code": 0,
            "input_hashes": review["input_hashes"],
            "source_fact_audit": {
                "passed": True,
                "audit": {"support_sections": {"version": 1, "passed": True}},
            },
            "extraction": {
                "native_replay_and_reader_completed": True,
                "native_hashes": {"field.tsv": hashlib.sha256(fields.read_bytes()).hexdigest()},
            },
        }
        terminal_path = directory / "terminal.json"
        terminal_path.write_text(json.dumps(terminal))
        replay["cases"][case] = {
            "directory": str(directory),
            "terminal_sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
        }
    replay_path = remote / "replay/family-replay.json"
    replay_path.write_text(json.dumps(replay))
    descriptor["replay_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    descriptor_path = tmp_path / "operator-descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor))
    return descriptor_path, remote, evidence


@pytest.mark.asyncio
async def test_retained_replay_reads_all_five_bound_cases_without_launching_physics(retained):
    descriptor, _, evidence = retained
    result = await native_task.replay_submission(
        {"_evaluator_retained_replay": str(descriptor)},
        LocalSession(),
        {"replay_eligible": True, "evidence_directory": str(evidence)},
    )
    assert set(result["observations"]) == set(CASES)
    receipt = json.loads((evidence / "retained-replay-verification.json").read_text())
    assert not receipt["native_histories_replayed"]
    assert receipt["prior_cumulative_native_compute_s"] == 7200
    assert not receipt["fresh_evaluation_runtime_validated"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["field", "observation", "terminal", "receipt", "review"])
async def test_changed_or_differently_reviewed_retained_state_is_unavailable(retained, mutation):
    descriptor, remote, evidence = retained
    if mutation == "review":
        path = evidence / "native-review.json"
        data = json.loads(path.read_text())
        data["input_hashes"] = {"model.mdpa": "b"}
        path.write_text(json.dumps(data))
    else:
        path = (
            remote
            / {
                "field": "pos_2m/field.tsv",
                "observation": "pos_2m/independent_monitors.json",
                "terminal": "pos_2m/terminal.json",
                "receipt": "replay/family-replay.json",
            }[mutation]
        )
        path.write_text(path.read_text() + " ")
    with pytest.raises(EvaluationUnavailableError):
        await native_task.read_retained_replay(
            descriptor,
            LocalSession(),
            {"replay_eligible": True, "evidence_directory": str(evidence)},
        )
    assert not (evidence / "retained-replay-verification.json").exists()


@pytest.mark.asyncio
async def test_retained_native_positive_cannot_bypass_unresolved_source_gate(retained):
    descriptor, _, evidence = retained
    with pytest.raises(EvaluationUnavailableError):
        await native_task.read_retained_replay(
            descriptor,
            LocalSession(),
            {"replay_eligible": False, "evidence_directory": str(evidence)},
        )


@pytest.mark.asyncio
async def test_legacy_audit_cannot_reuse_section_contaminated_history(retained):
    descriptor, remote, evidence = retained
    terminal_path = remote / "pos_0m/terminal.json"
    terminal = json.loads(terminal_path.read_text())
    del terminal["source_fact_audit"]["audit"]["support_sections"]
    terminal_path.write_text(json.dumps(terminal))
    replay_path = remote / "replay/family-replay.json"
    replay = json.loads(replay_path.read_text())
    replay["cases"]["pos_0m"]["terminal_sha256"] = hashlib.sha256(
        terminal_path.read_bytes()
    ).hexdigest()
    replay_path.write_text(json.dumps(replay))
    data = json.loads(descriptor.read_text())
    data["replay_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    descriptor.write_text(json.dumps(data))
    with pytest.raises(EvaluationUnavailableError, match="role-specific section audit"):
        await native_task.replay_submission(
            {"_evaluator_retained_replay": str(descriptor)},
            LocalSession(),
            {"replay_eligible": True, "evidence_directory": str(evidence)},
        )
    assert not (evidence / "retained-replay-verification.json").exists()
