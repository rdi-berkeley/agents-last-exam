import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.engineering.inner_support_elevation_optimization import main, native_task
from tasks.engineering.inner_support_elevation_optimization.scripts import native_resources
from tasks.engineering.inner_support_elevation_optimization.verification import (
    EvaluationUnavailableError,
)


def resource_receipt(resource):
    return {
        "completed": False,
        "failure_kind": "resource_exhausted",
        "exhausted_resource": resource,
        "runtime_budget_s": 6600,
        "elapsed_s": 6600 if resource == "wall_time" else 5,
        "attempts": [
            {
                "worker_exit_code": -9,
                "oom_kills": 1 if resource == "worker_memory" else 0,
                "memory_limit_bytes": 4 * 1024**3,
            }
        ],
        "error": "Published native replay resource limit exceeded",
    }


@pytest.fixture
def host_replay(monkeypatch):
    evidence = Path("/vm/evaluator/review")
    control = SimpleNamespace(
        evidence=evidence,
        review={"replay_eligible": True, "evidence_directory": str(evidence)},
        metadata={"remote_output_dir": "/submission"},
        receipt=resource_receipt("wall_time"),
        cpus=[0, 1, 2, 3, 4],
        status={"MainPID": "0", "ExecMainCode": "1", "ExecMainStatus": "1", "Result": "exit-code"},
        commands=[],
        reads=[],
        files={str(evidence / "native-review.json"): b"{}"},
        uploads=[],
    )

    async def upload(session, root, files):
        assert session is control.session
        control.uploads.append((str(root), set(files)))
        for name, content in files.items():
            assert isinstance(content, bytes)
            control.files[str(Path(root) / name)] = content

    async def command(session, command, **kwargs):
        arguments = shlex.split(command)
        control.commands.append(arguments)
        if arguments[0] == "python3" and arguments[1].endswith("/native_resources.py"):
            return {"return_code": 0, "stdout": json.dumps({"cpus": control.cpus})}
        if arguments[:3] == ["sudo", "-n", "systemd-run"]:
            return {"return_code": 0, "stdout": ""}
        assert arguments[:2] == ["systemctl", "show"]
        return {
            "return_code": 0,
            "stdout": "\n".join(f"{key}={value}" for key, value in control.status.items()),
        }

    async def review(*args):
        return control.review

    async def forbidden(*args, **kwargs):
        pytest.fail("Incomplete native replay reached delivery scoring")

    class Session:
        async def read_file(self, path):
            control.reads.append(path)
            if path.endswith("/family-replay.json"):
                return json.dumps(control.receipt)
            assert path.endswith("/independent_monitors.json")
            return json.dumps({"fixture_case": path.split("/")[-2]})

    control.session = Session()
    monkeypatch.setattr(native_task, "upload_files", upload)
    monkeypatch.setattr(main, "upload_files", upload)
    monkeypatch.setattr(native_task, "run_command", command)
    monkeypatch.setattr(main, "review_submission", review)
    monkeypatch.setattr(main, "evaluate_delivery", forbidden)
    return control


def test_published_replay_resource_limits():
    assert native_resources.REPLAY_SECONDS == 6600
    assert native_resources.REPLAY_WORKERS == 5
    assert native_resources.WORKER_MEMORY_MIB == 4096
    assert native_resources.COORDINATOR_MEMORY_MIB == 4096


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["wall_time", "worker_memory"])
@pytest.mark.parametrize("service_result", ["exit-code", "success"])
async def test_trusted_resource_receipt_raises_typed_failure_and_scores_zero(
    host_replay, resource, service_result
):
    control = host_replay
    control.receipt = resource_receipt(resource)
    control.status["Result"] = service_result
    with pytest.raises(native_resources.ReplayBudgetExceeded) as raised:
        await native_task.replay_submission(control.metadata, control.session, control.review)
    assert raised.value.resource == resource
    assert str(raised.value) == control.receipt["error"]
    assert await main.evaluate(SimpleNamespace(metadata=control.metadata), control.session) == [0.0]
    gate = json.loads(control.files[str(control.evidence / "evaluation-gate.json")])
    assert gate["evaluation_status"] == "resource_budget_exceeded"
    assert gate["exhausted_resource"] == resource
    assert gate["reason"] == control.receipt["error"]
    assert not gate["native_replay_completed"] and not gate["full_task_acceptance"]
    assert (
        json.loads(control.files[str(control.evidence / "family-replay.json")]) == control.receipt
    )
    assert all(path.endswith("/family-replay.json") for path in control.reads)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resource,changes",
    [
        pytest.param("wall_time", {"elapsed_s": 6599.999}, id="before-actual-deadline"),
        pytest.param(
            "wall_time", {"elapsed_s": 0, "native_compute_s": 33000}, id="cpu-is-not-wall"
        ),
        pytest.param("wall_time", {"runtime_budget_s": 6599}, id="shortened-receipt-budget"),
        pytest.param("wall_time", {"runtime_budget_s": None}, id="missing-receipt-budget"),
        pytest.param("wall_time", {"failure_kind": "evaluator_error"}, id="unrelated-failure"),
        pytest.param("wall_time", {"failure_kind": None}, id="unclassified-failure"),
        pytest.param("wall_time", {"exhausted_resource": "cpu"}, id="unknown-resource"),
        pytest.param("worker_memory", {"attempts": []}, id="no-worker-evidence"),
        pytest.param(
            "worker_memory",
            {
                "attempts": [
                    {"worker_exit_code": -9, "oom_kills": 0, "memory_limit_bytes": 4 * 1024**3}
                ]
            },
            id="unknown-sigkill",
        ),
        pytest.param(
            "worker_memory",
            {"attempts": [{"oom_kills": 1, "memory_limit_bytes": 2 * 1024**3}]},
            id="oom-at-smaller-limit",
        ),
        pytest.param(
            "worker_memory",
            {"attempts": [{"oom_kills": 1, "memory_limit_bytes": 8 * 1024**3}]},
            id="oom-at-larger-limit",
        ),
        pytest.param("worker_memory", {"attempts": [{"oom_kills": 1}]}, id="unknown-worker-limit"),
        pytest.param(
            "worker_memory",
            {"attempts": [{"memory_limit_bytes": 4 * 1024**3, "peak_memory_bytes": 4 * 1024**3}]},
            id="memory-peak-without-oom",
        ),
        pytest.param(
            "worker_memory",
            {
                "attempts": [
                    {"oom_kills": 1, "memory_limit_bytes": 2 * 1024**3},
                    {"oom_kills": 0, "memory_limit_bytes": 4 * 1024**3},
                ]
            },
            id="oom-and-limit-must-belong-to-same-attempt",
        ),
        pytest.param(
            "worker_memory",
            {"attempts": [], "resources": {"coordinator_oom_kills": 1}},
            id="coordinator-oom-is-not-worker-oom",
        ),
    ],
)
async def test_unconfirmed_resource_failures_remain_unavailable(host_replay, resource, changes):
    control = host_replay
    control.receipt = {**resource_receipt(resource), **changes}
    with pytest.raises(EvaluationUnavailableError, match="Native replay incomplete"):
        await main.evaluate(SimpleNamespace(metadata=control.metadata), control.session)
    assert str(control.evidence / "evaluation-gate.json") not in control.files
    assert (
        json.loads(control.files[str(control.evidence / "family-replay.json")]) == control.receipt
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["wall_time", "worker_memory"])
@pytest.mark.parametrize(
    "status",
    [
        {"Result": "oom-kill"},
        {"Result": "timeout"},
        {"Result": "signal", "ExecMainCode": "2", "ExecMainStatus": "9"},
        {"Result": "resources"},
        {"Result": ""},
        {"ExecMainCode": "2"},
        {"ExecMainCode": "3"},
        {"ExecMainCode": ""},
    ],
)
async def test_abnormal_service_exit_cannot_become_a_solver_zero(host_replay, resource, status):
    control = host_replay
    control.receipt = resource_receipt(resource)
    control.status.update(status)
    with pytest.raises(EvaluationUnavailableError, match="Native replay incomplete"):
        await main.evaluate(SimpleNamespace(metadata=control.metadata), control.session)
    assert str(control.evidence / "evaluation-gate.json") not in control.files
    live = json.loads(control.files[str(control.evidence / "native-live.json")])
    assert live["status"] == control.status


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["wall_time", "worker_memory"])
@pytest.mark.parametrize("receipt_budget", [120, 6600])
async def test_explicit_shortened_budget_cannot_score_zero(host_replay, resource, receipt_budget):
    control = host_replay
    control.metadata["native_replay_seconds"] = 120
    control.receipt = resource_receipt(resource)
    control.receipt["runtime_budget_s"] = receipt_budget
    with pytest.raises(EvaluationUnavailableError, match="Native replay incomplete"):
        await main.evaluate(SimpleNamespace(metadata=control.metadata), control.session)
    launch = json.loads(control.files[str(control.evidence / "native-launch.json")])
    assert launch["resources"]["runtime_budget_s"] == 120
    assert str(control.evidence / "evaluation-gate.json") not in control.files


@pytest.mark.asyncio
@pytest.mark.parametrize("slots", [0, 1, 4, 6])
async def test_host_refuses_any_allocation_other_than_five_slots(host_replay, slots):
    control = host_replay
    control.cpus = list(range(slots))
    with pytest.raises(EvaluationUnavailableError, match="five published worker slots"):
        await main.evaluate(SimpleNamespace(metadata=control.metadata), control.session)
    assert len(control.commands) == 1
    assert not control.reads
    assert str(control.evidence / "evaluation-gate.json") not in control.files


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [1000, 6600, 6659.999])
async def test_short_deadline_refuses_launch_without_reducing_budget(
    host_replay, monkeypatch, remaining
):
    control = host_replay
    monkeypatch.setattr(native_task, "time", SimpleNamespace(monotonic=lambda: 1000))
    with pytest.raises(EvaluationUnavailableError, match="full native replay budget"):
        await native_task.replay_submission(
            control.metadata, control.session, control.review, deadline=1000 + remaining
        )
    assert len(control.commands) == 1
    assert not control.reads
    assert str(control.evidence / "native-launch.json") not in control.files


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [119, 6601])
async def test_out_of_range_diagnostic_budget_cannot_launch(host_replay, budget):
    control = host_replay
    control.metadata["native_replay_seconds"] = budget
    with pytest.raises(EvaluationUnavailableError, match="Native replay budget"):
        await main.evaluate(SimpleNamespace(metadata=control.metadata), control.session)
    assert len(control.commands) == 1
    assert str(control.evidence / "evaluation-gate.json") not in control.files


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [120, 6600])
async def test_successful_replay_reads_all_cases_at_exact_launch_deadline(
    host_replay, monkeypatch, budget
):
    control = host_replay
    control.metadata["native_replay_seconds"] = budget
    control.receipt = {
        "completed": True,
        "runtime_budget_s": budget,
        "cases": {case: {"directory": f"/replay/{case}"} for case in native_task.CASES},
    }
    control.status.update(ExecMainStatus="0", Result="success")
    monkeypatch.setattr(native_task, "time", SimpleNamespace(monotonic=lambda: 1000))
    replay = await native_task.replay_submission(
        control.metadata, control.session, control.review, deadline=1000 + budget + 60
    )
    assert replay["replay"] == control.receipt
    assert replay["observations"] == {case: {"fixture_case": case} for case in native_task.CASES}
    assert len(control.reads) == 6
    assert (
        json.loads(control.files[str(control.evidence / "native-observations.json")])
        == replay["observations"]
    )
    launch = json.loads(control.files[str(control.evidence / "native-launch.json")])
    assert launch["resources"]["runtime_budget_s"] == budget
    assert launch["remote"] == str(control.evidence.parent)
    assert str(control.evidence.parent / "family") in launch["command"]
    assert str(control.evidence.parent / "review.json") in launch["command"]
    assert control.uploads[0][0] == str(control.evidence.parent)
    assert all(name.startswith(("scripts/", "assets/")) for name in control.uploads[0][1])
    assert all(root == str(control.evidence) for root, _ in control.uploads[1:])
