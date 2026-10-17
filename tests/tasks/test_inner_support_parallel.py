import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "tasks/engineering/inner_support_elevation_optimization/scripts"
)
FIXTURE_SCRIPT = """import argparse,hashlib,json,os,sys,time
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('family')
parser.add_argument('output',type=Path)
parser.add_argument('review',type=Path)
parser.add_argument('--position',type=int)
parser.add_argument('--runtime')
parser.add_argument('--max-iterations')
parser.add_argument('--restart',type=Path)
parser.add_argument('--until',default='full')
args=parser.parse_args()
started=time.monotonic()
args.output.mkdir()
(args.output/'restart').mkdir()
inputs=json.loads(args.review.read_text())['input_hashes']
mode=os.environ.get('SUPPORT_FIXTURE_MODE','cap')
capped=args.position==0 and args.restart is None and mode=='cap'
failed=mode=='audit-fail' and args.position==1
if mode=='missing-terminal' and args.position==0:
    time.sleep(0.06)
    sys.exit(9)
if mode=='hang' or (mode in ('worker-oom','unknown-sigkill') and args.until=='full'):
    (args.output/'ready').write_text('ready')
    if mode!='hang' and args.position==0:
        wait_deadline=time.monotonic()+5
        while not (args.output.parent/'pos_1m-attempt0/ready').exists():
            if time.monotonic()>=wait_deadline:
                sys.exit(8)
            time.sleep(0.01)
        os.kill(os.getpid(),9)
    if mode in ('hang','worker-oom'):
        time.sleep(30)
if args.until=='full':
    assert args.restart.with_suffix('.rest').read_bytes()==b'common-native-state'
time.sleep(0.06 if failed else 0.30)
state=b'prefix-capped' if capped else (b'common-native-state' if args.position==0 else str(args.position).encode())
metadata={'label':'stage_4_excavate_1' if capped else 'stage_7_excavate_1','native_state_sha256':hashlib.sha256(state).hexdigest(),'input_hashes':inputs}
for name in ('latest','branch'):
    (args.output/'restart'/name).with_suffix('.rest').write_bytes(state)
    (args.output/'restart'/name).with_suffix('.json').write_text(json.dumps(metadata))
if mode=='mutate-common' and args.position==1:
    args.restart.with_suffix('.rest').write_bytes(b'changed')
terminal={'native_exit_code':124 if capped else 0,'elapsed_s':2 if capped else 10,'common_stage7_ready':args.until=='stage7','source_fact_audit':{'passed':not failed},'extraction':{'native_replay_and_reader_completed':True},'pid':os.getpid(),'started':started,'finished':time.monotonic(),'restart':str(args.restart)}
if failed:
    terminal['source_audit_error']='Wrong native wall thickness'
(args.output/'terminal.json').write_text(json.dumps(terminal))
sys.exit(124 if capped else (1 if failed else 0))
"""


class FixtureWorkers:
    def __init__(self, clock=time.monotonic):
        self.cpus = (0, 1)
        self.clock = clock
        self.started = []
        self.finished = []
        self.stopped = []

    def start(self, command, destination, cpu):
        process = subprocess.Popen(command)
        worker = {"process": process, "cpu": cpu, "started": self.clock()}
        self.started.append(worker)
        return worker

    def finish(self, worker):
        self.finished.append(worker)
        return {
            "cpu_s": 0.25,
            "peak_memory_bytes": 1024,
            "memory_limit_bytes": 4 * 1024**3,
            "oom_kills": 0,
        }

    def stop(self, worker):
        self.stopped.append(worker)
        worker["process"].kill()
        worker["process"].wait()

    def aggregate_usage(self):
        return {"fixture_only": True, "aggregate_cpu_s": 0.25 * len(self.finished)}


@pytest.fixture
def family_control(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import native_replay_family as replay

    script = tmp_path / "case.py"
    script.write_text(FIXTURE_SCRIPT)
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"input_hashes": {"family": "fixture"}}))
    monkeypatch.setattr(replay, "verify_review", lambda *args: None)
    audits = []

    def audit(family, checkpoint, review, output):
        audits.append(checkpoint)
        assert checkpoint.with_suffix(".rest").read_bytes() == b"common-native-state"
        return {
            "native_state_sha256": hashlib.sha256(b"common-native-state").hexdigest(),
            "common_stage7_ready": True,
        }

    monkeypatch.setattr(replay, "verify_common_stage7", audit)
    return replay, script, review, audits


def test_native_family_continues_caps_and_branches_only_verified_common_state(
    family_control, tmp_path
):
    replay, script, review, audits = family_control
    workers = FixtureWorkers()
    result = replay.replay_family(
        tmp_path / "family", tmp_path / "output", review, workers=workers, case_script=script
    )
    assert result["completed"] and len(result["cases"]) == 5
    assert len(audits) == 1
    assert result["native_compute_s"] == 62
    assert result["worker_cpu_s"] == 1.75
    assert len(result["attempts"]) == 7
    assert result["attempts"][1]["restart"].endswith("common-prefix-attempt0/restart/latest")
    assert (
        Path(result["immutable_common_checkpoint"]["path"]).with_suffix(".rest").read_bytes()
        == b"common-native-state"
    )
    rows = [
        json.loads((Path(row["directory"]) / "terminal.json").read_text())
        for row in result["attempts"]
        if row["scope"] == "full"
    ]
    assert len({row["pid"] for row in rows}) == 5
    assert len({row["restart"] for row in rows}) == 1
    assert rows[0]["started"] < rows[1]["finished"] and rows[1]["started"] < rows[0]["finished"]
    events = sorted([(row["started"], 1) for row in rows] + [(row["finished"], -1) for row in rows])
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(active, peak)
    assert peak == 2
    for position in range(1, 5):
        case = Path(result["cases"][f"pos_{position}m"]["directory"])
        assert (case / "restart/latest.rest").read_bytes() == str(position).encode()


def test_all_five_branches_start_together_after_prefix(family_control, tmp_path):
    replay, script, review, _ = family_control
    workers = FixtureWorkers()
    workers.cpus = (0, 1, 2, 3, 4)
    result = replay.replay_family(
        tmp_path / "family", tmp_path / "output", review, workers=workers, case_script=script
    )
    prefixes = [row for row in result["attempts"] if row["scope"] == "stage7"]
    branches = [row for row in result["attempts"] if row["scope"] == "full"]
    terminals = [
        json.loads((Path(row["directory"]) / "terminal.json").read_text()) for row in branches
    ]
    prefix_terminal = json.loads((Path(prefixes[-1]["directory"]) / "terminal.json").read_text())
    assert len(branches) == 5
    assert {row["position"] for row in branches} == set(range(5))
    assert {row["cpu"] for row in branches} == set(workers.cpus)
    assert min(row["started"] for row in terminals) >= prefix_terminal["finished"]
    assert max(row["started"] for row in terminals) < min(row["finished"] for row in terminals)
    assert {row["restart"] for row in branches} == {result["immutable_common_checkpoint"]["path"]}
    assert result["native_compute_s"] == 62


def test_native_audit_failure_preserves_compute_and_finishes_inflight_worker(
    family_control, tmp_path, monkeypatch
):
    replay, script, review, _ = family_control
    monkeypatch.setenv("SUPPORT_FIXTURE_MODE", "audit-fail")
    workers = FixtureWorkers()
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="wall thickness"):
        replay.replay_family(
            tmp_path / "family", output, review, workers=workers, case_script=script
        )
    result = json.loads((output / "family-replay.json").read_text())
    assert not result["completed"]
    assert {row["position"] for row in result["attempts"]} == {0, 1}
    assert result["native_compute_s"] == 30
    assert all(worker["process"].poll() is not None for worker in workers.started)


def test_changed_common_state_cannot_yield_a_complete_family(family_control, tmp_path, monkeypatch):
    replay, script, review, _ = family_control
    monkeypatch.setenv("SUPPORT_FIXTURE_MODE", "mutate-common")
    with pytest.raises(RuntimeError, match="Common checkpoint changed"):
        replay.replay_family(
            tmp_path / "family",
            tmp_path / "output",
            review,
            workers=FixtureWorkers(),
            case_script=script,
        )


def test_worker_allocation_rejects_duplicate_or_unallocated_cpus(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from native_parallel import NativeWorkers

    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {1, 2})
    for cpus in ((1, 1), (0, 1), (1, 2, 3)):
        with pytest.raises(ValueError, match="distinct allocated"):
            NativeWorkers(cpus)


def test_missing_terminal_preserves_worker_time_in_failure_ledger(
    family_control, tmp_path, monkeypatch
):
    replay, script, review, _ = family_control
    monkeypatch.setenv("SUPPORT_FIXTURE_MODE", "missing-terminal")
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="worker exit 9"):
        replay.replay_family(
            tmp_path / "family", output, review, workers=FixtureWorkers(), case_script=script
        )
    record = json.loads((output / "family-replay.json").read_text())
    assert record["native_compute_s"] >= 0.06
    assert record["attempts"][0]["elapsed_source"] == "worker_wall_fallback"
    assert record["worker_cpu_s"] == 0.25
    assert not record["active_workers"]


def test_retained_prefix_ledger_counts_capped_time_without_replaying_pos0(family_control, tmp_path):
    replay, script, review, _ = family_control
    checkpoint = tmp_path / "common"
    checkpoint.with_suffix(".rest").write_bytes(b"common-native-state")
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {
                "label": "stage_7_excavate_1",
                "native_state_sha256": hashlib.sha256(b"common-native-state").hexdigest(),
            }
        )
    )
    completed = tmp_path / "completed-pos0"
    completed.mkdir()
    (completed / "native_model.mdpa").write_bytes(b"fixture-native-model")
    (completed / "terminal.json").write_text(
        json.dumps(
            {
                "native_exit_code": 0,
                "case": "pos_0m",
                "input_hashes": {"family": "fixture"},
                "source_fact_audit": {
                    "passed": True,
                    "audit": {"support_sections": {"version": 1, "passed": True}},
                },
                "extraction": {
                    "native_replay_and_reader_completed": True,
                    "native_hashes": {
                        "native_model.mdpa": hashlib.sha256(b"fixture-native-model").hexdigest()
                    },
                },
            }
        )
    )
    ledger = tmp_path / "attempted-time.json"
    ledger.write_text(
        json.dumps(
            [
                {"elapsed_s": 3570, "native_exit_code": 124},
                {"elapsed_s": 434, "native_exit_code": 0},
            ]
        )
    )
    output = tmp_path / "output"
    record = replay.replay_family(
        tmp_path / "family",
        output,
        review,
        common_checkpoint=checkpoint,
        completed_pos0=completed,
        prior_ledger=ledger,
        workers=FixtureWorkers(),
        case_script=script,
    )
    assert record["completed"]
    assert {row["position"] for row in record["attempts"]} == {1, 2, 3, 4}
    assert record["prior_native_compute_s"] == 4004
    assert record["cumulative_native_compute_s"] == 4044
    assert record["cases"]["pos_0m"]["reused_evaluator_owned_result"]
    assert checkpoint.with_suffix(".rest").read_bytes() == b"common-native-state"


def test_retained_prefix_runs_all_five_cases_with_four_slots_and_remaining_budget(
    family_control, tmp_path
):
    replay, script, review, audits = family_control
    checkpoint = tmp_path / "prefix"
    checkpoint.with_suffix(".rest").write_bytes(b"common-native-state")
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {
                "label": "stage_7_excavate_1",
                "native_state_sha256": hashlib.sha256(b"common-native-state").hexdigest(),
            }
        )
    )
    ledger = tmp_path / "prefix-ledger.json"
    ledger.write_text(json.dumps([{"elapsed_s": 2800, "scope": "stage7"}]))
    workers = FixtureWorkers()
    workers.cpus = (0, 1, 2, 3)
    result = replay.replay_family(
        tmp_path / "family",
        tmp_path / "output",
        review,
        total_seconds=3800,
        common_checkpoint=checkpoint,
        prior_ledger=ledger,
        workers=workers,
        case_script=script,
    )
    assert result["completed"] and len(result["cases"]) == 5
    assert len(result["attempts"]) == 5
    assert all(row["scope"] == "full" for row in result["attempts"])
    assert result["runtime_budget_s"] == 6600 - 2800
    assert result["cumulative_native_compute_s"] == 2800 + 50
    assert audits == [checkpoint]
    terminals = [
        json.loads((Path(row["directory"]) / "terminal.json").read_text())
        for row in result["attempts"]
    ]
    events = sorted(
        [(row["started"], 1) for row in terminals] + [(row["finished"], -1) for row in terminals]
    )
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert peak == 4


def test_unaffected_stage2_recovery_builds_new_common_state(family_control, tmp_path):
    replay, script, review, audits = family_control
    checkpoint = tmp_path / "stage2"
    checkpoint.with_suffix(".rest").write_bytes(b"safe-before-supports")
    metadata = {
        "label": "stage_2_outer_wall",
        "native_state_sha256": hashlib.sha256(b"safe-before-supports").hexdigest(),
        "input_hashes": {"family": "fixture"},
        "receipt": {"support_installations": []},
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"elapsed_s": 35}]))
    result = replay.replay_family(
        tmp_path / "family",
        tmp_path / "output",
        review,
        initial_checkpoint=checkpoint,
        prior_ledger=ledger,
        workers=FixtureWorkers(),
        case_script=script,
    )
    assert result["completed"]
    assert result["attempts"][0]["restart"] == str(checkpoint)
    assert result["prior_native_compute_s"] == 35
    assert audits[0] != checkpoint
    assert checkpoint.with_suffix(".rest").read_bytes() == b"safe-before-supports"
    metadata["receipt"]["support_installations"] = [{"label": "stage_3_outer_support_1"}]
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="precede support installation"):
        replay.replay_family(
            tmp_path / "family",
            tmp_path / "invalid-output",
            review,
            initial_checkpoint=checkpoint,
            prior_ledger=ledger,
            workers=FixtureWorkers(),
            case_script=script,
        )


@pytest.fixture
def resource_probe(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from native_resources import replay_resources

    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/parent/child\n")
    (proc / "meminfo").write_text(f"MemAvailable: {30 * 1024**2} kB\n")
    groups = tmp_path / "cgroup"
    (groups / "parent/child").mkdir(parents=True)
    for group in (groups, groups / "parent", groups / "parent/child"):
        (group / "cpu.max").write_text("max 100000")
        (group / "memory.max").write_text("max")
        (group / "memory.current").write_text("0")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2, 4, 6, 8, 10, 12, 14, 16})
    return replay_resources, proc, groups


def test_resource_selection_uses_guest_affinity_and_five_case_limit(resource_probe):
    probe, proc, groups = resource_probe
    assert probe(proc=proc, cgroups=groups)["cpus"] == (2, 4, 6, 8, 10)
    assert probe([14, 16], proc=proc, cgroups=groups)["cpus"] == (14, 16)
    for cpus in ([0], [2, 2], [True], list(range(6))):
        with pytest.raises(ValueError, match="allocation"):
            probe(cpus, proc=proc, cgroups=groups)


def test_four_cpu_guest_reserves_coordinator_and_host_memory(resource_probe, monkeypatch):
    probe, proc, groups = resource_probe
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    (proc / "meminfo").write_text(f"MemAvailable: {13 * 1024**2} kB\n")
    assert probe(proc=proc, cgroups=groups)["cpus"] == (0,)
    with pytest.raises(ValueError, match="headroom"):
        probe([0, 1], proc=proc, cgroups=groups)


def test_resource_selection_respects_ancestor_quota_and_memory_headroom(resource_probe):
    probe, proc, groups = resource_probe
    (groups / "parent/cpu.max").write_text("350000 100000")
    assert probe(proc=proc, cgroups=groups)["cpus"] == (2, 4, 6)
    (groups / "memory.max").write_text(str(17 * 1024**3))
    (groups / "memory.current").write_text(str(3 * 1024**3))
    assert probe(proc=proc, cgroups=groups)["cpus"] == (2, 4)
    with pytest.raises(ValueError, match="headroom"):
        probe([2, 4, 6], proc=proc, cgroups=groups)
    (proc / "meminfo").write_text(f"MemAvailable: {3 * 1024**2} kB\n")
    with pytest.raises(ValueError, match="headroom"):
        probe(proc=proc, cgroups=groups)


@pytest.mark.parametrize(
    "available_kib,slots",
    [
        (10 * 1024**2 - 1, 0),
        (10 * 1024**2, 1),
        (26 * 1024**2 - 1, 4),
        (26 * 1024**2, 5),
    ],
)
@pytest.mark.parametrize("limiter", ["host", "parent", "child"])
def test_worker_slots_require_four_gib_each_after_six_gib_reserve(
    resource_probe, available_kib, slots, limiter
):
    probe, proc, groups = resource_probe
    if limiter == "host":
        (proc / "meminfo").write_text(f"MemAvailable: {available_kib} kB\n")
    else:
        group = groups / ("parent" if limiter == "parent" else "parent/child")
        (group / "memory.max").write_text(str(available_kib * 1024 + 1024**3))
        (group / "memory.current").write_text(str(1024**3))
    if slots == 0:
        with pytest.raises(ValueError, match="headroom"):
            probe(proc=proc, cgroups=groups)
    else:
        result = probe(proc=proc, cgroups=groups)
        assert result["cpus"] == (2, 4, 6, 8, 10)[:slots]
        assert result["available_memory_bytes"] == available_kib * 1024


@pytest.fixture
def case_control(family_control, tmp_path, monkeypatch):
    import native_parallel as parallel

    _, script, review, _ = family_control
    clock = SimpleNamespace(now=1000.25)
    workers = FixtureWorkers(clock=lambda: clock.now)
    monkeypatch.setattr(
        parallel, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=time.sleep)
    )
    output = tmp_path / "cases"
    output.mkdir()
    checkpoint = tmp_path / "case-common"
    checkpoint.with_suffix(".rest").write_bytes(b"common-native-state")
    checkpoint.with_suffix(".json").write_text(
        json.dumps({"native_state_sha256": hashlib.sha256(b"common-native-state").hexdigest()})
    )
    record = {
        "started_monotonic": clock.now,
        "input_hashes": {"family": "fixture"},
        "attempts": [],
        "cases": {},
    }

    def run(seconds, positions=(0,)):
        parallel.run_cases(
            tmp_path / "family",
            output,
            review,
            positions,
            checkpoint,
            record,
            workers,
            clock.now + seconds,
            400,
            case_script=script,
            poll_seconds=0.01,
        )

    yield SimpleNamespace(
        parallel=parallel, clock=clock, workers=workers, output=output, record=record, run=run
    )
    for worker in workers.started:
        if worker["process"].poll() is None:
            worker["process"].kill()
        worker["process"].wait(timeout=5)


@pytest.mark.parametrize(
    "remaining,child_runtime", [(0.1, 1), (119.1, 120), (3570.1, 3571), (6599.25, 6600)]
)
def test_child_gets_ceiling_of_remaining_time_without_early_cutoff_or_cap(
    case_control, remaining, child_runtime
):
    control = case_control
    control.run(remaining)
    command = control.workers.started[0]["command"]
    assert int(command[command.index("--runtime") + 1]) == child_runtime
    assert set(control.record["cases"]) == {"pos_0m"}
    assert control.record["attempts"][0]["worker_exit_code"] == 0
    assert not control.workers.stopped
    assert not control.record["active_workers"]


@pytest.mark.parametrize("remaining", [0, -0.1])
def test_expired_deadline_never_launches_pending_cases(case_control, remaining):
    control = case_control
    with pytest.raises(control.parallel.ReplayBudgetExceeded) as raised:
        control.run(remaining)
    assert raised.value.resource == "wall_time"
    assert not control.workers.started
    ledger = json.loads((control.output / "family-replay.json").read_text())
    assert ledger["attempts"] == []
    assert ledger["native_compute_s"] == 0
    assert ledger["active_workers"] == []


def test_actual_deadline_stops_inflight_workers_and_retains_every_attempt(
    case_control, monkeypatch
):
    control = case_control
    monkeypatch.setenv("SUPPORT_FIXTURE_MODE", "hang")
    deadline = control.clock.now + 120.25
    ticks = []

    def advance_clock(_):
        assert not control.workers.stopped
        assert len(control.workers.started) == 2
        wait_deadline = time.monotonic() + 5
        while not all(
            (worker["destination"] / "ready").exists() for worker in control.workers.started
        ):
            assert time.monotonic() < wait_deadline, "Fixture workers failed to start"
            time.sleep(0.01)
        assert all(worker["process"].poll() is None for worker in control.workers.started)
        ticks.append(control.clock.now)
        assert len(ticks) <= 2, "Coordinator failed to enforce the actual deadline"
        control.clock.now = deadline - 0.01 if len(ticks) == 1 else deadline

    monkeypatch.setattr(control.parallel.time, "sleep", advance_clock)
    with pytest.raises(control.parallel.ReplayBudgetExceeded) as raised:
        control.run(120.25, positions=(0, 1, 2))
    assert raised.value.resource == "wall_time"
    assert len(ticks) == 2
    assert len(control.workers.stopped) == len(control.workers.finished) == 2
    assert all(worker["process"].poll() is not None for worker in control.workers.started)
    ledger = json.loads((control.output / "family-replay.json").read_text())
    assert {row["position"] for row in ledger["attempts"]} == {0, 1}
    assert any(row["coordinator_timeout"] for row in ledger["attempts"])
    assert all(row["elapsed_source"] == "worker_wall_fallback" for row in ledger["attempts"])
    assert ledger["elapsed_s"] == pytest.approx(120.25)
    assert ledger["native_compute_s"] == pytest.approx(240.5)
    assert ledger["worker_cpu_s"] == 0.5
    assert ledger["active_workers"] == []


@pytest.mark.parametrize("confirmed_oom", [True, False], ids=["worker-oom", "unknown-sigkill"])
def test_worker_kill_requires_oom_evidence_and_preserves_family_ledger(
    family_control, tmp_path, monkeypatch, confirmed_oom
):
    replay, script, review, _ = family_control
    monkeypatch.setenv("SUPPORT_FIXTURE_MODE", "worker-oom" if confirmed_oom else "unknown-sigkill")
    workers = FixtureWorkers()
    finish = workers.finish

    def measured(worker):
        usage = finish(worker)
        if confirmed_oom and worker["position"] == 0 and worker["process"].returncode == -9:
            usage["oom_kills"] = 1
        return usage

    monkeypatch.setattr(workers, "finish", measured)
    output = tmp_path / "killed-family"
    try:
        with pytest.raises(RuntimeError) as raised:
            replay.replay_family(
                tmp_path / "family", output, review, workers=workers, case_script=script
            )
        if confirmed_oom:
            assert isinstance(raised.value, replay.ReplayBudgetExceeded)
            assert raised.value.resource == "worker_memory"
            assert [worker["position"] for worker in workers.stopped] == [1]
        else:
            assert not isinstance(raised.value, replay.ReplayBudgetExceeded)
            assert "worker exit -9" in str(raised.value)
            assert not workers.stopped
        assert all(worker["process"].poll() is not None for worker in workers.started)
        record = json.loads((output / "family-replay.json").read_text())
        assert not record["completed"] and not record["active_workers"]
        assert record["failure_kind"] == (
            "resource_exhausted" if confirmed_oom else "evaluator_error"
        )
        assert record.get("exhausted_resource") == ("worker_memory" if confirmed_oom else None)
        assert len(record["attempts"]) == 3
        assert record["worker_cpu_s"] == 0.75
        assert record["native_compute_s"] == sum(row["elapsed_s"] for row in record["attempts"])
        killed = next(
            row for row in record["attempts"] if row["scope"] == "full" and row["position"] == 0
        )
        assert killed["worker_exit_code"] == -9
        assert killed["oom_kills"] == int(confirmed_oom)
        assert killed["memory_limit_bytes"] == 4 * 1024**3
        assert killed["elapsed_source"] == "worker_wall_fallback"
        assert all(Path(row["directory"]).is_dir() for row in record["attempts"])
        assert (
            Path(record["immutable_common_checkpoint"]["path"]).with_suffix(".rest").read_bytes()
            == b"common-native-state"
        )
    finally:
        for worker in workers.started:
            if worker["process"].poll() is None:
                worker["process"].kill()
            worker["process"].wait(timeout=5)


@pytest.mark.asyncio
async def test_task_launch_allocates_available_cpus_and_memory_with_time_left_for_delivery(
    monkeypatch,
):
    import shlex
    from tasks.engineering.inner_support_elevation_optimization import native_task

    evidence = Path("/vm/evaluator/review")
    commands = []
    files = {}
    uploads = []

    async def upload(session, root, payload):
        assert isinstance(session, Session)
        uploads.append((str(root), set(payload)))
        for name, content in payload.items():
            assert isinstance(content, bytes)
            files[str(Path(root) / name)] = content

    class Session:
        async def run_command(self, command, **kwargs):
            wrapper = shlex.split(command)
            assert wrapper[:2] == ["python3", "-c"]
            commands.append(shlex.split(wrapper[3]))
            if commands[-1][1].endswith("native_resources.py"):
                return {
                    "stdout": json.dumps(
                        {
                            "return_code": 0,
                            "stdout": json.dumps({"cpus": [0, 1, 2, 3, 4]}),
                            "stderr": "",
                        }
                    )
                }
            return {
                "return_code": 0,
                "stdout": json.dumps(
                    {"return_code": 0, "stdout": "MainPID=0\nExecMainStatus=0", "stderr": ""}
                ),
            }

        async def read_file(self, path):
            if path.endswith("family-replay.json"):
                return json.dumps(
                    {
                        "completed": True,
                        "cases": {case: {"directory": case} for case in native_task.CASES},
                    }
                )
            return "{}"

    monkeypatch.setattr(native_task, "upload_files", upload)
    replay = await native_task.replay_submission(
        {"remote_output_dir": "/submission"},
        Session(),
        {"evidence_directory": str(evidence)},
        deadline=time.monotonic() + 6910,
    )
    launch = commands[1]
    assert "AllowedCPUs=0 1 2 3 4" in launch and "CPUQuota=500%" in launch
    assert "MemoryMax=24576M" in launch and "Delegate=yes" in launch
    assert "MemorySwapMax=0" in launch
    assert "OOMPolicy=continue" in launch
    assert launch[-6:] == ["--cpus", "0", "1", "2", "3", "4"]
    budget = int(launch[launch.index("--total-seconds") + 1])
    assert budget == 6600
    assert f"RuntimeMaxSec={budget + 30}" in launch
    assert {"native_parallel.py", "native_worker.py"}.issubset(native_task.NATIVE_SCRIPTS)
    receipt = json.loads(files[str(evidence / "native-launch.json")])
    assert receipt["resources"]["worker_memory_mib"] == 4096
    assert receipt["resources"]["coordinator_memory_mib"] == 4096
    assert receipt["resources"]["aggregate_memory_mib"] == 24576
    assert receipt["resources"]["runtime_budget_s"] == 6600
    assert replay["remote"] == receipt["remote"] == str(evidence.parent)
    assert str(evidence.parent / "family") in launch
    assert str(evidence.parent / "review.json") in launch
    assert uploads[0][0] == str(evidence.parent)
    assert all(name.startswith(("scripts/", "assets/")) for name in uploads[0][1])
    assert all(root == str(evidence) for root, _ in uploads[1:])
    assert json.loads(files[str(evidence / "family-replay.json")]) == replay["replay"]
    assert json.loads(files[str(evidence / "native-observations.json")]) == replay["observations"]
