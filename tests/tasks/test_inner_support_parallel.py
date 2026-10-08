import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

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
if args.position:
    assert args.restart.with_suffix('.rest').read_bytes()==b'common-native-state'
time.sleep(0.06 if failed else 0.30)
state=b'prefix-capped' if capped else (b'common-native-state' if args.position==0 else str(args.position).encode())
metadata={'label':'stage_4_excavate_1' if capped else 'stage_7_excavate_1','native_state_sha256':hashlib.sha256(state).hexdigest(),'input_hashes':inputs}
for name in ('latest','branch'):
    (args.output/'restart'/name).with_suffix('.rest').write_bytes(state)
    (args.output/'restart'/name).with_suffix('.json').write_text(json.dumps(metadata))
if mode=='mutate-common' and args.position==1:
    args.restart.with_suffix('.rest').write_bytes(b'changed')
terminal={'native_exit_code':124 if capped else 0,'elapsed_s':2 if capped else 10,'source_fact_audit':{'passed':not failed},'extraction':{'native_replay_and_reader_completed':True},'pid':os.getpid(),'started':started,'finished':time.monotonic(),'restart':str(args.restart)}
if failed:
    terminal['source_audit_error']='Wrong native wall thickness'
(args.output/'terminal.json').write_text(json.dumps(terminal))
sys.exit(124 if capped else (1 if failed else 0))
"""


class FixtureWorkers:
    def __init__(self):
        self.cpus = (0, 1)
        self.started = []
        self.finished = []

    def start(self, command, destination, cpu):
        process = subprocess.Popen(command)
        worker = {"process": process, "cpu": cpu, "started": time.monotonic()}
        self.started.append(worker)
        return worker

    def finish(self, worker):
        self.finished.append(worker)
        return {"cpu_s": 0.25, "peak_memory_bytes": 1024, "oom_kills": 0}

    def stop(self, worker):
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
    assert result["native_compute_s"] == 52
    assert result["worker_cpu_s"] == 1.5
    assert len(result["attempts"]) == 6
    assert result["attempts"][1]["restart"].endswith("pos_0m-attempt0/restart/latest")
    assert (
        Path(result["immutable_common_checkpoint"]["path"]).with_suffix(".rest").read_bytes()
        == b"common-native-state"
    )
    rows = [
        json.loads((Path(row["directory"]) / "terminal.json").read_text())
        for row in result["attempts"]
        if row["position"]
    ]
    assert len({row["pid"] for row in rows}) == 4
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
    assert {row["position"] for row in result["attempts"]} == {0, 1, 2}
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


@pytest.mark.asyncio
async def test_task_launch_allocates_both_cpus_and_memory_with_time_left_for_delivery(
    monkeypatch, tmp_path
):
    import shlex
    from tasks.engineering.inner_support_elevation_optimization import native_task

    evidence = tmp_path / "review"
    evidence.mkdir()
    (evidence / "native-review.json").write_text("{}")
    (tmp_path / "family").mkdir()
    for name in ("family.json", "model.mdpa", "Materials.json"):
        (tmp_path / "family" / name).write_text("fixture")
    (tmp_path / "family/family.json").write_text(json.dumps({"engineering_choices": {}}))
    commands = []

    async def upload(*args):
        return None

    class Session:
        async def run_command(self, command, **kwargs):
            wrapper = shlex.split(command)
            assert wrapper[:2] == ["python3", "-c"]
            commands.append(shlex.split(wrapper[3]))
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
    await native_task.replay_submission(
        {"remote_output_dir": "/submission"},
        Session(),
        {"evidence_directory": str(evidence)},
        deadline=time.monotonic() + 1000,
    )
    launch = commands[0]
    assert "AllowedCPUs=1 2" in launch and "CPUQuota=200%" in launch
    assert "MemoryMax=5120M" in launch and "Delegate=yes" in launch
    assert "MemorySwapMax=0" in launch
    assert launch[-3:] == ["--cpus", "1", "2"]
    budget = int(launch[launch.index("--total-seconds") + 1])
    assert 900 <= budget <= 940
    assert f"RuntimeMaxSec={budget + 30}" in launch
    assert {"native_parallel.py", "native_worker.py"}.issubset(native_task.NATIVE_SCRIPTS)
