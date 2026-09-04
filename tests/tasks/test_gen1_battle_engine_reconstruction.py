"""End-to-end checks for computing_math/gen1_battle_engine_reconstruction.

These drive the task's real ``start()`` and ``evaluate()`` hooks against a local
session that runs bash and keeps files on disk, so staging, the leak check, the
on-VM runner and the host-side grader are all exercised without a VM.

The positive control needs a submission that is known correct. The reference
engine is a Zig binary, so the control here is a thin Python wrapper around it,
used only as a test fixture and never shipped to a VM. It proves the grading path
scores a correct engine at 1.0; it says nothing about solvability.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import types

import pytest

from tasks.computing_math.gen1_battle_engine_reconstruction import main as task
from tasks.computing_math.gen1_battle_engine_reconstruction.scripts import grade

ORACLE = pathlib.Path("/tmp/galaxy_srv_disk00/pengchx3/pkmn-spike/oracle-small/bin/oracle")

SHIM = '''
import json, subprocess, sys, tempfile, pathlib
sc = json.loads(pathlib.Path(sys.argv[1]).read_text())
with tempfile.NamedTemporaryFile("w", suffix=".hex", delete=False) as fh:
    fh.write(sc["tape"]); tape = fh.name
out = subprocess.run(["{oracle}", tape, sc["p1"], sc["p2"], str(sc["cap"])],
                     capture_output=True, text=True)
sys.stdout.write(out.stdout)
'''


class LocalSession:
    async def run_command(self, command, *, check=False, timeout=None):
        p = subprocess.run(  # noqa: ASYNC221 - the fake session is deliberately blocking
            ["bash", "-c", command], capture_output=True, text=True,
            timeout=timeout, check=False)
        if check and p.returncode != 0:
            raise RuntimeError(f"{command}\n{p.stderr[:400]}")
        return {"stdout": p.stdout, "stderr": p.stderr, "return_code": p.returncode}

    async def write_file(self, path, content):
        t = pathlib.Path(path)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(content, encoding="utf-8")

    async def read_file(self, path):
        return pathlib.Path(path).read_text(encoding="utf-8")

    async def file_exists(self, path):
        return pathlib.Path(path).is_file()


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.setattr(task, "EVAL_DIR", str(tmp_path / "eval"))
    cfg = task.TaskConfig(REMOTE_ROOT_DIR=str(tmp_path / "root"))
    tc = types.SimpleNamespace(metadata=cfg.to_metadata())
    asyncio.run(task.start(tc, LocalSession()))
    return tc


def test_start_stages_the_visible_corpus(staged):
    input_dir = pathlib.Path(staged.metadata["input_dir"])
    for rel in task.INPUT_FILES:
        assert (input_dir / rel).is_file(), rel
    scen = json.loads((input_dir / "corpus" / "scenarios.json").read_text())
    exp = json.loads((input_dir / "corpus" / "expected.json").read_text())
    assert len(scen) == len(exp) and len(scen) > 100
    assert all(isinstance(s["tape"], str) and len(s["tape"]) > 100 for s in scen), \
        "a visible scenario does not carry its roll tape inline"


def test_no_held_out_transcript_reaches_the_vm(staged):
    root = pathlib.Path(staged.metadata["task_dir"])
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                     for p in root.rglob("*") if p.is_file())
    leaked = [sid for sid in task._EXPECTED if sid in text]
    assert not leaked, f"held-out scenario ids present on the VM: {leaked[:5]}"
    sample = next(iter(task._EXPECTED.values()))
    assert sample["state"] not in text, "a held-out final state is on the VM"


def test_graded_scenarios_have_the_same_shape_as_visible_ones():
    """Grading against a shape the agent never saw is the harness lying."""
    vis = json.loads((task.DATA / "visible" / "scenarios.json").read_text())
    graded = [json.loads(t) for t in task._holdout_files().values()]
    assert {frozenset(s) for s in vis} == {frozenset(s) for s in graded}
    for s in graded:
        assert isinstance(s["tape"], str) and len(s["tape"]) == len(vis[0]["tape"])


def test_every_graded_mechanic_is_demonstrated():
    """A held-out family with no visible example would be unsolvable, not hard."""
    vis = json.loads((task.DATA / "visible" / "scenarios.json").read_text())
    seen = {grade.family_of(s["id"]) for s in vis}
    graded = {grade.family_of(s) for s in task._EXPECTED}
    assert not (graded - seen), f"graded but never demonstrated: {sorted(graded - seen)}"


def test_missing_submission_scores_zero(staged):
    assert asyncio.run(task.evaluate(staged, LocalSession())) == [0.0]


def test_grader_localises_a_broken_mechanic():
    """The point of per-family scoring: name the defect, do not just score it."""
    good = {sid: {"rc": 0, "stdout": "".join(u + "\n" for u in v["updates"])
                  + f"state {v['state']}\n"}
            for sid, v in task._EXPECTED.items()}
    target = grade.family_of(next(iter(task._EXPECTED)))
    broken = {sid: (r if grade.family_of(sid) != target else {"rc": 0, "stdout": ""})
              for sid, r in good.items()}
    rep = grade.score(broken, task._EXPECTED)
    assert rep["full_pass"] == 0.0
    assert rep["broken"] == [target]
    assert rep["mechanics"] > 0.9, "one broken family should not collapse the score"


def test_a_self_contained_engine_reaches_the_deliverable_form(staged):
    """The control the oracle shim cannot give: no oracle, just a Python file.

    The task asks for a single self-contained Python file, and the positive control
    above satisfies that shape only by shelling out to the Zig reference, which does
    not exist on a VM. This submits an engine that models one family outright, with
    its damage base, critical-hit rate and roll position fitted on visible scenarios
    alone. It reproduces every held-out scenario of that family byte for byte, which
    is what shows the required artefact can produce exact output at all.
    """
    sub = pathlib.Path(staged.metadata["submission_path"])
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_text((task.TASK_DIR / "assets" / "partial_engine.py").read_text(encoding="utf-8"),
                   encoding="utf-8")
    score = asyncio.run(task.evaluate(staged, LocalSession()))[0]
    assert score > 0.0, "a self-contained Python engine scored nothing"
    assert score == pytest.approx(1 / 47, abs=0.005), score


@pytest.mark.skipif(not ORACLE.exists(), reason="reference oracle not built on this host")
def test_reference_engine_scores_one_through_the_harness(staged):
    sub = pathlib.Path(staged.metadata["submission_path"])
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_text(SHIM.replace("{oracle}", str(ORACLE)), encoding="utf-8")
    assert asyncio.run(task.evaluate(staged, LocalSession())) == [1.0]
