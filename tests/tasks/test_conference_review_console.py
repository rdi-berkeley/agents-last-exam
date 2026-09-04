"""End-to-end and grader tests for education_info/conference_review_console.

Boots the real console server (subprocess, temp data copy, temp state dir),
drives the full enforced workflow over HTTP (screen -> bulk import -> checklist
-> finalize) using the reference truth, and asserts the grader awards 1.0.
Then exercises server-side gates and grader branches.
"""
from __future__ import annotations

import csv
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from tasks.education_info.conference_review_console.scripts.score_outputs import (
    score_console,
)

DATA = (
    Path(__file__).resolve().parents[2]
    / "task-data" / "education_info" / "conference_review_console" / "base"
)
SERVER = DATA / "software" / "server.py"

pytestmark = pytest.mark.skipif(
    not (DATA / "software" / "console_data.json").exists(),
    reason="task data not generated (run the generator first)",
)


class Client:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"
        self.cookie = ""

    def req(self, path: str, data: dict | None = None) -> tuple[int, str]:
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        r = urllib.request.Request(self.base + path, data=body)
        if self.cookie:
            r.add_header("Cookie", self.cookie)
        opener = urllib.request.build_opener(NoRedirect)
        try:
            resp = opener.open(r, timeout=10)
        except urllib.error.HTTPError as e:
            resp = e
        if resp.headers.get("Set-Cookie"):
            self.cookie = resp.headers["Set-Cookie"].split(";")[0]
        return resp.status, resp.read().decode(errors="replace")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


@pytest.fixture(scope="module")
def world():
    return json.loads((DATA / "reference" / "effective_world.json").read_text())


@pytest.fixture(scope="module")
def params():
    return json.loads((DATA / "reference" / "scoring.json").read_text())


def boot(tmp: Path):
    data_copy = tmp / "console_data.json"
    shutil.copy(DATA / "software" / "console_data.json", data_copy)
    state_dir = tmp / "state"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, str(SERVER), "--data", str(data_copy),
         "--state-dir", str(state_dir), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cli = Client(port)
    for _ in range(50):
        try:
            status, _ = cli.req("/healthz")
            if status == 200:
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("console did not boot")
    return cli, state_dir, data_copy, proc


@pytest.fixture(scope="module")
def console(tmp_path_factory):
    cli, state_dir, data_copy, proc = boot(tmp_path_factory.mktemp("console"))
    yield cli, state_dir, data_copy
    proc.kill()


def state_of(state_dir: Path) -> str:
    return (state_dir / "state.json").read_text()


def grade(state_json, audit, world, params):
    return score_console(state_json=state_json, audit_csv=audit,
                         effective_world=world, scoring_params=params)


def test_full_workflow_scores_one(console, world, params):
    cli, state_dir, data_copy = console
    assert not data_copy.exists(), "server must delete its data file at boot"

    status, _ = cli.req("/papers")
    assert status == 302, "unauthenticated access must redirect"
    cli.req("/login", {"user": "chair", "password": "prc-2026"})
    status, body = cli.req("/")
    assert status == 200 and "Dashboard" in body

    papers = {p["paper_id"]: p for p in world["papers"]}
    active = [pid for pid, p in papers.items() if p["status"] == "active"]

    # gate: import before screening is rejected
    rows = [r for r in csv.reader(
        (DATA / "reference" / "optimal_assignment.csv").open())][1:]
    status, _ = cli.req("/assignments/bulk",
                        {"tsv": f"{rows[0][0]},{rows[0][1]}"})
    _, report = cli.req("/assignments?report=x")
    st = json.loads(state_of(state_dir))
    assert st["assignments"] == {}, "unscreened paper must reject imports"

    # gate: wrong screening rejected
    pid0 = active[0]
    cli.req(f"/paper/{pid0}/screen",
            {"primary_area": papers[pid0]["primary_area"],
             "author_count": str(papers[pid0]["author_count"] + 1)})
    st = json.loads(state_of(state_dir))
    assert pid0 not in st["screenings"]

    for pid in active:
        cli.req(f"/paper/{pid}/screen",
                {"primary_area": papers[pid]["primary_area"],
                 "author_count": str(papers[pid]["author_count"])})
    st = json.loads(state_of(state_dir))
    assert len(st["screenings"]) == len(active)

    tsv = "\n".join(f"{p},{r}" for p, r in rows)
    cli.req("/assignments/bulk", {"tsv": tsv})
    st = json.loads(state_of(state_dir))
    assert sum(len(v) for v in st["assignments"].values()) == len(rows)

    # gate: finalize blocked before checklist
    cli.req("/finalize", {"confirm": "1"})
    st = json.loads(state_of(state_dir))
    assert not st["finalized"]

    # wrong checklist: not passed, count reported
    answers = {f"q{i + 1}": str(v) for i, v in enumerate(world["checklist_answers"])}
    bad = dict(answers)
    bad["q1"] = str(int(answers["q1"]) + 5)
    cli.req("/checklist", bad)
    st = json.loads(state_of(state_dir))
    assert not st["checklist_passed"]

    cli.req("/checklist", answers)
    st = json.loads(state_of(state_dir))
    assert st["checklist_passed"]

    cli.req("/finalize", {"confirm": "1"})
    st = json.loads(state_of(state_dir))
    assert st["finalized"]

    audit = (DATA / "reference" / "conflict_audit.csv").read_text()
    report = grade(state_of(state_dir), audit, world, params)
    assert report["violations"] == []
    assert report["screen_frac"] == 1.0
    assert report["checklist_frac"] == 1.0
    assert report["audit_f1"] == 1.0
    assert report["quality"] == 1.0
    assert report["score"] == 1.0

    # post-finalize mutations rejected
    cli.req("/assignments/clear", {})
    st = json.loads(state_of(state_dir))
    assert sum(len(v) for v in st["assignments"].values()) == len(rows)


def test_values_are_images_not_text(console, world):
    cli, _state_dir, _ = console
    papers = {p["paper_id"]: p for p in world["papers"]}
    pid = next(pid for pid, p in papers.items() if p["status"] == "active")
    status, body = cli.req(f"/paper/{pid}")
    assert status == 200
    assert '/img/' in body
    assert papers[pid]["primary_area"] not in body, "attribute leaked as text"
    author_name = papers[pid]["authors"][0][0]
    assert author_name not in body, "author name leaked as text"
    img_path = body.split('src="')[1].split('"')[0]
    status, png = cli.req(img_path)
    assert status == 200
    assert "PNG" in png[:8]

    status, body = cli.req("/reviewer/R01")
    assert '/img/' in body
    assert "Max load" not in body, "reviewer attributes leaked as text"


def test_screening_lock_and_checklist_lock(tmp_path, world):
    cli, state_dir, _, proc = boot(tmp_path)
    try:
        cli.req("/login", {"user": "chair", "password": "prc-2026"})
        papers = {p["paper_id"]: p for p in world["papers"]}
        pid = next(pid for pid, p in papers.items() if p["status"] == "active")
        for _ in range(3):
            cli.req(f"/paper/{pid}/screen",
                    {"primary_area": "", "author_count": "999"})
        # 4th attempt with CORRECT data must still be rejected (locked)
        cli.req(f"/paper/{pid}/screen",
                {"primary_area": papers[pid]["primary_area"],
                 "author_count": str(papers[pid]["author_count"])})
        st = json.loads(state_of(state_dir))
        assert pid not in st["screenings"]
        assert st["screen_attempts"][pid] == 3
        _, body = cli.req(f"/paper/{pid}")
        assert "locked" in body

        wrong = {f"q{i + 1}": "999999" for i in range(8)}
        for _ in range(10):
            cli.req("/checklist", wrong)
        good = {f"q{i + 1}": str(v)
                for i, v in enumerate(world["checklist_answers"])}
        cli.req("/checklist", good)
        st = json.loads(state_of(state_dir))
        assert st["checklist_attempts"] == 10
        assert not st["checklist_passed"], "checklist must lock after 10 attempts"
    finally:
        proc.kill()


def test_no_state_scores_zero(world, params):
    assert grade(None, None, world, params)["score"] == 0.0
    assert grade("not json", None, world, params)["score"] == 0.0


def make_good_state(world):
    papers = {p["paper_id"]: p for p in world["papers"]}
    active = [pid for pid, p in papers.items() if p["status"] == "active"]
    rows = [r for r in csv.reader(
        (DATA / "reference" / "optimal_assignment.csv").open())][1:]
    assignments: dict[str, list[str]] = {}
    for p, r in rows:
        assignments.setdefault(p, []).append(r)
    return {
        "screenings": {pid: {"primary_area": papers[pid]["primary_area"],
                             "author_count": papers[pid]["author_count"]}
                       for pid in active},
        "assignments": assignments,
        "checklist_passed": True,
        "checklist_answers": {f"q{i + 1}": str(v)
                              for i, v in enumerate(world["checklist_answers"])},
        "finalized": True,
    }


def test_forged_state_still_needs_truth(world, params):
    """Writing state.json directly cannot bypass knowledge: a wrong screening
    record or an ineligible reviewer collapses the score."""
    audit = (DATA / "reference" / "conflict_audit.csv").read_text()
    good = make_good_state(world)
    assert grade(json.dumps(good), audit, world, params)["score"] == 1.0

    wrong_screen = json.loads(json.dumps(good))
    pid = next(iter(wrong_screen["screenings"]))
    wrong_screen["screenings"][pid]["author_count"] += 1
    r = grade(json.dumps(wrong_screen), audit, world, params)
    assert r["score"] < 0.1
    assert any("screening" in v for v in r["violations"])

    emeritus = world["amendments"]["emeritus"]
    bad_assign = json.loads(json.dumps(good))
    pid = next(iter(bad_assign["assignments"]))
    bad_assign["assignments"][pid][0] = emeritus[0]
    r = grade(json.dumps(bad_assign), audit, world, params)
    assert r["score"] <= 0.1
    assert any("not eligible" in v for v in r["violations"])


def test_unfinalized_capped(world, params):
    audit = (DATA / "reference" / "conflict_audit.csv").read_text()
    st = make_good_state(world)
    st["finalized"] = False
    r = grade(json.dumps(st), audit, world, params)
    assert r["score"] == params["unfinalized_cap"]


def test_missing_audit_and_checklist_reduce_score(world, params):
    st = make_good_state(world)
    r = grade(json.dumps(st), None, world, params)
    assert r["score"] == pytest.approx(
        params["base_score"] + params["quality_weight"] + params["checklist_weight"])

    st["checklist_answers"] = {}
    r = grade(json.dumps(st), None, world, params)
    assert r["score"] == pytest.approx(
        params["base_score"] + params["quality_weight"])


def test_conflicted_assignment_collapses(world, params):
    audit = (DATA / "reference" / "conflict_audit.csv").read_text()
    st = make_good_state(world)
    rid, pid, _ = world["truth_triples"][0]
    if rid not in st["assignments"].get(pid, []):
        st["assignments"].setdefault(pid, []).pop()
        st["assignments"][pid].append(rid)
    r = grade(json.dumps(st), audit, world, params)
    assert r["score"] <= 0.1
    assert any("conflict of interest" in v for v in r["violations"])
