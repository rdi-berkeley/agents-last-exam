"""engineering/cad_session_brief_recovery: brief parsing, application matching and hooks with a faked judge."""
from __future__ import annotations

import json

import pytest

from tasks.engineering.cad_session_brief_recovery import main as T
from tasks.engineering.cad_session_brief_recovery.scripts import grade as g
from tests.tasks._cad_fake_session import FakeSession

RUBRICS = {"rubrics": [{"id": "R1", "requirement": "six profiles"}, {"id": "R2", "requirement": "rectangle"}, {"id": "R3", "requirement": "L shape"}, {"id": "R4", "requirement": "31 dimensions"}]}
TD = {"application": "AutoCAD 2021", "task": "Create six profiles", "description": "..."}


def test_parse_brief_and_application_family():
    assert g.parse_brief("") is None and g.parse_brief("{not json") is None
    assert g.parse_brief('{"requirements": []}')["requirements"] == []
    assert g.parse_brief('{"application": "x", "requirements": "a; b"}')["requirements"] == ["a", "b"]
    assert g.app_family("Siemens NX 12") == "nx" and g.app_family("SolidWorks 2023") == "solidworks" and g.app_family("AutoCAD") == "autocad" and g.app_family("") is None


@pytest.mark.asyncio
async def test_hooks_with_fake_judge(monkeypatch):
    def fake_chat(system, text, images=None, model=None, retries=4):
        return {"coverage": {"R1": "full", "R2": "partial", "R3": "none", "R4": "full"}}

    monkeypatch.setattr(g, "chat_json", fake_chat); monkeypatch.setenv("ALE_JUDGE_VOTES", "1")
    t = T.load()[0]; m = t.metadata; assert m["reference_dir"] not in t.description
    files = {m["reference_manifest"]: json.dumps([{"sid": "s01"}, {"sid": "s02"}])}
    for s in ("s01", "s02"):
        files[f"{m['reference_dir']}/{s}/rubrics.json"] = json.dumps(RUBRICS); files[f"{m['reference_dir']}/{s}/task_desc.json"] = json.dumps(TD)
    with pytest.raises(RuntimeError):
        await T.start(t, FakeSession(files, {m["sessions_dir"]: 5}))
    await T.start(t, FakeSession({}, {m["sessions_dir"]: 5}))
    assert await T.evaluate(t, FakeSession(files)) == [0.0]
    good = json.dumps({"application": "AutoCAD", "deliverable": "drawing", "objective": "profiles", "requirements": ["a", "b"]})
    files[f"{m['remote_output_dir']}/s01.json"] = good; files[f"{m['remote_output_dir']}/s02.json"] = "{not json"
    s = (await T.evaluate(t, FakeSession(files)))[0]
    exp = (0.15 * 1 + 0.85 * (1 + 0.5 + 0 + 1) / 4) / 2; assert abs(s - exp) < 1e-4, (s, exp)
    files[f"{m['remote_output_dir']}/s02.json"] = json.dumps({"application": "SOLIDWORKS", "requirements": ["x"]})  # wrong application: no app credit
    s2 = (await T.evaluate(t, FakeSession(files)))[0]; exp2 = (exp * 2 + 0.85 * 0.625) / 2; assert abs(s2 - exp2) < 1e-4


@pytest.mark.asyncio
async def test_judge_outage_scores_zero_without_raising(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(g, "chat_json", broken)
    t = T.load()[0]; m = t.metadata
    files = {m["reference_manifest"]: json.dumps([{"sid": "s01"}]), f"{m['reference_dir']}/s01/rubrics.json": json.dumps(RUBRICS), f"{m['reference_dir']}/s01/task_desc.json": json.dumps(TD), f"{m['remote_output_dir']}/s01.json": json.dumps({"application": "AutoCAD", "requirements": ["a"]})}
    assert await T.evaluate(t, FakeSession(files)) == [0.0]
