"""engineering/cad_part_from_drawing_rubric: renders, dimension factor and hooks with a faked judge."""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile

import pytest

cq = pytest.importorskip("cadquery")
pytest.importorskip("matplotlib")

from tasks.engineering.cad_part_from_drawing_rubric import main as T
from tasks.engineering.cad_part_from_drawing_rubric.scripts import grade as g
from tests.tasks._cad_fake_session import FakeSession

RUBRICS = {"rubrics": [{"id": "R1", "requirement": "single block"}, {"id": "R2", "requirement": "one through hole"}, {"id": "R3", "requirement": "two lugs"}, {"id": "R4", "requirement": "fillets"}]}
TD = {"application": "SOLIDWORKS", "task": "Create a block", "description": "A block with a hole."}


def _box_step(dims=(40, 20, 10)):
    with tempfile.TemporaryDirectory() as t:
        cq.exporters.export(cq.Workplane().box(*dims).faces(">Z").workplane().hole(6), f"{t}/b.step"); return open(f"{t}/b.step").read()


def test_renders_six_views_and_dimension_factor():
    with tempfile.TemporaryDirectory() as t:
        cq.exporters.export(cq.Workplane().box(100, 100, 60), f"{t}/b.step"); shape, err = g.load_solid(f"{t}/b.step"); assert err is None
        paths = g.render_views(shape, f"{t}/v"); assert len(paths) == 6 and all(os.path.getsize(p) > 2000 for p in paths)
        assert g.dim_factor(shape, {"extents_mm": [100, 100, 60]})[0] == 1.0
        assert g.dim_factor(shape, {"extents_mm": [100, 100, 40]})[0] == 0.0
        f, info = g.dim_factor(shape, {"extents_mm": [None, None, 50]}); assert abs(f - 0.4) < 1e-6 and info["max_rel_err"] == 0.2
        assert g.dim_factor(shape, None)[0] == 1.0
        assert g.load_solid(f"{t}/missing.step")[0] is None
        os.makedirs(f"{t}/ref/p01"); b = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (10, 10), "white").save(b, "PNG")
        json.dump({"images": [{"name": "d.png", "png_b64": base64.b64encode(b.getvalue()).decode()}]}, open(f"{t}/ref/p01/drawing_images.json", "w"))
        assert len(g.drawing_images(f"{t}/ref", "p01", f"{t}/work")) == 1 and g.drawing_images(f"{t}/ref", "p02", f"{t}/work") == []


@pytest.mark.asyncio
async def test_hooks_with_fake_judge(monkeypatch):
    calls = []

    def fake_chat(system, text, images=None, model=None, retries=4):
        calls.append(len(images or [])); return {"verdicts": {"R1": True, "R2": True, "R3": False, "R4": False}, "notes": "fake"}

    monkeypatch.setattr(g, "chat_json", fake_chat); monkeypatch.setenv("ALE_JUDGE_VOTES", "3")
    t = T.load()[0]; m = t.metadata; assert m["reference_dir"] not in t.description
    files = {m["reference_manifest"]: json.dumps([{"stem": "p01"}, {"stem": "p02"}])}
    for s in ("p01", "p02"):
        files[f"{m['reference_dir']}/{s}/rubrics.json"] = json.dumps(RUBRICS); files[f"{m['reference_dir']}/{s}/task_desc.json"] = json.dumps(TD)
    files[f"{m['reference_dir']}/p01/dims.json"] = json.dumps({"extents_mm": [40, 20, 10]})
    with pytest.raises(RuntimeError):
        await T.start(t, FakeSession(files, {m["parts_dir"]: 6}))
    await T.start(t, FakeSession({}, {m["parts_dir"]: 6}))
    assert await T.evaluate(t, FakeSession(files)) == [0.0]
    files[f"{m['remote_output_dir']}/p01.step"] = _box_step(); files[f"{m['remote_output_dir']}/p02.step"] = "not a step"
    s = (await T.evaluate(t, FakeSession(files)))[0]; assert abs(s - 0.25) < 1e-6, s     # p01: 2/4 rubrics x dim 1.0; p02 invalid -> 0
    assert calls == [6, 6, 6]                                                              # 3 votes, six renders, no drawing images staged
    files[f"{m['reference_dir']}/p01/dims.json"] = json.dumps({"extents_mm": [40, 20, 5]})  # 100 % off on the smallest extent -> factor 0
    assert (await T.evaluate(t, FakeSession(files)))[0] == 0.0


@pytest.mark.asyncio
async def test_judge_outage_scores_zero_without_raising(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(g, "chat_json", broken)
    t = T.load()[0]; m = t.metadata
    files = {m["reference_manifest"]: json.dumps([{"stem": "p01"}]), f"{m['reference_dir']}/p01/rubrics.json": json.dumps(RUBRICS), f"{m['reference_dir']}/p01/task_desc.json": json.dumps(TD), f"{m['remote_output_dir']}/p01.step": _box_step()}
    assert await T.evaluate(t, FakeSession(files)) == [0.0]
