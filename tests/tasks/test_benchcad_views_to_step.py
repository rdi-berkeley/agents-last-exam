"""engineering/benchcad_views_to_step: volume-IoU grader over the 24 axis-aligned rotations, and the hooks with a fake session."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

cq = pytest.importorskip("cadquery")

from tasks.engineering.benchcad_views_to_step import main as T
from tasks.engineering.benchcad_views_to_step.scripts.grade import N_ROTATIONS, ROTATIONS, load_reference, load_shape, score_parts
from tasks.engineering.benchcad_views_to_step.scripts.prompt import task_text
from tests.tasks._cad_fake_session import FakeSession

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ref():
    return cq.Workplane("XY").box(40, 20, 10).faces(">Z").workplane().hole(6)


def _write(wp, path):
    cq.exporters.export(wp, path); return path


def test_rotation_set_is_the_24_proper_axis_aligned_rotations():
    assert len(ROTATIONS) == N_ROTATIONS == 24 and len({tuple(map(tuple, m)) for m in ROTATIONS}) == 24
    for m in ROTATIONS:
        assert sorted(abs(x) for row in m for x in row) == [0.0] * 6 + [1.0] * 3
        det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        assert det == 1.0


def test_similarity_transforms_rotations_fusion_and_gates():
    with tempfile.TemporaryDirectory() as t:
        r = _write(_ref(), f"{t}/r.step")
        same = _write(_ref().translate((100, -30, 5)).rotate((0, 0, 0), (0, 0, 1), 90), f"{t}/p1.step")
        scaled = _write(cq.Workplane("XY").box(80, 40, 20).faces(">Z").workplane().hole(12), f"{t}/p2.step")
        res = score_parts({"a": same, "b": scaled}, {"a": r, "b": r})
        assert res["per_part"]["a"]["iou"] > 0.99 and res["per_part"]["b"]["iou"] > 0.99 and res["score"] > 0.99
        part = cq.Workplane("XY").box(40, 20, 10).union(cq.Workplane("XY").box(8, 8, 8).translate((12, -4, 8)))
        r2 = _write(part, f"{t}/r2.step"); preds = {}
        for i, (axis, deg) in enumerate((((1, 0, 0), 90), ((0, 1, 0), 180), ((0, 0, 1), 270))):
            preds[f"p{i}"] = _write(part.rotate((0, 0, 0), axis, deg).translate((7, 7, 7)), f"{t}/q{i}.step")
        assert all(v["iou"] > 0.99 for v in score_parts(preds, {k: r2 for k in preds})["per_part"].values())
        two = cq.Workplane("XY").box(25, 20, 10).translate((-7.5, 0, 0)).add(cq.Workplane("XY").box(25, 20, 10).translate((7.5, 0, 0)))
        p = _write(two, f"{t}/two.step"); shape, n, err = load_shape(p); assert err is None and n == 2
        rb = _write(cq.Workplane("XY").box(40, 20, 10), f"{t}/rb.step"); res2 = score_parts({"a": p}, {"a": rb})
        assert res2["per_part"]["a"]["iou"] > 0.99 and "2 solids fused" in res2["per_part"]["a"]["note"]
        apart = _write(cq.Workplane("XY").box(10, 10, 10).add(cq.Workplane("XY").box(10, 10, 10).translate((30, 0, 0))), f"{t}/apart.step")
        with pytest.raises(RuntimeError):
            load_reference(apart)
        cyl = _write(cq.Workplane("XY").cylinder(10, 15), f"{t}/cyl.step"); g = f"{t}/g.step"; open(g, "w").write("not a step file")
        res3 = score_parts({"a": cyl, "b": None, "c": g}, {"a": r, "b": r, "c": r})
        assert 0.0 < res3["per_part"]["a"]["iou"] < 0.7 and res3["per_part"]["b"]["note"] == "missing" and "unreadable" in res3["per_part"]["c"]["note"]


def test_prompt_and_task_card_agree():
    card = json.load(open(f"{HERE}/tasks/engineering/benchcad_views_to_step/task_card.json"))
    assert card["taskPrompt"] == task_text("input/parts", "input/README.md", "output")
    assert "All 24 proper axis-aligned rotations" in card["taskPrompt"]


@pytest.mark.asyncio
async def test_hooks():
    t = T.load()[0]; m = t.metadata; assert m["reference_dir"] not in t.description
    with tempfile.TemporaryDirectory() as tmp:
        ref = open(_write(_ref(), f"{tmp}/r.step")).read()
    files = {m["reference_manifest"]: json.dumps({"a": {"ok": True}, "b": {"ok": True}}), f"{m['reference_dir']}/a.step": ref, f"{m['reference_dir']}/b.step": ref, m["readme_path"]: "x"}
    with pytest.raises(RuntimeError):
        await T.start(t, FakeSession(files, {m["parts_dir"]: 12}))
    await T.start(t, FakeSession({m["readme_path"]: "x"}, {m["parts_dir"]: 12}))
    assert await T.evaluate(t, FakeSession(files)) == [0.0]
    files[f"{m['remote_output_dir']}/a.step"] = ref; files[f"{m['remote_output_dir']}/b.step"] = ref
    assert (await T.evaluate(t, FakeSession(files)))[0] > 0.98
    files[f"{m['remote_output_dir']}/b.step"] = "not a step"
    assert 0.4 < (await T.evaluate(t, FakeSession(files)))[0] < 0.6
