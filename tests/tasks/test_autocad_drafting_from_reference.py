"""engineering/autocad_drafting_from_reference: grader calibration on synthetic drawings and the three hooks with a fake session."""
from __future__ import annotations

import json
import tempfile

import pytest

ezdxf = pytest.importorskip("ezdxf")
pytest.importorskip("scipy")

from tasks.engineering.autocad_drafting_from_reference import main as T
from tasks.engineering.autocad_drafting_from_reference.scripts.grade import score_drawing
from tests.tasks._cad_fake_session import FakeSession


def _bracket(path, scale=1.0, dx=0.0, with_hole=True, with_dims=True):
    doc = ezdxf.new("R2010"); msp = doc.modelspace(); s = scale
    msp.add_lwpolyline([(dx, 0), (dx + 120 * s, 0), (dx + 120 * s, 20 * s), (dx + 40 * s, 20 * s), (dx + 40 * s, 80 * s), (dx, 80 * s)], close=True)
    if with_hole:
        msp.add_circle((dx + 20 * s, 50 * s), 6 * s); msp.add_circle((dx + 90 * s, 10 * s), 4 * s)
    if with_dims:
        msp.add_linear_dim(base=(dx, -10 * s), p1=(dx, 0), p2=(dx + 120 * s, 0)).render()
    msp.add_text("BRACKET", dxfattribs={"height": 5 * s}).set_placement((dx + 50 * s, 30 * s))
    doc.saveas(path); return path


def _text(path):
    return open(path).read()


def test_same_drawing_scaled_and_moved_scores_one():
    with tempfile.TemporaryDirectory() as t:
        r = _bracket(f"{t}/r.dxf"); p = _bracket(f"{t}/p.dxf", scale=2.5, dx=300)
        res = score_drawing(p, r); assert res["score"] > 0.97, res


def test_missing_features_and_dimensions_lower_score():
    with tempfile.TemporaryDirectory() as t:
        r = _bracket(f"{t}/r.dxf")
        nohole = score_drawing(_bracket(f"{t}/p.dxf", with_hole=False), r); assert 0.6 < nohole["score"] < 0.98, nohole
        nodim = score_drawing(_bracket(f"{t}/q.dxf", with_dims=False), r); assert abs(nodim["score"] - 0.8) < 0.03 and nodim["dim_ratio"] == 0.0, nodim


def test_wrong_drawing_and_garbage_score_low():
    with tempfile.TemporaryDirectory() as t:
        r = _bracket(f"{t}/r.dxf"); doc = ezdxf.new("R2010"); doc.modelspace().add_circle((0, 0), 50); doc.saveas(f"{t}/c.dxf")
        assert score_drawing(f"{t}/c.dxf", r)["score"] < 0.4
        open(f"{t}/g.dxf", "w").write("nonsense"); assert score_drawing(f"{t}/g.dxf", r)["score"] == 0.0


@pytest.mark.asyncio
async def test_hooks_hide_reference_and_never_raise_on_missing_output():
    t = T.load()[0]; m = t.metadata; assert m["reference_dir"] not in t.description
    with tempfile.TemporaryDirectory() as tmp:
        ref = _text(_bracket(f"{tmp}/r.dxf"))
    files = {m["reference_manifest"]: json.dumps([{"stem": "d01"}, {"stem": "d02"}]), f"{m['reference_dir']}/d01/expert.dxf": ref, f"{m['reference_dir']}/d02/expert.dxf": ref}
    with pytest.raises(RuntimeError):                      # reference visible during start -> refuse
        await T.start(t, FakeSession(files, {m["drawings_dir"]: 6}))
    await T.start(t, FakeSession({}, {m["drawings_dir"]: 6}))
    with pytest.raises(RuntimeError):                      # too few staged drawings -> staging error
        await T.start(t, FakeSession({}, {m["drawings_dir"]: 2}))
    assert await T.evaluate(t, FakeSession(files)) == [0.0]  # nothing delivered
    files[f"{m['remote_output_dir']}/d01.dxf"] = ref; files[f"{m['remote_output_dir']}/d02.dxf"] = "junk"
    s = (await T.evaluate(t, FakeSession(files)))[0]; assert abs(s - 0.5) < 0.02, s
