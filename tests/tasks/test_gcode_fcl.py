import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("kind", ["cylinder", "ball_cylinder"])
@pytest.mark.parametrize("segments", [32, 128])
def test_complete_fcl_sweeps_preserve_margin_and_reject_interior_crossings(
    tmp_path, monkeypatch, kind, segments
):
    pytest.importorskip("fcl")
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    script = Path(__file__).parents[2] / "tasks/engineering/gcode/scripts/native_safety.py"
    spec = importlib.util.spec_from_file_location("fcl_safety_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vertices = numpy.array(
        [[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0]], dtype="float64"
    )
    faces = numpy.array([[0, 1, 2], [0, 2, 3]], dtype="int32")
    vertices.tofile(tmp_path / "vertices.f64")
    faces.tofile(tmp_path / "faces.i32")
    mesh = tmp_path / "target.json"
    mesh.write_text(
        json.dumps(
            {
                "vertices": str(tmp_path / "vertices.f64"),
                "vertex_count": len(vertices),
                "faces": str(tmp_path / "faces.i32"),
                "face_count": len(faces),
            }
        )
    )
    enclosure = {"kind": kind, "radius": 1, "bottom": 0, "top": 3}
    if kind == "ball_cylinder":
        enclosure["ball_center"] = 1
    cases = [
        ((0, 0, 0.1), (0, 0, 0.1), True),
        ((-10, -5, 0.1), (10, 5, 0.1), True),
        ((-10, -5, 0.1), (10, 5, 4), True),
        ((0, 0, 0.0109), (0, 0, 0.0109), False),
        ((0, 0, 0.0111), (0, 0, 0.0111), True),
        ((0, 0, -1), (0, 0, -1), False),
        ((-10, -5, 4), (10, 5, -6), False),
        ((0, 0, -6), (0, 0, 4), False),
    ]
    requests = [
        dict(
            op="check", enclosure=enclosure, start=start, end=end, padding=0.011, segments=segments
        )
        for start, end, expected in cases
    ]
    output = io.StringIO()
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("".join(json.dumps(row) + "\n" for row in requests))
    )
    monkeypatch.setattr(sys, "stdout", output)
    module._fcl_worker(mesh)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(responses) == len(cases)
    for response, (start, end, expected) in zip(responses, cases):
        assert "error" not in response
        assert response["clear"] is expected, (start, end, response)


@pytest.mark.parametrize("gap", [-0.1, 0.0109, 0.014])
def test_refined_circumscribed_cylinder_preserves_side_clearance(tmp_path, monkeypatch, gap):
    pytest.importorskip("fcl")
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    script = Path(__file__).parents[2] / "tasks/engineering/gcode/scripts/native_safety.py"
    spec = importlib.util.spec_from_file_location("fcl_refinement_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vertices = numpy.array(
        [[1 + gap, -50, -50], [1 + gap, 50, -50], [1 + gap, 50, 50], [1 + gap, -50, 50]],
        dtype="float64",
    )
    faces = numpy.array([[0, 1, 2], [0, 2, 3]], dtype="int32")
    vertices.tofile(tmp_path / "vertices.f64")
    faces.tofile(tmp_path / "faces.i32")
    mesh = tmp_path / "target.json"
    mesh.write_text(
        json.dumps(
            {
                "vertices": str(tmp_path / "vertices.f64"),
                "vertex_count": len(vertices),
                "faces": str(tmp_path / "faces.i32"),
                "face_count": len(faces),
            }
        )
    )
    requests = [
        dict(
            op="check",
            enclosure={"kind": "cylinder", "radius": 1, "bottom": 0, "top": 3},
            start=(0, 0, 0),
            end=(0, 0, 5),
            padding=0.011,
            segments=segments,
        )
        for segments in (32, 128)
    ]
    output = io.StringIO()
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("".join(json.dumps(row) + "\n" for row in requests))
    )
    monkeypatch.setattr(sys, "stdout", output)
    module._fcl_worker(mesh)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(responses) == 2
    assert all("error" not in response for response in responses)
    assert not responses[0]["clear"]
    assert responses[1]["clear"] is (gap == 0.014)
    assert all(response["minimum_distance"] <= max(gap, 0) + 1e-8 for response in responses)
