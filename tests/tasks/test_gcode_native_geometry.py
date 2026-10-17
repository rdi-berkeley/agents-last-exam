import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh


MODULE_PATH = Path(__file__).parents[2] / "tasks/engineering/gcode/scripts/native_geometry.py"
SPEC = importlib.util.spec_from_file_location("gcode_native_geometry", MODULE_PATH)
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


def surface(quads):
    vertices = np.asarray(quads).reshape(-1, 3)
    faces = []
    for offset in range(0, len(vertices), 4):
        faces.extend(((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)))
    return trimesh.Trimesh(vertices, faces, process=False)


def pocket(half_width=2, depth=1):
    lower = -half_width
    upper = half_width
    floor = -depth
    return surface(
        [
            [(lower, -2, floor), (upper, -2, floor), (upper, 2, floor), (lower, 2, floor)],
            [(lower, -2, 0), (lower, 2, 0), (lower, 2, floor), (lower, -2, floor)],
            [(upper, -2, 0), (upper, 2, 0), (upper, 2, floor), (upper, -2, floor)],
            [(lower, -2, 0), (upper, -2, 0), (upper, -2, floor), (lower, -2, floor)],
            [(lower, 2, 0), (upper, 2, 0), (upper, 2, floor), (lower, 2, floor)],
        ]
    )


def finished_stock(half_width=2, depth=1):
    box = trimesh.creation.box(extents=(10, 10, 4))
    box.apply_translation((0, 0, -2))
    box.update_faces(box.face_normals[:, 2] < 0.5)
    top = surface(
        [
            [(-5, -5, 0), (5, -5, 0), (5, -2, 0), (-5, -2, 0)],
            [(-5, 2, 0), (5, 2, 0), (5, 5, 0), (-5, 5, 0)],
            [(-5, -2, 0), (-half_width, -2, 0), (-half_width, 2, 0), (-5, 2, 0)],
            [(half_width, -2, 0), (5, -2, 0), (5, 2, 0), (half_width, 2, 0)],
        ]
    )
    return trimesh.util.concatenate((box, top, pocket(half_width, depth)))


@pytest.fixture
def public_case(tmp_path):
    source = tmp_path / "original.stl"
    pocket().export(source)
    bounds = [-5, -5, -4, 5, 5, 0]
    region = geometry.prepare_region([source], bounds, tmp_path / "region.json")
    reference = tmp_path / "reference.stl"
    finished_stock().export(reference)
    return source, bounds, region, reference


def test_region_is_frozen_from_public_geometry(public_case):
    _, _, region, _ = public_case
    assert region["footprint_area_mm2"] == pytest.approx(16)
    assert not region["derived_from_reference_or_candidate"]
    points = np.array([[0, 0, -1], [3, 0, -1], [0, 0, 0], [0, 0, -4]])
    assert geometry.region_mask(points, region).tolist() == [True, False, False, False]


def test_identical_finished_stock_receives_full_geometry_credit(public_case):
    _, _, region, reference = public_case
    result = geometry.compare_surfaces(reference, reference, region, count=500)
    assert result["geometry_score"] == 1
    assert result["safety_checked"] is False


def test_untouched_exterior_cannot_hide_missing_pocket(public_case, tmp_path):
    _, _, region, reference = public_case
    blank = trimesh.creation.box(extents=(10, 10, 4))
    blank.apply_translation((0, 0, -2))
    candidate = tmp_path / "uncut.stl"
    blank.export(candidate)
    result = geometry.compare_surfaces(candidate, reference, region, count=500)
    assert result["geometry_score"] == 0
    assert result["regions"]["machining"]["candidate_samples"] == 0
    assert result["regions"]["remainder"]["score"] > 0.9


def test_missing_part_of_pocket_loses_geometry_credit(public_case, tmp_path):
    _, _, region, reference = public_case
    candidate = tmp_path / "partial.stl"
    finished_stock(half_width=0.5).export(candidate)
    result = geometry.compare_surfaces(candidate, reference, region, count=1000)
    assert 0 < result["geometry_score"] < 0.9


def test_removed_outer_stock_is_also_checked(public_case, tmp_path):
    _, _, region, reference = public_case
    candidate = tmp_path / "pocket-only.stl"
    pocket().export(candidate)
    result = geometry.compare_surfaces(candidate, reference, region, count=500)
    assert result["regions"]["machining"]["score"] == 1
    assert result["regions"]["remainder"]["score"] < 0.5
    assert result["geometry_score"] < 0.5


def test_bounded_distance_matches_full_native_index(public_case):
    _, _, _, reference = public_case
    points = np.array([[0, 0, -0.5], [0, 0, -1.5], [0, 0, -5], [7, 0, 0]])
    full = geometry.surface_distances(reference, points, chunk_size=10000)
    chunked = geometry.surface_distances(reference, points, chunk_size=3)
    np.testing.assert_allclose(full, chunked, rtol=0, atol=1e-9)


def test_truncated_mesh_never_produces_a_score(public_case, tmp_path):
    _, _, region, reference = public_case
    candidate = tmp_path / "truncated.stl"
    candidate.write_bytes(reference.read_bytes()[:-10])
    with pytest.raises(ValueError, match="complete"):
        geometry.compare_surfaces(candidate, reference, region, count=50)


def test_exact_geometry_cannot_bypass_failed_or_missing_safety(public_case):
    _, _, region, reference = public_case
    assert geometry.grade_replay({"status": "invalid_delivery"}, reference, region)["score"] == 0
    with pytest.raises(RuntimeError, match="no score"):
        geometry.grade_replay({"status": "unavailable"}, reference, region)
    report = {
        "status": "native_replay_complete",
        "safety": {"status": "clear", "collision_free": True},
        "stock": {"output": str(reference)},
    }
    with pytest.raises(RuntimeError, match="coverage"):
        geometry.grade_replay(report, reference, region)


def test_scored_stock_is_bound_to_complete_native_receipt(public_case):
    _, _, region, reference = public_case
    report = {
        "status": "native_replay_complete",
        "safety": {
            "status": "clear",
            "collision_free": True,
            "whole_path_checked": True,
            "tool_events_checked": True,
            "motion_count": 2,
            "motions_sha256": "same-physical-motions",
        },
        "stock": {
            "motion_count": 2,
            "motions_sha256": "same-physical-motions",
            "output": str(reference),
            "output_sha256": geometry.file_hash(reference),
        },
        "project": {"project_sha256": "saved-project"},
        "canonical": {"input_sha256": "executed-program"},
    }
    scored = geometry.grade_replay(report, reference, region, count=500)
    assert scored["score"] == 1
    assert scored["safety_checked"] is True
    report["stock"]["output_sha256"] = "changed-stock"
    with pytest.raises(RuntimeError, match="changed"):
        geometry.grade_replay(report, reference, region, count=500)
    report["stock"]["output_sha256"] = geometry.file_hash(reference)
    report["stock"]["motions_sha256"] = "same-count-but-different-physical-motions"
    with pytest.raises(RuntimeError, match="coverage"):
        geometry.grade_replay(report, reference, region, count=500)
