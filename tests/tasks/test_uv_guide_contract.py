from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tasks.visual_media.uv_reproduction.scripts import remote_hard_eval as scorer
from tasks.visual_media.uv_reproduction.scripts.render_uv_layout import (
    read_uv_faces,
    render_uv_layout,
)


RELEASE = Path("/home/allennie/ale-overall/release-staging/ale-task-fixes-20260913")
DATA = RELEASE / "data-v1.1/files/visual_media/uv_reproduction/uv_reproduction_anime_singing_girl"
OVERLAY = RELEASE / "post-validation-20260918/data-overrides/visual_media/uv_reproduction/base"
REFERENCE_OBJ_SHA256 = "2744dff649e9f94bab0093d752dfeb74d4fc49ca3da2f283a63a683bffd3479a"
OLD_GUIDE_SHA256 = "35ec288cc294267887c4f8c095faecfa8c717f41dbb9baa8552f52b671455ea7"


def edge_coverage(image, coordinates, faces):
    pixels = np.asarray(image.convert("RGB"))
    yellow = (pixels[:, :, 0] > 150) & (pixels[:, :, 1] > 100) & (pixels[:, :, 2] < 150)
    height, width = yellow.shape
    probes = []
    for face in faces:
        polygon = np.asarray([coordinates[index] for index in face])
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            probes.extend(start + fraction * (end - start) for fraction in (0, 0.25, 0.5, 0.75))
    probes = np.asarray(probes)
    columns = np.rint(probes[:, 0] * (width - 1)).astype(int)
    rows = np.rint((1 - probes[:, 1]) * (height - 1)).astype(int)
    covered = np.zeros(len(probes), dtype=bool)
    for delta_row in (-1, 0, 1):
        for delta_column in (-1, 0, 1):
            covered |= yellow[
                np.clip(rows + delta_row, 0, height - 1),
                np.clip(columns + delta_column, 0, width - 1),
            ]
    return float(covered.mean())


@pytest.fixture
def reference_uv():
    path = DATA / "reference/objects/singing.obj"
    if not path.exists():
        pytest.skip("frozen singing OBJ unavailable")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_OBJ_SHA256
    return read_uv_faces(path)


def test_frozen_reference_contains_full_area_uv_faces(reference_uv):
    coordinates, faces = reference_uv
    used = {index for face in faces for index in face}
    assert len(coordinates) == 760
    assert len(used) == 759
    assert len({coordinates[index] for index in used}) == 759
    assert len(faces) == 1990
    for face in faces:
        polygon = np.asarray([coordinates[index] for index in face])
        shifted = np.roll(polygon, -1, axis=0)
        area = abs(np.sum(polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1])) / 2
        assert area > 1e-12


def test_observed_five_ring_guide_is_a_negative_control(reference_uv):
    path = DATA / "input/reference_images/target_uv_layout.png"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == OLD_GUIDE_SHA256
    with Image.open(path) as image:
        assert edge_coverage(image, *reference_uv) < 0.01


def test_public_guide_follows_all_reference_face_edges(reference_uv):
    relative = "input/reference_images/target_uv_layout.png"
    path = OVERLAY / relative
    with Image.open(path if path.exists() else DATA / relative) as image:
        assert image.size == (1024, 1024)
        assert edge_coverage(image, *reference_uv) == 1.0


def test_public_and_hidden_guides_are_same_mesh_derived_wireframe(reference_uv):
    public = OVERLAY / "input/reference_images/target_uv_layout.png"
    hidden = OVERLAY / "reference/uv_targets/reference_uv_layout.png"
    assert public.read_bytes() == hidden.read_bytes()
    generated = render_uv_layout(DATA / "reference/objects/singing.obj")
    with Image.open(public) as staged:
        np.testing.assert_array_equal(np.asarray(staged), np.asarray(generated))
    assert edge_coverage(generated, *reference_uv) == 1.0
    manifest = json.loads((OVERLAY / "input/manifest.json").read_text())
    assert "input/reference_images/target_uv_layout.png" in manifest["reference_images"]
    assert "input/reference_images/target_material_preview.png" in manifest["reference_images"]


@pytest.mark.parametrize("face", ["1 2 3", "1//1 2//2 3//3", "1/0 2/2 3/3", "1/9 2/2 3/3"])
def test_generator_rejects_faces_without_valid_uvs(tmp_path, face):
    path = tmp_path / "invalid.obj"
    path.write_text(f"vt 0 0\nvt 1 0\nvt 0 1\nf {face}\n")
    with pytest.raises(ValueError, match="UV index"):
        render_uv_layout(path)


def test_generator_preserves_image_v_orientation_and_negative_indices(tmp_path):
    path = tmp_path / "triangle.obj"
    path.write_text("vt 0 0\nvt 1 0\nvt 0 1\nf 1/-3 2/-2 3/-1\n")
    image = render_uv_layout(path, 65)
    for point in ((0, 64), (64, 64), (0, 0), (32, 32)):
        assert image.getpixel(point) == (255, 201, 56)
    assert image.getpixel((48, 16)) != (255, 201, 56)


@pytest.mark.parametrize("identity_control", [False, True])
def test_actual_cached_render_replay_retains_material_criteria(
    monkeypatch, tmp_path, identity_control
):
    receipt_path = RELEASE / "validation-checks/uv-r02-final-check.json"
    if not receipt_path.exists():
        pytest.skip("r02 UV render artifacts unavailable")
    receipt = json.loads(receipt_path.read_text())
    selection = receipt["selection"]["kimi_after"]
    run_path = Path(selection["run_json"])
    assert hashlib.sha256(run_path.read_bytes()).hexdigest() == selection["sha256"]
    assets = RELEASE / "validation-checks/uv-r02-final-assets"
    bundle = (
        DATA / "reference/objects" if identity_control else run_path.parent / "output/submission"
    )
    renders = assets / ("reference_renders" if identity_control else "candidate_renders")
    calls = []

    def cached_render(obj_path, out_dir, blender_binary, renderer_script):
        assert obj_path == bundle / "singing.obj"
        calls.append(obj_path)
        out_dir.mkdir()
        for view in scorer.VIEW_NAMES:
            (out_dir / f"{view}.png").symlink_to(renders / f"{view}.png")

    monkeypatch.setattr(scorer, "_render_single", cached_render)
    args = SimpleNamespace(
        reference_obj=str(DATA / "reference/objects/singing.obj"),
        candidate_obj=str(bundle / "singing.obj"),
        candidate_mtl=str(bundle / ("singing.mtl" if identity_control else "material.mtl")),
        candidate_texture_dir=str(bundle / "textures"),
        reference_images_dir=str(DATA / "reference/images"),
        renderer_script=str(Path(scorer.__file__).with_name("blender_render_material_views.py")),
        blender_binary="NO_RENDERER_ALLOWED",
        requires_uv="1",
        requires_mtl="1",
        requires_basecolor_texture="1",
        requires_color_match_gate="1",
        color_match_gate_threshold=0.9,
    )
    for view in scorer.VIEW_NAMES:
        assert (DATA / f"reference/images/{view}.png").read_bytes() == (
            assets / f"reference_renders/{view}.png"
        ).read_bytes()
    result = scorer._eval_single(args, tmp_path / "not-written-report.json")
    assert len(calls) == 1
    assert all(result["metrics"]["gate"].values())
    if identity_control:
        assert result["score"] == 1.0
    else:
        expected = receipt["hard_scoring_from_original_frames"]["metrics"]
        assert result["metrics"] == expected
        soft = json.loads((assets / "soft_eval_report.json").read_text())
        assert [row["result"] for row in soft["results"]] == ["NO", "NO", "NO"]
        assert 0.3 * result["score"] + 0.7 * soft["final_score"] == selection["score"]
