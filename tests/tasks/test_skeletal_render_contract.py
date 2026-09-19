from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.visual_media.skeletal_animation_reproduction.scripts import remote_render_eval


RENDERER = (
    Path(__file__).resolve().parents[2]
    / "tasks/visual_media/skeletal_animation_reproduction/scripts/blender_render_submission.py"
)


@pytest.mark.parametrize("media_type", ["VIDEO", "IMAGE", None])
def test_renderer_normalizes_media_before_png(monkeypatch, tmp_path, media_type):
    class ImageSettings:
        def __init__(self):
            if media_type is not None:
                self.media_type = media_type
            self.formats = []

        @property
        def file_format(self):
            return self.formats[-1] if self.formats else "FFMPEG"

        @file_format.setter
        def file_format(self, value):
            assert getattr(self, "media_type", "IMAGE") == "IMAGE"
            self.formats.append(value)

    settings = ImageSettings()
    render = SimpleNamespace(image_settings=settings)
    bpy = SimpleNamespace(
        context=SimpleNamespace(scene=SimpleNamespace(render=render)),
        types=SimpleNamespace(
            RenderSettings=SimpleNamespace(
                bl_rna=SimpleNamespace(properties={"engine": SimpleNamespace(enum_items=[])})
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Vector=object))
    spec = importlib.util.spec_from_file_location("skeletal_renderer_under_test", RENDERER)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    monkeypatch.setattr(
        renderer,
        "parse_args",
        lambda: SimpleNamespace(
            output_dir=str(tmp_path), evaluation_config=None, image_width=512, image_height=500
        ),
    )

    class InitializationComplete(Exception):
        pass

    def stop_before_scene_changes(color):
        raise InitializationComplete

    monkeypatch.setattr(renderer, "_ensure_world", stop_before_scene_changes)
    with pytest.raises(InitializationComplete):
        renderer.main()

    assert settings.formats == ["PNG"]
    assert settings.color_mode == "RGBA"
    assert render.resolution_x == 512
    assert render.resolution_y == 500
    assert render.film_transparent is False
    if media_type is None:
        assert not hasattr(settings, "media_type")


@pytest.fixture
def wrapper_args(monkeypatch, tmp_path):
    config = tmp_path / "evaluation_config.json"
    config.write_text(json.dumps({"sample_count": 7, "image_width": 320, "image_height": 240}))
    args = SimpleNamespace(
        blend="original.blend",
        output_dir=str(tmp_path / "output"),
        renderer_script=str(RENDERER),
        evaluation_config=str(config),
    )
    monkeypatch.setattr(remote_render_eval, "parse_args", lambda: args)
    monkeypatch.setenv("BLENDER_BINARY", "blender")
    return args


@pytest.mark.parametrize("gate_passed", [True, False])
def test_wrapper_preserves_renderer_report(monkeypatch, wrapper_args, gate_passed):
    report = {"validity_gate_passed": gate_passed, "gate_fail_reasons": []}
    report_path = Path(wrapper_args.output_dir) / "render_report.json"

    def run(command, **kwargs):
        assert command[:7] == [
            "blender",
            "--background",
            "original.blend",
            "--python-exit-code",
            "1",
            "--python",
            str(RENDERER),
        ]
        assert command[7:] == [
            "--",
            "--output-dir",
            wrapper_args.output_dir,
            "--sample-count",
            "7",
            "--image-width",
            "320",
            "--image-height",
            "240",
            "--evaluation-config",
            wrapper_args.evaluation_config,
        ]
        assert kwargs == {"capture_output": True, "text": True}
        report_path.write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(remote_render_eval.subprocess, "run", run)
    assert remote_render_eval.main() == 0
    assert json.loads(report_path.read_text()) == report


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize("stale_report", [False, True])
def test_wrapper_reports_failure_without_accepting_stale_output(
    monkeypatch, wrapper_args, returncode, stale_report
):
    report_path = Path(wrapper_args.output_dir) / "render_report.json"
    if stale_report:
        report_path.parent.mkdir()
        report_path.write_text('{"validity_gate_passed": true}')

    def run(command, **kwargs):
        assert not report_path.exists()
        return subprocess.CompletedProcess(command, returncode, "o" * 5000, "e" * 5000)

    monkeypatch.setattr(remote_render_eval.subprocess, "run", run)
    assert remote_render_eval.main() == 1
    report = json.loads(report_path.read_text())
    assert report["validity_gate_passed"] is False
    assert report["gate_fail_reasons"] == [
        "blender_render_failed" if returncode else "missing_render_report"
    ]
    assert report["stdout"] == "o" * 4000
    assert report["stderr"] == "e" * 4000
    assert report["view_paths"] == {}
    assert report["silhouette_paths"] == []
    assert report["sample_positions"] == []


def test_wrapper_missing_binary_reports_without_launch(monkeypatch, wrapper_args):
    monkeypatch.delenv("BLENDER_BINARY")

    def unexpected_launch(*args, **kwargs):
        pytest.fail("Blender must not launch without BLENDER_BINARY")

    monkeypatch.setattr(remote_render_eval.subprocess, "run", unexpected_launch)
    assert remote_render_eval.main() == 1
    report = json.loads((Path(wrapper_args.output_dir) / "render_report.json").read_text())
    assert report["gate_fail_reasons"] == ["missing_blender_binary_env"]
