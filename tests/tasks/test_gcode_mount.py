import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[2] / "tasks/engineering/gcode/scripts"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native_project = load_module("mount_native_project", ROOT / "native_project.py")
native_pipeline = load_module("mount_native_pipeline", ROOT / "native_pipeline.py")
machine_setup = load_module("mount_machine_setup", ROOT / "machine_setup.py")


def environment(monkeypatch, tmp_path, number, mounted=False, safety_status="clear"):
    observations = SimpleNamespace(shapes=[], safety=[], stock=[], closed=[])
    identity = f"gcode-125162-319-{number:03d}"
    source_overhang = 40.0

    def controller():
        assembly = SimpleNamespace(
            Mount=SimpleNamespace(Overhang=SimpleNamespace(Value=13.0)) if mounted else None,
            HolderOverhang=SimpleNamespace(Value=source_overhang),
            Cutter=object(),
            Shank=None,
            Holder=object() if mounted else None,
        )
        return SimpleNamespace(
            ToolNumber=number,
            OriginalToolId=identity,
            OriginalAssembly=assembly,
            Tool=SimpleNamespace(Diameter=SimpleNamespace(Value=2.5)),
            SpindleSpeed=5000,
            Proxy=SimpleNamespace(execute=lambda current: None),
        )

    submitted = controller()
    original = controller()
    operation = SimpleNamespace(
        Name="NativeDrilling",
        State=[],
        Path=SimpleNamespace(Commands=["G1 Z-1 F1"]),
        ToolController=submitted,
    )
    target = object()
    stock = SimpleNamespace(BoundBox=SimpleNamespace(ZMax=0.15))
    stock.copy = lambda: stock
    submitted_job = SimpleNamespace(
        Tools=SimpleNamespace(Group=[submitted]), Operations=SimpleNamespace(Group=[operation])
    )
    original_job = SimpleNamespace(
        Tools=SimpleNamespace(Group=[original]),
        Operations=SimpleNamespace(Group=[]),
        Model=SimpleNamespace(Group=[SimpleNamespace(Shape=target)]),
        Stock=SimpleNamespace(Shape=stock),
    )
    project_path = tmp_path / "saved.FCStd"
    project_path.write_bytes(b"contract fixture, not a native document")
    source_path = tmp_path / "original.FCStd"
    source_path.write_bytes(b"source fixture, not a native document")
    submitted_document = SimpleNamespace(Name="Submitted", Objects=[submitted_job])
    original_document = SimpleNamespace(
        Name="Original", Objects=[original_job], recompute=lambda: None
    )
    manifest = {
        "tools": [{"id": identity, "number": number, "original_overhang_mm": source_overhang}]
    }

    def original_shapes(current):
        shapes = {
            role: getattr(current.OriginalAssembly, role)
            for role in ("Cutter", "Shank", "Holder")
            if getattr(current.OriginalAssembly, role) is not None
        }
        observations.shapes.append((current, shapes))
        return shapes

    class Safety:
        def __init__(self, actual_target, actual_stock, controllers, **kwargs):
            observations.safety.append((actual_target, actual_stock, controllers, kwargs))
            for current in controllers.values():
                original_shapes(current)

        def on_event(self, event):
            return None

        def consume(self, motions):
            assert list(motions) == ["fixture motion"]
            return {
                "status": safety_status,
                "feed_count": 1,
                "motion_count": 1,
                "motions_sha256": "fixture motion identity",
            }

    def replay_stock(actual_stock, controllers, motions, output, **kwargs):
        observations.stock.append((actual_stock, controllers, kwargs))
        assert list(motions) == ["fixture motion"]
        return {"motion_count": 1, "motions_sha256": "fixture motion identity"}

    modules = {
        "FreeCAD": SimpleNamespace(
            openDocument=lambda path: (
                submitted_document if Path(path) == project_path else original_document
            ),
            closeDocument=observations.closed.append,
        ),
        "FreeCADGui": SimpleNamespace(activateWorkbench=lambda name: None),
        "Part": SimpleNamespace(makeCompound=lambda shapes: tuple(shapes)),
        "Path": SimpleNamespace(),
        "Path.Base": SimpleNamespace(
            Util=SimpleNamespace(
                activeForOp=lambda current: True,
                toolControllerForOp=lambda current: current.ToolController,
            )
        ),
        "Path.Post": SimpleNamespace(),
        "Path.Post.Processor": SimpleNamespace(
            PostProcessorFactory=SimpleNamespace(
                get_post_processor=lambda job, name: SimpleNamespace(
                    export=lambda: [("fixture", f"G21 G90\nT{number} M6\nG1 Z-1 F60\nM2\n")]
                )
            )
        ),
        "tool_library": SimpleNamespace(
            register_library=lambda path: manifest, original_shapes=original_shapes
        ),
        "native_project": native_project,
        "machine_setup": machine_setup,
        "native_safety": SimpleNamespace(SafetyReplay=Safety),
        "native_stock": SimpleNamespace(replay_stock=replay_stock),
        "nc_motion": SimpleNamespace(
            iter_motions=lambda events, **kwargs: iter(["fixture motion"])
        ),
        "nc_replay": SimpleNamespace(
            iter_events=lambda path: iter([]),
            replay=lambda *args, **kwargs: {"events_file": "fixture"},
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        submitted=submitted,
        original=original,
        target=target,
        stock=stock,
        observations=observations,
        project=project_path,
        source=source_path,
        directory=tmp_path,
    )


def evaluate(pipeline, fixture):
    return pipeline.evaluate_native(
        fixture.project,
        fixture.source,
        fixture.directory / "library",
        fixture.directory / "replay",
        interpreter_runtime="fixture interpreter",
        collision_python="fixture collision backend",
        stock_binary="fixture stock backend",
    )


@pytest.mark.parametrize("number", [84, 85])
def test_preserves_original_cutter_and_fixed_overhang(monkeypatch, tmp_path, number):
    fixture = environment(monkeypatch, tmp_path, number)
    cutter = fixture.original.OriginalAssembly.Cutter
    result = evaluate(native_pipeline, fixture)
    assert result["status"] == "native_replay_complete"
    assert result["project"]["operations"][0]["holder_overhang_mm"] == 40.0
    assembly = fixture.original.OriginalAssembly
    assert assembly.Mount is None
    assert assembly.Shank is None
    assert assembly.Holder is None
    assert assembly.Cutter is cutter
    assert assembly.HolderOverhang.Value == 40.0
    target, stock, controllers, settings = fixture.observations.safety[0]
    assert target == (fixture.target,)
    assert stock is fixture.stock
    assert controllers[number] is fixture.original
    assert settings["linear_tolerance_mm"] == 0.001
    assert settings["surface_deflection_mm"] == 0.01
    assert settings["tool_gauge_offsets_mm"][number] == (0, 0, 0)
    assert fixture.observations.shapes[-1][1] == {"Cutter": cutter}
    assert fixture.observations.stock[0][1][number] is fixture.original
    assert "score" not in result


def test_mounted_tool_still_replays_selected_overhang(monkeypatch, tmp_path):
    fixture = environment(monkeypatch, tmp_path, 5, mounted=True)
    result = evaluate(native_pipeline, fixture)
    assert result["status"] == "native_replay_complete"
    assert result["project"]["operations"][0]["holder_overhang_mm"] == 13.0
    assert fixture.original.OriginalAssembly.Mount.Overhang == 13.0
    assert fixture.original.OriginalAssembly.Holder is not None


@pytest.mark.parametrize("overhang", [0.0, 41.0, -1.0, float("nan"), float("inf")])
def test_changed_holder_free_source_metadata_is_rejected(monkeypatch, tmp_path, overhang):
    fixture = environment(monkeypatch, tmp_path, 84)
    fixture.submitted.OriginalAssembly.HolderOverhang.Value = overhang
    result = evaluate(native_pipeline, fixture)
    assert result["status"] == "invalid_delivery"
    assert "overhang differs from its source" in result["error"]
    assert not fixture.observations.safety


def test_existing_holder_cannot_lose_its_mount(monkeypatch, tmp_path):
    fixture = environment(monkeypatch, tmp_path, 84)
    fixture.submitted.OriginalAssembly.Holder = object()
    result = evaluate(native_pipeline, fixture)
    assert result["status"] == "invalid_delivery"
    assert "requires its mounting control" in result["error"]
    assert not fixture.observations.safety


def test_original_package_holder_without_mount_remains_runtime_error(monkeypatch, tmp_path):
    fixture = environment(monkeypatch, tmp_path, 84)
    fixture.original.OriginalAssembly.Holder = object()
    result = evaluate(native_pipeline, fixture)
    assert result["status"] == "unavailable"
    assert "score" not in result
    assert not fixture.observations.safety


def test_replay_checks_fixed_overhang_against_original_document(monkeypatch, tmp_path):
    fixture = environment(monkeypatch, tmp_path, 84)
    fixture.original.OriginalAssembly.HolderOverhang.Value = 39.0
    result = evaluate(native_pipeline, fixture)
    assert result["status"] == "invalid_delivery"
    assert not fixture.observations.safety


@pytest.mark.parametrize("safety_status", ["collision", "unavailable"])
def test_mount_fix_does_not_bypass_safety_gate(monkeypatch, tmp_path, safety_status):
    fixture = environment(monkeypatch, tmp_path, 84, safety_status=safety_status)
    result = evaluate(native_pipeline, fixture)
    assert len(fixture.observations.safety) == 1
    assert not fixture.observations.stock
    if safety_status == "collision":
        assert result["status"] == "invalid_delivery"
        assert result["score"] == 0.0
    else:
        assert result["status"] == "unavailable"
        assert "score" not in result
