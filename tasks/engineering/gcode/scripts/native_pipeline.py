"""Replay a saved CAM project using original inputs and trusted native components."""

import json
from pathlib import Path
import time


def evaluate_native(
    project,
    source_document,
    library_root,
    work_dir,
    *,
    interpreter_runtime,
    collision_python,
    stock_binary,
    clearance_mm=5.0,
    safety_seconds=3000,
    resolution_mm=0.1,
):
    import FreeCAD
    import FreeCADGui
    import Part

    from machine_setup import prepare_program
    from native_project import InvalidDelivery, export_project
    from native_safety import SafetyReplay
    from native_stock import replay_stock
    from nc_motion import iter_motions
    from nc_replay import iter_events, replay
    from tool_library import register_library

    began = time.monotonic()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "unavailable", "geometry_scored": False, "full_task_accepted": False}
    document = None
    try:
        program = work_dir / "saved-project.ngc"
        exported = export_project(project, library_root, program)
        report["project"] = exported
        register_library(library_root)
        FreeCADGui.activateWorkbench("CAMWorkbench")
        document = FreeCAD.openDocument(str(source_document))
        jobs = [
            obj for obj in document.Objects if hasattr(obj, "Tools") and hasattr(obj, "Operations")
        ]
        if len(jobs) != 1:
            raise RuntimeError("Original public package does not contain one native CAM job")
        job = jobs[0]
        controllers = {int(obj.ToolNumber): obj for obj in job.Tools.Group}
        mounting = {}
        for operation in exported["operations"]:
            number = operation["tool"]
            if number in mounting and mounting[number] != operation["holder_overhang_mm"]:
                raise InvalidDelivery(
                    "One tool number has inconsistent mounting in saved operations"
                )
            mounting[number] = operation["holder_overhang_mm"]
        for number, overhang in mounting.items():
            assembly = controllers[number].OriginalAssembly
            if assembly.Mount is None:
                if assembly.Holder is not None:
                    raise RuntimeError("Original holder requires its mounting control")
                if overhang != float(assembly.HolderOverhang.Value):
                    raise InvalidDelivery("Holder-free assembly overhang differs from its source")
            else:
                assembly.Mount.Overhang = overhang
        document.recompute()
        target = Part.makeCompound([obj.Shape for obj in job.Model.Group])
        stock = job.Stock.Shape.copy()
        table = work_dir / "tools.tbl"
        table.write_text(
            "".join(
                f"T{number} P{index} D{float(controllers[number].Tool.Diameter.Value):.12g} Z0\n"
                for index, number in enumerate(sorted(mounting), 1)
            )
        )
        prepared = work_dir / "initialized.ngc"
        setup = prepare_program(
            program, prepared, stock_top_mm=stock.BoundBox.ZMax, clearance_mm=clearance_mm
        )
        report["setup"] = setup
        machine = work_dir / "machine.ini"
        machine.write_text("[TRAJ]\nLINEAR_UNITS = mm\n")
        canonical = replay(
            prepared,
            table,
            runtime_root=interpreter_runtime,
            work_root=work_dir,
            ini_file=machine,
            timeout=120,
        )
        report["canonical"] = canonical
        calibration = {number: (0, 0, 0) for number in (0, *mounting)}
        settings = {
            "initial_machine_mm": setup["initial_machine_mm"],
            "initial_tool": setup["initial_tool"],
            "tool_gauge_offsets_mm": calibration,
            "pocket_tools": {index: number for index, number in enumerate(sorted(mounting), 1)},
        }
        safety = SafetyReplay(
            target,
            stock,
            controllers,
            work_dir=work_dir,
            linear_tolerance_mm=0.001,
            time_limit_s=safety_seconds,
            collision_python=collision_python,
            surface_deflection_mm=0.01,
            **settings,
        )
        motions = iter_motions(
            iter_events(canonical["events_file"]),
            chord_error_mm=0.001,
            on_event=safety.on_event,
            **settings,
        )
        report["safety"] = safety.consume(motions)
        if report["safety"]["status"] == "collision":
            report.update(status="invalid_delivery", score=0.0)
        elif report["safety"]["status"] == "clear":
            if report["safety"]["feed_count"] == 0:
                raise InvalidDelivery("Saved project contains no cutting motions")
            motions = iter_motions(
                iter_events(canonical["events_file"]), chord_error_mm=0.001, **settings
            )
            report["stock"] = replay_stock(
                stock,
                {number: controllers[number] for number in mounting},
                motions,
                work_dir / "replayed-stock.stl",
                resolution_mm=resolution_mm,
                tool_tolerance_mm=0.001,
                native_binary=stock_binary,
                library_root=library_root,
            )
            if report["stock"]["motion_count"] != report["safety"]["motion_count"]:
                raise RuntimeError("Safety and stock replay consumed different motion counts")
            if report["stock"]["motions_sha256"] != report["safety"]["motions_sha256"]:
                raise RuntimeError("Safety and stock replay consumed different physical motions")
            report["status"] = "native_replay_complete"
    except InvalidDelivery as error:
        report.update(status="invalid_delivery", score=0.0, error=str(error))
    except Exception as error:
        report.update(status="unavailable", error=f"{type(error).__name__}: {error}")
    finally:
        if document is not None:
            FreeCAD.closeDocument(document.Name)
        report["seconds"] = time.monotonic() - began
        (work_dir / "result.json").write_text(json.dumps(report, indent=2))
    return report
