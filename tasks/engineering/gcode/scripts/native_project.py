"""Export a saved native CAM job using the installed LinuxCNC postprocessor."""

import hashlib
import json
import math
from pathlib import Path


class InvalidDelivery(ValueError):
    """A required saved project or original tool selection is invalid."""


def export_project(project, library_root, output):
    import FreeCAD
    import FreeCADGui
    from Path.Base import Util
    from Path.Post.Processor import PostProcessorFactory

    from tool_library import original_shapes, register_library

    project, library_root, output = map(Path, (project, library_root, output))
    if not project.is_file():
        raise InvalidDelivery("Required saved CAM project is missing")
    if output.exists():
        raise FileExistsError(output)
    before = hashlib.sha256(project.read_bytes()).hexdigest()
    manifest = register_library(library_root)
    entries = {entry["id"]: entry for entry in manifest["tools"]}
    FreeCADGui.activateWorkbench("CAMWorkbench")
    document = FreeCAD.openDocument(str(project))
    try:
        jobs = [
            obj for obj in document.Objects if hasattr(obj, "Tools") and hasattr(obj, "Operations")
        ]
        if len(jobs) != 1:
            raise InvalidDelivery("Submit one saved CAM job for the original workpiece")
        job = jobs[0]
        operations = [obj for obj in job.Operations.Group if Util.activeForOp(obj)]
        if not operations:
            raise InvalidDelivery("Saved project contains no active machining operation")
        controllers = {}
        operation_rows = []
        for operation in operations:
            if "Invalid" in operation.State or not operation.Path.Commands:
                raise InvalidDelivery(f"Saved operation {operation.Name} has no valid toolpath")
            controller = Util.toolControllerForOp(operation)
            if controller is None:
                raise InvalidDelivery(f"Saved operation {operation.Name} has no toolcontroller")
            number = int(controller.ToolNumber)
            identity = getattr(controller, "OriginalToolId", None)
            entry = entries.get(identity)
            if entry is None or number != entry["number"]:
                raise InvalidDelivery("Operation uses a tool outside the original numbered library")
            if number in controllers and controllers[number] is not controller:
                raise InvalidDelivery("Distinct controllers share one executed tool number")
            original_shapes(controller)
            assembly = controller.OriginalAssembly
            if assembly.Mount is None:
                if assembly.Holder is not None:
                    raise InvalidDelivery("Original holder requires its mounting control")
                overhang = float(assembly.HolderOverhang.Value)
                if overhang != float(entry["original_overhang_mm"]):
                    raise InvalidDelivery("Holder-free assembly overhang differs from its source")
            else:
                overhang = float(assembly.Mount.Overhang.Value)
            if not math.isfinite(overhang) or overhang < 0:
                raise InvalidDelivery("Tool mounting overhang must be finite and nonnegative")
            controllers[number] = controller
            operation_rows.append(
                {
                    "name": operation.Name,
                    "tool": number,
                    "original_tool_id": identity,
                    "commands": len(operation.Path.Commands),
                    "holder_overhang_mm": overhang,
                    "spindle_rpm": float(controller.SpindleSpeed),
                }
            )
        for controller in controllers.values():
            controller.Proxy.execute(controller)
        job.PostProcessorArgs = "--no-show-editor --no-header --precision 6 --no-tlo"
        job.SplitOutput = False
        processor = PostProcessorFactory.get_post_processor(job, "linuxcnc")
        if processor is None:
            raise RuntimeError("Installed FreeCAD LinuxCNC postprocessor is unavailable")
        sections = processor.export()
        if len(sections) != 1 or not isinstance(sections[0][1], str) or not sections[0][1].strip():
            raise RuntimeError("Native postprocessor did not produce one complete program")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(sections[0][1])
        report = {
            "project_sha256": before,
            "nc_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "operations": operation_rows,
            "tool_numbers": sorted(controllers),
            "exporter": "FreeCAD native LinuxCNC postprocessor",
            "coordinate_contract": "Millimetre tool-tip virtual machine; no tool-length compensation",
            "paths_recomputed": False,
            "safety_checked": False,
        }
    finally:
        FreeCAD.closeDocument(document.Name)
    if hashlib.sha256(project.read_bytes()).hexdigest() != before:
        raise RuntimeError("Read-only export changed the submitted project")
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
