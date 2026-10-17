"""Load public CAM assets and bind original components to native toolcontrollers.

Persistent component links describe geometry, not completed collision checks.
"""

import json
from pathlib import Path


def source_flutes(settings):
    value = int(settings["NumberOfFlutes"])
    if value < 0 or float(settings["NumberOfFlutes"]) != value:
        raise ValueError("Source flute count must be a nonnegative integer")
    return value, value != 0


def register_library(library_root):
    from Path import Preferences
    from Path.Tool.camassets import cam_assets

    library_root = Path(library_root)
    manifest = json.loads((library_root / "original-tools.json").read_text())
    for filename in manifest["templates"]:
        cam_assets.add_file("toolbitshape", library_root / "Tools/Shape" / filename)
    for entry in manifest["tools"]:
        cam_assets.add_file("toolbit", library_root / entry["bit_file"])
    library_uri = cam_assets.add_file("toolbitlibrary", library_root / manifest["library_file"])
    Preferences.setLastToolLibrary(str(library_uri))
    return manifest


def bind_controller(controller, assembly, entry):
    if controller.Tool.ToolBitID != entry["id"]:
        raise ValueError("Controller and source tool identities differ")
    if assembly.SourceSHA256 != entry["source_sha256"]:
        raise ValueError("Assembly and source tool identities differ")
    bindings = {
        "OriginalAssembly": assembly,
        "OriginalCutter": assembly.Cutter,
        "OriginalShank": assembly.Shank,
        "OriginalHolder": assembly.Holder,
    }
    for name, component in bindings.items():
        if component is not None and component.Document != controller.Document:
            raise ValueError("Original components must belong to the controller document")
        if name not in controller.PropertiesList:
            controller.addProperty("App::PropertyLink", name, "Original tool")
        setattr(controller, name, component)
    flute_count, known = source_flutes(entry["settings"])
    values = (
        ("App::PropertyString", "OriginalToolId", entry["id"]),
        ("App::PropertyString", "SourceSHA256", entry["source_sha256"]),
        ("App::PropertyInteger", "SourceToolNumber", entry["source_tool_number"]),
        ("App::PropertyInteger", "SourceFlutes", flute_count),
        ("App::PropertyBool", "SourceFlutesKnown", known),
        (
            "App::PropertyString",
            "SourceFlutesMeaning",
            "Specified by source" if known else "Unknown: source records zero",
        ),
        (
            "App::PropertyString",
            "OriginalParameters",
            json.dumps(entry["parameters"], sort_keys=True),
        ),
    )
    for owner in (controller, controller.Tool):
        for kind, name, value in values:
            if name not in owner.PropertiesList:
                owner.addProperty(kind, name, "Original tool")
            setattr(owner, name, value)
            owner.setEditorMode(name, 1)
    return controller


def create_controller(job, library_root, entry, assembly):
    import FreeCAD
    from Path.Tool import Controller
    from Path.Tool.toolbit import ToolBit

    document = job.Document
    FreeCAD.setActiveDocument(document.Name)
    tool = ToolBit.from_file(Path(library_root) / entry["bit_file"]).attach_to_doc(doc=document)
    tool.ToolBitID = entry["id"]
    tool.Visibility = False
    controller = Controller.Create(
        name=f"TC: {entry['name']}", tool=tool, toolNumber=entry["number"]
    )
    bind_controller(controller, assembly, entry)
    job.Proxy.addToolController(controller)
    return controller


def bind_job(job, library_root):
    manifest = json.loads((Path(library_root) / "original-tools.json").read_text())
    entries = {entry["id"]: entry for entry in manifest["tools"]}
    assemblies = {
        obj.OriginalToolId: obj
        for obj in job.Document.Objects
        if obj.TypeId == "App::DocumentObjectGroup" and hasattr(obj, "OriginalToolId")
    }
    for controller in job.Tools.Group:
        identity = controller.Tool.ToolBitID
        bind_controller(controller, assemblies[identity], entries[identity])
    return len(job.Tools.Group)


def restore_document(document, library_root):
    import FreeCAD
    from Path.Tool.toolbit import ToolBit

    library_root = Path(library_root)
    manifest = json.loads((library_root / "original-tools.json").read_text())
    entries = {entry["id"]: entry for entry in manifest["tools"]}
    restored = set()
    jobs = [obj for obj in document.Objects if hasattr(obj, "Tools") and hasattr(obj, "Operations")]
    for job in jobs:
        for controller in job.Tools.Group:
            tool = controller.Tool
            if tool.Name in restored:
                continue
            entry = entries[tool.ToolBitID]
            attributes = json.loads((library_root / entry["bit_file"]).read_text())
            for name in attributes["parameter"]:
                if hasattr(tool, name):
                    value = getattr(tool, name)
                    attributes["parameter"][name] = (
                        value if isinstance(value, (str, int, float, bool)) else str(value)
                    )
            proxy = tool.Proxy
            proxy._visual_update_queued = False
            observer = getattr(proxy, "_recompute_observer", None)
            if observer is not None:
                FreeCAD.removeDocumentObserver(observer)
                del proxy._recompute_observer
            proxy._tool_bit_shape = ToolBit.from_dict(attributes)._tool_bit_shape
            restored.add(tool.Name)
        bind_job(job, library_root)
    document.recompute()
    return len(restored)


def original_shapes(controller):
    import FreeCAD

    tool = controller.Tool
    if tool.ToolBitID != controller.OriginalToolId:
        raise ValueError("Controller tool no longer matches its original components")
    parameters = json.loads(controller.OriginalParameters)
    for name in (
        "Diameter",
        "Length",
        "CuttingEdgeHeight",
        "ShankDiameter",
        "CornerRadius",
        "TipAngle",
    ):
        if name not in parameters:
            continue
        actual = float(getattr(tool, name).Value)
        expected = float(FreeCAD.Units.Quantity(parameters[name]).Value)
        if abs(actual - expected) > 1e-8:
            raise ValueError(f"{name} differs from the original tool; component binding is stale")
    shapes = {}
    for role in ("Cutter", "Shank", "Holder"):
        component = getattr(controller, f"Original{role}")
        if component != getattr(controller.OriginalAssembly, role):
            raise ValueError("Controller component differs from its source assembly")
        if component is not None:
            if component.Document != controller.Document:
                raise ValueError("Original component is an external document link")
            shape = component.Shape.copy()
            if shape.isNull() or not shape.isValid():
                raise ValueError(f"Invalid original {role.lower()} shape")
            shapes[role] = shape
    if "Cutter" not in shapes:
        raise ValueError("Original cutter is missing")
    return shapes


def set_simulator_tool(simulator, controller, tolerance):
    if tolerance <= 0:
        raise ValueError("Simulation tessellation tolerance must be positive")
    shapes = original_shapes(controller)
    simulator.SetToolShape(shapes["Cutter"], tolerance)
    return shapes
