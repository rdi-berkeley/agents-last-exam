"""Bind original rotational cutters to native CAMotics primitive components."""

import math


def cutter_profile(controller, entry, tolerance_mm):
    import FreeCAD
    import Part

    from tool_library import original_shapes

    if not math.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Cutter profile tolerance must be finite and positive")
    if controller.SourceSHA256 != entry["source_sha256"]:
        raise ValueError("Original cutter source identity changed")
    if controller.OriginalToolId != entry["id"]:
        raise ValueError("Original cutter identifier changed")
    settings = entry["settings"]
    radius = float(settings["Diameter"]) / 2
    height = float(settings["Length"])
    kind = settings["Type"]
    if not all(math.isfinite(value) and value > 0 for value in (radius, height)):
        raise ValueError("Invalid original cutter dimensions")
    if kind == "end_mill" or (kind == "drill" and float(controller.Tool.TipAngle.Value) == 180):
        expected = Part.makeCylinder(radius, height)
        components = [["N", height, radius, radius, 0]]
    elif kind == "ball_nosed":
        if height < radius:
            raise ValueError("Original ball cutter is shorter than its hemispherical tip")
        expected = Part.makeSphere(radius, FreeCAD.Vector(0, 0, radius))
        expected = expected.common(
            Part.makeBox(
                2 * radius + 2,
                2 * radius + 2,
                radius,
                FreeCAD.Vector(-radius - 1, -radius - 1, 0),
            )
        )
        if height > radius:
            expected = expected.fuse(
                Part.makeCylinder(radius, height - radius, FreeCAD.Vector(0, 0, radius))
            )
        components = [["B", height, radius, 0, 0]]
    elif kind == "drill":
        angle = float(controller.Tool.TipAngle.Value)
        if not 0 < angle < 180:
            raise ValueError("Invalid original drill angle")
        tip = radius / math.tan(math.radians(angle / 2))
        if tip >= height:
            raise ValueError("Original drill tip exceeds its cutting length")
        expected = Part.makeCone(0, radius, tip).fuse(
            Part.makeCylinder(radius, height - tip, FreeCAD.Vector(0, 0, tip))
        )
        components = [["N", tip, radius, 0, 0], ["N", height - tip, radius, radius, tip]]
    elif kind == "tip_radiused":
        corner = float(settings["TipRadius"])
        if not tolerance_mm < corner < min(radius, height):
            raise ValueError("Original corner radius is outside the supported profile")
        start = FreeCAD.Vector(radius - corner, 0, 0)
        middle = FreeCAD.Vector(
            radius - corner + corner / math.sqrt(2), 0, corner - corner / math.sqrt(2)
        )
        end = FreeCAD.Vector(radius, 0, corner)
        edges = [
            Part.makeLine(FreeCAD.Vector(), start),
            Part.Arc(start, middle, end).toShape(),
            Part.makeLine(end, FreeCAD.Vector(radius, 0, height)),
            Part.makeLine(FreeCAD.Vector(radius, 0, height), FreeCAD.Vector(0, 0, height)),
            Part.makeLine(FreeCAD.Vector(0, 0, height), FreeCAD.Vector()),
        ]
        expected = Part.Face(Part.Wire(edges)).revolve(
            FreeCAD.Vector(), FreeCAD.Vector(0, 0, 1), 360
        )
        count = math.ceil((math.pi / 2) / (2 * math.acos(1 - tolerance_mm / corner)))
        previous_z, previous_radius = 0.0, radius - corner
        components = []
        for index in range(1, count + 1):
            angle = -math.pi / 2 + index / count * math.pi / 2
            next_z = corner + corner * math.sin(angle)
            next_radius = radius - corner + corner * math.cos(angle)
            components.append(["N", next_z - previous_z, next_radius, previous_radius, previous_z])
            previous_z, previous_radius = next_z, next_radius
        components.append(["N", height - corner, radius, radius, corner])
    else:
        raise ValueError(f"Unsupported original cutter profile: {kind}")
    original = original_shapes(controller)["Cutter"]
    extra = expected.cut(original).Volume
    missing = original.cut(expected).Volume
    if max(extra, missing) > 1e-6:
        raise ValueError("Original cutter does not match its source profile")
    return {
        "original_tool_id": entry["id"],
        "source_sha256": entry["source_sha256"],
        "components": components,
        "profile_tolerance_mm": tolerance_mm,
        "expected_minus_original_mm3": extra,
        "original_minus_expected_mm3": missing,
    }
