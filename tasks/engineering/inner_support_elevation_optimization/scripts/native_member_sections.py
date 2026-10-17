"""Bind declared crown and support sections to members of the submitted graph."""

import math


def on_perimeter(first, second, perimeter, tolerance=1e-7):
    direction = [second[axis] - first[axis] for axis in range(2)]
    length = math.hypot(*direction)
    if length == 0:
        raise ValueError("Zero-length declared support member")
    intervals = []
    for start, end in perimeter:
        edge = [end[axis] - start[axis] for axis in range(2)]
        edge_length = math.hypot(*edge)
        if edge_length == 0:
            raise ValueError("Zero-length declared perimeter edge")
        member_offsets = [
            [point[axis] - start[axis] for axis in range(2)] for point in (first, second)
        ]
        offsets = [[point[axis] - first[axis] for axis in range(2)] for point in (start, end)]
        if any(
            abs(offset[0] * edge[1] - offset[1] * edge[0]) / edge_length > tolerance
            for offset in member_offsets
        ):
            continue
        projections = [
            sum(value * delta for value, delta in zip(offset, direction)) / length
            for offset in offsets
        ]
        intervals.append((max(0.0, min(projections)), min(length, max(projections))))
    covered = 0.0
    for lower, upper in sorted(intervals):
        if upper < 0 or lower > length:
            continue
        if lower > covered + tolerance:
            return False
        covered = max(covered, upper)
    return covered >= length - tolerance


def member_sections(family, system, elevation, *, crown_only=False):
    definition = next(item for item in family["braces"] if item["system"] == system)
    choices = family["engineering_choices"]
    crown = choices.get("crowns", {}).get(system)
    at_crown = crown is not None and abs(elevation - crown["axis_z_m"]) < 1e-8
    if crown_only and not at_crown:
        raise ValueError("Crown-only installation must use its declared elevation")
    polygon = family[f"{system}_polygon"]
    perimeter = list(zip(polygon, polygon[1:] + polygon[:1]))
    segments = definition["crown_segments_xy_m" if crown_only else "segments_xy_m"]
    members = []
    for segment in segments:
        first, second = segment
        perimeter_member = on_perimeter(first, second, perimeter)
        if crown_only and not perimeter_member:
            raise ValueError("Declared crown member does not lie on the submitted wall perimeter")
        role = "crown" if at_crown and perimeter_member else "support"
        section = crown["section_m"] if role == "crown" else choices["beam_section_m"]
        section_properties(section)
        members.append({"segment": segment, "role": role, "section_m": section})
    return members


def section_properties(section):
    width, height = section["width"], section["height"]
    if any(not math.isfinite(value) or value <= 0 for value in (width, height)):
        raise ValueError("Declared beam section dimensions must be finite and positive")
    major, minor = max(width, height), min(width, height)
    return {
        "CROSS_AREA": width * height,
        "I22": width * height**3 / 12,
        "I33": height * width**3 / 12,
        "TORSIONAL_INERTIA": (
            major * minor**3 * (1 / 3 - 0.21 * minor / major * (1 - minor**4 / (12 * major**4)))
        ),
    }


def audit_support_sections(runner):
    checked = []
    for installation in runner.receipt.get("support_installations", []):
        members = member_sections(
            runner.family,
            installation["system"],
            installation["elevation_m"],
            crown_only=installation["crown_only"],
        )
        identifiers = installation["native_element_ids"]
        if len(identifiers) != len(members) or len(set(identifiers)) != len(identifiers):
            raise ValueError("Native member count differs from declared support graph")
        counts = {"crown": 0, "support": 0}
        properties_by_role = {"crown": set(), "support": set()}
        for identifier, member in zip(identifiers, members):
            element = runner.part.GetElement(identifier)
            if not element.Is(runner.Kratos.ACTIVE):
                raise ValueError("Installed native support member is inactive")
            for name, expected in section_properties(member["section_m"]).items():
                actual = element.Properties[runner.Kratos.KratosGlobals.GetVariable(name)]
                if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(
                        f"Native role section mismatch at {installation['label']} "
                        f"element {identifier} ({member['role']}) {name}: "
                        f"actual={actual}, expected={expected}"
                    )
            counts[member["role"]] += 1
            properties_by_role[member["role"]].add(element.Properties.Id)
        checked.append(
            {
                "label": installation["label"],
                "member_counts": counts,
                "property_ids_by_role": {
                    role: sorted(identifiers) for role, identifiers in properties_by_role.items()
                },
            }
        )
    return {"version": 1, "passed": True, "installations": checked}
