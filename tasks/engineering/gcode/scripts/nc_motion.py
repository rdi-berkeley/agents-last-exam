"""Stream LinuxCNC SAI canonical records as absolute millimeter XYZ motions.

Gauge positions follow LinuxCNC: Rz(program + G92) + G5x + compensation.
Returned tip positions subtract the installed tool's declared gauge calibration,
which is independent of active compensation. Original component placements must
not be applied twice. No calibration, blending, cutting or collision is inferred.
"""

import math


class MotionError(ValueError):
    """The canonical stream cannot be represented faithfully by this adapter."""


def _numbers(arguments, count):
    values = tuple(float(value) for value in arguments.replace(",", " ").split())
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise MotionError(f"Expected {count} finite canonical arguments")
    return values


def _integer(value):
    if not math.isfinite(value) or value != int(value):
        raise MotionError("Expected an integer canonical argument")
    return int(value)


def _xyz(values, scale):
    if any(value != 0 for value in values[3:]):
        raise MotionError("Nonzero rotary or auxiliary axes are not supported by XYZ replay")
    return tuple(value * scale for value in values[:3])


def _rotate(point, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        point[0] * cosine - point[1] * sine,
        point[0] * sine + point[1] * cosine,
        point[2],
    )


def _arc_points(start, end, center, turns, axes, chord_error_mm):
    """Yield native OCCT circle evaluations with canonical linear helical lift.

    Subdivision bounds radial sagitta plus the final endpoint radius mismatch.
    Helical lift is linear in angle, so it adds no deviation from each chord at
    the same interpolation fraction. Coordinates are those printed by SAI;
    upstream decimal rounding is a separate uncertainty, reported on each row.
    """
    first, second, axial = axes
    if turns == 0:
        if any(start[axis] != end[axis] for axis in (first, second)):
            raise MotionError("Zero-turn ARC_FEED changes its in-plane position")
        yield end, 0.0
        return
    radius = math.hypot(start[first] - center[first], start[second] - center[second])
    end_radius = math.hypot(end[first] - center[first], end[second] - center[second])
    mismatch = abs(radius - end_radius)
    budget = chord_error_mm - mismatch
    if radius <= 0 or budget <= 0:
        raise MotionError(
            "Arc radius or endpoint mismatch exceeds the requested chord-error budget"
        )
    start_angle = math.atan2(start[second] - center[second], start[first] - center[first])
    end_angle = math.atan2(end[second] - center[second], end[first] - center[first])
    direction = 1 if turns > 0 else -1
    partial = (direction * (end_angle - start_angle)) % math.tau
    roundoff = 32 * max(math.ulp(value) for value in (*start, *end, *center))
    if math.hypot(start[first] - end[first], start[second] - end[second]) <= roundoff:
        partial = math.tau
    sweep = direction * ((partial or math.tau) + (abs(turns) - 1) * math.tau)
    max_angle = min(math.pi / 2, 4 * math.asin(math.sqrt(min(budget / (2 * radius), 0.5))))
    count = max(1, math.ceil(abs(sweep) / max_angle))
    error_bound = 2 * radius * math.sin(abs(sweep) / (4 * count)) ** 2 + mismatch
    import FreeCAD
    import Part

    circle = Part.Circle(FreeCAD.Vector(), FreeCAD.Vector(0, 0, 1), radius)
    for index in range(1, count + 1):
        if index == count:
            yield end, error_bound
        else:
            fraction = index / count
            radial = circle.value(start_angle + sweep * fraction)
            point = list(start)
            point[first] = center[first] + radial.x
            point[second] = center[second] + radial.y
            point[axial] = start[axial] + (end[axial] - start[axial]) * fraction
            yield tuple(point), error_bound


def iter_motions(
    events,
    *,
    initial_machine_mm,
    initial_tool,
    tool_gauge_offsets_mm,
    chord_error_mm,
    pocket_tools=None,
    on_event=None,
):
    """Yield motion dictionaries for native stock/safety consumers, one at a time.

    initial_machine_mm is the physical gauge start corresponding to the
    interpreter setup, not the first endpoint. tool_gauge_offsets_mm maps each
    installed tool number to its XYZ gauge-minus-tip calibration in millimeters.
    This mapping is required: a no-TLO virtual tip setup declares zero vectors;
    normal G43 uses actual declared calibration, never an inferred tool length.
    initial_tool is its installed public ordinal (0 means empty).
    SELECT_TOOL carries a tool number, whereas
    CHANGE_TOOL/CHANGE_TOOL_NUMBER carry internal table indices. pocket_tools
    maps those indices, not table P words, to tool numbers; M61 requires it.

    Arcs use installed FreeCAD Part.Circle evaluations; straight-only streams
    require no FreeCAD import. Units, G5x/G92, rotation, tool compensation and
    physical position persist independently. Unit/offset/tool events do not
    create motion. LinuxCNC has already resolved distance modes and cycles.

    on_event receives every supported original record, including dwell, stop,
    coolant, override and diagnostic events that do not alter nominal geometry.
    Feed/spindle/path-control state is retained on rows; G64 is the commanded
    path, not an emulation of the controller's blending. Unknown events, probe,
    tapping, NURBS, I/O, tool-table mutation and non-XYZ motion fail explicitly.
    A consumer must treat late errors as failure of the entire replay.
    """
    position = tuple(float(value) for value in initial_machine_mm)
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise MotionError("initial_machine_mm must contain three finite millimeter coordinates")
    tool = _integer(initial_tool)
    if tool < 0 or not math.isfinite(chord_error_mm) or chord_error_mm <= 0:
        raise MotionError("Tool number must be nonnegative and chord_error_mm finite and positive")
    calibration = {}
    for number, offset in tool_gauge_offsets_mm.items():
        number = _integer(number)
        offset = tuple(float(value) for value in offset)
        if number < 0 or len(offset) != 3 or not all(math.isfinite(value) for value in offset):
            raise MotionError(
                "Tool gauge calibration requires nonnegative tools and finite XYZ offsets"
            )
        calibration[number] = offset
    if tool not in calibration:
        raise MotionError("Initial tool has no declared gauge calibration")
    pocket_tools = dict(pocket_tools or {})
    scale = None
    g5x = (0.0, 0.0, 0.0)
    g92 = (0.0, 0.0, 0.0)
    tool_offset = (0.0, 0.0, 0.0)
    rotation = 0.0
    plane = "CANON_PLANE_XY"
    planes = {"CANON_PLANE_XY": (0, 1, 2), "CANON_PLANE_XZ": (2, 0, 1), "CANON_PLANE_YZ": (1, 2, 0)}
    selected_tool = None
    feed = 0.0
    feed_mode = 0
    spindle_rpm = 0.0
    spindle_direction = 0
    motion_mode = "CANON_CONTINUOUS"
    path_tolerance = 0.0
    naivecam_tolerance = 0.0
    ended = False
    previous_ordinal = 0
    no_arguments = {
        "ON_RESET",
        "START_CHANGE",
        "PROGRAM_STOP",
        "OPTIONAL_PROGRAM_STOP",
        "FLOOD_ON",
        "FLOOD_OFF",
        "MIST_ON",
        "MIST_OFF",
        "ENABLE_FEED_OVERRIDE",
        "DISABLE_FEED_OVERRIDE",
        "ENABLE_FEED_HOLD",
        "DISABLE_FEED_HOLD",
        "DISABLE_ADAPTIVE_FEED",
        "STOP_CUTTER_RADIUS_COMPENSATION",
    }
    text_events = {"COMMENT", "MESSAGE", "LOG", "LOGOPEN", "LOGAPPEND"}

    def machine(point):
        rotated = _rotate(tuple(point[axis] + g92[axis] for axis in range(3)), rotation)
        return tuple(rotated[axis] + g5x[axis] + tool_offset[axis] for axis in range(3))

    def program(point):
        translated = tuple(point[axis] - g5x[axis] - tool_offset[axis] for axis in range(3))
        rotated = _rotate(translated, -rotation)
        return tuple(rotated[axis] - g92[axis] for axis in range(3))

    for event in events:
        try:
            ordinal = _integer(event["ordinal"])
            if ordinal <= previous_ordinal:
                raise MotionError("Canonical ordinals must be strictly increasing")
            previous_ordinal = ordinal
            name, arguments = event["name"], event["arguments"]
            points = None
            turns = 0
            if name == "USE_LENGTH_UNITS":
                units = {"CANON_UNITS_MM": 1.0, "CANON_UNITS_INCHES": 25.4}
                if arguments not in units:
                    raise MotionError(f"Unsupported units: {arguments}")
                scale = units[arguments]
            elif scale is None:
                raise MotionError("Canonical stream must establish units before other events")
            elif name == "SET_G5X_OFFSET":
                values = _numbers(arguments, 7)
                if not 1 <= _integer(values[0]) <= 9:
                    raise MotionError("Invalid G5x frame index")
                g5x = _xyz(values[1:], scale)
            elif name == "SET_G92_OFFSET":
                g92 = _xyz(_numbers(arguments, 6), scale)
            elif name == "USE_TOOL_LENGTH_OFFSET":
                tool_offset = _xyz(_numbers(arguments, 9), scale)
            elif name == "SET_XY_ROTATION":
                rotation = math.radians(_numbers(arguments, 1)[0])
            elif name == "SELECT_PLANE":
                if arguments not in planes:
                    raise MotionError(f"Unsupported plane: {arguments}")
                plane = arguments
            elif name == "SELECT_TOOL":
                selected_tool = _integer(_numbers(arguments, 1)[0])
                if selected_tool < 0:
                    raise MotionError("Negative selected tool")
            elif name in {"CHANGE_TOOL", "CHANGE_TOOL_NUMBER"}:
                pocket = _integer(_numbers(arguments, 1)[0])
                if name == "CHANGE_TOOL" and selected_tool is not None:
                    if pocket in pocket_tools and pocket_tools[pocket] != selected_tool:
                        raise MotionError(
                            "Selected tool disagrees with the internal pocket mapping"
                        )
                    tool = selected_tool
                    pocket_tools[pocket] = tool
                elif pocket in pocket_tools:
                    tool = _integer(pocket_tools[pocket])
                else:
                    raise MotionError(
                        "Tool change requires a selected tool or explicit pocket mapping"
                    )
            elif name == "SET_FEED_RATE":
                feed = _numbers(arguments, 1)[0] * scale
                if feed < 0:
                    raise MotionError("Negative feed")
            elif name == "SET_FEED_MODE":
                spindle, mode = _numbers(arguments, 2)
                if spindle != 0 or mode not in (0, 1):
                    raise MotionError("Unsupported feed mode or spindle")
                feed_mode = int(mode)
            elif name == "SET_FEED_REFERENCE":
                if arguments not in {"CANON_XYZ", "CANON_WORKPIECE"}:
                    raise MotionError("Unsupported feed reference")
            elif name == "SET_SPINDLE_SPEED":
                spindle, spindle_rpm = _numbers(arguments, 2)
                if spindle != 0 or spindle_rpm < 0:
                    raise MotionError("Unsupported spindle or negative spindle speed")
            elif name in {
                "START_SPINDLE_CLOCKWISE",
                "START_SPINDLE_COUNTERCLOCKWISE",
                "STOP_SPINDLE_TURNING",
            }:
                if _numbers(arguments, 1)[0] != 0:
                    raise MotionError("Only spindle zero is supported")
                spindle_direction = {
                    "START_SPINDLE_CLOCKWISE": 1,
                    "START_SPINDLE_COUNTERCLOCKWISE": -1,
                    "STOP_SPINDLE_TURNING": 0,
                }[name]
            elif name == "SET_SPINDLE_MODE":
                if _numbers(arguments, 2) != (0.0, 0.0):
                    raise MotionError("Constant surface speed or secondary spindle is unsupported")
            elif name == "SET_MOTION_CONTROL_MODE":
                fields = [field.strip() for field in arguments.split(",")]
                motion_mode = fields[0]
                if motion_mode in {"CANON_EXACT_PATH", "CANON_EXACT_STOP"} and len(fields) == 1:
                    path_tolerance = 0.0
                elif motion_mode == "CANON_CONTINUOUS" and len(fields) == 2:
                    path_tolerance = _numbers(fields[1], 1)[0] * scale
                    if path_tolerance < 0:
                        raise MotionError("Negative path tolerance")
                else:
                    raise MotionError("Unsupported path-control mode")
            elif name == "SET_NAIVECAM_TOLERANCE":
                naivecam_tolerance = _numbers(arguments, 1)[0] * scale
                if naivecam_tolerance < 0:
                    raise MotionError("Negative naive-CAM tolerance")
            elif name in {"STRAIGHT_TRAVERSE", "STRAIGHT_FEED"}:
                endpoint = _xyz(_numbers(arguments, 6), scale)
                points = ((endpoint, 0.0),)
            elif name == "ARC_FEED":
                values = _numbers(arguments, 9)
                _xyz((0.0, 0.0, 0.0) + values[6:], scale)
                first, second, axial = planes[plane]
                start = program(position)
                endpoint, center = list(start), list(start)
                endpoint[first], endpoint[second], endpoint[axial] = (
                    values[0] * scale,
                    values[1] * scale,
                    values[5] * scale,
                )
                center[first], center[second] = values[2] * scale, values[3] * scale
                turns = _integer(values[4])
                points = _arc_points(
                    start, tuple(endpoint), tuple(center), turns, planes[plane], chord_error_mm
                )
            elif name == "DWELL":
                if _numbers(arguments, 1)[0] < 0:
                    raise MotionError("Negative dwell")
            elif name in {"ENABLE_SPEED_OVERRIDE", "DISABLE_SPEED_OVERRIDE"}:
                if _numbers(arguments, 1)[0] != 0:
                    raise MotionError("Only spindle zero is supported")
            elif name in no_arguments or name == "PROGRAM_END":
                _numbers(arguments, 0)
                if name == "PROGRAM_END":
                    ended = True
            elif name in text_events:
                if not arguments.startswith('"') or not arguments.endswith('"'):
                    raise MotionError("Malformed canonical diagnostic text")
            else:
                raise MotionError(f"Unsupported canonical event: {name}")
            if on_event is not None:
                on_event(event)
            if points is not None:
                rapid = name == "STRAIGHT_TRAVERSE"
                if tool not in calibration:
                    raise MotionError(f"Tool {tool} has no declared gauge calibration")
                if ended:
                    raise MotionError("Motion after PROGRAM_END")
                if not rapid and (tool <= 0 or feed <= 0):
                    raise MotionError("Cutting motion requires an installed tool and positive feed")
                for endpoint, error_bound in points:
                    end = machine(endpoint)
                    yield {
                        "start_mm": tuple(
                            position[axis] - calibration[tool][axis] for axis in range(3)
                        ),
                        "end_mm": tuple(end[axis] - calibration[tool][axis] for axis in range(3)),
                        "gauge_start_mm": position,
                        "gauge_end_mm": end,
                        "tool_gauge_offset_mm": calibration[tool],
                        "tool": tool,
                        "rapid": rapid,
                        "chord_error_mm": error_bound,
                        "source_ordinal": ordinal,
                        "source_block": event.get("source_block"),
                        "canonical_name": name,
                        "source_units": "mm" if scale == 1.0 else "inch",
                        "canonical_rounding_mm": 0.00005 * scale,
                        "plane": plane,
                        "arc_turns": turns,
                        "tool_offset_mm": tool_offset,
                        "feed_mm_min": feed if feed_mode == 0 else None,
                        "feed_mm_rev": feed if feed_mode == 1 else None,
                        "spindle_rpm": spindle_rpm,
                        "spindle_direction": spindle_direction,
                        "motion_mode": motion_mode,
                        "path_tolerance_mm": path_tolerance,
                        "naivecam_tolerance_mm": naivecam_tolerance,
                    }
                    position = end
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise MotionError(f"Canonical event {event.get('ordinal')}: {error}") from error
