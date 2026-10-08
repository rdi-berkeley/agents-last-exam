"""Stream physical XYZ motions through native OCCT safety proofs.

Remaining-stock bounds are conservative certificates, not a material-removal
score. Unresolved geometry and incomplete event coverage cannot certify safety.
"""

import hashlib
from array import array
import json
import math
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time


class SafetyUnavailable(RuntimeError):
    """The supplied geometry, stream or resource budget cannot establish safety."""


def _point(values):
    point = tuple(float(value) for value in values)
    if len(point) != 3 or not all(map(math.isfinite, point)):
        raise ValueError("Safety replay requires finite XYZ coordinates")
    return point


def _number(value):
    if type(value) is not int or value < 0:
        raise ValueError("Tool numbers must be nonnegative integers")
    return value


def _bounds(shape):
    box = shape.BoundBox
    return (box.XMin, box.YMin, box.ZMin), (box.XMax, box.YMax, box.ZMax)


def _swept_bounds(bounds, start, end, padding=0.0):
    lower, upper = bounds
    return (
        tuple(lower[axis] + min(start[axis], end[axis]) - padding for axis in range(3)),
        tuple(upper[axis] + max(start[axis], end[axis]) + padding for axis in range(3)),
    )


def _separated(first, second, margin=0.0):
    return any(
        first[1][axis] + margin < second[0][axis] or second[1][axis] + margin < first[0][axis]
        for axis in range(3)
    )


class SafetyReplay:
    """Consume iter_motions with on_event bound to this instance.

    All geometry uses the original machine frame and tip-relative component
    placements. Calibration maps tool ordinals to independent gauge-minus-tip
    XYZ offsets. work_dir stores streamed evidence on caller-selected storage.
    A report's clear status requires exhaustion and observed PROGRAM_END.
    """

    def __init__(
        self,
        target,
        initial_stock,
        controllers,
        *,
        work_dir,
        initial_machine_mm,
        initial_tool,
        tool_gauge_offsets_mm,
        linear_tolerance_mm,
        time_limit_s,
        pocket_tools=None,
        collision_python=None,
        surface_deflection_mm=0.01,
    ):
        import FreeCAD
        import Part

        for value in (linear_tolerance_mm, time_limit_s, surface_deflection_mm):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Safety tolerance and time budget must be finite and positive")
        self.app, self.part = FreeCAD, Part
        self.tolerance = float(linear_tolerance_mm)
        self.started = time.monotonic()
        self.deadline = self.started + time_limit_s
        self.controllers = dict(controllers)
        self.calibration = {
            _number(number): _point(offset) for number, offset in tool_gauge_offsets_mm.items()
        }
        self.tool = _number(initial_tool)
        self.gauge = _point(initial_machine_mm)
        self.pockets = dict(pocket_tools or {})
        self.selected = None
        self.target = target.copy()
        self.initial_stock = initial_stock.copy()
        if self.target.isNull() or not self.target.Faces or not self.target.isValid():
            raise SafetyUnavailable(
                "Original target requires valid native faces; open faces are supported"
            )
        self._validate_solid(self.initial_stock, "Original initial stock")
        self.target_bounds = _bounds(self.target)
        self.target_faces = [(face, _bounds(face)) for face in self.target.Faces]
        self.stock_bounds = _bounds(self.initial_stock)
        self.shapes = {}
        self.enclosures = {}
        self.inscribed_cylinders = {}
        self.collision_python = collision_python
        self.surface_deflection = float(surface_deflection_mm)
        self.worker = None
        self.worker_errors = None
        self.feed_index = []
        self.stock_regions = []
        self.applied_feed_offset = 0
        self.previous_tip = None
        self.previous_gauge = None
        self.previous_tool = None
        self.program_end = False
        self.observed_events = False
        self.last_ordinal = 0
        self.running = False
        self.finished = False
        self.digest = hashlib.sha256()
        directory = Path(work_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="safety-", dir=directory))
        self.feeds = (self.directory / "feeds.jsonl").open("w+", encoding="utf-8")
        self.evidence = (self.directory / "checks.jsonl").open("w", encoding="utf-8")
        self.report = {
            "status": "unavailable",
            "motion_count": 0,
            "feed_count": 0,
            "rapid_count": 0,
            "placement_count": 0,
            "whole_path_checked": False,
            "tool_events_checked": False,
            "initial_stock_fast_proofs": 0,
            "initial_stock_native_proofs": 0,
            "remaining_stock_bounds_used": False,
            "remaining_stock_queries": 0,
            "remaining_stock_candidate_feeds": 0,
            "remaining_stock_boolean_cuts": 0,
            "remaining_stock_empty_proofs": 0,
            "remaining_stock_partial_proofs": 0,
            "remaining_stock_cache_hits": 0,
            "remaining_stock_failed_upper_cuts": 0,
            "remaining_stock_refinement_failures": [],
            "remaining_stock_swept_cylinders": 0,
            "target_representation": "original native faces, including open surfaces",
            "target_face_count": len(self.target.Faces),
            "surface_deflection_mm": self.surface_deflection,
            "fcl_fast_proofs": 0,
            "occt_target_refinements": 0,
            "confirmed_collision_count": 0,
            "unavailable_check_count": 0,
            "checks": {},
            "tools": {},
            "first_issues": [],
            "linear_tolerance_mm": self.tolerance,
            "physical_removal_score": False,
            "full_task_accepted": False,
            "machine_fixtures_checked": False,
            "toolchanger_trajectory_checked": False,
            "evidence_file": str(self.directory / "checks.jsonl"),
        }

    def _budget(self):
        if time.monotonic() > self.deadline:
            raise SafetyUnavailable("Native safety time budget exhausted")

    def _validate_solid(self, shape, label, allow_empty=False):
        if allow_empty and (shape.isNull() or (not shape.Faces and not shape.Edges)):
            return
        if (
            shape.isNull()
            or not shape.isValid()
            or not shape.isClosed()
            or not shape.Solids
            or not all(solid.isClosed() for solid in shape.Solids)
            or shape.Volume <= 0
        ):
            raise SafetyUnavailable(f"{label} requires valid closed native solids")

    def _components(self, tool):
        if tool == 0:
            return {}
        if tool not in self.shapes:
            from tool_library import original_shapes

            if tool not in self.controllers or tool not in self.calibration:
                raise SafetyUnavailable(f"Tool {tool} lacks its original controller or calibration")
            controller = self.controllers[tool]
            if int(controller.ToolNumber) != tool:
                raise SafetyUnavailable("Controller ordinal differs from the motion tool")
            shapes = original_shapes(controller)
            for role, shape in shapes.items():
                self._validate_solid(shape, f"Original tool {tool} {role}")
            self.shapes[tool] = shapes
            self.report["tools"][str(tool)] = {
                "original_tool_id": controller.OriginalToolId,
                "source_sha256": controller.SourceSHA256,
                "present_components": sorted(shapes),
                "source_absent_components": sorted({"Cutter", "Shank", "Holder"} - shapes.keys()),
            }
        return self.shapes[tool]

    def _placed(self, shape, point):
        placed = shape.copy()
        placed.translate(self.app.Vector(*point))
        return placed

    def _worker_query(self, message):
        self._budget()
        self.worker.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.worker.stdin.flush()
        ready, unused_write, unused_error = select.select(
            [self.worker.stdout], [], [], max(0, self.deadline - time.monotonic())
        )
        if not ready:
            raise SafetyUnavailable("Native FCL query exceeded the safety time budget")
        line = self.worker.stdout.readline()
        if not line:
            raise SafetyUnavailable("Native FCL worker exited without a result")
        result = json.loads(line)
        if "error" in result:
            raise SafetyUnavailable(result["error"])
        return result

    def _start_worker(self):
        self._budget()
        vertices, triangles = self.target.tessellate(self.surface_deflection)
        mesh_file = self.directory / "target.json"
        vertex_file = self.directory / "target.vertices.f64"
        face_file = self.directory / "target.faces.i32"
        with vertex_file.open("wb") as stream:
            array(
                "d",
                (coordinate for point in vertices for coordinate in (point.x, point.y, point.z)),
            ).tofile(stream)
        with face_file.open("wb") as stream:
            array("i", (index for triangle in triangles for index in triangle)).tofile(stream)
        with mesh_file.open("w") as stream:
            json.dump(
                {
                    "vertices": str(vertex_file),
                    "faces": str(face_file),
                    "vertex_count": len(vertices),
                    "face_count": len(triangles),
                },
                stream,
                separators=(",", ":"),
            )
        del vertices, triangles
        self.worker_errors = (self.directory / "fcl-stderr.txt").open("w")
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        for name in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD"):
            environment.pop(name, None)
        self.worker = subprocess.Popen(
            [
                str(self.collision_python),
                str(Path(__file__).resolve()),
                "--fcl-worker",
                str(mesh_file),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.worker_errors,
            text=True,
            bufsize=1,
            env=environment,
        )
        result = self._worker_query({"op": "ready"})
        self.report["fcl_target_triangles"] = result["triangles"]

    def _enclosure(self, tool):
        if tool not in self.enclosures:
            cutter = self._components(tool)["Cutter"]
            lower, upper = _bounds(cutter)
            radius = max(abs(value) for value in (*lower[:2], *upper[:2]))
            height = upper[2] - lower[2]
            ball = self.part.makeSphere(
                radius + self.tolerance, self.app.Vector(0, 0, lower[2] + radius)
            )
            shapes = [ball]
            if height > radius:
                shapes.append(
                    self.part.makeCylinder(
                        radius + self.tolerance,
                        height - radius + self.tolerance,
                        self.app.Vector(0, 0, lower[2] + radius),
                    )
                )
            remainder = cutter.cut(shapes)
            if remainder.isNull() or (
                remainder.isValid()
                and (not remainder.Solids or remainder.Volume <= self.tolerance**3)
            ):
                enclosure = {
                    "kind": "ball_cylinder",
                    "radius": radius + self.tolerance,
                    "bottom": lower[2],
                    "ball_center": lower[2] + radius,
                    "top": upper[2] + self.tolerance,
                }
            else:
                cylinder = self.part.makeCylinder(
                    radius + self.tolerance,
                    height + 2 * self.tolerance,
                    self.app.Vector(0, 0, lower[2] - self.tolerance),
                )
                remainder = cutter.cut(cylinder)
                if not remainder.isNull() and (
                    not remainder.isValid()
                    or (remainder.Solids and remainder.Volume > self.tolerance**3)
                ):
                    self.enclosures[tool] = None
                    return None
                enclosure = {
                    "kind": "cylinder",
                    "radius": radius + self.tolerance,
                    "bottom": lower[2] - self.tolerance,
                    "top": upper[2] + self.tolerance,
                }
            self.enclosures[tool] = enclosure
            self.report["tools"][str(tool)]["native_verified_cutter_enclosure"] = enclosure
        return self.enclosures[tool]

    def _target_sweep(self, tool, shape, start, end, error):
        if _separated(
            _swept_bounds(_bounds(shape), start, end, error), self.target_bounds, self.tolerance
        ):
            return {"status": "clear", "proof": "complete swept bounding box versus original faces"}
        if self.collision_python is not None:
            if self.worker is None:
                self._start_worker()
            enclosure = self._enclosure(tool)
            if enclosure is not None:
                result = self._worker_query(
                    {
                        "op": "check",
                        "enclosure": enclosure,
                        "start": start,
                        "end": end,
                        "padding": self.surface_deflection + error + self.tolerance,
                    }
                )
                if result["clear"]:
                    self.report["fcl_fast_proofs"] += 1
                    return {
                        "status": "clear",
                        "proof": "FCL complete enclosing sweep versus original face BVH",
                        "surface_margin_mm": result["minimum_distance"],
                    }
        self.report["occt_target_refinements"] += 1
        swept = _swept_bounds(_bounds(shape), start, end, error + self.tolerance)
        faces = [face for face, bounds in self.target_faces if not _separated(swept, bounds)]
        if not faces:
            return {
                "status": "clear",
                "proof": "complete sweep disjoint from every original face bound",
            }
        target = self.part.makeCompound(faces)
        return self._sweep(shape, start, end, target, target, error, surface=True)

    def _box(self, bounds):
        lower, upper = bounds
        return self.part.makeBox(
            *(upper[axis] - lower[axis] for axis in range(3)), self.app.Vector(*lower)
        )

    def _subtract(self, stock, cutters, *, upper_bound=False, on_success=None):
        self._budget()
        if stock.isNull() or not stock.Solids:
            return stock
        try:
            result = stock.cut(cutters)
            self.report["remaining_stock_boolean_cuts"] += 1
            self._validate_solid(result, "Native remaining-stock bound", allow_empty=True)
        except (ValueError, SafetyUnavailable) as error:
            if not upper_bound or (isinstance(error, ValueError) and str(error) != "Null shape"):
                raise
            self._validate_solid(stock, "Unrefined remaining-stock upper bound")
            self.report["remaining_stock_failed_upper_cuts"] += 1
            failures = self.report["remaining_stock_refinement_failures"]
            if len(failures) < 20:
                failures.append({"motion": self.report["motion_count"], "reason": str(error)})
            if isinstance(cutters, list) and len(cutters) > 1:
                completed = []
                for cutter in cutters:
                    stock = self._subtract(
                        stock,
                        cutter,
                        upper_bound=True,
                        on_success=lambda: completed.append(True),
                    )
                if len(completed) == len(cutters) and on_success is not None:
                    on_success()
            return stock
        if on_success is not None:
            on_success()
        return result

    def _feed_inner_sweep(self, row):
        """Return guaranteed removal inside a verified cylindrical part of the cutter."""
        start, end = row["start_mm"], row["end_mm"]
        if row["chord_error_mm"] != 0 or (start[2] != end[2] and start[:2] != end[:2]):
            return None
        tool = row["tool"]
        if tool not in self.inscribed_cylinders:
            cutter = self._components(tool)["Cutter"]
            lower, upper = _bounds(cutter)
            radius = max(abs(value) for value in (*lower[:2], *upper[:2])) - self.tolerance
            self.inscribed_cylinders[tool] = None
            if radius > 0:
                for bottom in (lower[2] + self.tolerance, lower[2] + radius + self.tolerance):
                    height = upper[2] - self.tolerance - bottom
                    if height <= 0:
                        continue
                    cylinder = self.part.makeCylinder(radius, height, self.app.Vector(0, 0, bottom))
                    self._budget()
                    remainder = cylinder.cut(cutter)
                    if remainder.isNull() or (remainder.isValid() and not remainder.Solids):
                        self.inscribed_cylinders[tool] = radius, bottom, height
                        self.report["tools"].setdefault(str(tool), {})[
                            "native_verified_inscribed_cylinder_mm"
                        ] = {"radius": radius, "bottom": bottom, "height": height}
                        break
        dimensions = self.inscribed_cylinders[tool]
        if dimensions is None:
            return None
        radius, bottom, height = dimensions
        distance = math.dist(start[:2], end[:2])
        if distance == 0:
            sweep = self.part.makeCylinder(
                radius,
                height + abs(end[2] - start[2]),
                self.app.Vector(start[0], start[1], min(start[2], end[2]) + bottom),
            )
        else:
            sweep = self.part.makeBox(
                distance,
                2 * radius,
                height,
                self.app.Vector(start[0], start[1] - radius, start[2] + bottom),
            )
            sweep.rotate(
                self.app.Vector(start[0], start[1], start[2] + bottom),
                self.app.Vector(0, 0, 1),
                math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])),
            )
        self.report["remaining_stock_swept_cylinders"] += 1
        return sweep

    def _remaining(self, query_bounds, clearance=None):
        """Bound stock inside a box containing the complete queried sweep and margin."""
        self.report["remaining_stock_bounds_used"] = True
        self.report["remaining_stock_queries"] += 1
        self.feeds.flush()
        self.feeds.seek(self.applied_feed_offset)
        while line := self.feeds.readline():
            self._budget()
            row = json.loads(line)
            bounds = _swept_bounds(
                _bounds(self._components(row["tool"])["Cutter"]),
                row["start_mm"],
                row["end_mm"],
                row["chord_error_mm"] + self.tolerance,
            )
            self.feed_index.append((self.applied_feed_offset, bounds))
            self.applied_feed_offset = self.feeds.tell()
        try:
            region = None
            if clearance is not None:
                for cached in reversed(self.stock_regions):
                    if all(
                        cached["bounds"][0][axis] <= query_bounds[0][axis]
                        and cached["bounds"][1][axis] >= query_bounds[1][axis]
                        for axis in range(3)
                    ):
                        region = cached
                        self.report["remaining_stock_cache_hits"] += 1
                        break
            if region is None:
                padding = [
                    query_bounds[1][axis] - query_bounds[0][axis]
                    if clearance is not None and axis < 2
                    else 0
                    for axis in range(3)
                ]
                region_bounds = (
                    tuple(query_bounds[0][axis] - padding[axis] for axis in range(3)),
                    tuple(query_bounds[1][axis] + padding[axis] for axis in range(3)),
                )
                upper = self.initial_stock.common(self._box(region_bounds))
                self._validate_solid(upper, "Local initial stock", allow_empty=True)
                region = {
                    "bounds": region_bounds,
                    "upper": upper,
                    "applied_inner_offsets": set(),
                    "applied_endpoints": set(),
                }
                if clearance is not None:
                    self.stock_regions.append(region)
                    self.stock_regions = self.stock_regions[-8:]
            upper = region["upper"]
            if clearance is not None and clearance(upper):
                self.report["remaining_stock_partial_proofs"] += 1
                return upper, self.part.Shape()
            candidates = [
                (offset, bounds)
                for offset, bounds in reversed(self.feed_index)
                if not _separated(bounds, query_bounds)
            ]
            self.report["remaining_stock_candidate_feeds"] += len(candidates)
            cutters = []
            pending_endpoints = set()
            refinements = 0
            next_clearance = 1

            def cleared():
                nonlocal refinements, next_clearance
                refinements += 1
                if clearance is None or refinements < next_clearance:
                    return False
                next_clearance += min(next_clearance, 32)
                return clearance(upper)

            seen = set()
            for offset, bounds in reversed(candidates):
                self._budget()
                self.feeds.seek(offset)
                row = json.loads(self.feeds.readline())
                cutter = self._components(row["tool"])["Cutter"]
                inner = (
                    self._feed_inner_sweep(row)
                    if offset not in region["applied_inner_offsets"]
                    else None
                )
                if inner is not None and not _separated(_bounds(inner), query_bounds):
                    upper = self._subtract(
                        upper,
                        inner,
                        upper_bound=True,
                        on_success=lambda: region["applied_inner_offsets"].add(offset),
                    )
                    region["upper"] = upper
                    if upper.isNull() or not upper.Solids:
                        cutters = []
                        break
                    if cleared():
                        self.report["remaining_stock_partial_proofs"] += 1
                        return upper, self.part.Shape()
                for key in ("end_mm", "start_mm"):
                    point = tuple(row[key])
                    identity = (row["tool"], point)
                    if identity in seen or identity in region["applied_endpoints"]:
                        continue
                    seen.add(identity)
                    if not _separated(_swept_bounds(_bounds(cutter), point, point), query_bounds):
                        cutters.append(self._placed(cutter, point))
                        pending_endpoints.add(identity)
                if len(cutters) >= 16:
                    upper = self._subtract(
                        upper,
                        cutters,
                        upper_bound=True,
                        on_success=lambda: region["applied_endpoints"].update(pending_endpoints),
                    )
                    region["upper"] = upper
                    cutters = []
                    pending_endpoints.clear()
                    if upper.isNull() or not upper.Solids:
                        break
                    if cleared():
                        self.report["remaining_stock_partial_proofs"] += 1
                        return upper, self.part.Shape()
            if cutters:
                upper = self._subtract(
                    upper,
                    cutters,
                    upper_bound=True,
                    on_success=lambda: region["applied_endpoints"].update(pending_endpoints),
                )
                region["upper"] = upper
            if upper.isNull() or not upper.Solids:
                self.report["remaining_stock_empty_proofs"] += 1
                return upper, upper
            if cutters and clearance is not None and clearance(upper):
                self.report["remaining_stock_partial_proofs"] += 1
                return upper, self.part.Shape()
            lower = self.initial_stock.common(self._box(query_bounds))
            self._validate_solid(lower, "Local initial stock", allow_empty=True)
            for beginning in range(0, len(candidates), 16):
                lower = self._subtract(
                    lower,
                    [
                        self._box(bounds)
                        for offset, bounds in candidates[beginning : beginning + 16]
                    ],
                )
                if lower.isNull() or not lower.Solids:
                    break
            return upper, lower
        finally:
            self.feeds.seek(0, 2)

    def _penetration(self, placed, obstacle, uncertainty, surface=False):
        if obstacle.isNull() or not obstacle.Faces:
            return None
        if _separated(_bounds(placed), _bounds(obstacle)):
            return None
        self._budget()
        common = placed.common(obstacle)
        if not common.isNull() and not common.isValid():
            raise SafetyUnavailable("Native intersection returned invalid geometry")
        if surface:
            boundary = self.part.makeCompound(placed.Faces)
            for face in common.Faces:
                self._budget()
                point = face.distToShape(self.part.Vertex(face.CenterOfMass))[1][0][0]
                if not any(solid.isInside(point, self.tolerance, False) for solid in placed.Solids):
                    continue
                clearance = boundary.distToShape(self.part.Vertex(point))[0]
                if clearance > uncertainty + self.tolerance:
                    return {
                        "point_mm": (point.x, point.y, point.z),
                        "interior_clearance_mm": clearance,
                        "target_surface_area_inside_tool_mm2": common.Area,
                    }
            return None
        volume = common.Volume if common.Solids else 0.0
        if volume <= self.tolerance**3:
            return None
        for solid in common.Solids:
            center = solid.CenterOfMass
            faces = self.part.makeCompound(solid.Faces)
            candidates = [center]
            for face in solid.Faces:
                point = face.CenterOfMass
                candidates.extend(
                    point * weight + center * (1 - weight) for weight in (1, 0.99, 0.9)
                )
            for point in candidates:
                self._budget()
                if not solid.isInside(point, self.tolerance, False):
                    continue
                clearance = faces.distToShape(self.part.Vertex(point))[0]
                if clearance > uncertainty + self.tolerance:
                    return {
                        "point_mm": (point.x, point.y, point.z),
                        "intersection_volume_mm3": volume,
                        "interior_clearance_mm": clearance,
                    }
        return None

    def _sweep(self, shape, start, end, upper, lower, error, surface=False, clearance_only=False):
        if upper.isNull() or not upper.Faces:
            return {"status": "clear", "proof": "empty remaining-stock upper bound"}
        moving_bounds, obstacle_bounds = _bounds(shape), _bounds(upper)
        if _separated(
            _swept_bounds(moving_bounds, start, end, error), obstacle_bounds, self.tolerance
        ):
            return {"status": "clear", "proof": "complete swept bounding box"}
        for fraction, point in ((0.0, start), (1.0, end)):
            placed = self._placed(shape, point)
            if clearance_only and start != end:
                distance = placed.distToShape(upper, min(self.tolerance, 1e-7))[0]
                if distance <= error + self.tolerance:
                    return {
                        "status": "unavailable",
                        "reason": "Upper-bound clearance not established",
                    }
            else:
                witness = self._penetration(placed, lower, 0.0, surface)
                if witness:
                    return {"status": "collision", "fraction": fraction, **witness}
        if start == end:
            placed = self._placed(shape, start)
            common = placed.common(upper)
            empty = (
                not common.Faces or common.Area <= self.tolerance**2
                if surface
                else (not common.Solids or common.Volume <= self.tolerance**3)
            )
            if error == 0 and empty:
                return {"status": "clear", "proof": "native static nonpenetration"}
            return {"status": "unavailable", "reason": "Static stock-bound or tolerance ambiguity"}
        pending = [(0.0, 1.0)]
        while pending:
            self._budget()
            beginning, ending = pending.pop()
            fraction = (beginning + ending) * 0.5
            center = tuple(start[axis] + fraction * (end[axis] - start[axis]) for axis in range(3))
            placed = self._placed(shape, center)
            distance = placed.distToShape(upper, min(self.tolerance, 1e-7))[0]
            if clearance_only and distance <= error + self.tolerance:
                return {"status": "unavailable", "reason": "Upper-bound clearance not established"}
            travel = math.dist(start, end) * (ending - beginning) * 0.5
            if distance > travel + error + self.tolerance:
                continue
            witness = self._penetration(placed, lower, error, surface)
            if witness:
                return {"status": "collision", "fraction": fraction, **witness}
            if travel <= self.tolerance:
                return {
                    "status": "unavailable",
                    "reason": "Native distance cannot separate the complete sweep from stock/target",
                    "fraction_interval": [beginning, ending],
                }
            pending.extend(((fraction, ending), (beginning, fraction)))
        return {"status": "clear", "proof": "native distance bounds cover the complete segment"}

    def _record(self, check, role, tool, result, context):
        status = result["status"]
        counts = self.report["checks"].setdefault(
            check, {"clear": 0, "collision": 0, "unavailable": 0}
        )
        counts[status] += 1
        row = {"check": check, "component": role, "tool": tool, **context, **result}
        self.evidence.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        if status != "clear":
            field = (
                "confirmed_collision_count" if status == "collision" else "unavailable_check_count"
            )
            self.report[field] += 1
            if len(self.report["first_issues"]) < 20:
                self.report["first_issues"].append(row)

    def _stock_check(self, check, role, tool, shape, start, end, error, context, feed=None):
        if _separated(
            _swept_bounds(_bounds(shape), start, end, error), self.stock_bounds, self.tolerance
        ):
            result = {"status": "clear", "proof": "complete sweep disjoint from initial stock"}
            self.report["initial_stock_fast_proofs"] += 1
        else:
            result = self._sweep(shape, start, end, self.initial_stock, self.initial_stock, error)
            if result["status"] == "clear":
                result = {**result, "obstacle": "original initial stock"}
                self.report["initial_stock_native_proofs"] += 1
                self._record(check, role, tool, result, context)
                return
            query_bounds = _swept_bounds(_bounds(shape), start, end, error + 2 * self.tolerance)
            upper, lower = self._remaining(
                query_bounds,
                clearance=lambda upper: (
                    self._sweep(
                        shape, start, end, upper, self.part.Shape(), error, clearance_only=True
                    )["status"]
                    == "clear"
                ),
            )
            if feed is not None:
                cutter = self._components(tool)["Cutter"]
                enclosure = self._box(
                    _swept_bounds(_bounds(cutter), start, end, error + self.tolerance)
                )
                lower = self._subtract(lower, enclosure)
            result = self._sweep(shape, start, end, upper, lower, error)
        self._record(check, role, tool, result, context)

    def _placement(self, tool, gauge, context):
        if tool not in self.calibration:
            raise SafetyUnavailable(f"Tool {tool} has no declared gauge calibration")
        tip = tuple(gauge[axis] - self.calibration[tool][axis] for axis in range(3))
        for role, shape in self._components(tool).items():
            if role == "Cutter":
                self._record(
                    "static_target",
                    role,
                    tool,
                    self._target_sweep(tool, shape, tip, tip, 0.0),
                    context,
                )
            self._stock_check("static_stock", role, tool, shape, tip, tip, 0.0, context)
        self.report["placement_count"] += 1

    def on_event(self, event):
        """Observe canonical tool installations, including ones after the last move."""
        if not self.running:
            raise SafetyUnavailable("Canonical events must be consumed within SafetyReplay.consume")
        self._budget()
        ordinal = event["ordinal"]
        if type(ordinal) is not int or ordinal <= self.last_ordinal:
            raise SafetyUnavailable("Safety canonical ordinals must increase")
        self.last_ordinal = ordinal
        self.observed_events = True
        name = event["name"]
        if name == "PROGRAM_END":
            self.program_end = True
        if name == "SELECT_TOOL":
            self.selected = _number(int(event["arguments"]))
        elif name in {"CHANGE_TOOL", "CHANGE_TOOL_NUMBER"}:
            pocket = int(event["arguments"])
            if name == "CHANGE_TOOL" and self.selected is not None:
                tool = self.selected
                if pocket in self.pockets and self.pockets[pocket] != tool:
                    raise SafetyUnavailable("Selected tool and internal pocket mapping differ")
                self.pockets[pocket] = tool
            elif pocket in self.pockets:
                tool = _number(self.pockets[pocket])
            else:
                raise SafetyUnavailable("Tool installation lacks its internal pocket mapping")
            self.tool = tool
            self._placement(tool, self.gauge, {"event_ordinal": ordinal, "placement": name})

    def _motion(self, motion):
        self._budget()
        start, end = _point(motion["start_mm"]), _point(motion["end_mm"])
        tool = _number(motion["tool"])
        rapid = motion["rapid"]
        error = float(motion.get("chord_error_mm", 0.0))
        if type(rapid) is not bool or not math.isfinite(error) or error < 0:
            raise ValueError("Safety replay requires rapid bool and finite nonnegative chord error")
        if tool == 0 and not rapid:
            raise SafetyUnavailable("Cutting feed has no installed original cutter")
        keys = ("gauge_start_mm", "gauge_end_mm", "tool_gauge_offset_mm")
        if not all(key in motion for key in keys):
            raise SafetyUnavailable("Safety replay requires complete gauge/calibration metadata")
        gauge_start, gauge_end, calibration = (_point(motion[key]) for key in keys)
        if tool not in self.calibration or math.dist(calibration, self.calibration[tool]) > 1e-7:
            raise SafetyUnavailable("Motion calibration differs from the declared installed tool")
        for point, gauge in ((start, gauge_start), (end, gauge_end)):
            if math.dist(point, tuple(gauge[axis] - calibration[axis] for axis in range(3))) > 1e-7:
                raise ValueError("Motion physical tip differs from its declared gauge/calibration")
        if math.dist(self.gauge, gauge_start) > 1e-7:
            raise ValueError("Safety motion stream has a gauge discontinuity")
        if self.previous_tool == tool and math.dist(self.previous_tip, start) > 1e-7:
            raise ValueError("Safety motion stream has a same-tool tip discontinuity")
        if self.observed_events and tool != self.tool:
            raise SafetyUnavailable("Motion tool differs from observed canonical installation")
        if not self.observed_events and tool != self.tool:
            self.tool = tool
            self._placement(tool, gauge_start, {"placement": "inferred from motion"})
        context = {
            "motion": self.report["motion_count"],
            "source_ordinal": motion.get("source_ordinal"),
        }
        components = self._components(tool)
        for role, shape in components.items():
            if role == "Cutter":
                self._record(
                    "cutter_target",
                    role,
                    tool,
                    self._target_sweep(tool, shape, start, end, error),
                    context,
                )
            if role != "Cutter" or rapid:
                self._stock_check(
                    "rapid_stock" if rapid else "noncutting_stock",
                    role,
                    tool,
                    shape,
                    start,
                    end,
                    error,
                    context,
                    feed=None if rapid else motion,
                )
        row = {
            "start_mm": start,
            "end_mm": end,
            "tool": tool,
            "rapid": rapid,
            "chord_error_mm": error,
            "gauge_start_mm": gauge_start,
            "gauge_end_mm": gauge_end,
            "tool_gauge_offset_mm": calibration,
        }
        encoded = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
        self.digest.update(encoded.encode())
        if not rapid:
            self.feeds.write(encoded)
            self.report["feed_count"] += 1
        else:
            self.report["rapid_count"] += 1
        self.previous_tip, self.previous_gauge, self.previous_tool = end, gauge_end, tool
        self.gauge = gauge_end
        self.report["motion_count"] += 1

    def consume(self, motions):
        """Return a compact report; detailed checks and feed history remain JSONL."""
        if self.running or self.finished:
            raise ValueError("A safety replay instance can consume one stream only")
        self.running = True
        try:
            self._placement(self.tool, self.gauge, {"placement": "initial"})
            for motion in motions:
                self._motion(motion)
            self.report["whole_path_checked"] = True
            self.report["tool_events_checked"] = self.observed_events and self.program_end
            if not self.report["tool_events_checked"]:
                raise SafetyUnavailable(
                    "Missing canonical event hook or PROGRAM_END; final tool placement unverified"
                )
            if self.report["confirmed_collision_count"]:
                self.report["status"] = "collision"
            elif self.report["unavailable_check_count"] == 0:
                self.report["status"] = "clear"
        except Exception as error:
            self.report["failure"] = {"type": type(error).__name__, "reason": str(error)}
            self.report["status"] = "unavailable"
        finally:
            self.running = False
            self.finished = True
            self.feeds.close()
            self.evidence.close()
            if self.worker is not None:
                try:
                    self.worker.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    self.worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.worker.kill()
                    self.worker.wait(timeout=5)
                self.worker.stdout.close()
            if self.worker_errors is not None:
                self.worker_errors.close()
        self.report.update(
            elapsed_seconds=time.monotonic() - self.started,
            motions_sha256=self.digest.hexdigest(),
            collision_free=True
            if self.report["status"] == "clear"
            else (False if self.report["status"] == "collision" else None),
        )
        (self.directory / "receipt.json").write_text(json.dumps(self.report, indent=2))
        return self.report


def _fcl_worker(mesh_file):
    import sys

    import fcl
    import numpy as np
    from scipy.spatial import ConvexHull

    mesh = json.loads(Path(mesh_file).read_text())
    vertices = np.memmap(
        mesh["vertices"], dtype=np.float64, mode="r", shape=(mesh["vertex_count"], 3)
    )
    faces = np.memmap(mesh["faces"], dtype=np.int32, mode="r", shape=(mesh["face_count"], 3))
    model = fcl.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(vertices, faces)
    model.endModel()
    target = fcl.CollisionObject(model)
    angles = np.arange(32) * (2 * np.pi / 32)
    circle = np.column_stack((np.cos(angles), np.sin(angles))) / np.cos(np.pi / 32)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["op"] == "ready":
                print(json.dumps({"triangles": len(faces)}), flush=True)
                continue
            if request["op"] != "check":
                raise ValueError("Unsupported FCL worker operation")
            definition = request["enclosure"]
            start, end = np.array(request["start"]), np.array(request["end"])
            radius = definition["radius"]
            bottom = definition.get("ball_center", definition["bottom"])
            top = definition["top"]
            bodies = []
            if top > bottom:
                ring = circle * radius
                points = np.vstack(
                    [
                        np.column_stack((ring, np.full(32, height))) + point
                        for point in (start, end)
                        for height in (bottom, top)
                    ]
                )
                hull = ConvexHull(points)
                triangles = hull.simplices.copy()
                for index, triangle in enumerate(triangles):
                    first, second, third = points[triangle]
                    if (
                        np.dot(np.cross(second - first, third - first), hull.equations[index, :3])
                        < 0
                    ):
                        triangles[index] = triangle[::-1]
                convex = fcl.Convex(
                    points,
                    len(triangles),
                    np.column_stack((np.full(len(triangles), 3), triangles)).ravel(),
                )
                bodies.append(fcl.CollisionObject(convex))
            if definition["kind"] == "ball_cylinder":
                delta = end - start
                length = np.linalg.norm(delta)
                center = (start + end) / 2 + [0, 0, definition["ball_center"]]
                if length > 1e-12:
                    direction = delta / length
                    reference = (
                        np.array([1.0, 0.0, 0.0])
                        if abs(direction[0]) < 0.9
                        else np.array([0.0, 1.0, 0.0])
                    )
                    first_axis = np.cross(reference, direction)
                    first_axis /= np.linalg.norm(first_axis)
                    rotation = np.column_stack(
                        (first_axis, np.cross(direction, first_axis), direction)
                    )
                    bodies.append(
                        fcl.CollisionObject(
                            fcl.Capsule(radius, length), fcl.Transform(rotation, center)
                        )
                    )
                else:
                    bodies.append(fcl.CollisionObject(fcl.Sphere(radius), fcl.Transform(center)))
            minimum = math.inf
            for body in bodies:
                if fcl.collide(body, target, fcl.CollisionRequest(), fcl.CollisionResult()):
                    minimum = 0.0
                    break
                distance = fcl.distance(body, target, fcl.DistanceRequest(), fcl.DistanceResult())
                if not math.isfinite(distance) or distance < 0:
                    raise ValueError("Native FCL returned an invalid distance")
                minimum = min(minimum, distance)
            if not bodies:
                raise ValueError("No verified cutter enclosure")
            print(
                json.dumps({"clear": minimum > request["padding"], "minimum_distance": minimum}),
                flush=True,
            )
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Internal native FCL surface-sweep worker")
    parser.add_argument("--fcl-worker", required=True, metavar="TARGET_MESH_JSON")
    _fcl_worker(parser.parse_args().fcl_worker)
