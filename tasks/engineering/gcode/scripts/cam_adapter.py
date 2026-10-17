import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def enable(operation):
    properties = (
        ("App::PropertyBool", "AdaptiveSampling", True),
        ("App::PropertyLength", "MinimumSampling", 0.001),
        ("App::PropertyFloat", "SamplingCosineLimit", 0.999999),
        ("App::PropertyLength", "RadialStockAllowance", 0.0),
    )
    for kind, name, default in properties:
        if name not in operation.PropertiesList:
            operation.addProperty(kind, name, "Adaptive surface")
            setattr(operation, name, default)
    operation.AdaptiveSampling = True
    operation.touch()


def install(cache_root, verify_interval=0):
    import FreeCAD
    import Path.Op.Surface as Surface
    import Path.Op.SurfaceSupport as SurfaceSupport

    if hasattr(Surface.ObjectSurface, "_ale_cam_report"):
        return Surface.ObjectSurface._ale_cam_report
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    original_prepare = Surface.ObjectSurface._planarGetPDC
    original_scan = Surface.ObjectSurface._planarDropCutScan
    original_execute = Surface.ObjectSurface.opExecute
    original_cutter = SurfaceSupport.OCL_Tool.getOclTool
    code_identity = {
        "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "surface": hashlib.sha256(Path(Surface.__file__).read_bytes()).hexdigest(),
        "native": hashlib.sha256(Path(Surface.ocl.__file__).read_bytes()).hexdigest(),
    }
    surfaces = {}
    report = {"computed_scans": 0, "cache_hits": 0, "full_comparisons": []}

    def cutter_with_allowance(instance):
        cutter = original_cutter(instance)
        value = getattr(instance.obj, "RadialStockAllowance", 0.0)
        allowance = float(getattr(value, "Value", value))
        if not math.isfinite(allowance) or allowance < 0:
            raise ValueError("Radial stock allowance must be finite and nonnegative")
        return cutter.offsetCutter(allowance) if allowance else cutter

    def prepare(instance, native_surface, depth, sampling, cutter):
        operation = getattr(instance, "_ale_operation", None)
        if not getattr(operation, "AdaptiveSampling", False):
            return original_prepare(instance, native_surface, depth, sampling, cutter)
        minimum = float(operation.MinimumSampling.Value)
        cosine = float(operation.SamplingCosineLimit)
        if not all(math.isfinite(value) for value in (depth, sampling, minimum, cosine)):
            raise ValueError("Nonfinite adaptive scan setting")
        if not 0 < minimum <= sampling or not 0 <= cosine <= 1:
            raise ValueError("Invalid adaptive scan setting")
        identity = id(native_surface)
        if identity not in surfaces:
            if len(surfaces) >= 2:
                del surfaces[next(iter(surfaces))]
            triangles = native_surface.getTriangles()
            if not triangles:
                raise ValueError("Cannot compute a surface operation without model triangles")
            coordinates = np.empty((len(triangles), 3, 3), dtype=np.float64)
            for number, triangle in enumerate(triangles):
                coordinates[number] = [
                    [point.x, point.y, point.z] for point in triangle.getPoints()
                ]
            if not np.isfinite(coordinates).all():
                raise ValueError("Nonfinite native surface")
            surfaces[identity] = {
                "native": native_surface,
                "coordinates": coordinates,
                "lower": coordinates[:, :, :2].min(axis=1),
                "upper": coordinates[:, :, :2].max(axis=1),
                "sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
            }
        source = surfaces[identity]
        settings = {
            "source": source["sha256"],
            "depth": float(depth),
            "sampling": float(sampling),
            "cutter": str(cutter),
            "radius": float(cutter.getRadius()),
            "length": float(cutter.getLength()),
            "minimum": minimum,
            "cosine": cosine,
            "code": code_identity,
        }
        return {
            "source": source,
            "settings": settings,
            "cutter": cutter,
            "identity": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        }

    def native_scan(descriptor, native_surface, first, last):
        settings = descriptor["settings"]
        solver = Surface.ocl.AdaptivePathDropCutter()
        solver.setSTL(native_surface)
        solver.setCutter(descriptor["cutter"])
        solver.setZ(settings["depth"])
        solver.setSampling(settings["sampling"])
        solver.setMinSampling(settings["minimum"])
        solver.setCosLimit(settings["cosine"])
        path = Surface.ocl.Path()
        path.append(
            Surface.ocl.Line(
                Surface.ocl.Point(first[0], first[1], settings["depth"]),
                Surface.ocl.Point(last[0], last[1], settings["depth"]),
            )
        )
        solver.setPath(path)
        solver.run()
        return [[point.x, point.y, point.z] for point in solver.getCLPoints()]

    def scan(instance, descriptor, first, last):
        if not isinstance(descriptor, dict):
            return original_scan(instance, descriptor, first, last)
        payload = {"descriptor": descriptor["identity"], "first": list(first), "last": list(last)}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cached = cache_root / f"{digest}.json"
        if cached.exists():
            value = json.loads(cached.read_text())
            points = np.asarray(value["points"], dtype=float)
            if (
                value["input"] != payload
                or points.ndim != 2
                or points.shape[1] != 3
                or not np.isfinite(points).all()
            ):
                raise ValueError("Invalid cached native scan")
            report["cache_hits"] += 1
            return [FreeCAD.Vector(*point) for point in points]
        source = descriptor["source"]
        radius = descriptor["settings"]["radius"] + 1e-8
        lower = np.minimum(first, last) - radius
        upper = np.maximum(first, last) + radius
        selected = np.flatnonzero(
            (source["upper"] >= lower).all(axis=1) & (source["lower"] <= upper).all(axis=1)
        )
        clipped = Surface.ocl.STLSurf()
        for triangle in source["coordinates"][selected]:
            clipped.addTriangle(
                Surface.ocl.Triangle(*[Surface.ocl.Point(*point) for point in triangle])
            )
        scan_surface = clipped if len(selected) else source["native"]
        points = native_scan(descriptor, scan_surface, first, last)
        if verify_interval and report["computed_scans"] % verify_interval == 0:
            expected = native_scan(descriptor, source["native"], first, last)
            if np.shape(points) != np.shape(expected) or not np.allclose(
                points, expected, atol=1e-9, rtol=0
            ):
                raise ValueError("Clipped native scan differs from complete surface")
            report["full_comparisons"].append({"digest": digest, "points": len(points)})
        pending = cached.with_suffix(f".{os.getpid()}.pending")
        pending.write_text(json.dumps({"input": payload, "points": points}, separators=(",", ":")))
        pending.replace(cached)
        report["computed_scans"] += 1
        return [FreeCAD.Vector(*point) for point in points]

    def execute(instance, operation):
        instance._ale_operation = operation
        try:
            return original_execute(instance, operation)
        finally:
            del instance._ale_operation

    SurfaceSupport.OCL_Tool.getOclTool = cutter_with_allowance
    Surface.ObjectSurface.opExecute = execute
    Surface.ObjectSurface._planarGetPDC = prepare
    Surface.ObjectSurface._planarDropCutScan = scan
    Surface.ObjectSurface._ale_cam_report = report
    return report
