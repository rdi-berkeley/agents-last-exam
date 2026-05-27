"""
extract_features.py — Feature extraction from STEP/PRT files (runs on remote Windows VM)

Analyzes a 3D CAD file using pythonocc-core (OCC) to extract:
- Global geometry: volume, bounding box
- Cylindrical features: holes (with diameter, location, axis)
- Hole histogram: count of holes grouped by diameter

This script is used both for:
1. Pre-processing ground truth .step files → gt_features.json
2. Runtime evaluation of agent's .step output → agent_features.json

Requirements:
    pythonocc-core (conda install -c conda-forge pythonocc-core)
    numpy

Usage:
    python extract_features.py --file PATH_TO_STEP_OR_PRT --output PATH_TO_JSON

Output:
    JSON file with structure:
    {
        "meta": {"filename": "..."},
        "geometry": {"volume": ..., "bbox_dims": [...], "bbox_min": [...], "bbox_max": [...]},
        "features": {"hole_count_unique": ..., "hole_histogram": {...}, "holes_details": [...]}
    }

    Also prints the JSON to stdout for pipeline integration.

Exit codes:
    0 = success
    1 = error (file not found, OCC import failure, etc.)
"""

import sys
import json
import math
import os
import argparse

# Try OCP (cadquery/build123d) first, then fall back to OCC.Core (pythonocc-core)
try:
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.BRepLProp import BRepLProp_SLProps

    # OCP uses TopoDS.Face_s() instead of topods.Face()
    class _topods_compat:
        @staticmethod
        def Face(shape):
            return TopoDS.Face_s(shape)

    topods = _topods_compat()

    # OCP static methods use _s suffix
    class _brepgprop_compat:
        @staticmethod
        def VolumeProperties(shape, props):
            return BRepGProp.VolumeProperties_s(shape, props)

    class _brepbndlib_compat:
        @staticmethod
        def Add(shape, bbox):
            return BRepBndLib.Add_s(shape, bbox)

    brepgprop = _brepgprop_compat()
    brepbndlib = _brepbndlib_compat()
    _USING_OCP = True
except ImportError:
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Cylinder
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.gp import gp_Pnt, gp_Vec
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        _USING_OCP = False
    except ImportError:
        print(json.dumps({"error": "Neither OCP (cadquery/build123d) nor pythonocc-core installed"}), flush=True)
        sys.exit(1)


class StepFeatureExtractor:
    """Extract geometric features from a STEP/PRT file using OCC."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.shape = None
        self._load_step(file_path)

    def _load_step(self, path):
        """Load a STEP file using OCC STEPControl_Reader."""
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status == IFSelect_RetDone:
            reader.TransferRoot()
            self.shape = reader.Shape()
        else:
            raise ValueError(f"Failed to read STEP/PRT file: {path}")

    def _is_coaxial(self, loc1, dir1, loc2, dir2, tol_loc=0.1, tol_ang=0.02):
        """Check if two cylinders are coaxial (same axis line)."""
        v1 = gp_Vec(dir1[0], dir1[1], dir1[2])
        v2 = gp_Vec(dir2[0], dir2[1], dir2[2])
        if not (v1.IsParallel(v2, tol_ang) or v1.IsParallel(v2.Reversed(), tol_ang)):
            return False
        p1 = gp_Pnt(loc1[0], loc1[1], loc1[2])
        p2 = gp_Pnt(loc2[0], loc2[1], loc2[2])
        vec_p1p2 = gp_Vec(p1, p2)
        cross_prod = vec_p1p2.Crossed(v1)
        return cross_prod.Magnitude() < tol_loc

    def _classify_cylinder_geometric(self, face, surf_adaptor):
        """Classify a cylindrical face as 'hole' or 'pin' by normal direction."""
        u_min, u_max = surf_adaptor.FirstUParameter(), surf_adaptor.LastUParameter()
        v_min, v_max = surf_adaptor.FirstVParameter(), surf_adaptor.LastVParameter()
        u_mid, v_mid = (u_min + u_max) / 2.0, (v_min + v_max) / 2.0

        sl_props = BRepLProp_SLProps(surf_adaptor, u_mid, v_mid, 1, 1e-6)
        if not sl_props.IsNormalDefined():
            return "unknown"

        p_surf = sl_props.Value()
        n_geom = sl_props.Normal()

        if face.Orientation() == TopAbs_REVERSED:
            n_face = n_geom.Reversed()
        else:
            n_face = n_geom

        cyl = surf_adaptor.Cylinder()
        axis = cyl.Axis()

        vec_loc_to_surf = gp_Vec(axis.Location(), p_surf)
        vec_axis_dir = gp_Vec(
            axis.Direction().X(), axis.Direction().Y(), axis.Direction().Z()
        )

        projection = vec_loc_to_surf.Dot(vec_axis_dir)
        vec_parallel = vec_axis_dir.Multiplied(projection)
        vec_radial = vec_loc_to_surf.Subtracted(vec_parallel)

        vec_n_face = gp_Vec(n_face.X(), n_face.Y(), n_face.Z())
        dot_prod = vec_n_face.Dot(vec_radial)

        if dot_prod > 0:
            return "pin"
        else:
            return "hole"

    def analyze_features(self):
        """Analyze cylindrical features with angle-based filtering."""
        raw_cylinders = []
        topo_exp = TopExp_Explorer(self.shape, TopAbs_FACE)

        while topo_exp.More():
            face = topods.Face(topo_exp.Current())
            surf_adaptor = BRepAdaptor_Surface(face, True)

            if surf_adaptor.GetType() == GeomAbs_Cylinder:
                cyl = surf_adaptor.Cylinder()
                radius = cyl.Radius()
                axis = cyl.Axis()
                loc = [
                    axis.Location().X(),
                    axis.Location().Y(),
                    axis.Location().Z(),
                ]
                direc = [
                    axis.Direction().X(),
                    axis.Direction().Y(),
                    axis.Direction().Z(),
                ]

                u_start = surf_adaptor.FirstUParameter()
                u_end = surf_adaptor.LastUParameter()
                angle = abs(u_end - u_start)

                f_type = self._classify_cylinder_geometric(face, surf_adaptor)

                raw_cylinders.append(
                    {
                        "radius": radius,
                        "location": loc,
                        "direction": direc,
                        "type": f_type,
                        "angle": angle,
                    }
                )
            topo_exp.Next()

        # --- Merge coaxial cylinders ---
        merged_features = []
        for raw in raw_cylinders:
            matched = False
            for exist in merged_features:
                if (
                    raw["type"] == exist["type"]
                    and math.isclose(raw["radius"], exist["radius"], abs_tol=0.01)
                    and self._is_coaxial(
                        raw["location"],
                        raw["direction"],
                        exist["location"],
                        exist["direction"],
                    )
                ):
                    exist["total_angle"] += raw["angle"]
                    matched = True
                    break

            if not matched:
                raw["total_angle"] = raw["angle"]
                merged_features.append(raw)

        final_features = {"holes": [], "pins": [], "fillets": []}

        for feat in merged_features:
            diameter = feat["radius"] * 2.0
            data = {
                "diameter": round(diameter, 4),
                "location": [round(x, 4) for x in feat["location"]],
                "axis": [round(x, 4) for x in feat["direction"]],
                "angle_deg": round(math.degrees(feat["total_angle"]), 1),
            }

            if feat["type"] == "hole":
                if feat["total_angle"] > 5.0:  # 5.0 rad ~ 286 deg
                    final_features["holes"].append(data)
                else:
                    final_features["fillets"].append(data)
            else:
                if feat["radius"] <= 2.1:
                    final_features["fillets"].append(data)
                else:
                    final_features["pins"].append(data)

        return final_features

    def analyze_global_properties(self):
        """Compute volume and bounding box."""
        props = GProp_GProps()
        brepgprop.VolumeProperties(self.shape, props)
        volume = props.Mass()

        bbox = Bnd_Box()
        brepbndlib.Add(self.shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        return {
            "volume": round(volume, 4),
            "bbox_dims": [
                round(xmax - xmin, 4),
                round(ymax - ymin, 4),
                round(zmax - zmin, 4),
            ],
            "bbox_min": [round(xmin, 4), round(ymin, 4), round(zmin, 4)],
            "bbox_max": [round(xmax, 4), round(ymax, 4), round(zmax, 4)],
        }

    def generate_report(self):
        """Generate full analysis report."""
        feats = self.analyze_features()
        global_props = self.analyze_global_properties()

        hole_hist = {}
        for h in feats["holes"]:
            d_str = f"{h['diameter']:.2f}"
            hole_hist[d_str] = hole_hist.get(d_str, 0) + 1

        return {
            "meta": {"filename": os.path.basename(self.file_path)},
            "geometry": global_props,
            "features": {
                "hole_count_unique": len(feats["holes"]),
                "pin_count_unique": len(feats["pins"]),
                "fillet_count_unique": len(feats["fillets"]),
                "hole_histogram": hole_hist,
                "holes_details": feats["holes"],
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="Extract geometric features from a STEP/PRT file"
    )
    parser.add_argument("--file", required=True, help="Path to STEP/PRT file")
    parser.add_argument(
        "--output",
        required=False,
        help="Path to output JSON file (also prints to stdout)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        result = {"error": f"File not found: {args.file}"}
        print(json.dumps(result))
        sys.exit(1)

    try:
        extractor = StepFeatureExtractor(args.file)
        report = extractor.generate_report()

        # Save to file if --output specified
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {args.output}", file=sys.stderr)

        # Always print to stdout for pipeline integration
        print(json.dumps(report))
        sys.exit(0)
    except Exception as e:
        result = {"error": str(e)}
        print(json.dumps(result))
        sys.exit(1)


if __name__ == "__main__":
    main()
