"""Semantic scorer for the G1 asset-package-to-URDF task."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POS_TOL = 1e-2
ORI_TOL_DEG = 2.0
XYZ_TOL = 5e-3
RPY_TOL = 3.5e-2
REVOLUTE_LIMIT_TOL = 5e-2
PRISMATIC_LIMIT_TOL = 5e-3
VEL_REL_TOL = 0.10
EFF_REL_TOL = 0.10
MASS_REL_TOL = 0.01
INERTIA_REL_TOL = 0.02
AXIS_ANG_TOL_DEG = 2.0


@dataclass
class EvalResult:
    passed: bool
    errors: list[str]
    warnings: list[str]

    def add(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def is_nan_like(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none"}
    return False


def parse_vec(text: str | None, n: int = 3) -> np.ndarray | None:
    if text is None or is_nan_like(text):
        return None
    values = [float(x) for x in str(text).replace(",", " ").split()]
    if len(values) != n:
        raise ValueError(f"Expected {n} values, got {text!r}")
    return np.array(values, dtype=float)


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def angle_diff(left: float, right: float) -> float:
    return abs(wrap_angle(left - right))


def rel_err(left: float, right: float) -> float:
    denom = max(abs(right), 1e-12)
    return abs(left - right) / denom


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=float)
    trace = np.trace(matrix)
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quat[3] = 0.25 * scale
        quat[0] = (matrix[2, 1] - matrix[1, 2]) / scale
        quat[1] = (matrix[0, 2] - matrix[2, 0]) / scale
        quat[2] = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        idx = int(np.argmax(np.diag(matrix)))
        if idx == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quat[3] = (matrix[2, 1] - matrix[1, 2]) / scale
            quat[0] = 0.25 * scale
            quat[1] = (matrix[0, 1] + matrix[1, 0]) / scale
            quat[2] = (matrix[0, 2] + matrix[2, 0]) / scale
        elif idx == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quat[3] = (matrix[0, 2] - matrix[2, 0]) / scale
            quat[0] = (matrix[0, 1] + matrix[1, 0]) / scale
            quat[1] = 0.25 * scale
            quat[2] = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quat[3] = (matrix[1, 0] - matrix[0, 1]) / scale
            quat[0] = (matrix[0, 2] + matrix[2, 0]) / scale
            quat[1] = (matrix[1, 2] + matrix[2, 1]) / scale
            quat[2] = 0.25 * scale
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    return quat


def quat_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = left / max(np.linalg.norm(left), 1e-12)
    right = right / max(np.linalg.norm(right), 1e-12)
    dot = float(np.clip(abs(np.dot(left, right)), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def transform_from_xyz_rpy(xyz: np.ndarray | None, rpy: np.ndarray | None) -> np.ndarray:
    transform = np.eye(4)
    if xyz is not None:
        transform[:3, 3] = xyz
    if rpy is not None:
        transform[:3, :3] = rpy_to_matrix(rpy)
    return transform


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1 - c
    return np.array(
        [
            [x * x * one_minus_c + c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, y * y * one_minus_c + c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, z * z * one_minus_c + c],
        ],
        dtype=float,
    )


def motion_transform(joint_type: str, axis: np.ndarray | None, joint_value: float) -> np.ndarray:
    transform = np.eye(4)
    if joint_type in {"revolute", "continuous"}:
        if axis is None:
            raise ValueError("Revolute joint missing axis")
        transform[:3, :3] = axis_angle_to_matrix(axis, joint_value)
    elif joint_type == "prismatic":
        if axis is None:
            raise ValueError("Prismatic joint missing axis")
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        transform[:3, 3] = axis * joint_value
    elif joint_type in {"fixed", "floating"}:
        pass
    else:
        raise NotImplementedError(f"Unsupported joint type: {joint_type}")
    return transform


def geometry_to_record(geom_elem: ET.Element) -> tuple[str, str | None, str | None]:
    if geom_elem.find("mesh") is not None:
        mesh = geom_elem.find("mesh")
        assert mesh is not None
        filename = mesh.attrib.get("filename")
        params = None
        scale = mesh.attrib.get("scale")
        if scale:
            params = json.dumps({"scale": scale}, separators=(",", ":"), sort_keys=True)
        return "mesh", filename, params
    if geom_elem.find("box") is not None:
        box = geom_elem.find("box")
        assert box is not None
        return "box", None, json.dumps({"size": box.attrib.get("size")}, separators=(",", ":"), sort_keys=True)
    if geom_elem.find("sphere") is not None:
        sphere = geom_elem.find("sphere")
        assert sphere is not None
        return "sphere", None, json.dumps({"radius": sphere.attrib.get("radius")}, separators=(",", ":"), sort_keys=True)
    if geom_elem.find("cylinder") is not None:
        cylinder = geom_elem.find("cylinder")
        assert cylinder is not None
        return (
            "cylinder",
            None,
            json.dumps(
                {"length": cylinder.attrib.get("length"), "radius": cylinder.attrib.get("radius")},
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    raise NotImplementedError("Unsupported geometry element")


def canonical_jsonish(text: str | None) -> str | None:
    if text is None or is_nan_like(text):
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return str(text).strip()
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def parse_urdf(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError("Root element is not <robot>")

    model = {"name": root.attrib.get("name"), "links": {}, "joints": {}}
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        info = {"visuals": [], "collisions": [], "inertial": None}
        for index, visual in enumerate(link.findall("visual"), start=1):
            origin = visual.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else "0 0 0")
            rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else "0 0 0")
            geom = visual.find("geometry")
            if geom is None:
                raise ValueError(f"visual on link {link_name} is missing geometry")
            geometry_type, filename, params = geometry_to_record(geom)
            info["visuals"].append(
                {
                    "link_name": link_name,
                    "visual_index": index,
                    "origin_xyz": xyz,
                    "origin_rpy": rpy,
                    "geometry_type": geometry_type,
                    "filename": filename,
                    "params": canonical_jsonish(params),
                }
            )

        for index, collision in enumerate(link.findall("collision"), start=1):
            origin = collision.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else "0 0 0")
            rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else "0 0 0")
            geom = collision.find("geometry")
            if geom is None:
                raise ValueError(f"collision on link {link_name} is missing geometry")
            geometry_type, filename, params = geometry_to_record(geom)
            info["collisions"].append(
                {
                    "link_name": link_name,
                    "collision_name": collision.attrib.get("name"),
                    "collision_index": index,
                    "origin_xyz": xyz,
                    "origin_rpy": rpy,
                    "geometry_type": geometry_type,
                    "filename": filename,
                    "params": canonical_jsonish(params),
                }
            )

        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else "0 0 0")
            rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else "0 0 0")
            mass_elem = inertial.find("mass")
            inertia_elem = inertial.find("inertia")
            info["inertial"] = {
                "link_name": link_name,
                "origin_xyz": xyz,
                "origin_rpy": rpy,
                "mass": float(mass_elem.attrib["value"]) if mass_elem is not None else 0.0,
                "ixx": float(inertia_elem.attrib.get("ixx", 0.0)) if inertia_elem is not None else 0.0,
                "ixy": float(inertia_elem.attrib.get("ixy", 0.0)) if inertia_elem is not None else 0.0,
                "ixz": float(inertia_elem.attrib.get("ixz", 0.0)) if inertia_elem is not None else 0.0,
                "iyy": float(inertia_elem.attrib.get("iyy", 0.0)) if inertia_elem is not None else 0.0,
                "iyz": float(inertia_elem.attrib.get("iyz", 0.0)) if inertia_elem is not None else 0.0,
                "izz": float(inertia_elem.attrib.get("izz", 0.0)) if inertia_elem is not None else 0.0,
            }
        model["links"][link_name] = info

    for joint in root.findall("joint"):
        joint_name = joint.attrib["name"]
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint_name} is missing parent or child")
        origin = joint.find("origin")
        axis_elem = joint.find("axis")
        limit_elem = joint.find("limit")
        mimic_elem = joint.find("mimic")
        model["joints"][joint_name] = {
            "joint_name": joint_name,
            "joint_type": joint.attrib["type"],
            "parent_link": parent.attrib["link"],
            "child_link": child.attrib["link"],
            "origin_xyz": parse_vec(origin.attrib.get("xyz") if origin is not None else "0 0 0"),
            "origin_rpy": parse_vec(origin.attrib.get("rpy") if origin is not None else "0 0 0"),
            "axis_xyz": (
                parse_vec(axis_elem.attrib.get("xyz"))
                if axis_elem is not None and "xyz" in axis_elem.attrib
                else None
            ),
            "lower": None if limit_elem is None or "lower" not in limit_elem.attrib else float(limit_elem.attrib["lower"]),
            "upper": None if limit_elem is None or "upper" not in limit_elem.attrib else float(limit_elem.attrib["upper"]),
            "velocity": (
                None if limit_elem is None or "velocity" not in limit_elem.attrib else float(limit_elem.attrib["velocity"])
            ),
            "effort": None if limit_elem is None or "effort" not in limit_elem.attrib else float(limit_elem.attrib["effort"]),
            "mimic_source": None if mimic_elem is None else mimic_elem.attrib.get("joint"),
            "mimic_multiplier": (
                None if mimic_elem is None or "multiplier" not in mimic_elem.attrib else float(mimic_elem.attrib["multiplier"])
            ),
            "mimic_offset": None if mimic_elem is None or "offset" not in mimic_elem.attrib else float(mimic_elem.attrib["offset"]),
        }
    return model


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_from_zip(zip_path: Path, member: str) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(member))


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_csv_rows_from_zip(zip_path: Path, member: str) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as handle:
        wrapper = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        return list(csv.DictReader(wrapper))


def maybe_check_urdf(path: Path, result: EvalResult) -> None:
    executable = shutil.which("check_urdf")
    if not executable:
        result.warn("check_urdf not found on PATH; using XML and structural validation only")
        return
    proc = subprocess.run(
        [executable, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        result.add(f"check_urdf failed: {proc.stdout.strip()}")


def compare_required_sets(model: dict, input_zip: Path, result: EvalResult) -> None:
    link_manifest = load_json_from_zip(input_zip, "metadata/link_manifest.json")
    expected_links = set(link_manifest["all_links"])
    actual_links = set(model["links"].keys())
    if actual_links != expected_links:
        missing = sorted(expected_links - actual_links)
        extra = sorted(actual_links - expected_links)
        if missing:
            result.add(f"Missing links: {missing}")
        if extra:
            result.add(f"Extra links: {extra}")

    joint_manifest = load_json_from_zip(input_zip, "metadata/joint_manifest.json")
    expected_joints = {joint["name"] for joint in joint_manifest["all_joints"]}
    actual_joints = set(model["joints"].keys())
    if actual_joints != expected_joints:
        missing = sorted(expected_joints - actual_joints)
        extra = sorted(expected_joints ^ actual_joints)
        if missing:
            result.add(f"Missing joints: {missing}")
        if extra:
            result.add(f"Extra joints: {sorted(actual_joints - expected_joints)}")


def compare_tree(model: dict, input_zip: Path, result: EvalResult) -> None:
    tree_hint = load_json_from_zip(input_zip, "metadata/kinematic_tree_hint.json")
    expected_edges = {tuple(edge) for edge in tree_hint["edges"]}
    actual_edges = {(joint["parent_link"], joint["child_link"]) for joint in model["joints"].values()}
    if actual_edges != expected_edges:
        missing = sorted(expected_edges - actual_edges)
        extra = sorted(actual_edges - expected_edges)
        if missing:
            result.add(f"Missing tree edges: {missing}")
        if extra:
            result.add(f"Extra tree edges: {extra}")

    floating_base_joint = model["joints"].get("floating_base_joint")
    if (
        not floating_base_joint
        or floating_base_joint["parent_link"] != "world"
        or floating_base_joint["child_link"] != "pelvis"
        or floating_base_joint["joint_type"] != "floating"
    ):
        result.add("floating_base_joint structure is incorrect")


def parse_joint_rows(rows: list[dict]) -> dict[str, dict]:
    parsed = {}
    for row in rows:
        parsed[row["joint_name"]] = {
            "joint_name": row["joint_name"],
            "joint_type": row["joint_type"],
            "parent_link": row["parent_link"],
            "child_link": row["child_link"],
            "origin_xyz": parse_vec(row["origin_xyz"]),
            "origin_rpy": parse_vec(row["origin_rpy"]),
            "axis_xyz": parse_vec(row["axis_xyz"]) if not is_nan_like(row["axis_xyz"]) else None,
            "lower": None if is_nan_like(row["lower"]) else float(row["lower"]),
            "upper": None if is_nan_like(row["upper"]) else float(row["upper"]),
            "velocity": None if is_nan_like(row["velocity"]) else float(row["velocity"]),
            "effort": None if is_nan_like(row["effort"]) else float(row["effort"]),
            "mimic_source": None if is_nan_like(row["mimic_source"]) else row["mimic_source"],
            "mimic_multiplier": None if is_nan_like(row["mimic_multiplier"]) else float(row["mimic_multiplier"]),
            "mimic_offset": None if is_nan_like(row["mimic_offset"]) else float(row["mimic_offset"]),
        }
    return parsed


def compare_joint_semantics(model: dict, input_zip: Path, result: EvalResult) -> None:
    expected = parse_joint_rows(load_csv_rows_from_zip(input_zip, "metadata/joint_parameters.csv"))
    for joint_name, exp in expected.items():
        act = model["joints"].get(joint_name)
        if act is None:
            continue
        for field in ["joint_type", "parent_link", "child_link"]:
            if act[field] != exp[field]:
                result.add(f"Joint {joint_name}: {field} mismatch ({act[field]!r} != {exp[field]!r})")
        if np.any(np.abs(act["origin_xyz"] - exp["origin_xyz"]) > XYZ_TOL):
            result.add(f"Joint {joint_name}: origin_xyz mismatch")
        if any(angle_diff(a, b) > RPY_TOL for a, b in zip(act["origin_rpy"], exp["origin_rpy"])):
            result.add(f"Joint {joint_name}: origin_rpy mismatch")
        if exp["joint_type"] not in {"fixed", "floating"}:
            if act["axis_xyz"] is None or exp["axis_xyz"] is None:
                result.add(f"Joint {joint_name}: missing axis")
            else:
                left = act["axis_xyz"] / max(np.linalg.norm(act["axis_xyz"]), 1e-12)
                right = exp["axis_xyz"] / max(np.linalg.norm(exp["axis_xyz"]), 1e-12)
                angle = math.degrees(math.acos(float(np.clip(np.dot(left, right), -1.0, 1.0))))
                if angle > AXIS_ANG_TOL_DEG:
                    result.add(f"Joint {joint_name}: axis mismatch ({angle:.3f} deg)")

        if exp["joint_type"] == "revolute":
            if act["lower"] is None or abs(act["lower"] - exp["lower"]) > REVOLUTE_LIMIT_TOL:
                result.add(f"Joint {joint_name}: lower limit mismatch")
            if act["upper"] is None or abs(act["upper"] - exp["upper"]) > REVOLUTE_LIMIT_TOL:
                result.add(f"Joint {joint_name}: upper limit mismatch")

        if exp["joint_type"] == "prismatic":
            if act["lower"] is None or abs(act["lower"] - exp["lower"]) > PRISMATIC_LIMIT_TOL:
                result.add(f"Joint {joint_name}: lower limit mismatch")
            if act["upper"] is None or abs(act["upper"] - exp["upper"]) > PRISMATIC_LIMIT_TOL:
                result.add(f"Joint {joint_name}: upper limit mismatch")

        for field, tol in [("velocity", VEL_REL_TOL), ("effort", EFF_REL_TOL)]:
            expected_value = exp[field]
            actual_value = act[field]
            if expected_value is None:
                continue
            if actual_value is None or rel_err(actual_value, expected_value) > tol:
                result.add(f"Joint {joint_name}: {field} mismatch")

        for field in ["mimic_source", "mimic_multiplier", "mimic_offset"]:
            expected_value = exp[field]
            actual_value = act[field]
            if expected_value is None and actual_value is None:
                continue
            if expected_value is None and actual_value is not None:
                result.add(f"Joint {joint_name}: unexpected {field}")
            elif expected_value is not None and actual_value is None:
                result.add(f"Joint {joint_name}: missing {field}")
            elif field == "mimic_source" and actual_value != expected_value:
                result.add(f"Joint {joint_name}: {field} mismatch")
            elif field != "mimic_source" and abs(float(actual_value) - float(expected_value)) > 1e-9:
                result.add(f"Joint {joint_name}: {field} mismatch")


def parse_geometry_rows(rows: list[dict], *, kind: str) -> dict[str, list[dict]]:
    parsed: dict[str, list[dict]] = {}
    for row in rows:
        entry = {
            "link_name": row["link_name"],
            "origin_xyz": parse_vec(row["origin_xyz"]),
            "origin_rpy": parse_vec(row["origin_rpy"]),
            "geometry_type": row["geometry_type"],
            "filename": None if is_nan_like(row.get("filename")) else row.get("filename"),
            "params": canonical_jsonish(row.get("params")),
        }
        if kind == "collision":
            entry["collision_name"] = None if is_nan_like(row.get("collision_name")) else row.get("collision_name")
        parsed.setdefault(row["link_name"], []).append(entry)
    return parsed


def geometry_rows_match(left: dict, right: dict, *, check_collision_name: bool) -> bool:
    if left["geometry_type"] != right["geometry_type"]:
        return False
    if (left.get("filename") or None) != (right.get("filename") or None):
        return False
    if canonical_jsonish(left.get("params")) != canonical_jsonish(right.get("params")):
        return False
    if np.any(np.abs(left["origin_xyz"] - right["origin_xyz"]) > XYZ_TOL):
        return False
    if any(angle_diff(a, b) > RPY_TOL for a, b in zip(left["origin_rpy"], right["origin_rpy"])):
        return False
    if check_collision_name:
        left_name = left.get("collision_name")
        right_name = right.get("collision_name")
        if left_name is not None and right_name is not None and left_name != right_name:
            return False
    return True


def compare_geometry(model: dict, input_zip: Path, result: EvalResult, *, which: str) -> None:
    if which == "visual":
        expected = parse_geometry_rows(load_csv_rows_from_zip(input_zip, "metadata/visual_geometry_table.csv"), kind="visual")
        actual = {link_name: link_info["visuals"] for link_name, link_info in model["links"].items()}
        check_collision_name = False
    else:
        expected = parse_geometry_rows(
            load_csv_rows_from_zip(input_zip, "metadata/collision_geometry_table.csv"),
            kind="collision",
        )
        actual = {link_name: link_info["collisions"] for link_name, link_info in model["links"].items()}
        check_collision_name = True

    for link_name, expected_rows in expected.items():
        actual_rows = list(actual.get(link_name, []))
        used = [False] * len(actual_rows)
        for expected_row in expected_rows:
            found = False
            for index, actual_row in enumerate(actual_rows):
                if used[index]:
                    continue
                if geometry_rows_match(
                    expected_row,
                    actual_row,
                    check_collision_name=check_collision_name,
                ):
                    used[index] = True
                    found = True
                    break
            if not found:
                result.add(
                    f"{which} geometry mismatch for link {link_name}: "
                    f"could not match expected entry {expected_row}"
                )
        extras = [actual_rows[index] for index, used_flag in enumerate(used) if not used_flag]
        if extras:
            result.add(f"Extra {which} geometry entries for link {link_name}: {len(extras)}")


def compare_inertials(model: dict, input_zip: Path, result: EvalResult) -> None:
    rows = load_csv_rows_from_zip(input_zip, "metadata/inertial_table.csv")
    for row in rows:
        link_name = row["link_name"]
        expected = {
            "origin_xyz": parse_vec(row["origin_xyz"]),
            "origin_rpy": parse_vec(row["origin_rpy"]),
            "mass": float(row["mass"]),
            "ixx": float(row["ixx"]),
            "ixy": float(row["ixy"]),
            "ixz": float(row["ixz"]),
            "iyy": float(row["iyy"]),
            "iyz": float(row["iyz"]),
            "izz": float(row["izz"]),
        }
        actual = model["links"].get(link_name, {}).get("inertial")
        if actual is None:
            result.add(f"Link {link_name}: missing inertial")
            continue
        if np.any(np.abs(actual["origin_xyz"] - expected["origin_xyz"]) > XYZ_TOL):
            result.add(f"Link {link_name}: inertial origin_xyz mismatch")
        if any(angle_diff(a, b) > RPY_TOL for a, b in zip(actual["origin_rpy"], expected["origin_rpy"])):
            result.add(f"Link {link_name}: inertial origin_rpy mismatch")
        if rel_err(actual["mass"], expected["mass"]) > MASS_REL_TOL:
            result.add(f"Link {link_name}: mass mismatch")
        for key in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
            if rel_err(float(actual[key]), float(expected[key])) > INERTIA_REL_TOL:
                result.add(f"Link {link_name}: {key} mismatch")


def build_children_map(model: dict) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for joint_name, joint in model["joints"].items():
        children.setdefault(joint["parent_link"], []).append(joint_name)
    return children


def compute_fk(model: dict, active_joint_positions: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    children = build_children_map(model)
    poses: dict[str, np.ndarray] = {"world": np.eye(4)}

    def recurse(parent_link: str) -> None:
        parent_pose = poses[parent_link]
        for joint_name in children.get(parent_link, []):
            joint = model["joints"][joint_name]
            joint_value = active_joint_positions.get(joint_name, 0.0)
            transform = (
                parent_pose
                @ transform_from_xyz_rpy(joint["origin_xyz"], joint["origin_rpy"])
                @ motion_transform(joint["joint_type"], joint["axis_xyz"], joint_value)
            )
            poses[joint["child_link"]] = transform
            recurse(joint["child_link"])

    recurse("world")
    return {
        link_name: (transform[:3, 3].copy(), matrix_to_quat_xyzw(transform[:3, :3]))
        for link_name, transform in poses.items()
    }


def compare_fk(model: dict, reference_dir: Path, result: EvalResult) -> None:
    samples = load_json(reference_dir / "gold_fk_samples.json")
    for sample in samples["samples"]:
        poses = compute_fk(model, sample["active_joint_positions"])
        for link_name, gold_pose in sample["link_poses"].items():
            if link_name not in poses:
                result.add(f"FK sample {sample['sample_name']}: missing link {link_name}")
                continue
            position, quat = poses[link_name]
            gold_position = np.array(gold_pose["xyz"], dtype=float)
            gold_quat = np.array(gold_pose["quat_xyzw"], dtype=float)
            if np.linalg.norm(position - gold_position) > POS_TOL:
                result.add(f"FK sample {sample['sample_name']}: position mismatch for {link_name}")
            if quat_angle_deg(quat, gold_quat) > ORI_TOL_DEG:
                result.add(f"FK sample {sample['sample_name']}: orientation mismatch for {link_name}")


def evaluate_files(*, output_file: Path, input_zip: Path, reference_dir: Path) -> dict:
    result = EvalResult(passed=True, errors=[], warnings=[])

    if output_file.name != "submission.urdf":
        result.add("Submission file must be named submission.urdf")

    try:
        ET.parse(output_file)
    except Exception as exc:
        result.add(f"XML parsing failed: {exc}")

    if not result.errors:
        maybe_check_urdf(output_file, result)

    model = None
    if not result.errors:
        try:
            model = parse_urdf(output_file)
        except Exception as exc:
            result.add(f"URDF structural parsing failed: {exc}")

    if model is not None:
        compare_required_sets(model, input_zip, result)
        compare_tree(model, input_zip, result)
        compare_joint_semantics(model, input_zip, result)
        compare_geometry(model, input_zip, result, which="visual")
        compare_geometry(model, input_zip, result, which="collision")
        compare_inertials(model, input_zip, result)
        compare_fk(model, reference_dir, result)

    result.passed = not result.errors
    return {
        "score": 1.0 if result.passed else 0.0,
        "passed": result.passed,
        "num_errors": len(result.errors),
        "num_warnings": len(result.warnings),
        "errors": result.errors,
        "warnings": result.warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zip", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()

    evaluation = evaluate_files(
        output_file=args.submission,
        input_zip=args.input_zip,
        reference_dir=args.reference_dir,
    )
    print(json.dumps(evaluation, indent=2))
    sys.exit(0 if evaluation["passed"] else 1)


if __name__ == "__main__":
    main()
