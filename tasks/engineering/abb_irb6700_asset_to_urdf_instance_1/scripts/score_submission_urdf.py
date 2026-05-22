"""Semantic scorer for the ABB IRB6700 URDF reconstruction task."""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

EPS = 1e-9
SEMANTIC_TOL = 1e-4
POSE_POSITION_TOL = 1e-3
POSE_ANGLE_TOL_RAD = 1e-2

WEIGHT_LINKS = 0.05
WEIGHT_JOINTS = 0.25
WEIGHT_FK = 0.70
FK_POSITION_FALLOFF = 0.05
FK_ORIENTATION_FALLOFF = math.pi / 6

JOINT_ATTR_WEIGHTS: dict[str, float] = {
    "type": 1,
    "parent_child": 1,
    "origin_xyz": 8,
    "origin_rpy": 3,
    "axis": 3,
    "lower": 1,
    "upper": 1,
    "velocity": 1,
    "effort": 1,
    "mimic_source": 1,
    "mimic_multiplier": 1,
}

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


@dataclass(frozen=True)
class JointModel:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: Vec3
    origin_rpy: Vec3
    axis_xyz: Vec3 | None
    lower: float | None
    upper: float | None
    velocity: float | None
    effort: float | None
    mimic_source: str | None
    mimic_multiplier: float | None


@dataclass(frozen=True)
class LinkModel:
    name: str
    visual_meshes: tuple[str, ...]
    collision_meshes: tuple[str, ...]


@dataclass(frozen=True)
class ParsedUrdf:
    robot_name: str
    links: dict[str, LinkModel]
    joints: dict[str, JointModel]


def _parse_vec3(raw: str | None, *, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    if raw is None or raw.strip() == "":
        return default
    parts = raw.split()
    if len(parts) != 3:
        raise ValueError(f"expected 3-vector, got: {raw!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return float(value)


def _norm(vec: Vec3) -> float:
    return math.sqrt(sum(component * component for component in vec))


def _normalize_axis(vec: Vec3 | None) -> Vec3 | None:
    if vec is None:
        return None
    length = _norm(vec)
    if length <= EPS:
        return (0.0, 0.0, 0.0)
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _vectors_close(left: Vec3 | None, right: Vec3 | None, tol: float = SEMANTIC_TOL) -> bool:
    if left is None or right is None:
        return left is right
    return all(abs(a - b) <= tol for a, b in zip(left, right))


def _floats_close(left: float | None, right: float | None, tol: float = SEMANTIC_TOL) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= tol


def _identity_matrix() -> Mat3:
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def _mat_mul(left: Mat3, right: Mat3) -> Mat3:
    rows: list[Vec3] = []
    for row in left:
        values = []
        for col_idx in range(3):
            values.append(
                row[0] * right[0][col_idx]
                + row[1] * right[1][col_idx]
                + row[2] * right[2][col_idx]
            )
        rows.append((values[0], values[1], values[2]))
    return (rows[0], rows[1], rows[2])


def _mat_transpose(matrix: Mat3) -> Mat3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _mat_vec_mul(matrix: Mat3, vec: Vec3) -> Vec3:
    return (
        matrix[0][0] * vec[0] + matrix[0][1] * vec[1] + matrix[0][2] * vec[2],
        matrix[1][0] * vec[0] + matrix[1][1] * vec[1] + matrix[1][2] * vec[2],
        matrix[2][0] * vec[0] + matrix[2][1] * vec[1] + matrix[2][2] * vec[2],
    )


def _vec_add(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _rpy_to_matrix(rpy: Vec3) -> Mat3:
    roll, pitch, yaw = rpy
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    rz = (
        (cy, -sy, 0.0),
        (sy, cy, 0.0),
        (0.0, 0.0, 1.0),
    )
    ry = (
        (cp, 0.0, sp),
        (0.0, 1.0, 0.0),
        (-sp, 0.0, cp),
    )
    rx = (
        (1.0, 0.0, 0.0),
        (0.0, cr, -sr),
        (0.0, sr, cr),
    )
    return _mat_mul(_mat_mul(rz, ry), rx)


def _axis_angle_to_matrix(axis: Vec3, angle: float) -> Mat3:
    axis = _normalize_axis(axis)
    if axis is None:
        return _identity_matrix()
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


def _quat_to_matrix(quat_xyzw: Vec3 | tuple[float, float, float, float]) -> Mat3:
    x, y, z, w = quat_xyzw
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _rotation_angle(left: Mat3, right: Mat3) -> float:
    delta = _mat_mul(_mat_transpose(left), right)
    trace = delta[0][0] + delta[1][1] + delta[2][2]
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.acos(cos_angle)


def _collect_meshes(link_element: ET.Element, tag_name: str) -> tuple[str, ...]:
    meshes: list[str] = []
    for element in link_element.findall(tag_name):
        mesh = element.find("./geometry/mesh")
        if mesh is None:
            continue
        filename = mesh.attrib.get("filename", "").strip()
        if filename:
            meshes.append(filename)
    return tuple(meshes)


def parse_urdf(urdf_path: Path) -> ParsedUrdf:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"{urdf_path} does not have a <robot> root")

    links: dict[str, LinkModel] = {}
    joints: dict[str, JointModel] = {}
    for link_element in root.findall("link"):
        name = link_element.attrib.get("name", "").strip()
        if not name:
            raise ValueError(f"{urdf_path} contains a link without a name")
        if name in links:
            raise ValueError(f"{urdf_path} contains duplicate link {name}")
        links[name] = LinkModel(
            name=name,
            visual_meshes=_collect_meshes(link_element, "visual"),
            collision_meshes=_collect_meshes(link_element, "collision"),
        )

    for joint_element in root.findall("joint"):
        name = joint_element.attrib.get("name", "").strip()
        if not name:
            raise ValueError(f"{urdf_path} contains a joint without a name")
        if name in joints:
            raise ValueError(f"{urdf_path} contains duplicate joint {name}")
        parent = joint_element.find("parent")
        child = joint_element.find("child")
        if parent is None or child is None:
            raise ValueError(f"{urdf_path} joint {name} is missing parent or child")
        joint_type = joint_element.attrib.get("type", "").strip()
        if not joint_type:
            raise ValueError(f"{urdf_path} joint {name} is missing a type")
        origin = joint_element.find("origin")
        axis = joint_element.find("axis")
        limit = joint_element.find("limit")
        mimic = joint_element.find("mimic")
        joints[name] = JointModel(
            name=name,
            joint_type=joint_type,
            parent=parent.attrib["link"],
            child=child.attrib["link"],
            origin_xyz=_parse_vec3(None if origin is None else origin.attrib.get("xyz")),
            origin_rpy=_parse_vec3(None if origin is None else origin.attrib.get("rpy")),
            axis_xyz=(
                None
                if joint_type == "fixed"
                else _normalize_axis(_parse_vec3(None if axis is None else axis.attrib.get("xyz"), default=(1.0, 0.0, 0.0)))
            ),
            lower=None if limit is None else _parse_optional_float(limit.attrib.get("lower")),
            upper=None if limit is None else _parse_optional_float(limit.attrib.get("upper")),
            velocity=None if limit is None else _parse_optional_float(limit.attrib.get("velocity")),
            effort=None if limit is None else _parse_optional_float(limit.attrib.get("effort")),
            mimic_source=None if mimic is None else mimic.attrib.get("joint"),
            mimic_multiplier=(
                None
                if mimic is None
                else float(mimic.attrib.get("multiplier", "1.0"))
            ),
        )

    return ParsedUrdf(
        robot_name=root.attrib.get("name", ""),
        links=links,
        joints=joints,
    )


def load_expected_joint_table(path: Path) -> dict[str, JointModel]:
    expected: dict[str, JointModel] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["joint_name"]
            joint_type = row["joint_type"]
            expected[name] = JointModel(
                name=name,
                joint_type=joint_type,
                parent=row["parent_link"],
                child=row["child_link"],
                origin_xyz=_parse_vec3(row["origin_xyz"]),
                origin_rpy=_parse_vec3(row["origin_rpy"]),
                axis_xyz=(
                    None
                    if joint_type == "fixed"
                    else _normalize_axis(_parse_vec3(row["axis_xyz"], default=(1.0, 0.0, 0.0)))
                ),
                lower=_parse_optional_float(row["lower"]),
                upper=_parse_optional_float(row["upper"]),
                velocity=_parse_optional_float(row["velocity"]),
                effort=_parse_optional_float(row["effort"]),
                mimic_source=row["mimic_source"] or None,
                mimic_multiplier=_parse_optional_float(row["mimic_multiplier"]),
            )
    return expected


def load_expected_link_table(path: Path) -> dict[str, LinkModel]:
    expected: dict[str, LinkModel] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["link_name"]
            visual = tuple(filter(None, [row["visual_mesh"].strip()]))
            collision = tuple(filter(None, [row["collision_mesh"].strip()]))
            expected[name] = LinkModel(
                name=name,
                visual_meshes=visual,
                collision_meshes=collision,
            )
    return expected


def _score_links(candidate: ParsedUrdf, expected_links: dict[str, LinkModel]) -> tuple[float, str]:
    if not expected_links:
        return 1.0, "no expected links"
    correct = 0
    total = len(expected_links)
    for name, expected in expected_links.items():
        actual = candidate.links.get(name)
        if actual is None:
            continue
        if actual.visual_meshes == expected.visual_meshes and actual.collision_meshes == expected.collision_meshes:
            correct += 1
    score = correct / total
    return score, f"{correct}/{total} links correct"


def _score_joint_semantics(candidate: ParsedUrdf, expected_joints: dict[str, JointModel]) -> tuple[float, str]:
    if not expected_joints:
        return 1.0, "no expected joints"
    joint_scores: list[float] = []
    details: list[str] = []
    for name, expected in expected_joints.items():
        actual = candidate.joints.get(name)
        if actual is None:
            joint_scores.append(0.0)
            details.append(f"{name}: missing")
            continue
        W = JOINT_ATTR_WEIGHTS
        scored = 0.0
        total_w = 0.0
        missed: list[str] = []

        def _check(attr: str, ok: bool) -> None:
            nonlocal scored, total_w
            w = W[attr]
            total_w += w
            if ok:
                scored += w
            else:
                missed.append(attr)

        _check("type", actual.joint_type == expected.joint_type)
        _check("parent_child", actual.parent == expected.parent and actual.child == expected.child)
        _check("origin_xyz", _vectors_close(actual.origin_xyz, expected.origin_xyz))
        _check("origin_rpy", _vectors_close(actual.origin_rpy, expected.origin_rpy))
        if expected.joint_type != "fixed":
            _check("axis", _vectors_close(actual.axis_xyz, expected.axis_xyz))
        _check("lower", _floats_close(actual.lower, expected.lower))
        _check("upper", _floats_close(actual.upper, expected.upper))
        _check("velocity", _floats_close(actual.velocity, expected.velocity))
        _check("effort", _floats_close(actual.effort, expected.effort))
        _check("mimic_source", actual.mimic_source == expected.mimic_source)
        _check("mimic_multiplier", _floats_close(actual.mimic_multiplier, expected.mimic_multiplier))
        js = scored / total_w if total_w > 0 else 0.0
        joint_scores.append(js)
        if js < 1.0:
            details.append(f"{name}: {scored:.0f}/{total_w:.0f} [{','.join(missed)}]")
    score = sum(joint_scores) / len(joint_scores)
    detail_str = "; ".join(details) if details else "all joints fully correct"
    return score, detail_str


def _compose(parent_rot: Mat3, parent_xyz: Vec3, child_rot: Mat3, child_xyz: Vec3) -> tuple[Mat3, Vec3]:
    return (
        _mat_mul(parent_rot, child_rot),
        _vec_add(_mat_vec_mul(parent_rot, child_xyz), parent_xyz),
    )


def _joint_motion_transform(joint: JointModel, joint_value: float) -> tuple[Mat3, Vec3]:
    if joint.joint_type == "fixed":
        return (_identity_matrix(), (0.0, 0.0, 0.0))
    axis = joint.axis_xyz or (1.0, 0.0, 0.0)
    if joint.joint_type in {"revolute", "continuous"}:
        return (_axis_angle_to_matrix(axis, joint_value), (0.0, 0.0, 0.0))
    if joint.joint_type == "prismatic":
        return (_identity_matrix(), (axis[0] * joint_value, axis[1] * joint_value, axis[2] * joint_value))
    raise ValueError(f"unsupported joint type: {joint.joint_type}")


def _resolve_joint_positions(joints: dict[str, JointModel], active_positions: dict[str, float]) -> dict[str, float]:
    resolved = {name: 0.0 for name, joint in joints.items() if joint.joint_type != "fixed"}
    resolved.update(active_positions)

    changed = True
    for _ in range(len(joints)):
        if not changed:
            break
        changed = False
        for joint in joints.values():
            if joint.mimic_source is None:
                continue
            if joint.mimic_source not in resolved:
                continue
            value = resolved[joint.mimic_source] * (joint.mimic_multiplier or 1.0)
            if resolved.get(joint.name) != value:
                resolved[joint.name] = value
                changed = True
    return resolved


def _fk_link_poses(model: ParsedUrdf, root_link: str, active_positions: dict[str, float]) -> dict[str, tuple[Vec3, Mat3]]:
    children: dict[str, list[JointModel]] = {}
    child_links: set[str] = set()
    for joint in model.joints.values():
        if joint.parent not in model.links or joint.child not in model.links:
            raise ValueError(f"joint {joint.name} references an unknown link")
        children.setdefault(joint.parent, []).append(joint)
        if joint.child in child_links:
            raise ValueError(f"link {joint.child} has multiple parents")
        child_links.add(joint.child)

    if root_link not in model.links:
        raise ValueError(f"reference frame {root_link} is missing from the URDF")

    resolved_positions = _resolve_joint_positions(model.joints, active_positions)
    poses: dict[str, tuple[Vec3, Mat3]] = {root_link: ((0.0, 0.0, 0.0), _identity_matrix())}
    stack = [root_link]
    while stack:
        parent_name = stack.pop()
        parent_xyz, parent_rot = poses[parent_name]
        for joint in children.get(parent_name, []):
            if joint.child in poses:
                raise ValueError(f"cycle detected at link {joint.child}")
            origin_transform = (_rpy_to_matrix(joint.origin_rpy), joint.origin_xyz)
            joint_transform = _joint_motion_transform(joint, resolved_positions.get(joint.name, 0.0))
            world_transform = _compose(parent_rot, parent_xyz, origin_transform[0], origin_transform[1])
            child_rot, child_xyz = _compose(
                world_transform[0],
                world_transform[1],
                joint_transform[0],
                joint_transform[1],
            )
            poses[joint.child] = (child_xyz, child_rot)
            stack.append(joint.child)

    if set(poses) != set(model.links):
        missing = sorted(set(model.links) - set(poses))
        raise ValueError(f"unable to reach all links from {root_link}: {missing}")
    return poses


def _score_pose_samples(
    candidate: ParsedUrdf,
    reference_model: ParsedUrdf,
    reference_manifest_path: Path,
) -> tuple[float, str]:
    try:
        manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return 0.0, f"failed to load pose manifest: {exc}"

    root_link = manifest["reference_frame"]
    sample_scores: list[float] = []
    for sample in manifest["samples"]:
        active_positions = {
            name: float(value)
            for name, value in sample["active_joint_positions"].items()
        }
        try:
            candidate_poses = _fk_link_poses(candidate, root_link, active_positions)
            reference_poses = _fk_link_poses(reference_model, root_link, active_positions)
        except Exception:
            sample_scores.append(0.0)
            continue
        link_scores: list[float] = []
        for link_name, (expected_xyz, expected_rot) in reference_poses.items():
            if link_name not in candidate_poses:
                link_scores.append(0.0)
                continue
            candidate_xyz, candidate_rot = candidate_poses[link_name]
            pos_err = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(candidate_xyz, expected_xyz))
            )
            orient_err = _rotation_angle(candidate_rot, expected_rot)
            pos_score = max(0.0, 1.0 - pos_err / FK_POSITION_FALLOFF)
            orient_score = max(0.0, 1.0 - orient_err / FK_ORIENTATION_FALLOFF)
            link_scores.append(0.5 * pos_score + 0.5 * orient_score)
        if link_scores:
            sample_scores.append(sum(link_scores) / len(link_scores))
        else:
            sample_scores.append(0.0)

    if not sample_scores:
        return 0.0, "no samples evaluated"
    score = sum(sample_scores) / len(sample_scores)
    return score, f"average FK score across {len(sample_scores)} samples"


def evaluate_files(*, output_file: Path, reference_dir: Path) -> dict[str, object]:
    zero = {"score": 0.0, "link_score": 0.0, "joint_score": 0.0, "fk_score": 0.0}

    if not output_file.exists():
        return {**zero, "reason": "missing submission.urdf"}

    try:
        candidate = parse_urdf(output_file)
    except Exception as exc:
        return {**zero, "reason": f"failed to parse candidate URDF: {exc}"}

    try:
        expected_links = load_expected_link_table(reference_dir / "gold_link_mesh_table.csv")
        expected_joints = load_expected_joint_table(reference_dir / "gold_joint_table.csv")
        reference_model = parse_urdf(reference_dir / "abb_irb6700_200_260.urdf")
    except Exception as exc:
        return {**zero, "reason": f"failed to parse hidden reference data: {exc}"}

    link_score, link_detail = _score_links(candidate, expected_links)
    joint_score, joint_detail = _score_joint_semantics(candidate, expected_joints)
    fk_score, fk_detail = _score_pose_samples(
        candidate, reference_model, reference_dir / "joint_manifest.json",
    )

    total = (
        WEIGHT_LINKS * link_score
        + WEIGHT_JOINTS * joint_score
        + WEIGHT_FK * fk_score
    )

    reason = (
        f"links={link_score:.3f} ({link_detail}), "
        f"joints={joint_score:.3f} ({joint_detail}), "
        f"fk={fk_score:.3f} ({fk_detail})"
    )

    return {
        "score": round(total, 4),
        "link_score": round(link_score, 4),
        "joint_score": round(joint_score, 4),
        "fk_score": round(fk_score, 4),
        "reason": reason,
    }
