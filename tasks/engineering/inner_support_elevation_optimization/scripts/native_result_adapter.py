"""Extract reviewed physical monitors from evaluator-owned native Kratos results."""

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


COMPONENTS = ("DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z")
ROTATIONS = ("ROTATION_X", "ROTATION_Y", "ROTATION_Z")
GEOMETRY_ATOL_M = 0.002
FIELD_ATOL_M = 2e-10
FIELD_RTOL = 2e-5


@dataclass
class NativeResult:
    nodes: dict = field(default_factory=dict)
    elements: dict = field(default_factory=dict)
    fields: dict = field(default_factory=lambda: {name: {} for name in COMPONENTS})
    constraints: dict = field(default_factory=dict)
    rotation_constraints: dict = field(default_factory=dict)
    constraint_variables: dict = field(default_factory=dict)
    active: set = field(default_factory=set)
    active_constraints: set = field(default_factory=set)
    soil_nodes: set = field(default_factory=set)
    rotations: dict = field(default_factory=lambda: {name: {} for name in ROTATIONS})
    properties: dict = field(default_factory=dict)

    def displacement(self, identifier):
        return np.array([self.fields[name][identifier] for name in COMPONENTS])


def trusted_bytes(path, expected_sha256):
    """Bind an artifact to a digest supplied by the evaluator, never the submission."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000_000:
        raise ValueError(f"Unsafe, missing or oversized native artifact: {path.name}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"Evaluator-owned artifact digest mismatch: {path.name}")
    return content


def read_native_result(
    path,
    *,
    expected_sha256,
    coordinates_path=None,
    coordinates_sha256=None,
    precise_translations=False,
):
    """Read native WRITE output, including activity and installed-wall constraints."""
    content = trusted_bytes(path, expected_sha256)
    result = NativeResult()
    stack = []
    constraint_ids = set()
    for raw in io.StringIO(content.decode("utf-8")):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        cells = line.split()
        if cells[0] == "Begin":
            stack.append(cells[1:])
            continue
        if cells[0] == "End":
            if not stack or stack.pop()[0] != cells[1]:
                raise ValueError("Unbalanced native MDPA block")
            continue
        if not stack:
            raise ValueError("Native data outside a block")
        block = stack[-1]
        if block[0] == "Nodes":
            identifier = int(cells[0])
            coordinates = np.array([float(value) for value in cells[1:]])
            if identifier in result.nodes or coordinates.shape != (3,):
                raise ValueError("Duplicate or malformed native node")
            if not np.isfinite(coordinates).all():
                raise ValueError("Nonfinite native coordinates")
            result.nodes[identifier] = coordinates
        elif block[0] == "Elements":
            identifier, property_id, *connectivity = map(int, cells)
            if identifier in result.elements or len(connectivity) != len(set(connectivity)):
                raise ValueError("Duplicate native element or repeated vertex")
            result.elements[identifier] = (block[1], property_id, tuple(connectivity))
        elif block[0] == "Properties" and len(cells) == 2:
            try:
                value = float(cells[1])
            except ValueError:
                continue
            if not np.isfinite(value):
                raise ValueError("Nonfinite native property")
            result.properties.setdefault(int(block[1]), {})[cells[0]] = value
        elif block[0] == "NodalData" and block[1] in COMPONENTS + ROTATIONS:
            identifier, _, value = cells
            identifier, value = int(identifier), float(value)
            target = result.fields if block[1] in COMPONENTS else result.rotations
            if identifier in target[block[1]] or not np.isfinite(value):
                raise ValueError("Duplicate or nonfinite native displacement")
            target[block[1]][identifier] = value
        elif block[0] == "Constraints" and block[2] in COMPONENTS + ROTATIONS:
            match = re.fullmatch(r"(\d+)\s+(\S+)\s+\[([^]]+)\]\s+(\d+)\s+(.+)", line)
            if not match or block[1] != "LinearMasterSlaveConstraint":
                raise ValueError("Unsupported native translation constraint")
            identifier, offset, weights, slave, masters = match.groups()
            identifier, slave = int(identifier), int(slave)
            weights = tuple(float(value) for value in re.split(r"[,\s]+", weights.strip()))
            masters = tuple(int(value) for value in masters.split())
            target = result.constraints if block[2] in COMPONENTS else result.rotation_constraints
            if (
                len(weights) != len(masters)
                or len(block[2:]) != len(masters) + 1
                or not set(block[2:]).issubset(COMPONENTS + ROTATIONS)
                or identifier in constraint_ids
                or (slave, block[2]) in target
                or not np.isfinite([float(offset), *weights]).all()
            ):
                raise ValueError("Malformed or repeated native translation constraint")
            constraint_ids.add(identifier)
            target[slave, block[2]] = (identifier, float(offset), weights, masters)
            result.constraint_variables[slave, block[2]] = tuple(block[3:])
        elif len(stack) > 1 and stack[-2] == ["SubModelPart", "Active"]:
            if block[0] == "SubModelPartElements":
                result.active.add(int(line))
            elif block[0] == "SubModelPartConstraints":
                result.active_constraints.add(int(line))
        elif len(stack) > 1 and stack[-2] == ["SubModelPart", "Soil"]:
            if block[0] == "SubModelPartNodes":
                result.soil_nodes.add(int(line))
    if stack or not result.nodes or not result.active or not result.soil_nodes:
        raise ValueError("Missing native mesh, active membership or soil membership")
    if not result.active.issubset(result.elements) or not result.soil_nodes.issubset(result.nodes):
        raise ValueError("Unknown native membership")
    if any(set(values) != set(result.nodes) for values in result.fields.values()):
        raise ValueError("Missing native displacement field/node")
    if any(
        not set(connectivity).issubset(result.nodes)
        for _, _, connectivity in result.elements.values()
    ):
        raise ValueError("Unknown element node")
    if coordinates_path is not None:
        read_native_coordinates(result, coordinates_path, expected_sha256=coordinates_sha256)
        if precise_translations:
            read_native_translations(result, coordinates_path, expected_sha256=coordinates_sha256)
    elif coordinates_sha256 is not None:
        raise ValueError("Native coordinate digest requires its artifact")
    elif precise_translations:
        raise ValueError("Precise translations require the pinned native capture")
    return result


def read_native_coordinates(result, path, *, expected_sha256):
    """Restore pinned native X0 coordinates after checking MDPA decimal serialization."""
    content = trusted_bytes(path, expected_sha256)
    rows = csv.DictReader(io.StringIO(content.decode("utf-8")), delimiter="\t")
    required = {"node", "x0_m", "y0_m", "z0_m"}
    if not required.issubset(rows.fieldnames or ()):
        raise ValueError("Missing native coordinate columns")
    coordinates = {}
    for row in rows:
        try:
            identifier = int(row["node"])
            point = np.array([float(row[name]) for name in ("x0_m", "y0_m", "z0_m")])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed native coordinate row") from error
        if identifier in coordinates or identifier not in result.nodes:
            raise ValueError("Duplicate or unknown native coordinate node")
        if not np.isfinite(point).all():
            raise ValueError("Nonfinite native coordinate")
        rounded = np.array([float(format(value, ".6g")) for value in point])
        if not np.array_equal(result.nodes[identifier], rounded):
            raise ValueError("Native coordinates disagree with MDPA six-digit serialization")
        coordinates[identifier] = point
    if set(coordinates) != set(result.nodes):
        raise ValueError("Missing native coordinate node")
    result.nodes = coordinates


def read_native_translations(result, path, *, expected_sha256):
    """Restore evaluator-captured native translations that round exactly to native MDPA."""
    content = trusted_bytes(path, expected_sha256)
    rows = csv.DictReader(io.StringIO(content.decode("utf-8")), delimiter="\t")
    columns = ("ux_m", "uy_m", "uz_m")
    if not {"node", *columns}.issubset(rows.fieldnames or ()):
        raise ValueError("Missing native translation columns")
    translations = {}
    for row in rows:
        try:
            identifier = int(row["node"])
            values = np.array([float(row[name]) for name in columns])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed native translation row") from error
        if identifier in translations or identifier not in result.nodes:
            raise ValueError("Duplicate or unknown native translation node")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite native translation")
        rounded = np.array([float(format(value, ".6g")) for value in values])
        if not np.array_equal(result.displacement(identifier), rounded):
            raise ValueError("Native translations disagree with MDPA six-digit serialization")
        translations[identifier] = values
    if set(translations) != set(result.nodes):
        raise ValueError("Missing native translation node")
    result.fields = {
        component: {identifier: values[index] for identifier, values in translations.items()}
        for index, component in enumerate(COMPONENTS)
    }


def read_stage1_baseline(path, *, expected_sha256, soil_nodes):
    """Read the pinned runner Stage1 capture, not a candidate NPZ or restart state."""
    content = trusted_bytes(path, expected_sha256)
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        identifiers = archive["node_ids"]
        values = archive["displacement_m"]
    if (
        identifiers.ndim != 1
        or not np.issubdtype(identifiers.dtype, np.integer)
        or len(set(identifiers)) != len(identifiers)
        or set(identifiers) != set(soil_nodes)
        or values.shape != (len(identifiers), 3)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Missing or malformed native Stage1 baseline")
    return dict(zip(map(int, identifiers), values))


def wall_vertex(result, baseline, identifier):
    """Recover original geometry and construction motion from native attachment offsets."""
    constraints = [result.constraints.get((identifier, component)) for component in COMPONENTS]
    if any(constraint is None for constraint in constraints):
        raise ValueError("Missing wall translation attachment")
    masters = {constraint[3] for constraint in constraints}
    if len(masters) != 1 or any(constraint[2] != (1.0,) for constraint in constraints):
        raise ValueError("Unsupported wall attachment; requires one soil master per vertex")
    if any(
        result.constraint_variables.get((identifier, component), (component,)) != (component,)
        for component in COMPONENTS
    ):
        raise ValueError("Unsupported wall attachment variable; requires soil translation")
    master = next(iter(masters))[0]
    if master not in baseline or not all(
        constraint[0] in result.active_constraints for constraint in constraints
    ):
        raise ValueError("Inactive wall attachment or missing Stage1 master")
    offsets = np.array([constraint[1] for constraint in constraints])
    displacement = result.displacement(identifier)
    master_displacement = result.displacement(master)
    serialization_budget = FIELD_ATOL_M + 6e-6 * (
        np.abs(displacement) + np.abs(master_displacement) + np.abs(offsets)
    )
    if np.any(np.abs(displacement - master_displacement - offsets) > serialization_budget):
        raise ValueError("Native wall displacement violates its attachment")
    if not np.allclose(
        result.nodes[identifier], result.nodes[master] - offsets, rtol=0, atol=GEOMETRY_ATOL_M
    ):
        raise ValueError("Native wall installation geometry disagrees with attachment")
    return result.nodes[master], displacement - offsets - baseline[master]


def ground_triangles(result, baseline, ground_z):
    triangles = []
    for identifier in sorted(result.active):
        kind, _, connectivity = result.elements[identifier]
        if kind != "UPwSmallStrainElement3D4N":
            continue
        surface = [node for node in connectivity if abs(result.nodes[node][2] - ground_z) < 1e-8]
        if len(surface) != 3:
            continue
        if not set(surface).issubset(baseline):
            raise ValueError("Missing Stage1 ground baseline")
        coordinates = np.array([result.nodes[node][:2] for node in surface])
        matrix = np.vstack([coordinates.T, np.ones(3)])
        if abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("Degenerate ground face")
        displacement = np.array([result.displacement(node) - baseline[node] for node in surface])
        triangles.append((identifier, np.linalg.inv(matrix), displacement))
    return triangles


def wall_panels(result, baseline, property_ids):
    panels = []
    vertices = {}
    for identifier in sorted(result.active):
        kind, property_id, connectivity = result.elements[identifier]
        if property_id not in property_ids:
            continue
        if kind not in {"MITCThickShellElement3D4N", "ShellThickElement3D4N"}:
            raise ValueError("Reviewed wall property contains an unsupported element")
        for node in connectivity:
            if node not in vertices:
                vertices[node] = wall_vertex(result, baseline, node)
        coordinates = np.array([vertices[node][0] for node in connectivity])
        displacements = np.array([vertices[node][1] for node in connectivity])
        lower, upper = coordinates[:, 2].min(), coordinates[:, 2].max()
        bottom = np.flatnonzero(np.abs(coordinates[:, 2] - lower) < 1e-8)
        top = np.flatnonzero(np.abs(coordinates[:, 2] - upper) < 1e-8)
        if len(connectivity) != 4 or len(bottom) != 2 or len(top) != 2 or upper <= lower:
            raise ValueError("Reader supports vertically extruded Q4 wall panels only")
        start, end = coordinates[bottom, :2]
        top_order = [
            top[np.argmin(np.linalg.norm(coordinates[top, :2] - point, axis=1))]
            for point in (start, end)
        ]
        if len(set(top_order)) != 2 or not np.allclose(
            coordinates[top_order, :2], [start, end], rtol=0, atol=GEOMETRY_ATOL_M
        ):
            raise ValueError("Nonvertical native wall panel")
        panels.append(
            (identifier, start, end, lower, upper, displacements[bottom], displacements[top_order])
        )
    if not panels:
        raise ValueError("No active reviewed outer-wall panels")
    return panels


def ground_quadrilaterals(result, baseline, ground_z):
    faces = []
    face_indices = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    for identifier in sorted(result.active):
        kind, _, connectivity = result.elements[identifier]
        if kind != "UPwSmallStrainElement3D8N":
            continue
        if len(connectivity) != 8:
            raise ValueError("Malformed native eight-node hexahedron")
        coordinates = np.array([result.nodes[node] for node in connectivity])
        at_ground = set(np.flatnonzero(np.abs(coordinates[:, 2] - ground_z) < 1e-8))
        if len(at_ground) != 4 or coordinates[:, 2].max() > ground_z + 1e-8:
            continue
        face = next((indices for indices in face_indices if set(indices) == at_ground), None)
        if face is None:
            raise ValueError("Ground vertices do not form a native hexahedron face")
        nodes = [connectivity[index] for index in face]
        if not set(nodes).issubset(baseline):
            raise ValueError("Missing Stage1 ground baseline")
        points = coordinates[list(face), :2]
        local = points - points[0]
        scale = float(np.linalg.norm(np.ptp(local, axis=0)))
        coefficients = (
            np.array([[1, 1, 1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1], [1, -1, 1, -1]]) @ local / 4
        )
        determinants = [
            np.linalg.det(
                np.column_stack(
                    [
                        coefficients[1] + eta * coefficients[3],
                        coefficients[2] + xi * coefficients[3],
                    ]
                )
            )
            for xi, eta in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        if scale == 0 or not (
            min(determinants) > 1e-12 * scale**2 or max(determinants) < -1e-12 * scale**2
        ):
            raise ValueError("Degenerate or folded native Q4 ground face")
        displacements = np.array([result.displacement(node) - baseline[node] for node in nodes])
        faces.append((identifier, points[0], coefficients, scale, displacements))
    return faces


def quadrilateral_weights(origin, coefficients, scale, point):
    target = np.asarray(point) - origin
    local = np.zeros(2)
    for _ in range(30):
        xi, eta = local
        residual = (
            coefficients[0]
            + xi * coefficients[1]
            + eta * coefficients[2]
            + xi * eta * coefficients[3]
            - target
        )
        if np.linalg.norm(residual) <= 1e-11 * scale:
            if np.any(np.abs(local) > 1 + 1e-8):
                return None
            return (
                np.array(
                    [
                        (1 - xi) * (1 - eta),
                        (1 + xi) * (1 - eta),
                        (1 + xi) * (1 + eta),
                        (1 - xi) * (1 + eta),
                    ]
                )
                / 4
            )
        jacobian = np.column_stack(
            [coefficients[1] + eta * coefficients[3], coefficients[2] + xi * coefficients[3]]
        )
        if abs(np.linalg.det(jacobian)) <= 1e-12 * scale**2:
            return None
        local -= np.linalg.solve(jacobian, residual)
        if not np.isfinite(local).all():
            return None
    return None


def beam_physical_point(result, definition, point):
    """Read a reviewed rectangular Euler-Bernoulli beam material point in global axes."""
    point = np.asarray(point, dtype=float)
    identifier = definition["element"]
    if identifier not in result.active:
        raise ValueError("Inactive crown beam")
    kind, property_id, nodes = result.elements[identifier]
    if kind != "CrLinearBeamElement3D2N" or len(nodes) != 2:
        raise ValueError("Unsupported crown element")
    properties = result.properties[property_id]
    if properties.get("AREA_EFFECTIVE_Y", 0) or properties.get("AREA_EFFECTIVE_Z", 0):
        raise ValueError("Crown reader requires native shear-rigid beam interpolation")
    reference = np.asarray(definition["reference_axis_m"], dtype=float)
    frame = np.asarray(definition["native_initial_frame"], dtype=float)
    reference_frame = np.asarray(definition["reference_frame"], dtype=float)
    baseline = np.asarray(definition["stage1_axis_displacement_m"], dtype=float)
    if (
        reference.shape != (2, 3)
        or baseline.shape != (2, 3)
        or frame.shape != (3, 3)
        or reference_frame.shape != (3, 3)
        or not all(
            np.isfinite(value).all() for value in (reference, baseline, frame, reference_frame)
        )
        or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-8)
        or not np.allclose(reference_frame.T @ reference_frame, np.eye(3), atol=1e-8)
        or np.linalg.det(frame) < 0.999999
        or np.linalg.det(reference_frame) < 0.999999
    ):
        raise ValueError("Invalid evaluator-owned crown geometry")
    direction = reference[1] - reference[0]
    length_reference = np.linalg.norm(direction)
    fraction = float(np.dot(point - reference[0], direction) / length_reference**2)
    if not -1e-8 <= fraction <= 1 + 1e-8:
        raise ValueError("Physical point outside crown span")
    axis_point = (1 - fraction) * reference[0] + fraction * reference[1]
    offset_local = reference_frame.T @ (np.asarray(point) - axis_point)
    width = np.sqrt(12 * properties["I33"] / properties["CROSS_AREA"])
    height = np.sqrt(12 * properties["I22"] / properties["CROSS_AREA"])
    if not np.isclose(width * height, properties["CROSS_AREA"], rtol=2e-5):
        raise ValueError("Crown properties do not match a rectangular section")
    if (
        abs(offset_local[0]) > GEOMETRY_ATOL_M
        or abs(offset_local[1]) > width / 2 + 2e-6 * width
        or abs(offset_local[2]) > height / 2 + 2e-6 * height
    ):
        raise ValueError("Physical point outside native crown section")
    initial = np.array([result.nodes[node] for node in nodes])
    length = float(np.linalg.norm(initial[1] - initial[0]))
    if not np.allclose(
        frame[:, 0], (initial[1] - initial[0]) / length, rtol=0, atol=2 * GEOMETRY_ATOL_M / length
    ):
        raise ValueError("Crown frame disagrees with native reference geometry")
    translations = np.array([frame.T @ result.displacement(node) for node in nodes])
    try:
        rotations = np.array(
            [
                frame.T @ np.array([result.rotations[name][node] for name in ROTATIONS])
                for node in nodes
            ]
        )
    except KeyError as error:
        raise ValueError("Missing native crown rotation") from error
    if not np.isfinite(rotations).all():
        raise ValueError("Nonfinite native crown rotation")
    cubic = np.array(
        [
            1 - 3 * fraction**2 + 2 * fraction**3,
            length * (fraction - 2 * fraction**2 + fraction**3),
            3 * fraction**2 - 2 * fraction**3,
            length * (-(fraction**2) + fraction**3),
        ]
    )
    derivative = np.array(
        [
            (-6 * fraction + 6 * fraction**2) / length,
            1 - 4 * fraction + 3 * fraction**2,
            (6 * fraction - 6 * fraction**2) / length,
            -2 * fraction + 3 * fraction**2,
        ]
    )
    bending_y = np.array([translations[0, 1], rotations[0, 2], translations[1, 1], rotations[1, 2]])
    bending_z = np.array(
        [translations[0, 2], -rotations[0, 1], translations[1, 2], -rotations[1, 1]]
    )
    axis_displacement = np.array(
        [
            (1 - fraction) * translations[0, 0] + fraction * translations[1, 0],
            cubic @ bending_y,
            cubic @ bending_z,
        ]
    )
    section_rotation = np.array(
        [
            (1 - fraction) * rotations[0, 0] + fraction * rotations[1, 0],
            -derivative @ bending_z,
            derivative @ bending_y,
        ]
    )
    installed_axis = (1 - fraction) * initial[0] + fraction * initial[1]
    initial_offset = frame @ offset_local
    return (
        installed_axis
        - axis_point
        + frame @ axis_displacement
        + np.cross(frame @ section_rotation, initial_offset)
        + initial_offset
        - (np.asarray(point) - axis_point)
        - ((1 - fraction) * baseline[0] + fraction * baseline[1])
    )


def crown_intervals(result, definitions, point, wall_range):
    intervals = []
    for definition in definitions:
        reference = np.asarray(definition["reference_axis_m"], dtype=float)
        if np.ptp(reference[:, 2]) > 1e-8:
            raise ValueError("Crown profiles require a horizontal reviewed beam axis")
        direction = reference[1, :2] - reference[0, :2]
        fraction = np.dot(point[:2] - reference[0, :2], direction) / np.dot(direction, direction)
        distance = np.linalg.norm(point[:2] - reference[0, :2] - fraction * direction)
        if not -1e-8 <= fraction <= 1 + 1e-8 or distance > GEOMETRY_ATOL_M:
            continue
        lower, upper = definition["profile_range_z_m"]
        lower, upper = max(lower, wall_range[0]), min(upper, wall_range[1])
        if upper <= lower:
            continue
        values = [
            beam_physical_point(result, definition, np.array([*point[:2], elevation]))
            for elevation in (lower, upper)
        ]
        intervals.append((lower, upper, definition["element"], values))
    return intervals


def crown_section_placement(definition, point):
    """Return installation geometry motion of a section offset, excluding FE displacement."""
    reference = np.asarray(definition["reference_axis_m"], dtype=float)
    direction = reference[1] - reference[0]
    fraction = np.dot(point - reference[0], direction) / np.dot(direction, direction)
    offset = point - ((1 - fraction) * reference[0] + fraction * reference[1])
    frame = np.asarray(definition["native_initial_frame"], dtype=float)
    reference_frame = np.asarray(definition["reference_frame"], dtype=float)
    return frame @ reference_frame.T @ offset - offset


def validate_crown_attachments(result, baseline, panels, definitions):
    """Recognize constructor attachments only when relaxing a mixed-boundary rejection."""
    vertices = {}
    edges = set()
    for identifier, _, _, _, upper, _, _ in panels:
        connectivity = result.elements[identifier][2]
        for node in connectivity:
            if node not in vertices:
                vertices[node] = wall_vertex(result, baseline, node)
        top = tuple(
            sorted(node for node in connectivity if abs(vertices[node][0][2] - upper) < 1e-8)
        )
        edges.add(top)
    seen = set()
    for definition in definitions:
        identifier = definition["element"]
        if identifier in seen:
            raise ValueError("Duplicate native crown owner")
        seen.add(identifier)
        reference = np.asarray(definition["reference_axis_m"], dtype=float)
        lower, upper = definition["profile_range_z_m"]
        for elevation in (lower, upper):
            beam_physical_point(result, definition, [*reference[0, :2], elevation])
        nodes = result.elements[identifier][2]
        eccentricity = reference[0, 2] - lower
        for endpoint, node in enumerate(nodes):
            constraints = [
                (result.constraints if variable in COMPONENTS else result.rotation_constraints).get(
                    (node, variable)
                )
                for variable in COMPONENTS + ROTATIONS
            ]
            if any(constraint is None for constraint in constraints):
                raise ValueError("Missing native crown attachment")
            if any(constraint[0] not in result.active_constraints for constraint in constraints):
                raise ValueError("Inactive native crown attachment")
            wall_masters = constraints[3][3]
            if len(wall_masters) != 2 or tuple(sorted(wall_masters)) not in edges:
                raise ValueError("Native crown attachment does not own a reviewed wall edge")
            start, end = [vertices[master][0] for master in wall_masters]
            edge_direction = end - start
            edge_length = np.linalg.norm(edge_direction)
            if edge_length <= GEOMETRY_ATOL_M or not np.allclose(
                [start[2], end[2]], lower, rtol=0, atol=1e-8
            ):
                raise ValueError("Native crown attachment elevation disagrees with owner boundary")
            attachment = reference[endpoint] - [0, 0, eccentricity]
            fraction = float(np.dot(attachment - start, edge_direction) / edge_length**2)
            if (
                not -1e-8 <= fraction <= 1 + 1e-8
                or np.linalg.norm(attachment - start - fraction * edge_direction) > GEOMETRY_ATOL_M
            ):
                raise ValueError("Native crown endpoint does not lie on its attachment edge")
            fraction = float(np.clip(fraction, 0, 1))
            weights = (1 - fraction, fraction)
            soil_masters = tuple(
                result.constraints[master, COMPONENTS[0]][3][0] for master in wall_masters
            )
            if not set(soil_masters).issubset(result.soil_nodes):
                raise ValueError("Native crown attachment requires native soil masters")
            for component, (variable, constraint) in enumerate(
                zip(COMPONENTS + ROTATIONS, constraints)
            ):
                _, offset, actual_weights, masters = constraint
                expected_masters = soil_masters if component < 3 else wall_masters
                expected_variables = (variable, variable)
                expected_weights = weights
                if eccentricity and component < 2:
                    expected_masters += wall_masters
                    expected_variables += (ROTATIONS[1 - component],) * 2
                    sign = 1 if component == 0 else -1
                    expected_weights += tuple(sign * eccentricity * weight for weight in weights)
                if (
                    masters != expected_masters
                    or result.constraint_variables.get((node, variable)) != expected_variables
                    or len(actual_weights) != len(expected_weights)
                    or not np.allclose(
                        actual_weights,
                        expected_weights,
                        rtol=6e-6,
                        atol=2e-9 * max(1, abs(eccentricity)) / edge_length,
                    )
                ):
                    raise ValueError(
                        "Native crown attachment masters or interpolation disagree with geometry"
                    )
                try:
                    slave_values = result.fields if variable in COMPONENTS else result.rotations
                    slave_value = slave_values[variable][node]
                    terms = np.array(
                        [
                            coefficient
                            * (
                                result.fields if master_variable in COMPONENTS else result.rotations
                            )[master_variable][master]
                            for coefficient, master_variable, master in zip(
                                actual_weights, expected_variables, masters
                            )
                        ]
                    )
                except KeyError as error:
                    raise ValueError("Missing native crown attachment field") from error
                budget = FIELD_ATOL_M + 6e-6 * (
                    abs(slave_value) + abs(offset) + np.abs(terms).sum()
                )
                if (
                    not np.isfinite([slave_value, offset, *terms]).all()
                    or abs(slave_value - offset - terms.sum()) > budget
                ):
                    raise ValueError("Native crown displacement/rotation violates its attachment")
            initial_motion = result.nodes[node] - reference[endpoint]
            rotation_offsets = np.array([constraint[1] for constraint in constraints[3:]])
            rotation_motion = np.cross(-rotation_offsets, [0, 0, eccentricity])
            translation_offsets = np.array([constraint[1] for constraint in constraints[:3]])
            budget = FIELD_ATOL_M + 6e-6 * (
                np.abs(initial_motion) + np.abs(rotation_motion) + np.abs(translation_offsets)
            )
            if np.any(np.abs(initial_motion + rotation_motion + translation_offsets) > budget):
                raise ValueError(
                    "Native crown installation geometry disagrees with attachment offsets"
                )
            expected_baseline = sum(
                weight * baseline[master] for weight, master in zip(weights, soil_masters)
            )
            if not np.allclose(
                definition["stage1_axis_displacement_m"][endpoint],
                expected_baseline,
                rtol=FIELD_RTOL,
                atol=FIELD_ATOL_M,
            ):
                raise ValueError("Native crown baseline disagrees with attachment masters")


def extract_monitors(
    result, baseline, registration, *, outer_wall_property_ids, wall_range, crown_geometry=()
):
    """Sample all reviewed stations with native shape functions; never extrapolate."""
    if set(registration) != {
        f"{prefix}{index}" for prefix in ("DBC", "CX") for index in range(1, 11)
    }:
        raise ValueError("Exactly DBC1..10 and CX1..10 are required")
    coordinates = {name: np.asarray(point, dtype=float) for name, point in registration.items()}
    if any(point.shape != (3,) or not np.isfinite(point).all() for point in coordinates.values()):
        raise ValueError("Invalid monitor coordinates")
    soil_triangles = {}
    soil_quadrilaterals = {}
    panels = wall_panels(result, baseline, set(outer_wall_property_ids))
    crowns = {definition["element"]: definition for definition in crown_geometry}
    panel_tops = {panel[0]: panel[4] for panel in panels}
    owner_boundaries = []
    validated_crowns = set()
    stations, profiles = [], []
    for name, point in coordinates.items():
        if not name.startswith("DBC"):
            continue
        if point[2] not in soil_triangles:
            soil_triangles[point[2]] = ground_triangles(result, baseline, point[2])
            soil_quadrilaterals[point[2]] = ground_quadrilaterals(result, baseline, point[2])
        matches = []
        for identifier, inverse, displacement in soil_triangles[point[2]]:
            weights = inverse @ np.array([*point[:2], 1.0])
            if weights.min() >= -1e-8 and weights.max() <= 1 + 1e-8:
                matches.append((identifier, weights @ displacement))
        for identifier, origin, coefficients, scale, displacement in soil_quadrilaterals[point[2]]:
            weights = quadrilateral_weights(origin, coefficients, scale, point[:2])
            if weights is not None:
                matches.append((identifier, weights @ displacement))
        if not matches:
            raise ValueError(f"No active ground interpolation coverage at {name}")
        if any(
            not np.allclose(row[1], matches[0][1], atol=FIELD_ATOL_M, rtol=FIELD_RTOL)
            for row in matches
        ):
            raise ValueError(f"Discontinuous ground interpolation at {name}")
        stations.append(
            {
                "monitor": name,
                "xyz_m": point.tolist(),
                "u_m": matches[0][1].tolist(),
                "element_ids": [row[0] for row in matches],
            }
        )
    toe, wall_top = map(float, wall_range)
    if not np.isfinite([toe, wall_top]).all() or wall_top <= toe:
        raise ValueError("Invalid reviewed wall range")
    for name, point in coordinates.items():
        if not name.startswith("CX"):
            continue
        intervals = []
        for identifier, start, end, lower, upper, bottom_u, top_u in panels:
            direction = end - start
            length = np.linalg.norm(direction)
            if length <= GEOMETRY_ATOL_M:
                raise ValueError("Degenerate native wall span")
            fraction = np.dot(point[:2] - start, direction) / length**2
            distance = np.linalg.norm(point[:2] - start - fraction * direction)
            if distance > GEOMETRY_ATOL_M or not 0 <= fraction <= 1:
                continue
            if upper < toe or lower > wall_top:
                continue
            lower_u = (1 - fraction) * bottom_u[0] + fraction * bottom_u[1]
            upper_u = (1 - fraction) * top_u[0] + fraction * top_u[1]
            clipped_lower, clipped_upper = max(lower, toe), min(upper, wall_top)
            if clipped_upper <= clipped_lower:
                continue
            values = [
                lower_u + (level - lower) / (upper - lower) * (upper_u - lower_u)
                for level in (clipped_lower, clipped_upper)
            ]
            intervals.append((clipped_lower, clipped_upper, identifier, values))
        intervals.extend(crown_intervals(result, crown_geometry, point, (toe, wall_top)))
        intervals.sort(key=lambda row: (row[0], row[1], row[2]))
        for interval_index, first in enumerate(intervals):
            for second in intervals[interval_index + 1 :]:
                if second[0] >= first[1] - 1e-8:
                    break
                if (first[2] in crowns) != (second[2] in crowns):
                    raise ValueError(f"Overlapping native shell/crown owners at {name}")
        reached = toe
        at_elevation = {}
        if not intervals:
            raise ValueError(f"No native wall interpolation coverage at {name}")
        for lower, upper, identifier, values in intervals:
            if lower > reached + 1e-8:
                raise ValueError(f"Wall profile gap at {name}: {reached:g} to {lower:g} m")
            reached = max(reached, upper)
            for elevation, displacement in zip((lower, upper), values):
                physical_point = np.array([*point[:2], elevation])
                placement = (
                    crown_section_placement(crowns[identifier], physical_point)
                    if identifier in crowns
                    else np.zeros(3)
                )
                attachment_motion = displacement - placement
                for previous in at_elevation.get(elevation, []):
                    difference = (attachment_motion - previous[1]).tolist()
                    continuous = np.allclose(
                        attachment_motion, previous[1], rtol=FIELD_RTOL, atol=FIELD_ATOL_M
                    )
                    discontinuity = (
                        f"Discontinuous native wall profile at {name}: elements "
                        f"{previous[0]}/{identifier}, after native section placement "
                        f"difference_m={difference}"
                    )
                    mixed_owners = (identifier in crowns) != (previous[0] in crowns)
                    if mixed_owners:
                        crown_id = identifier if identifier in crowns else previous[0]
                        shell_id = previous[0] if identifier in crowns else identifier
                        boundary = crowns[crown_id]["profile_range_z_m"][0]
                        valid_boundary = not (
                            abs(elevation - boundary) > 1e-8
                            or abs(panel_tops[shell_id] - boundary) > 1e-8
                        )
                        if not continuous:
                            if not valid_boundary:
                                raise ValueError(discontinuity)
                            if crown_id not in validated_crowns:
                                try:
                                    validate_crown_attachments(
                                        result,
                                        baseline,
                                        panels,
                                        [
                                            row
                                            for row in crown_geometry
                                            if row["element"] == crown_id
                                        ],
                                    )
                                except ValueError as error:
                                    raise ValueError(discontinuity) from error
                                validated_crowns.add(crown_id)
                        owner_boundaries.append(
                            {
                                "monitor": name,
                                "xyz_m": physical_point.tolist(),
                                "element_ids": [previous[0], identifier],
                                "attachment_motion_difference_m": difference,
                            }
                        )
                    elif not continuous:
                        raise ValueError(discontinuity)
                at_elevation.setdefault(elevation, []).append((identifier, attachment_motion))
                profiles.append(
                    {
                        "monitor": name,
                        "xyz_m": [*point[:2].tolist(), elevation],
                        "u_m": displacement.tolist(),
                        "element_ids": [identifier],
                        "initial_section_placement_m": placement.tolist(),
                    }
                )
        if reached < wall_top - 1e-8:
            raise ValueError(f"Wall profile gap at {name}: {reached:g} to {wall_top:g} m")
    return {
        "stations": stations,
        "profiles": profiles,
        "wall_range_z_m": [toe, wall_top],
        "native_owner_boundaries": owner_boundaries,
        "max_settlement_mm": max(max(0.0, -row["u_m"][2]) * 1000 for row in stations),
        "max_disp_mm": max(float(np.linalg.norm(row["u_m"][:2])) * 1000 for row in profiles),
        "profile_extrema": "all Q4 vertical interval endpoints; horizontal norm is convex on each linear interval",
    }


def check_reported_monitors(candidate, independently_extracted):
    """Compare a report to trusted extraction, including locations and complete profiles."""
    for table in ("stations", "profiles"):
        expected = independently_extracted[table]
        actual = candidate.get(table)
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Missing or extra reported {table}")

        def key(row):
            return row["monitor"], *row["xyz_m"]

        try:
            pairs = zip(sorted(actual, key=key), sorted(expected, key=key))
            for reported, reference in pairs:
                point = np.asarray(reported["xyz_m"], dtype=float)
                if (
                    point.shape != (3,)
                    or not np.isfinite(point).all()
                    or reported["monitor"] != reference["monitor"]
                    or not np.allclose(point, reference["xyz_m"], rtol=0, atol=1e-8)
                ):
                    raise ValueError("Wrong reported monitor/location")
                values = np.asarray(reported["u_m"], dtype=float)
                if (
                    values.shape != (3,)
                    or not np.isfinite(values).all()
                    or not np.allclose(values, reference["u_m"], atol=1e-6, rtol=1e-6)
                ):
                    raise ValueError("Fabricated or inconsistent reported monitor displacement")
        except (KeyError, TypeError) as error:
            raise ValueError("Malformed reported monitor table") from error
    for quantity in ("max_settlement_mm", "max_disp_mm"):
        value = candidate.get(quantity)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or abs(value - independently_extracted[quantity])
            > max(0.30, 0.05 * abs(independently_extracted[quantity]))
        ):
            raise ValueError(f"Reported {quantity} disagrees with independent native extraction")
    return {"passed": True, "physical_replay_or_source_compliance": False}
