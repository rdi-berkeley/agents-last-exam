"""Read raw native Kratos fields and precision exports without extracting monitors."""

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


COMPONENTS = ("DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z")
ROTATIONS = ("ROTATION_X", "ROTATION_Y", "ROTATION_Z")


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
    """Read bytes matching the caller-supplied digest; this does not certify provenance."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000_000:
        raise ValueError(f"Unsafe, missing or oversized native artifact: {path.name}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"Native artifact digest mismatch: {path.name}")
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
    """Restore captured native translations that round exactly to native MDPA."""
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
    """Read a pinned Stage1 displacement array with the requested soil node set."""
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
