import json
from pathlib import Path

import numpy as np
import pytest

from tasks.engineering.inner_support_elevation_optimization.scripts import (
    native_model_contract as contract,
)


def geometry_fixture():
    outer = [[-60, -40], [60, -40], [60, 40], [40, 40], [40, 10], [20, 10], [20, 40], [-60, 40]]
    inner = [[-45, -25], [45, -25], [45, 5], [-45, 5]]
    family = {
        "outer_polygon": outer,
        "inner_polygon": inner,
        "domain": [-115, 115, -92.5, 92.5, -31.74, 6.5],
        "layers": [{"bottom": -31.74}],
        "engineering_choices": {
            "beam_section_m": {"width": 1, "height": 1.2},
            "crowns": {
                "outer": {
                    "axis_z_m": 6.5,
                    "shell_top_m": 6.5,
                    "section_m": {"width": 1, "height": 1},
                }
            },
        },
        "braces": [],
        "monitors": {
            f"{prefix}{number}": [
                -55 + 10 * number,
                -42 if prefix == "DBC" else -40,
                6.5 if prefix == "DBC" else 7,
            ]
            for prefix in ("DBC", "CX")
            for number in range(1, 11)
        },
    }
    nodes = {}
    for system, polygon, toe, top in (("outer", outer, -26.5, 6.5), ("inner", inner, -21.6, -10.1)):
        faces = []
        perimeter = list(zip(polygon, polygon[1:] + polygon[:1]))
        for start, end in perimeter:
            face = []
            for point, height in ((start, toe), (end, toe), (end, top), (start, top)):
                identifier = len(nodes) + 1
                nodes[identifier] = np.array([*point, height])
                face.append(identifier)
            faces.append(face)
        family[system + "_wall_faces"] = faces
        diagonal = [polygon[0], polygon[5] if system == "outer" else polygon[2]]
        family["braces"].append({"system": system, "segments_xy_m": [*perimeter, diagonal]})
    elements = {}
    for center, size in (([-59, -39], 1), ([-110, -85], 10)):
        connectivity = []
        for offset in ((0, 0, 0), (size, 0, 0), (0, size, 0), (0, 0, -1)):
            identifier = len(nodes) + 1
            nodes[identifier] = np.array([center[0] + offset[0], center[1] + offset[1], offset[2]])
            connectivity.append(identifier)
        elements[len(elements) + 1] = (1, connectivity)
    nodes[len(nodes) + 1] = np.array([-115, -92.5, -31.74])
    nodes[len(nodes) + 1] = np.array([115, 92.5, 6.5])
    return family, nodes, elements


@pytest.fixture
def declared_family(tmp_path):
    family, nodes, elements = geometry_fixture()
    assets = Path(contract.__file__).parents[1] / "assets"
    benchmark = json.loads((assets / "benchmark.json").read_text())
    family.update(
        format="support-native-family-1",
        units="m-kN-s",
        ground_elevation=6.5,
        source_hashes=json.loads((assets / "source_manifest.json").read_text())["sha256"],
    )
    family["layers"] = []
    elevation = 6.5
    materials = []
    for index, row in enumerate(benchmark["soils"], 1):
        family["layers"].append({"id": row[0], "top": elevation, "bottom": elevation - row[1]})
        elevation -= row[1]
        variables = dict(
            zip(("GEO_COHESION", "GEO_FRICTION_ANGLE", "K0_NC", "POISSON_RATIO"), row[3:7])
        )
        variables.update(
            YOUNG_MODULUS=(row[7] or 100) * 1000,
            POROSITY=0.3,
            DENSITY_WATER=1,
            DENSITY_SOLID=(row[2] / 9.81 - 0.3) / 0.7,
            K0_MAIN_DIRECTION=2,
        )
        materials.append(
            {
                "model_part_name": f"Support.Layer{index:02}",
                "properties_id": index,
                "Material": {
                    "constitutive_law": {"name": "GeoMohrCoulombLaw3D"},
                    "Variables": variables,
                    "Tables": {},
                },
            }
        )
    (tmp_path / "family.json").write_text(json.dumps(family))
    (tmp_path / "Materials.json").write_text(json.dumps({"properties": materials}))
    lines = [
        "Begin Nodes",
        *(f"{identifier} " + " ".join(map(str, point)) for identifier, point in nodes.items()),
        "End Nodes",
        "Begin Elements UPwSmallStrainElement3D4N",
        *(
            f"{identifier} {property_id} " + " ".join(map(str, connectivity))
            for identifier, (property_id, connectivity) in elements.items()
        ),
        "End Elements",
    ]
    (tmp_path / "model.mdpa").write_text("\n".join(lines))
    return tmp_path


def test_preflight_binds_inputs_without_claiming_physical_completion(declared_family):
    result = contract.validate_model(declared_family)
    assert result["contract"] == "support-deterministic-20261006"
    assert set(result["input_hashes"]) == {"family.json", "model.mdpa", "Materials.json"}
    assert not result["native_compliance_completed"] and not result["full_task_acceptance"]
    assert result["geometry"]["near_median_m"] < result["geometry"]["far_median_m"]


@pytest.mark.parametrize("elevation", [-26.5, 0, 6.5, 7])
def test_cx_marker_elevation_does_not_clip_the_required_profile(elevation):
    family, nodes, elements = geometry_fixture()
    for name, point in family["monitors"].items():
        if name.startswith("CX"):
            point[2] = elevation
    contract.check_geometry(family, nodes, elements)


@pytest.mark.parametrize(
    "damage,match",
    [
        ("wall_gap", "coverage"),
        ("wall_overlap", "overlap"),
        ("duplicate_monitor", "1m apart"),
        ("inside_dbc", "DBC"),
        ("off_wall", "CX"),
        ("domain", "bounds"),
        ("uniform", "refinement"),
        ("wrong_crown", "installation level"),
    ],
)
def test_geometric_negatives(damage, match):
    family, nodes, elements = geometry_fixture()
    if damage == "wall_gap":
        family["outer_wall_faces"].pop()
    elif damage == "wall_overlap":
        family["outer_wall_faces"].append(family["outer_wall_faces"][0])
    elif damage == "duplicate_monitor":
        family["monitors"]["DBC2"] = family["monitors"]["DBC1"]
    elif damage == "inside_dbc":
        family["monitors"]["DBC1"] = [0, 0, 6.5]
    elif damage == "off_wall":
        family["monitors"]["CX1"][1] -= 1
    elif damage == "domain":
        family["domain"][0] -= 1
    elif damage == "uniform":
        first = elements[1][1]
        second = elements[2][1]
        for original, changed in zip(first, second):
            nodes[changed] = nodes[original] + [-50, -40, 0]
    else:
        family["engineering_choices"]["crowns"]["outer"]["axis_z_m"] = 5
    with pytest.raises(ValueError, match=match):
        contract.check_geometry(family, nodes, elements)


@pytest.mark.parametrize("size", [(0.8, 1.0), (1.0, 1.2)])
def test_supported_sections_and_hydraulic_prose_do_not_invent_alternative_intent(size):
    family, _, _ = geometry_fixture()
    family["engineering_choices"].update(
        beam_section_m=dict(zip(("width", "height"), size)),
        hydraulic_model="Prescribed uniform hydrostatic pore pressure",
        unresolved=["No transient flow or per-member section is requested."],
    )
    contract.check_braces(family)


@pytest.mark.parametrize("damage", ["disconnected", "missing_waler", "missing_diagonal", "outside"])
def test_brace_topology_negatives(damage):
    family, _, _ = geometry_fixture()
    graph = family["braces"][0]["segments_xy_m"]
    if damage == "disconnected":
        graph.append([[0, 0], [1, 1]])
    elif damage == "missing_waler":
        graph[:] = graph[-1:]
    elif damage == "missing_diagonal":
        graph.pop()
    else:
        graph.append([family["outer_polygon"][0], [500, 500]])
    with pytest.raises(ValueError):
        contract.check_braces(family)


@pytest.mark.parametrize("damage", ["material", "tolerance", "layer", "duplicate_node"])
def test_invalid_native_inputs_cannot_pass_preflight(declared_family, damage):
    filename = "Materials.json" if damage == "material" else "family.json"
    path = declared_family / filename
    value = json.loads(path.read_text())
    if damage == "material":
        value["properties"][0]["Material"]["Variables"]["GEO_COHESION"] += 1
    elif damage == "tolerance":
        value["engineering_choices"]["solver_controls"] = {"residual_relative_tolerance": 0.1}
    elif damage == "layer":
        value["layers"].pop()
    else:
        mesh = declared_family / "model.mdpa"
        mesh.write_text(mesh.read_text().replace("End Nodes", "1 0 0 0\nEnd Nodes"))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        contract.validate_model(declared_family)


def test_polygon_translation_rotation_and_self_intersection():
    family, _, _ = geometry_fixture()
    polygon = np.array(family["outer_polygon"])
    baseline = contract.polygon_geometry(polygon)[2:]
    angle = 0.37
    frame = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    assert contract.polygon_geometry(polygon @ frame + 123)[2:] == pytest.approx(baseline)
    with pytest.raises(ValueError, match="intersecting"):
        contract.polygon_geometry([[0, 0], [1, 1], [0, 1], [1, 0]])
