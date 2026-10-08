"""Read a bounded declarative native Kratos family; never execute candidate code."""

import hashlib
import json
import math
from pathlib import Path


def contains(point, polygon):
    inside = False
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        if (first[1] > point[1]) != (second[1] > point[1]):
            cross = first[0] + (point[1] - first[1]) * (second[0] - first[0]) / (
                second[1] - first[1]
            )
            if point[0] < cross:
                inside = not inside
    return inside


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def effective_layers(family):
    layers = [dict(layer) for layer in family["layers"]]
    extension = family["engineering_choices"].get("deep_rock_continuation")
    if extension is None:
        if family["domain"][4] < layers[-1]["bottom"] - 1e-8:
            raise ValueError("Domain below source strata needs an explicit supported extension")
        return layers
    if (
        set(extension) != {"source_layer_id", "bottom_m", "rationale"}
        or extension["source_layer_id"] != layers[-1]["id"]
        or not isinstance(extension["rationale"], str)
        or not extension["rationale"].strip()
        or isinstance(extension["bottom_m"], bool)
        or not isinstance(extension["bottom_m"], (int, float))
        or not math.isfinite(extension["bottom_m"])
        or extension["bottom_m"] >= layers[-1]["bottom"]
        or not math.isclose(extension["bottom_m"], family["domain"][4], abs_tol=1e-8)
    ):
        raise ValueError("Invalid declared continuation of the deepest source rock")
    layers[-1]["bottom"] = extension["bottom_m"]
    return layers


def read_family(directory):
    root = Path(directory).resolve()
    paths = {name: root / name for name in ("family.json", "model.mdpa", "Materials.json")}
    for path in paths.values():
        if path.is_symlink() or path.resolve().parent != root or not path.is_file():
            raise ValueError(f"Unsafe or missing native input: {path.name}")
        if path.stat().st_size > 100_000_000:
            raise ValueError(f"Native input exceeds preparation limit: {path.name}")
    family = read_json(paths["family.json"])
    if set(family) != {
        "format",
        "units",
        "ground_elevation",
        "domain",
        "layers",
        "outer_polygon",
        "inner_polygon",
        "outer_wall_faces",
        "inner_wall_faces",
        "braces",
        "monitors",
        "source_hashes",
        "engineering_choices",
    }:
        raise ValueError("Unsupported family keys; executable processes are not accepted")
    if family["format"] != "support-native-family-1" or family["units"] != "m-kN-s":
        raise ValueError("Unsupported family format or units")
    if family["ground_elevation"] != 6.5:
        raise ValueError("This adapter requires the published absolute elevation datum")
    benchmark = read_json(Path(__file__).parents[1] / "assets/benchmark.json")
    manifest = read_json(Path(__file__).parents[1] / "assets/source_manifest.json")
    if family["source_hashes"] != manifest["sha256"]:
        raise ValueError("Family source provenance differs from preserved public inputs")
    layers = family["layers"]
    if len(layers) != 14 or [row["id"] for row in layers] != [row[0] for row in benchmark["soils"]]:
        raise ValueError("All fourteen source strata are required in order")
    elevation = family["ground_elevation"]
    for layer, source in zip(layers, benchmark["soils"]):
        if not math.isclose(layer["top"], elevation, abs_tol=1e-8):
            raise ValueError(f"Wrong layer top: {layer['id']}")
        elevation -= source[1]
        if not math.isclose(layer["bottom"], elevation, abs_tol=1e-8):
            raise ValueError(f"Wrong layer thickness: {layer['id']}")
    effective_layers(family)
    controls = family["engineering_choices"].get("solver_controls", {})
    allowed_controls = {
        "residual_relative_tolerance",
        "residual_absolute_tolerance",
        "displacement_relative_tolerance",
        "displacement_absolute_tolerance",
    }
    if not set(controls).issubset(allowed_controls) or any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value <= 0
        for value in controls.values()
    ):
        raise ValueError("Unsupported or nonpositive declared solver tolerance")
    material_document = read_json(paths["Materials.json"])
    if set(material_document) != {"properties"}:
        raise ValueError("Unsupported native material document")
    allowed_variables = {
        "GEO_DRAINAGE_TYPE",
        "YOUNG_MODULUS",
        "POISSON_RATIO",
        "DENSITY_SOLID",
        "DENSITY_WATER",
        "POROSITY",
        "BULK_MODULUS_SOLID",
        "BULK_MODULUS_FLUID",
        "DYNAMIC_VISCOSITY",
        "BIOT_COEFFICIENT",
        "RETENTION_LAW",
        "SATURATED_SATURATION",
        "GEO_COHESION",
        "GEO_FRICTION_ANGLE",
        "GEO_DILATANCY_ANGLE",
        "GEO_ENABLE_TENSION_CUT_OFF",
        "GEO_TENSILE_STRENGTH",
        "PERMEABILITY_XX",
        "PERMEABILITY_YY",
        "PERMEABILITY_ZZ",
        "PERMEABILITY_XY",
        "PERMEABILITY_YZ",
        "PERMEABILITY_ZX",
        "K0_MAIN_DIRECTION",
        "K0_NC",
    }
    if len(material_document["properties"]) != 14:
        raise ValueError("Expected fourteen effective soil materials")
    property_ids = set()
    model_parts = set()
    for material in material_document["properties"]:
        if set(material) != {"model_part_name", "properties_id", "Material"}:
            raise ValueError("Unsupported material property fields")
        values = material["Material"]
        if set(values) != {"constitutive_law", "Variables", "Tables"} or values["Tables"]:
            raise ValueError("Only constant allowlisted native properties are supported")
        if values["constitutive_law"] != {"name": "GeoMohrCoulombLaw3D"}:
            raise ValueError("All soil must use native GeoMohrCoulombLaw3D")
        if not set(values["Variables"]).issubset(allowed_variables):
            raise ValueError("Unsupported material variable")
        if material["model_part_name"] not in {
            f"Support.Layer{index:02}" for index in range(1, 15)
        }:
            raise ValueError("Unsupported model part binding")
        if material["properties_id"] in property_ids or material["model_part_name"] in model_parts:
            raise ValueError("Repeated native property or model part assignment")
        property_ids.add(material["properties_id"])
        model_parts.add(material["model_part_name"])
    for line in paths["model.mdpa"].read_text().splitlines():
        if line.startswith("Begin ") and line.split()[1] not in {
            "ModelPartData",
            "Properties",
            "Nodes",
            "Elements",
            "SubModelPart",
            "SubModelPartData",
            "SubModelPartNodes",
            "SubModelPartElements",
            "SubModelPartConditions",
        }:
            raise ValueError("Unsupported native mesh block")
        if line.startswith("Begin Elements ") and line.split()[2] not in {
            "UPwSmallStrainElement3D4N",
            "UPwSmallStrainElement3D8N",
        }:
            raise ValueError("Unsupported soil element type")
    family["input_hashes"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    }
    return family, benchmark


def native_linear_settings(backend):
    if backend == "sparse-lu":
        return {"solver_type": "LinearSolversApplication.sparse_lu"}
    if backend == "amgcl-gmres":
        return {
            "solver_type": "amgcl",
            "smoother_type": "ilut",
            "krylov_type": "gmres",
            "coarsening_type": "aggregation",
            "tolerance": 1e-10,
            "max_iteration": 1000,
            "block_size": 1,
            "use_block_matrices_if_possible": False,
            "scaling": True,
            "verbosity": 1,
            "gmres_krylov_space_dimension": 100,
        }
    raise ValueError(f"Unsupported installed native backend: {backend}")


def load_native(directory, backend="sparse-lu"):
    import KratosMultiphysics as Kratos
    import KratosMultiphysics.GeoMechanicsApplication as Geo
    from KratosMultiphysics.GeoMechanicsApplication.geomechanics_U_Pw_solver import UPwSolver

    family, benchmark = read_family(directory)
    native_layers = effective_layers(family)
    model = Kratos.Model()
    solver = UPwSolver(
        model,
        Kratos.Parameters(
            json.dumps(
                {
                    "model_part_name": "Support",
                    "domain_size": 3,
                    "rotation_dofs": True,
                    "scheme_type": "Backward_Euler",
                    "solution_type": "quasi_static",
                    "strategy_type": "newton_raphson",
                    "linear_solver_settings": native_linear_settings(backend),
                    "convergence_criterion": "and_criterion",
                    "residual_relative_tolerance": 1e-9,
                    "residual_absolute_tolerance": 1e-7,
                    "displacement_relative_tolerance": 1e-8,
                    "displacement_absolute_tolerance": 1e-10,
                    "max_iterations": 40,
                    "compute_reactions": True,
                    "reform_dofs_at_each_step": True,
                    "move_mesh_flag": False,
                    "reset_totals": False,
                    "echo_level": 0,
                }
            )
        ),
    )
    for name, value in family["engineering_choices"].get("solver_controls", {}).items():
        solver.settings[name].SetDouble(value)
    solver.AddVariables()
    part = model["Support"]
    part.SetBufferSize(2)
    Kratos.ModelPartIO(str(Path(directory) / "model")).ReadModelPart(part)
    Kratos.ReadMaterialsUtility(
        Kratos.Parameters(
            json.dumps(
                {"Parameters": {"materials_filename": str(Path(directory) / "Materials.json")}}
            )
        ),
        model,
    )
    active = part.CreateSubModelPart("Active")
    soil = part.CreateSubModelPart("Soil")
    active.AddNodes([node.Id for node in part.Nodes])
    active.AddElements([element.Id for element in part.Elements])
    soil.AddElements([element.Id for element in part.Elements])
    soil.AddNodes([node.Id for node in part.Nodes])
    solver.computing_model_part_name = "Active"
    part.ProcessInfo[Kratos.DELTA_TIME] = 1.0
    part.ProcessInfo[Geo.DT_PRESSURE_COEFFICIENT] = 1.0
    part.ProcessInfo[Geo.VELOCITY_COEFFICIENT] = 1.0
    used = set()
    for element in part.Elements:
        element.Set(Kratos.ACTIVE, True)
        geometry = element.GetGeometry()
        if geometry.DomainSize() <= 0:
            raise ValueError(f"Invalid soil volume: {element.Id}")
        depth = sum(node.Z0 for node in geometry) / len(geometry)
        layer_index = next(
            (
                index
                for index, layer in enumerate(native_layers)
                if layer["bottom"] - 1e-8 <= depth <= layer["top"] + 1e-8
            ),
            None,
        )
        if layer_index is None:
            raise ValueError("Element outside source stratigraphy")
        layer = native_layers[layer_index]
        if any(
            node.Z0 < layer["bottom"] - 1e-8 or node.Z0 > layer["top"] + 1e-8 for node in geometry
        ):
            raise ValueError("Soil element crosses a source layer boundary")
        source = benchmark["soils"][layer_index]
        properties = element.Properties
        variables = {
            name: Kratos.KratosGlobals.GetVariable(name)
            for name in (
                "GEO_COHESION",
                "GEO_FRICTION_ANGLE",
                "K0_NC",
                "POISSON_RATIO",
                "YOUNG_MODULUS",
                "POROSITY",
                "DENSITY_SOLID",
                "DENSITY_WATER",
            )
        }
        for name, expected in zip(
            ("GEO_COHESION", "GEO_FRICTION_ANGLE", "K0_NC", "POISSON_RATIO"), source[3:7]
        ):
            if not math.isclose(properties[variables[name]], expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Wrong effective {name} in layer {source[0]}")
        modulus = properties[variables["YOUNG_MODULUS"]]
        if source[7] is not None and not math.isclose(modulus, source[7] * 1000, rel_tol=1e-9):
            raise ValueError("Wrong effective soil stiffness")
        if source[7] is None and modulus <= max(row[7] or 0 for row in benchmark["soils"]) * 1000:
            raise ValueError("Rock must be stiffer than the supplied soil strata")
        porosity = properties[variables["POROSITY"]]
        weight = 9.81 * (
            (1 - porosity) * properties[variables["DENSITY_SOLID"]]
            + porosity * properties[variables["DENSITY_WATER"]]
        )
        if not math.isclose(weight, source[2], rel_tol=1e-9):
            raise ValueError("Wrong effective bulk unit weight")
        if properties[Kratos.KratosGlobals.GetVariable("K0_MAIN_DIRECTION")] != 2:
            raise ValueError("K0 must use the vertical z direction")
        used.add(layer_index)
    if used != set(range(14)):
        raise ValueError("Native mesh does not actually contain all fourteen strata")
    solver.AddDofs()
    return model, solver, family, benchmark
