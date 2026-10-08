"""Check fixed source facts on evaluator-owned native state without solving FE."""

import argparse
import json
import math
import resource
import time
from pathlib import Path

from native_family_runner import FamilyRunner
from native_input_adapter import effective_layers
from native_member_sections import audit_support_sections
from native_replay_case import verify_review


def close(actual, expected, description):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError(f"{description}: actual={actual}, expected={expected}")


def audit(runner):
    Kratos = runner.Kratos
    active_ids = {element.Id for element in runner.active.Elements}
    soil_ids = {element.Id for element in runner.soil.Elements}
    if active_ids & soil_ids != soil_ids - runner.excavated:
        raise ValueError("Native active soil differs from committed excavation state")
    layer_counts = [0] * 14
    native_layers = effective_layers(runner.family)
    active_soil_nodes = set()
    property_layers = {}
    total_volume = 0.0
    for element in runner.soil.Elements:
        geometry = element.GetGeometry()
        volume = geometry.DomainSize()
        if volume <= 0:
            raise ValueError("Nonpositive native soil volume")
        total_volume += volume
        elevations = [node.Z0 for node in geometry]
        center = sum(elevations) / len(elevations)
        layer_index = next(
            (
                index
                for index, layer in enumerate(native_layers)
                if layer["bottom"] - 1e-8 <= center <= layer["top"] + 1e-8
            ),
            None,
        )
        if layer_index is None:
            raise ValueError("Native soil lies outside listed source strata")
        layer = native_layers[layer_index]
        if min(elevations) < layer["bottom"] - 1e-8 or max(elevations) > layer["top"] + 1e-8:
            raise ValueError("Native element crosses a source stratum")
        layer_counts[layer_index] += 1
        properties = element.Properties
        binding = (properties.Id, layer_index)
        if binding not in property_layers:
            source = runner.benchmark["soils"][layer_index]
            variables = {
                name: Kratos.KratosGlobals.GetVariable(name)
                for name in (
                    "GEO_COHESION",
                    "GEO_FRICTION_ANGLE",
                    "K0_NC",
                    "K0_MAIN_DIRECTION",
                    "POROSITY",
                    "DENSITY_SOLID",
                    "DENSITY_WATER",
                )
            }
            for name, expected in zip(("GEO_COHESION", "GEO_FRICTION_ANGLE", "K0_NC"), source[3:6]):
                close(properties[variables[name]], expected, f"Source {name} in {source[0]}")
            close(properties[Kratos.POISSON_RATIO], source[6], "Source soil Poisson ratio")
            modulus = properties[Kratos.YOUNG_MODULUS]
            if source[7] is not None:
                close(modulus, source[7] * 1000, "Source soil modulus")
            elif modulus <= max(row[7] or 0 for row in runner.benchmark["soils"]) * 1000:
                raise ValueError("Declared rock stiffness is not above supplied soil stiffness")
            porosity = properties[variables["POROSITY"]]
            weight = 9.81 * (
                (1 - porosity) * properties[variables["DENSITY_SOLID"]]
                + porosity * properties[variables["DENSITY_WATER"]]
            )
            close(weight, source[2], "Source bulk soil weight")
            close(properties[variables["K0_MAIN_DIRECTION"]], 2, "Native vertical K0 axis")
            property_layers[binding] = True
        if element.Is(Kratos.ACTIVE) != (element.Id in active_ids):
            raise ValueError("Native soil active flag differs from assembled membership")
        if element.Id in active_ids:
            active_soil_nodes.update(node.Id for node in geometry)
    if any(count == 0 for count in layer_counts):
        raise ValueError("Native state is missing a source stratum")
    minimum_x, maximum_x, minimum_y, maximum_y, bottom, top = runner.family["domain"]
    close(
        total_volume,
        (maximum_x - minimum_x) * (maximum_y - minimum_y) * (top - bottom),
        "Native initial soil/domain volume",
    )
    components = (Kratos.DISPLACEMENT_X, Kratos.DISPLACEMENT_Y, Kratos.DISPLACEMENT_Z)
    checked_fixities = 0
    for identifier in active_soil_nodes:
        node = runner.part.GetNode(identifier)
        on_bottom = abs(node.Z0 - bottom) < 1e-7
        expected = (
            on_bottom or min(abs(node.X0 - minimum_x), abs(node.X0 - maximum_x)) < 1e-7,
            on_bottom or min(abs(node.Y0 - minimum_y), abs(node.Y0 - maximum_y)) < 1e-7,
            on_bottom,
        )
        for variable, fixed in zip(components, expected):
            if node.IsFixed(variable) != fixed:
                raise ValueError(f"Source displacement boundary differs at node {identifier}")
            checked_fixities += 1
    for node in runner.part.Nodes:
        for value, expected in zip(
            node.GetSolutionStepValue(Kratos.VOLUME_ACCELERATION), (0, 0, -9.81)
        ):
            close(value, expected, "Native gravity acceleration")
        if not node.IsFixed(Kratos.WATER_PRESSURE):
            raise ValueError("Native hydraulic fixity differs from declared hydrostatic model")
        close(
            node.GetSolutionStepValue(Kratos.WATER_PRESSURE),
            -9.81 * max(runner.water_elevation - node.Z0, 0),
            "Native pressure differs from declared hydrostatic water level",
        )
    structural_properties = set()
    structural_elements = 0
    for element in runner.active.Elements:
        if element.Id in soil_ids:
            continue
        structural_elements += 1
        properties = element.Properties
        if properties.Id in structural_properties:
            continue
        structural_properties.add(properties.Id)
        close(properties[Kratos.YOUNG_MODULUS], 30e6, "Source C35 stiffness")
        close(properties[Kratos.POISSON_RATIO], 0.2, "Source C35 Poisson ratio")
        close(properties[Kratos.DENSITY] * 9.81, 25, "Source structural selfweight")
    wall_ranges = {}
    for label, property_id in runner.receipt.get("wall_properties", {}).items():
        inner = label == "stage_8_inner_wall"
        close(
            runner.part.Properties[property_id][Kratos.THICKNESS],
            0.62 if inner else 1.0,
            "Source equivalent wall thickness",
        )
        elevations = [runner.part.GetNode(master).Z0 for master in runner.wall_nodes[label]]
        close(min(elevations), -21.6 if inner else -26.5, "Source wall toe")
        wall_ranges[label] = [min(elevations), max(elevations)]
    return {
        "source_fact_checks_passed": True,
        "support_sections": audit_support_sections(runner),
        "soil_elements_by_source_layer": layer_counts,
        "deep_rock_continuation": runner.family["engineering_choices"].get(
            "deep_rock_continuation"
        ),
        "initial_native_soil_volume_m3": total_volume,
        "native_excavated_elements": len(runner.excavated),
        "checked_active_soil_fixities": checked_fixities,
        "checked_native_gravity_and_pressure_nodes": runner.part.NumberOfNodes(),
        "structural_elements": structural_elements,
        "structural_property_ids": sorted(structural_properties),
        "native_wall_reference_ranges_m": wall_ranges,
        "hydraulics": "Checks declared uniform hydrostatic choice, not a public prescription",
        "source_compliance_accepted": False,
        "remaining_review": [
            "full construction and independent case histories",
            "source registration and undimensioned-member engineering adequacy",
            "mesh/domain/hydraulic adequacy and full physical monitor extraction",
        ],
    }


def negative_controls(runner):
    Kratos = runner.Kratos
    soil_property = next(iter(runner.soil.Elements)).Properties
    wall_property = runner.part.Properties[runner.receipt["wall_properties"]["stage_2_outer_wall"]]
    results = []
    for name, properties, variable, changed, message in (
        (
            "wrong_cohesion",
            soil_property,
            Kratos.KratosGlobals.GetVariable("GEO_COHESION"),
            -1.0,
            "Source GEO_COHESION",
        ),
        ("zero_wall_weight", wall_property, Kratos.DENSITY, 0.0, "structural selfweight"),
        ("wrong_wall_thickness", wall_property, Kratos.THICKNESS, 0.5, "wall thickness"),
    ):
        original = properties[variable]
        error = None
        try:
            properties[variable] = changed
            audit(runner)
        except ValueError as exception:
            error = str(exception)
        finally:
            properties[variable] = original
        if error is None or message not in error:
            raise AssertionError(f"Native mutation not rejected as intended: {name}: {error}")
        results.append({"mutation": name, "rejected": True, "reason": error})
    bottom = runner.family["domain"][4]
    node = next(node for node in runner.soil.Nodes if abs(node.Z0 - bottom) < 1e-7)
    error = None
    try:
        node.Free(Kratos.DISPLACEMENT_Z)
        audit(runner)
    except ValueError as exception:
        error = str(exception)
    finally:
        node.Fix(Kratos.DISPLACEMENT_Z)
    if error is None or "displacement boundary" not in error:
        raise AssertionError(f"Freed native bottom was not rejected: {error}")
    results.append({"mutation": "free_bottom", "rejected": True, "reason": error})
    if runner.excavated:
        identifier = min(runner.excavated)
        error = None
        try:
            runner.active.AddElement(runner.part.GetElement(identifier), 0)
            audit(runner)
        except ValueError as exception:
            error = str(exception)
        finally:
            runner.active.RemoveElement(identifier)
        if error is None or "active soil" not in error:
            raise AssertionError(f"Reactivated excavated native soil was not rejected: {error}")
        results.append({"mutation": "reactivated_excavation", "rejected": True, "reason": error})
    return results


def main(family, checkpoint, review, output, controls=False):
    started = time.monotonic()
    output = Path(output)
    if output.exists():
        raise ValueError("Audit output must be a fresh evaluator-owned directory")
    review_data = json.loads(Path(review).read_text())
    verify_review(family, review_data)
    runner = FamilyRunner.from_checkpoint(family, output, checkpoint)
    record = {
        "native_solve_performed": False,
        "input_hashes": runner.family["input_hashes"],
        "checkpoint": runner.receipt["restarted_from"],
        "full_task_acceptance": False,
    }
    try:
        record["audit"] = audit(runner)
        if controls:
            record["negative_controls"] = negative_controls(runner)
        record["passed"] = True
    except Exception as error:
        record.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_s"] = time.monotonic() - started
        record["peak_rss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (output / "source-audit.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    parser.add_argument("checkpoint")
    parser.add_argument("review")
    parser.add_argument("output")
    parser.add_argument("--controls", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.family,
        arguments.checkpoint,
        arguments.review,
        arguments.output,
        arguments.controls,
    )
