"""Native source-strata initialization and persistent construction staging."""

import argparse
import csv
import hashlib
import json
import math
import resource
import time
import traceback
from pathlib import Path

import numpy as np

from native_input_adapter import effective_layers, load_native, native_linear_settings, read_family
from native_member_sections import audit_support_sections, member_sections, section_properties


def compare_native_fields(actual_fields, reference_fields):
    result = {
        "passed": True,
        "fields": {},
        "displacement_accuracy_basis": (
            "1 micrometre absolute diagnostic budget, 300 times smaller than the "
            "original task's 0.30 mm absolute response tolerance; not a source-prescribed gate"
        ),
    }
    if set(actual_fields) != set(reference_fields):
        return {"passed": False, "fields": {}, "error": "Native field set differs"}
    for name, actual in actual_fields.items():
        expected = reference_fields[name]
        absolute = 1e-4 if name.endswith("kPa") else 1e-8
        if name == "displacement_m":
            absolute = 1e-6
        if name.endswith("ids"):
            passed = np.array_equal(actual, expected)
            difference = 0.0 if passed else None
        else:
            passed = actual.shape == expected.shape and np.allclose(
                actual, expected, rtol=1e-6, atol=absolute
            )
            difference = (
                float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else None
            )
        result["fields"][name] = {
            "passed": bool(passed),
            "max_abs_difference": difference,
            "absolute_tolerance": absolute,
            "relative_tolerance": 1e-6,
        }
        result["passed"] &= bool(passed)
    return result


class FamilyRunner:
    def __init__(
        self,
        directory,
        output,
        backend="sparse-lu",
        reference=None,
        max_iterations=40,
        position=0,
        capture_matrix=False,
        write_checkpoints=False,
    ):
        import KratosMultiphysics as Kratos
        import KratosMultiphysics.GeoMechanicsApplication as Geo

        self.Kratos = Kratos
        self.Geo = Geo
        self.directory = directory
        if position not in range(5):
            raise ValueError("Support position must be 0..4 m")
        self.position = position
        self.capture_matrix = capture_matrix
        self.write_checkpoints = write_checkpoints
        self.completed_labels = set()
        self.crown_geometry = []
        self.model, self.solver, self.family, self.benchmark = load_native(directory, backend)
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or not 1 <= max_iterations <= 1000
        ):
            raise ValueError("Native iteration budget must be an integer within 1..1000")
        self.solver.settings["max_iterations"].SetInt(max_iterations)
        self.solver.settings["echo_level"].SetInt(1)
        self.part = self.model["Support"]
        self.active = self.part.GetSubModelPart("Active")
        self.soil = self.part.GetSubModelPart("Soil")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.reference = Path(reference) if reference else None
        self.comparison_labels = {
            "stage_1_k0_equilibrium_hold",
            "stage_2_outer_wall",
            "stage_3_outer_support_1",
            "stage_4_dewater_1",
        }
        self.receipt = {
            "input_hashes": self.family["input_hashes"],
            "stages": [],
            "scope": self.family["engineering_choices"].get("scope", "submitted native family"),
            "full_case_complete": False,
            "native_loader_passed": True,
            "soil_elements": self.soil.NumberOfElements(),
            "nodes": self.part.NumberOfNodes(),
            "source_layers": 14,
            "linear_solver_settings": native_linear_settings(backend),
            "max_nonlinear_iterations": max_iterations,
            "state_comparisons": [],
            "case": f"pos_{position}m",
            "independent_initial_history": True,
        }
        if self.reference is not None:
            reference_bytes = (self.reference / "receipt.json").read_bytes()
            reference_receipt = json.loads(reference_bytes)
            completed_labels = {row["label"] for row in reference_receipt["stages"]}
            if not self.comparison_labels.issubset(completed_labels):
                raise ValueError("State comparison requires completed reference checkpoints")
            if reference_receipt["input_hashes"] != self.family["input_hashes"]:
                raise ValueError("Backend comparison must use identical native input bytes")
            self.receipt["reference_receipt_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
        import os

        self.receipt["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        self.started = time.monotonic()
        self.failed = False
        self.initialized = False
        self.connections = {}
        self.wall_nodes = {}
        self.baseline = {}
        self.boundary_dofs = set()
        self.water_elevation = 6.5
        self.excavated = set()
        self.receipt["engineering_choices"] = self.family["engineering_choices"]
        self.receipt["excavation_events"] = []
        self.receipt["support_installations"] = []
        self.apply_boundaries()
        self.write_receipt()

    @classmethod
    def from_checkpoint(cls, directory, output, checkpoint, max_iterations=200, position=None):
        import os

        import KratosMultiphysics as Kratos
        import KratosMultiphysics.GeoMechanicsApplication as Geo
        from KratosMultiphysics.GeoMechanicsApplication.geomechanics_U_Pw_solver import UPwSolver

        class RestartSolver(UPwSolver):
            def _FillBuffer(self):
                if not self.main_model_part.ProcessInfo[Kratos.IS_RESTARTED]:
                    super()._FillBuffer()

            def _BaseConstructScheme(self):
                scheme = super()._BaseConstructScheme()
                if self.main_model_part.ProcessInfo[Kratos.IS_RESTARTED]:
                    empty_model = Kratos.Model()
                    empty_part = empty_model.CreateModelPart("RestoredEntitiesAlreadyInitialized")
                    scheme.InitializeElements(empty_part)
                    scheme.InitializeConditions(empty_part)
                return scheme

        checkpoint = Path(checkpoint)
        metadata = json.loads(checkpoint.with_suffix(".json").read_text())
        native_file = checkpoint.with_suffix(".rest")
        if hashlib.sha256(native_file.read_bytes()).hexdigest() != metadata["native_state_sha256"]:
            raise ValueError("Native checkpoint digest mismatch")
        family, benchmark = read_family(directory)
        if family["input_hashes"] != metadata["input_hashes"]:
            raise ValueError("Checkpoint belongs to different frozen inputs")
        old_position = int(metadata["receipt"]["case"][4])
        position = old_position if position is None else position
        if position not in range(5) or not 1 <= max_iterations <= 1000:
            raise ValueError("Invalid restart position or iteration budget")
        if position != old_position and int(metadata["label"].split("_")[1]) >= 8:
            raise ValueError("Case alternatives must branch before inner-system installation")
        runner = cls.__new__(cls)
        runner.Kratos, runner.Geo = Kratos, Geo
        runner.directory, runner.output = directory, Path(output)
        runner.output.mkdir(parents=True, exist_ok=True)
        runner.family, runner.benchmark = family, benchmark
        runner.model = Kratos.Model()
        serializer = Kratos.FileSerializer(str(checkpoint))
        serializer.Load("Model", runner.model)
        settings = Kratos.Parameters(json.dumps(metadata["solver_settings"]))
        settings["model_import_settings"]["input_type"].SetString("rest")
        settings["max_iterations"].SetInt(max_iterations)
        runner.solver = RestartSolver(runner.model, settings)
        runner.solver.computing_model_part_name = "Active"
        runner.part = runner.model["Support"]
        runner.active = runner.part.GetSubModelPart("Active")
        runner.soil = runner.part.GetSubModelPart("Soil")
        runner.position, runner.capture_matrix, runner.write_checkpoints = position, False, True
        runner.reference, runner.comparison_labels = None, set()
        runner.receipt = metadata["receipt"]
        runner.receipt.update(
            case=f"pos_{position}m",
            max_nonlinear_iterations=max_iterations,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            independent_initial_history=False,
            restarted_from={
                "label": metadata["label"],
                "native_state_sha256": metadata["native_state_sha256"],
                "source_case": f"pos_{old_position}m",
                "prefix_elapsed_s": metadata["receipt"]["elapsed_s"],
                "validation_required": True,
            },
        )
        runner.receipt.pop("error", None)
        runner.receipt.pop("requested_scope_completed", None)
        runner.receipt.pop("failed_iterations", None)
        runner.completed_labels = {row["label"] for row in runner.receipt["stages"]}
        runner.started = time.monotonic()
        runner.failed, runner.initialized = False, False
        runner.connections = {}
        runner.wall_nodes = {
            name: {
                int(master): runner.part.GetNode(identifier) for master, identifier in nodes.items()
            }
            for name, nodes in metadata["wall_nodes"].items()
        }
        runner.baseline = {
            int(identifier): value for identifier, value in metadata["baseline"].items()
        }
        runner.boundary_dofs = {tuple(row) for row in metadata["boundary_dofs"]}
        runner.water_elevation = metadata["water_elevation"]
        runner.excavated = set(metadata["excavated"])
        runner.crown_geometry = metadata["crown_geometry"]
        runner.receipt["support_section_audit"] = audit_support_sections(runner)
        identifiers = sorted(runner.baseline)
        np.savez_compressed(
            runner.output / "stage_1_k0_equilibrium_hold.npz",
            node_ids=identifiers,
            displacement_m=[runner.baseline[identifier] for identifier in identifiers],
        )
        runner.write_receipt()
        return runner

    def apply_boundaries(self):
        Kratos = self.Kratos
        minimum_x, maximum_x, minimum_y, maximum_y, bottom, _ = self.family["domain"]
        for node in self.part.Nodes:
            node.Fix(Kratos.WATER_PRESSURE)
            fixed = []
            if abs(node.Z0 - bottom) < 1e-7:
                fixed.extend((Kratos.DISPLACEMENT_X, Kratos.DISPLACEMENT_Y, Kratos.DISPLACEMENT_Z))
            if min(abs(node.X0 - minimum_x), abs(node.X0 - maximum_x)) < 1e-7:
                fixed.append(Kratos.DISPLACEMENT_X)
            if min(abs(node.Y0 - minimum_y), abs(node.Y0 - maximum_y)) < 1e-7:
                fixed.append(Kratos.DISPLACEMENT_Y)
            for variable in fixed:
                node.Fix(variable)
                self.boundary_dofs.add((node.Id, variable.Name()))
            node.SetSolutionStepValue(Kratos.VOLUME_ACCELERATION, [0, 0, -9.81])
        self.water(6.5)

    def write_receipt(self):
        self.receipt["elapsed_s"] = time.monotonic() - self.started
        self.receipt["peak_rss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.output.joinpath("receipt.json").write_text(json.dumps(self.receipt, indent=2))

    def water(self, elevation):
        for node in self.part.Nodes:
            node.SetSolutionStepValue(
                self.Kratos.WATER_PRESSURE, -9.81 * max(elevation - node.Z0, 0)
            )
        self.water_elevation = elevation

    def step(self, label):
        if self.failed:
            raise RuntimeError("Failed native run must restart from immutable inputs")
        self.failed = True
        step_started = time.monotonic()
        self.receipt["attempting_step"] = label
        self.write_receipt()
        self.solver.AddDofs()
        if not self.initialized:
            self.solver.Initialize()
            self.solver.convergence_criterion.SetActualizeRHSFlag(True)
            self.initialized = True
        self.solver.Check()
        self.solver.AdvanceInTime(self.part.ProcessInfo[self.Kratos.TIME])
        self.solver.InitializeSolutionStep()
        self.solver.Predict()
        if not self.solver.SolveSolutionStep():
            self.receipt["failed_iterations"] = self.part.ProcessInfo[
                self.Kratos.NL_ITERATION_NUMBER
            ]
            self.write_receipt()
            raise RuntimeError(f"Native step {label} did not converge")
        matrix = self.solver.solving_strategy.GetSystemMatrix()
        matrix_size = [matrix.Size1(), matrix.Size2()]
        if label == "stage_4_dewater_1" and self.reference is None and self.capture_matrix:
            self.Kratos.WriteMatrixMarketMatrix(
                str(self.output / "first_water_matrix.mtx"), matrix, False
            )
        self.solver.FinalizeSolutionStep()
        for node in self.part.Nodes:
            if not all(
                math.isfinite(value)
                for value in node.GetSolutionStepValue(self.Kratos.DISPLACEMENT)
            ):
                raise RuntimeError("Nonfinite native displacement")
        record = {
            "label": label,
            "active_elements": self.active.NumberOfElements(),
            "iterations": self.part.ProcessInfo[self.Kratos.NL_ITERATION_NUMBER],
            "matrix_size": matrix_size,
            "native_constraints": self.active.NumberOfMasterSlaveConstraints(),
            "water_elevation_m": self.water_elevation,
            "excavated_soil_elements": len(self.excavated),
            "step_elapsed_s": time.monotonic() - step_started,
        }
        self.receipt["stages"].append(record)
        self.failed = False
        self.write_receipt()
        print(json.dumps(record), flush=True)
        if label in self.comparison_labels:
            self.compare_state(label)
        if self.write_checkpoints and self.baseline:
            self.save_checkpoint(label)

    def save_checkpoint(self, label):
        import os
        import shutil

        if shutil.disk_usage(self.output).free < 2 * 1024**3:
            self.receipt["checkpoint_error"] = "Less than 2 GiB guest disk headroom"
            self.write_receipt()
            return
        directory = self.output / "restart"
        directory.mkdir(exist_ok=True)
        pending = directory / "pending"
        try:
            serializer = self.Kratos.FileSerializer(str(pending))
            serializer.Save("Model", self.model)
            del serializer
            native_path = pending.with_suffix(".rest")
            metadata = {
                "label": label,
                "input_hashes": self.family["input_hashes"],
                "solver_settings": json.loads(self.solver.settings.WriteJsonString()),
                "wall_nodes": {
                    name: {str(master): node.Id for master, node in nodes.items()}
                    for name, nodes in self.wall_nodes.items()
                },
                "baseline": self.baseline,
                "water_elevation": self.water_elevation,
                "excavated": sorted(self.excavated),
                "boundary_dofs": sorted(self.boundary_dofs),
                "crown_geometry": self.crown_geometry,
                "receipt": self.receipt,
                "native_state_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
                "constitutive_restart_validated": False,
            }
            metadata_path = pending.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata))
            os.replace(native_path, directory / "latest.rest")
            os.replace(metadata_path, directory / "latest.json")
            if label == "stage_4_dewater_1" or label.startswith("stage_7_excavate"):
                for extension in ("rest", "json"):
                    branch_pending = directory / f"branch-pending.{extension}"
                    branch_pending.unlink(missing_ok=True)
                    os.link(directory / f"latest.{extension}", branch_pending)
                    os.replace(branch_pending, directory / f"branch.{extension}")
            self.receipt["latest_native_checkpoint"] = {
                "label": label,
                "bytes": (directory / "latest.rest").stat().st_size,
                "restart_validated": False,
            }
        except Exception as error:
            self.receipt["checkpoint_error"] = f"{type(error).__name__}: {error}"
        self.write_receipt()

    def compare_state(self, label):
        Kratos = self.Kratos
        nodes = sorted(self.part.Nodes, key=lambda node: node.Id)
        elements = sorted(self.soil.Elements, key=lambda element: element.Id)
        fields = {
            "node_ids": np.array([node.Id for node in nodes]),
            "element_ids": np.array([element.Id for element in elements]),
            "displacement_m": np.array(
                [list(node.GetSolutionStepValue(Kratos.DISPLACEMENT)) for node in nodes]
            ),
            "rotation_rad": np.array(
                [list(node.GetSolutionStepValue(Kratos.ROTATION)) for node in nodes]
            ),
            "water_pressure_kPa": np.array(
                [node.GetSolutionStepValue(Kratos.WATER_PRESSURE) for node in nodes]
            ),
            "effective_stress_kPa": np.array(
                [
                    np.asarray(
                        element.CalculateOnIntegrationPoints(
                            Kratos.CAUCHY_STRESS_VECTOR, self.part.ProcessInfo
                        )
                    )
                    for element in elements
                ]
            ),
        }
        np.savez_compressed(self.output / f"{label}.npz", **fields)
        if self.reference is None:
            return
        reference_path = self.reference / f"{label}.npz"
        reference = np.load(reference_path, allow_pickle=False)
        result = {"label": label, **compare_native_fields(fields, reference)}
        result["reference_state_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        self.receipt["state_comparisons"].append(result)
        self.write_receipt()
        print(json.dumps({"state_comparison": result}), flush=True)
        if not result["passed"]:
            raise RuntimeError(f"Native backend state equivalence failed at {label}")

    def k0_initialize(self):
        if "stage_1_k0_equilibrium_hold" in self.completed_labels:
            return
        Kratos = self.Kratos
        native_layers = effective_layers(self.family)
        for element in self.soil.Elements:
            geometry = element.GetGeometry()
            if len(geometry) == 4:
                lower, upper = 0.13819660, 0.58541020
                shapes = np.array(
                    [
                        [lower, upper, lower, lower],
                        [lower, lower, upper, lower],
                        [lower, lower, lower, upper],
                        [upper, lower, lower, lower],
                    ]
                )
            else:
                shapes = np.asarray(geometry.ShapeFunctionsValues())
            elevations = shapes @ np.array([node.Z0 for node in geometry])
            stresses = []
            for elevation in elevations:
                vertical = sum(
                    max(0, layer["top"] - max(elevation, layer["bottom"])) * (source[2] - 9.81)
                    for layer, source in zip(native_layers, self.benchmark["soils"])
                )
                stresses.append(Kratos.Vector([0, 0, -vertical, 0, 0, 0]))
            element.SetValuesOnIntegrationPoints(
                Kratos.CAUCHY_STRESS_VECTOR, stresses, 6, self.part.ProcessInfo
            )
        process = self.Geo.ApplyK0ProcedureProcess(
            self.model,
            Kratos.Parameters(
                json.dumps(
                    {
                        "model_part_name": "Support.Soil",
                        "use_standard_procedure": False,
                    }
                )
            ),
        )
        process.Check()
        process.ExecuteInitialize()
        process.ExecuteFinalizeSolutionStep()
        process.ExecuteFinalize()
        errors = []
        overburden_errors = []
        layer_sums = {}
        for element in self.soil.Elements:
            stress = np.asarray(
                element.CalculateOnIntegrationPoints(
                    Kratos.CAUCHY_STRESS_VECTOR, self.part.ProcessInfo
                )
            )
            expected_k0 = element.Properties[Kratos.KratosGlobals.GetVariable("K0_NC")]
            errors.append(float(np.max(np.abs(stress[:, :2] - expected_k0 * stress[:, 2, None]))))
            if np.max(np.abs(stress[:, 3:])) > 1e-8:
                raise RuntimeError("Native K0 process retained shear")
            depth = np.mean([node.Z0 for node in element.GetGeometry()])
            effective_vertical = 0.0
            for layer, source in zip(native_layers, self.benchmark["soils"]):
                thickness = max(0, layer["top"] - max(depth, layer["bottom"]))
                effective_vertical += thickness * (source[2] - 9.81)
            overburden_errors.append(abs(float(np.mean(stress[:, 2])) + effective_vertical))
            layer_index = next(
                index
                for index, layer in enumerate(native_layers)
                if layer["bottom"] <= depth <= layer["top"]
            )
            sums = layer_sums.setdefault(layer_index, [0.0, 0.0, 0.0])
            volume = element.GetGeometry().DomainSize()
            sums[0] += volume * float(np.mean(stress[:, 2]))
            sums[1] += volume * effective_vertical
            sums[2] += volume
        layer_error = max(
            abs((stress_sum + expected_sum) / volume)
            for stress_sum, expected_sum, volume in layer_sums.values()
        )
        self.receipt["k0"] = {
            "native_process": "ApplyK0ProcedureProcess",
            "max_horizontal_relation_error_kPa": max(errors),
            "max_element_mean_vertical_overburden_error_kPa": max(overburden_errors),
            "max_layer_mean_vertical_overburden_error_kPa": layer_error,
        }
        self.write_receipt()
        if max(errors) > 1e-8 or layer_error > 1e-5:
            raise RuntimeError("K0 stress does not match source overburden and layer ratios")
        before = {
            element.Id: np.asarray(
                element.CalculateOnIntegrationPoints(
                    Kratos.CAUCHY_STRESS_VECTOR, self.part.ProcessInfo
                )
            )
            for element in self.soil.Elements
        }
        self.solver.Initialize()
        self.solver.convergence_criterion.SetActualizeRHSFlag(True)
        self.initialized = True
        self.receipt["k0"]["initialization_transfer"] = (
            "source_effective_overburden_at_native_quadrature_points_then_native_K0_before_first_MC_initialization"
        )
        self.step("stage_1_k0_equilibrium_hold")
        drift = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(
                            element.CalculateOnIntegrationPoints(
                                Kratos.CAUCHY_STRESS_VECTOR, self.part.ProcessInfo
                            )
                        )
                        - before[element.Id]
                    )
                )
            )
            for element in self.soil.Elements
        )
        self.receipt["k0"]["hold_stress_drift_kPa"] = drift
        self.write_receipt()
        if drift > 1e-4:
            raise RuntimeError("K0 state was not preserved by the native equilibrium hold")
        self.baseline = {
            node.Id: list(node.GetSolutionStepValue(Kratos.DISPLACEMENT))
            for node in self.part.Nodes
        }
        self.output.joinpath("stage_1_baseline.json").write_text(json.dumps(self.baseline))

    def initialize_new_shell(self, element):
        process_info = self.part.ProcessInfo
        restarted = process_info[self.Kratos.IS_RESTARTED]
        try:
            process_info[self.Kratos.IS_RESTARTED] = False
            element.Initialize(process_info)
        finally:
            process_info[self.Kratos.IS_RESTARTED] = restarted

    def install_wall(self, faces, thickness, label):
        if label in self.completed_labels:
            return
        Kratos = self.Kratos
        copies = {}
        for face in faces:
            for identifier in face:
                if identifier in copies:
                    continue
                master = self.part.GetNode(identifier)
                offset = master.GetSolutionStepValue(Kratos.DISPLACEMENT)
                node = self.part.CreateNewNode(
                    self.part.NumberOfNodes() + 1,
                    master.X0 + offset[0],
                    master.Y0 + offset[1],
                    master.Z0 + offset[2],
                )
                self.active.AddNode(node, 0)
                for variable in (
                    Kratos.DISPLACEMENT_X,
                    Kratos.DISPLACEMENT_Y,
                    Kratos.DISPLACEMENT_Z,
                    Kratos.ROTATION_X,
                    Kratos.ROTATION_Y,
                    Kratos.ROTATION_Z,
                ):
                    node.AddDof(variable)
                node.AddDof(Kratos.WATER_PRESSURE)
                node.Fix(Kratos.WATER_PRESSURE)
                node.SetSolutionStepValue(Kratos.VOLUME_ACCELERATION, [0, 0, -9.81])
                for component, variable in enumerate(
                    (Kratos.DISPLACEMENT_X, Kratos.DISPLACEMENT_Y, Kratos.DISPLACEMENT_Z)
                ):
                    self.active.CreateNewMasterSlaveConstraint(
                        "LinearMasterSlaveConstraint",
                        self.active.NumberOfMasterSlaveConstraints() + 1,
                        master,
                        variable,
                        node,
                        variable,
                        1.0,
                        -offset[component],
                    )
                copies[identifier] = node
        properties = self.part.CreateNewProperties(
            max(prop.Id for prop in self.part.Properties) + 1
        )
        for variable, value in (
            (Kratos.YOUNG_MODULUS, 30e6),
            (Kratos.POISSON_RATIO, 0.2),
            (Kratos.DENSITY, 25 / 9.81),
            (Kratos.THICKNESS, thickness),
        ):
            properties[variable] = value
        properties[Kratos.CONSTITUTIVE_LAW] = Kratos.KratosGlobals.GetConstitutiveLaw(
            "ReissnerMindlinShellElasticConstitutiveLaw"
        ).Clone()
        for face in faces:
            element = self.part.CreateNewElement(
                "ShellThickElement3D4N",
                self.part.NumberOfElements() + 1,
                [copies[identifier].Id for identifier in face],
                properties,
            )
            self.active.AddElement(element, 0)
            element.Set(Kratos.ACTIVE, True)
            self.initialize_new_shell(element)
        self.wall_nodes[label] = copies
        self.receipt.setdefault("wall_properties", {})[label] = properties.Id
        self.step(label)

    def install_support(self, system, elevation, wall_label, label, *, crown_only=False):
        if label in self.completed_labels:
            return
        Kratos = self.Kratos
        choices = self.family["engineering_choices"]
        crown = choices.get("crowns", {}).get(system)
        is_crown = crown is not None and abs(elevation - crown["axis_z_m"]) < 1e-8
        members = member_sections(self.family, system, elevation, crown_only=crown_only)
        copies = self.wall_nodes[wall_label]
        faces = self.family[f"{system}_wall_faces"]
        attachment_z = crown["shell_top_m"] if is_crown else elevation
        eccentricity = elevation - attachment_z
        segments = [member["segment"] for member in members]
        edges = set()
        for face in faces:
            for first, second in zip(face, face[1:] + face[:1]):
                if (
                    abs(self.part.GetNode(first).Z0 - attachment_z) < 1e-7
                    and abs(self.part.GetNode(second).Z0 - attachment_z) < 1e-7
                ):
                    edges.add(tuple(sorted((first, second))))
        if not edges:
            raise ValueError(f"Wall mesh does not expose support level {elevation}")
        translations = (Kratos.DISPLACEMENT_X, Kratos.DISPLACEMENT_Y, Kratos.DISPLACEMENT_Z)
        rotations = (Kratos.ROTATION_X, Kratos.ROTATION_Y, Kratos.ROTATION_Z)
        beam_nodes = {}
        beam_baselines = {}
        attached = 0
        max_initial_residual = 0.0
        for segment in segments:
            for point in segment:
                key = tuple(point)
                if key in beam_nodes:
                    continue
                nearest = None
                for first, second in edges:
                    masters = [self.part.GetNode(first), self.part.GetNode(second)]
                    start = np.array([masters[0].X0, masters[0].Y0])
                    direction = np.array([masters[1].X0, masters[1].Y0]) - start
                    fraction = float(
                        np.clip(
                            np.dot(np.array(point) - start, direction)
                            / np.dot(direction, direction),
                            0,
                            1,
                        )
                    )
                    distance = float(np.linalg.norm(np.array(point) - start - fraction * direction))
                    if nearest is None or distance < nearest[0]:
                        nearest = (distance, first, second, fraction)
                node = self.part.CreateNewNode(self.part.NumberOfNodes() + 1, *point, elevation)
                self.active.AddNode(node, 0)
                for variable in (*translations, *rotations, Kratos.WATER_PRESSURE):
                    node.AddDof(variable)
                node.Fix(Kratos.WATER_PRESSURE)
                if nearest[0] < 1e-5:
                    attached += 1
                    _, first, second, fraction = nearest
                    weights = [1 - fraction, fraction]
                    soil_masters = [self.part.GetNode(first), self.part.GetNode(second)]
                    wall_masters = [copies[first], copies[second]]
                    displacement = sum(
                        weight * np.array(master.GetSolutionStepValue(Kratos.DISPLACEMENT))
                        for weight, master in zip(weights, soil_masters)
                    )
                    beam_baselines[key] = sum(
                        weight * np.array(self.baseline[master.Id])
                        for weight, master in zip(weights, soil_masters)
                    ).tolist()
                    node.X0 += displacement[0]
                    node.Y0 += displacement[1]
                    node.Z0 += displacement[2]
                    node.X, node.Y, node.Z = node.X0, node.Y0, node.Z0
                    for component, variable in enumerate((*translations, *rotations)):
                        masters = (
                            soil_masters
                            if variable in translations
                            else [copies[first], copies[second]]
                        )
                        master_dofs = [master.GetDof(variable) for master in masters]
                        coefficients = list(weights)
                        if eccentricity and component < 2:
                            rotation = rotations[1 - component]
                            sign = 1 if component == 0 else -1
                            master_dofs.extend(master.GetDof(rotation) for master in wall_masters)
                            coefficients.extend(sign * eccentricity * weight for weight in weights)
                        offset = -sum(
                            coefficient * dof.GetSolutionStepValue()
                            for coefficient, dof in zip(coefficients, master_dofs)
                        )
                        relation = Kratos.Matrix(1, len(coefficients))
                        for index, coefficient in enumerate(coefficients):
                            relation[0, index] = coefficient
                        self.active.CreateNewMasterSlaveConstraint(
                            "LinearMasterSlaveConstraint",
                            self.active.NumberOfMasterSlaveConstraints() + 1,
                            master_dofs,
                            [node.GetDof(variable)],
                            relation,
                            Kratos.Vector([offset]),
                        )
                beam_nodes[key] = node
        role_properties = {}
        for member in members:
            if member["role"] in role_properties:
                continue
            properties = self.part.CreateNewProperties(
                max(prop.Id for prop in self.part.Properties) + 1
            )
            for name, value in {
                "YOUNG_MODULUS": 30e6,
                "POISSON_RATIO": 0.2,
                "DENSITY": 25 / 9.81,
                **section_properties(member["section_m"]),
            }.items():
                properties[Kratos.KratosGlobals.GetVariable(name)] = value
            role_properties[member["role"]] = properties
        element_ids = []
        for member in members:
            first, second = member["segment"]
            element = self.part.CreateNewElement(
                "CrLinearBeamElement3D2N",
                self.part.NumberOfElements() + 1,
                [beam_nodes[tuple(first)].Id, beam_nodes[tuple(second)].Id],
                role_properties[member["role"]],
            )
            self.active.AddElement(element, 0)
            element.Set(Kratos.ACTIVE, True)
            element.Initialize(self.part.ProcessInfo)
            residual = Kratos.Vector()
            element.CalculateRightHandSide(residual, self.part.ProcessInfo)
            max_initial_residual = max(max_initial_residual, float(np.linalg.norm(residual)))
            element_ids.append(element.Id)
            if member["role"] == "crown":
                import KratosMultiphysics.StructuralMechanicsApplication as Structural

                orientation = np.asarray(
                    element.Calculate(Structural.LOCAL_ELEMENT_ORIENTATION, self.part.ProcessInfo)
                )[:3, :3]
                direction = np.subtract(second, first)
                axis = np.array([*direction, 0.0]) / np.linalg.norm(direction)
                transverse = np.cross([0, 0, 1], axis)
                reference_frame = np.column_stack([axis, transverse, np.cross(axis, transverse)])
                self.crown_geometry.append(
                    {
                        "system": system,
                        "element": element.Id,
                        "reference_axis_m": [[*first, elevation], [*second, elevation]],
                        "reference_frame": reference_frame.tolist(),
                        "native_initial_frame": orientation.tolist(),
                        "stage1_axis_displacement_m": [
                            beam_baselines[tuple(first)],
                            beam_baselines[tuple(second)],
                        ],
                        "profile_range_z_m": [
                            attachment_z,
                            elevation + member["section_m"]["height"] / 2,
                        ],
                    }
                )
        if max_initial_residual > 1e-6:
            raise RuntimeError("Support carries load before installation selfweight")
        for node in beam_nodes.values():
            node.SetSolutionStepValue(Kratos.VOLUME_ACCELERATION, [0, 0, -9.81])
        if is_crown and eccentricity:
            by_coordinates = {
                (round(node.X0, 7), round(node.Y0, 7), round(node.Z0, 7)): node
                for node in self.soil.Nodes
            }
            for identifier in sorted({identifier for edge in edges for identifier in edge}):
                master = self.part.GetNode(identifier)
                slave = by_coordinates.get(
                    (round(master.X0, 7), round(master.Y0, 7), round(elevation, 7))
                )
                if slave is None:
                    raise ValueError("Crown/soil interface requires an existing axis-level node")
                for component, variable in enumerate(translations):
                    master_dofs = [master.GetDof(variable)]
                    coefficients = [1.0]
                    if component < 2:
                        master_dofs.append(copies[identifier].GetDof(rotations[1 - component]))
                        coefficients.append(eccentricity if component == 0 else -eccentricity)
                    offset = slave.GetSolutionStepValue(variable) - sum(
                        coefficient * dof.GetSolutionStepValue()
                        for coefficient, dof in zip(coefficients, master_dofs)
                    )
                    relation = Kratos.Matrix(1, len(coefficients))
                    for index, coefficient in enumerate(coefficients):
                        relation[0, index] = coefficient
                    self.active.CreateNewMasterSlaveConstraint(
                        "LinearMasterSlaveConstraint",
                        self.active.NumberOfMasterSlaveConstraints() + 1,
                        master_dofs,
                        [slave.GetDof(variable)],
                        relation,
                        Kratos.Vector([offset]),
                    )
        self.receipt["support_installations"].append(
            {
                "label": label,
                "system": system,
                "elevation_m": elevation,
                "beam_elements": len(element_ids),
                "wall_attachment_nodes": attached,
                "unloaded_installation_residual": max_initial_residual,
                "native_element_ids": element_ids,
                "crown_only": crown_only,
                "attachment_elevation_m": attachment_z,
                "eccentricity_m": eccentricity,
            }
        )
        self.receipt["support_section_audit"] = audit_support_sections(self)
        self.step(label)

    def dewater(self, elevation, label, initial=None):
        initial = self.water_elevation if initial is None else initial
        steps = max(1, math.ceil(abs(elevation - initial) / 2.0))
        for index in range(1, steps + 1):
            if f"{label}_dewater_{index}" in self.completed_labels:
                continue
            self.water(initial + (elevation - initial) * index / steps)
            self.step(f"{label}_dewater_{index}")

    def excavate(self, system, elevation, label):
        from native_input_adapter import contains

        polygon = self.family[f"{system}_polygon"]
        groups = {}
        for element in self.soil.Elements:
            if element.Id in self.excavated:
                continue
            geometry = element.GetGeometry()
            if min(node.Z0 for node in geometry) < elevation - 1e-7:
                continue
            center = [
                sum(getattr(node, axis) for node in geometry) / len(geometry)
                for axis in ("X0", "Y0")
            ]
            if contains(center, polygon):
                groups.setdefault(min(node.Z0 for node in geometry), []).append(element.Id)
        if not groups:
            if any(step.startswith(f"{label}_excavate_") for step in self.completed_labels):
                return
            raise ValueError(f"No soil selected for {label}")
        first_substep = 1 + sum(
            step.startswith(f"{label}_excavate_") for step in self.completed_labels
        )
        for substep, (bottom, identifiers) in enumerate(
            sorted(groups.items(), reverse=True), first_substep
        ):
            for identifier in identifiers:
                self.part.GetElement(identifier).Set(self.Kratos.ACTIVE, False)
                self.active.RemoveElement(identifier)
                self.excavated.add(identifier)
            self.receipt["excavation_events"].append(
                {
                    "label": label,
                    "system": system,
                    "bottom_m": bottom,
                    "removed_elements": len(identifiers),
                }
            )
            self.step(f"{label}_excavate_{substep}")

    def run(self, until):
        self.k0_initialize()
        if until == "k0":
            return
        if not self.family["outer_wall_faces"]:
            raise ValueError("Native input has no outer wall faces")
        self.install_wall(self.family["outer_wall_faces"], 1.0, "stage_2_outer_wall")
        if until == "outer-wall":
            return
        self.install_support("outer", 6.5, "stage_2_outer_wall", "stage_3_outer_support_1")
        if until == "first-water":
            self.water(4.925)
            self.step("stage_4_dewater_1")
            return
        initial_water = 6.5
        for stage in self.benchmark["stages"][3:7]:
            label = f"stage_{stage['stage']}"
            target_water = 6.5 - stage["water_depth_m"]
            self.dewater(target_water, label, initial=initial_water)
            initial_water = target_water
            elevation = 6.5 - stage["excavation_depth_m"]
            self.excavate("outer", elevation, label)
            if stage["stage"] < 7:
                self.install_support(
                    "outer",
                    elevation,
                    "stage_2_outer_wall",
                    f"{label}_outer_support_{stage['stage'] - 2}",
                )
        self.install_wall(self.family["inner_wall_faces"], 0.62, "stage_8_inner_wall")
        if self.position == 0:
            self.install_support(
                "inner", -10.1, "stage_8_inner_wall", "stage_8_inner_support_pos_0m"
            )
        else:
            self.install_support(
                "inner", -10.1, "stage_8_inner_wall", "stage_8_inner_crown", crown_only=True
            )
        self.dewater(
            6.5 - self.benchmark["stages"][8]["water_depth_m"], "stage_9", initial=initial_water
        )
        if self.position:
            self.excavate("inner", -10.1 - self.position, "stage_9_expose_support")
            self.install_support(
                "inner",
                -10.1 - self.position,
                "stage_8_inner_wall",
                f"stage_9_inner_support_pos_{self.position}m",
            )
        self.excavate("inner", -15.1, "stage_9")
        self.receipt["stage_9_native_solve_completed"] = True
        self.receipt["full_case_complete"] = False
        self.receipt["full_case_limitations"] = self.family["engineering_choices"].get(
            "unresolved", []
        )
        self.write_receipt()

    def export_native(self):
        Kratos = self.Kratos
        Kratos.ModelPartIO(str(self.output / "native_model"), Kratos.IO.WRITE).WriteModelPart(
            self.part
        )
        with self.output.joinpath("native_nodal_fields.tsv").open("w") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["node", "x0_m", "y0_m", "z0_m", "ux_m", "uy_m", "uz_m", "pw_kPa"])
            for node in self.part.Nodes:
                writer.writerow(
                    [
                        node.Id,
                        node.X0,
                        node.Y0,
                        node.Z0,
                        *node.GetSolutionStepValue(Kratos.DISPLACEMENT),
                        node.GetSolutionStepValue(Kratos.WATER_PRESSURE),
                    ]
                )
        with self.output.joinpath("native_soil_stresses.jsonl").open("w") as stream:
            for element in self.soil.Elements:
                stresses = element.CalculateOnIntegrationPoints(
                    Kratos.CAUCHY_STRESS_VECTOR, self.part.ProcessInfo
                )
                stream.write(
                    json.dumps(
                        {
                            "element": element.Id,
                            "property": element.Properties.Id,
                            "stress_kPa": [list(stress) for stress in stresses],
                        }
                    )
                    + "\n"
                )
        self.receipt["native_fields_exported"] = True
        self.output.joinpath("crown_geometry.json").write_text(
            json.dumps(self.crown_geometry, indent=2)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    parser.add_argument("output")
    parser.add_argument(
        "--until", choices=("k0", "outer-wall", "first-water", "full"), default="full"
    )
    parser.add_argument(
        "--linear-backend", choices=("sparse-lu", "amgcl-gmres"), default="sparse-lu"
    )
    parser.add_argument("--equivalence-reference")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--position", type=int, choices=range(5), default=0)
    parser.add_argument("--capture-backend-matrix", action="store_true")
    parser.add_argument("--write-checkpoints", action="store_true")
    parser.add_argument("--restart")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    runner = None
    try:
        if arguments.restart:
            runner = FamilyRunner.from_checkpoint(
                arguments.family,
                output,
                arguments.restart,
                arguments.max_iterations,
                arguments.position,
            )
        else:
            runner = FamilyRunner(
                arguments.family,
                output,
                arguments.linear_backend,
                arguments.equivalence_reference,
                arguments.max_iterations,
                arguments.position,
                arguments.capture_backend_matrix,
                arguments.write_checkpoints,
            )
        runner.run(arguments.until)
        runner.export_native()
        runner.receipt["requested_scope_completed"] = True
        runner.write_receipt()
    except Exception:
        if runner is None:
            receipt = {"native_loader_passed": False, "full_case_complete": False}
        else:
            receipt = runner.receipt
            runner.write_receipt()
        receipt["error"] = traceback.format_exc()
        output.joinpath("receipt.json").write_text(json.dumps(receipt, indent=2))
        print(receipt["error"], flush=True)
        raise


if __name__ == "__main__":
    main()
