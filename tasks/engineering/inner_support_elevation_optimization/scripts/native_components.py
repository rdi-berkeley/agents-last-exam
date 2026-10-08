"""Bounded native component runner in m/kN, not the full task evaluator."""

import math

import KratosMultiphysics as Kratos
import KratosMultiphysics.GeoMechanicsApplication as Geo
import KratosMultiphysics.StructuralMechanicsApplication as Structural
import numpy as np
from KratosMultiphysics.GeoMechanicsApplication.geomechanics_U_Pw_solver import UPwSolver


MECHANICAL_DOFS = (
    Kratos.DISPLACEMENT_X,
    Kratos.DISPLACEMENT_Y,
    Kratos.DISPLACEMENT_Z,
    Kratos.ROTATION_X,
    Kratos.ROTATION_Y,
    Kratos.ROTATION_Z,
)


class NativeStageModel:
    """Dense small-strain component assembly with native constitutive history.

    Water pressure is prescribed at every soil node. No transient seepage is
    solved. Ties enforce increments from the installation configuration. The
    bounded dense solve supports at most 300 free mechanical degrees of freedom.
    Failed steps invalidate this runner; they are never committed or retried.
    """

    def __init__(self):
        self.model = Kratos.Model()
        settings = Kratos.Parameters(
            '{"model_part_name":"Component","domain_size":3,"rotation_dofs":true}'
        )
        self.variable_provider = UPwSolver(self.model, settings)
        self.variable_provider.AddVariables()
        self.part = self.model["Component"]
        self.part.SetBufferSize(2)
        self.part.ProcessInfo[Kratos.DELTA_TIME] = 1.0
        self.part.ProcessInfo[Geo.DT_PRESSURE_COEFFICIENT] = 1.0
        self.part.ProcessInfo[Geo.VELOCITY_COEFFICIENT] = 1.0
        self.active = self.part.CreateSubModelPart("Active")
        self.scheme = Kratos.ResidualBasedIncrementalUpdateStaticScheme()
        self.scheme.Initialize(self.active)
        self.lifecycle_arguments = (
            self.active,
            Kratos.CompressedMatrix(),
            Kratos.Vector(),
            Kratos.Vector(),
        )
        self.ties = {}
        self.external = {}
        self.soil_ids = set()
        self.failed = False
        self.time = 0

    def node(self, coordinates, fixed=()):
        node = self.part.CreateNewNode(self.part.NumberOfNodes() + 1, *coordinates)
        self.active.AddNode(node, 0)
        for variable in MECHANICAL_DOFS:
            node.AddDof(variable)
        node.AddDof(Kratos.WATER_PRESSURE)
        node.Fix(Kratos.WATER_PRESSURE)
        for variable in fixed:
            node.Fix(variable)
        return node

    def _element(self, name, nodes, values, law=None):
        identifier = self.part.NumberOfElements() + 1
        properties = self.part.CreateNewProperties(identifier)
        for name_or_variable, value in values.items():
            variable = (
                Kratos.KratosGlobals.GetVariable(name_or_variable)
                if isinstance(name_or_variable, str)
                else name_or_variable
            )
            properties[variable] = value
        if law is not None:
            properties[Kratos.CONSTITUTIVE_LAW] = Kratos.KratosGlobals.GetConstitutiveLaw(
                law
            ).Clone()
        element = self.part.CreateNewElement(
            name, identifier, [node.Id for node in nodes], properties
        )
        self.active.AddElement(element, 0)
        element.Set(Kratos.ACTIVE, True)
        element.Initialize(self.part.ProcessInfo)
        return element

    def soil(
        self,
        nodes,
        *,
        young_kpa,
        poisson,
        unit_weight,
        cohesion_kpa,
        friction_deg,
        dilation_deg,
        porosity,
        tension_cutoff=False,
        tensile_strength=0.0,
    ):
        if not 0 <= porosity < 1:
            raise ValueError("Porosity must be in [0,1)")
        values = {
            "GEO_DRAINAGE_TYPE": "CONSTANT_PW_FIELD",
            "YOUNG_MODULUS": young_kpa,
            "POISSON_RATIO": poisson,
            "DENSITY_SOLID": (unit_weight / 9.81 - porosity) / (1 - porosity),
            "DENSITY_WATER": 1.0,
            "POROSITY": porosity,
            "BULK_MODULUS_SOLID": 1e9,
            "BULK_MODULUS_FLUID": 2e6,
            "DYNAMIC_VISCOSITY": 1e-6,
            "BIOT_COEFFICIENT": 1.0,
            "RETENTION_LAW": "SaturatedLaw",
            "SATURATED_SATURATION": 1.0,
            "GEO_COHESION": cohesion_kpa,
            "GEO_FRICTION_ANGLE": friction_deg,
            "GEO_DILATANCY_ANGLE": dilation_deg,
            "GEO_ENABLE_TENSION_CUT_OFF": tension_cutoff,
            "PERMEABILITY_XX": 1e-9,
            "PERMEABILITY_YY": 1e-9,
            "PERMEABILITY_ZZ": 1e-9,
            "PERMEABILITY_XY": 0.0,
            "PERMEABILITY_YZ": 0.0,
            "PERMEABILITY_ZX": 0.0,
        }
        if tension_cutoff:
            values["GEO_TENSILE_STRENGTH"] = tensile_strength
        element = self._element("UPwSmallStrainElement3D8N", nodes, values, "GeoMohrCoulombLaw3D")
        self.soil_ids.add(element.Id)
        return element

    def wall(self, nodes, *, thickness):
        return self._element(
            "ShellThickElement3D4N",
            nodes,
            {
                "YOUNG_MODULUS": 30e6,
                "POISSON_RATIO": 0.2,
                "DENSITY": 25 / 9.81,
                "THICKNESS": thickness,
            },
            "ReissnerMindlinShellElasticConstitutiveLaw",
        )

    def install_beam(self, master, anchor_coordinates, *, area, inertia_y, inertia_z, torsion):
        displacement = np.array(master.GetSolutionStepValue(Kratos.DISPLACEMENT))
        coordinates = np.array([master.X0, master.Y0, master.Z0]) + displacement
        tip = self.node(coordinates.tolist())
        anchor = self.node(anchor_coordinates, fixed=MECHANICAL_DOFS)
        for variable in MECHANICAL_DOFS:
            self.ties[(tip.Id, variable.Name())] = (master.Id, variable.Name())
        beam = self._element(
            "CrLinearBeamElement3D2N",
            [tip, anchor],
            {
                "YOUNG_MODULUS": 30e6,
                "POISSON_RATIO": 0.2,
                "DENSITY": 25 / 9.81,
                Structural.CROSS_AREA: area,
                Structural.I22: inertia_y,
                Structural.I33: inertia_z,
                Structural.TORSIONAL_INERTIA: torsion,
            },
        )
        _, residual = self.local_system(beam)
        if np.linalg.norm(residual) > 1e-10:
            raise RuntimeError("New unloaded beam has nonzero installation stress")
        return beam, tip, anchor

    def install_wall(self, faces, *, thickness, fixed_rotations=()):
        """Install connected shell faces, tying only translations to existing soil."""
        copies = {}
        for face in faces:
            for master in face:
                if master.Id in copies:
                    continue
                displacement = np.array(master.GetSolutionStepValue(Kratos.DISPLACEMENT))
                coordinates = np.array([master.X0, master.Y0, master.Z0]) + displacement
                fixed = MECHANICAL_DOFS[3:] if master.Id in fixed_rotations else ()
                node = self.node(coordinates.tolist(), fixed=fixed)
                copies[master.Id] = node
                for variable in MECHANICAL_DOFS[:3]:
                    self.ties[(node.Id, variable.Name())] = (master.Id, variable.Name())
        elements = [
            self.wall([copies[node.Id] for node in face], thickness=thickness) for face in faces
        ]
        for element in elements:
            _, residual = self.local_system(element)
            if np.linalg.norm(residual) > 1e-10:
                raise RuntimeError("New unloaded wall has nonzero installation stress")
        return elements, copies

    def gravity(self, fraction=1.0):
        for node in self.active.Nodes:
            node.SetSolutionStepValue(Kratos.VOLUME_ACCELERATION, [0.0, 0.0, -9.81 * fraction])

    def water_table(self, elevation):
        for node in self.active.Nodes:
            node.SetSolutionStepValue(Kratos.WATER_PRESSURE, -9.81 * max(elevation - node.Z0, 0.0))

    def excavate(self, identifiers):
        for identifier in identifiers:
            if identifier not in self.soil_ids or not self.active.HasElement(identifier):
                raise ValueError("Excavation must select active soil elements")
        for identifier in identifiers:
            self.part.GetElement(identifier).Set(Kratos.ACTIVE, False)
            self.active.RemoveElement(identifier)

    def local_system(self, element):
        tangent = Kratos.Matrix()
        residual = Kratos.Vector()
        element.CalculateLocalSystem(tangent, residual, self.part.ProcessInfo)
        return np.array(tangent), np.array(residual)

    def _dofs(self):
        dofs = {}
        for element in self.active.Elements:
            for dof in element.GetDofList(self.part.ProcessInfo):
                key = (dof.Id(), dof.GetVariable().Name())
                dofs[key] = dof
        for key in list(dofs):
            target = self._target(key)
            if target not in dofs:
                dofs[target] = self.part.GetNode(target[0]).GetDof(
                    Kratos.KratosGlobals.GetVariable(target[1])
                )
        keys = sorted({self._target(key) for key in dofs if not dofs[key].IsFixed()})
        keys = [
            key
            for key in keys
            if not self.part.GetNode(key[0]).IsFixed(Kratos.KratosGlobals.GetVariable(key[1]))
        ]
        if len(keys) > 300:
            raise ValueError("Component solve exceeds 300 free DOFs; use a validated sparse runner")
        indices = {key: index for index, key in enumerate(keys)}
        return dofs, indices

    def _target(self, key):
        visited = set()
        while key in self.ties:
            if key in visited:
                raise ValueError("Cyclic incremental connection")
            visited.add(key)
            key = self.ties[key]
        return key

    def assemble(self):
        dofs, indices = self._dofs()
        tangent = np.zeros((len(indices), len(indices)))
        residual = np.zeros(len(indices))
        full_residual = dict.fromkeys(dofs, 0.0)
        for element in self.active.Elements:
            local_tangent, local_residual = self.local_system(element)
            local_dofs = element.GetDofList(self.part.ProcessInfo)
            transform = np.zeros((len(local_dofs), len(indices)))
            for local_index, dof in enumerate(local_dofs):
                key = (dof.Id(), dof.GetVariable().Name())
                full_residual[key] += float(local_residual[local_index])
                index = indices.get(self._target(key))
                if index is not None:
                    transform[local_index, index] = 1.0
            tangent += transform.T @ local_tangent @ transform
            residual += transform.T @ local_residual
        for key, value in self.external.items():
            if key not in dofs:
                raise ValueError("Applied load has no active mechanical DOF")
            full_residual[key] += value
            index = indices.get(self._target(key))
            if index is not None:
                residual[index] += value
        return tangent, residual, full_residual, dofs, indices

    def step(self, label, *, tolerance=1e-8, max_iterations=40):
        if self.failed:
            raise RuntimeError(
                "Runner invalidated by a failed step; reconstruct from initial inputs"
            )
        self.failed = True
        self.scheme.Check(self.active)
        self.time += 1
        self.part.CloneTimeStep(float(self.time))
        self.part.ProcessInfo[Kratos.STEP] = self.time
        self.part.ProcessInfo[Kratos.DELTA_TIME] = 1.0
        self.scheme.InitializeSolutionStep(*self.lifecycle_arguments)
        for iteration in range(max_iterations):
            self.part.ProcessInfo[Kratos.NL_ITERATION_NUMBER] = iteration + 1
            self.scheme.InitializeNonLinIteration(*self.lifecycle_arguments)
            tangent, residual, full_residual, dofs, indices = self.assemble()
            norm = float(np.linalg.norm(residual))
            if not math.isfinite(norm):
                raise RuntimeError("Non-finite native residual")
            if norm < tolerance:
                self.scheme.FinalizeSolutionStep(*self.lifecycle_arguments)
                self.failed = False
                reactions = {}
                for key, value in full_residual.items():
                    target = self._target(key)
                    node = self.part.GetNode(target[0])
                    variable = Kratos.KratosGlobals.GetVariable(target[1])
                    if node.IsFixed(variable):
                        reactions[target[1]] = reactions.get(target[1], 0.0) - value
                return {
                    "label": label,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "reaction_sums": reactions,
                    "active_elements": self.active.NumberOfElements(),
                }
            increment = np.linalg.solve(tangent, residual)
            for key, dof in dofs.items():
                index = indices.get(self._target(key))
                if index is not None:
                    node = self.part.GetNode(key[0])
                    variable = dof.GetVariable()
                    node.SetSolutionStepValue(
                        variable, node.GetSolutionStepValue(variable) + float(increment[index])
                    )
            self.scheme.FinalizeNonLinIteration(*self.lifecycle_arguments)
        raise RuntimeError(
            f"Component step {label!r} failed to converge after {max_iterations} iterations"
        )

    def free_residual_norm(self):
        return float(np.linalg.norm(self.assemble()[1]))

    def effective_stresses(self, identifier):
        element = self.part.GetElement(identifier)
        return np.array(
            element.CalculateOnIntegrationPoints(Kratos.CAUCHY_STRESS_VECTOR, self.part.ProcessInfo)
        )
