"""Programmatic component fixtures driven by the maintained native UPwSolver."""

import json
import math

from native_components import MECHANICAL_DOFS as MECHANICAL_DOFS
from native_components import NativeStageModel as FixtureModel
from native_components import Kratos, Structural, UPwSolver
from native_components import np as np


class NativeStageModel(FixtureModel):
    """Reuse fixture construction; delegate every solution to UPwSolver."""

    def __init__(self):
        super().__init__()
        self.solver = UPwSolver(
            self.model,
            Kratos.Parameters(
                json.dumps(
                    {
                        "model_part_name": "Component",
                        "domain_size": 3,
                        "rotation_dofs": True,
                        "scheme_type": "Backward_Euler",
                        "solution_type": "quasi_static",
                        "strategy_type": "newton_raphson",
                        "linear_solver_settings": {
                            "solver_type": "LinearSolversApplication.sparse_lu"
                        },
                        "convergence_criterion": "and_criterion",
                        "residual_relative_tolerance": 1e-13,
                        "residual_absolute_tolerance": 1e-14,
                        "displacement_relative_tolerance": 1e-10,
                        "displacement_absolute_tolerance": 1e-13,
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
        self.solver.computing_model_part_name = self.active.Name
        self.initialized = False
        self.connection_offsets = {}
        self.load_conditions = {}

    def assemble(self):
        raise RuntimeError("Dense assembly is forbidden on the native sparse path")

    def _sync_connections(self):
        for slave_key in self.ties:
            if slave_key in self.connection_offsets:
                continue
            master_key = self._target(slave_key)
            slave = self.part.GetNode(slave_key[0])
            master = self.part.GetNode(master_key[0])
            variable = Kratos.KratosGlobals.GetVariable(slave_key[1])
            offset = slave.GetSolutionStepValue(variable) - master.GetSolutionStepValue(variable)
            self.connection_offsets[slave_key] = (master_key, offset)
            self.active.CreateNewMasterSlaveConstraint(
                "LinearMasterSlaveConstraint",
                len(self.connection_offsets),
                master,
                variable,
                slave,
                variable,
                1.0,
                offset,
            )

    def _sync_loads(self):
        for node in self.active.Nodes:
            node.SetSolutionStepValue(Structural.POINT_LOAD, [0.0, 0.0, 0.0])
        for (node_id, variable_name), value in self.external.items():
            if variable_name not in ("DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z"):
                raise ValueError("This component adapter supports only nodal forces")
            node = self.part.GetNode(node_id)
            load = list(node.GetSolutionStepValue(Structural.POINT_LOAD))
            load[("DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z").index(variable_name)] = (
                value
            )
            node.SetSolutionStepValue(Structural.POINT_LOAD, load)
            if node_id not in self.load_conditions:
                properties = next(iter(self.part.Properties))
                condition = self.part.CreateNewCondition(
                    "PointLoadCondition3D1N",
                    self.part.NumberOfConditions() + 1,
                    [node_id],
                    properties,
                )
                self.active.AddCondition(condition, 0)
                condition.Set(Kratos.ACTIVE, True)
                self.load_conditions[node_id] = condition

    def force_balance(self):
        """Independently inspect residuals; never assemble or solve a tangent."""
        forces = {}
        for entity in [*self.active.Elements, *self.active.Conditions]:
            residual = Kratos.Vector()
            entity.CalculateRightHandSide(residual, self.part.ProcessInfo)
            for dof, value in zip(entity.GetDofList(self.part.ProcessInfo), residual):
                key = self._target((dof.Id(), dof.GetVariable().Name()))
                forces[key] = forces.get(key, 0.0) + float(value)
        residual_square = 0.0
        reactions = {}
        for (node_id, variable_name), value in forces.items():
            node = self.part.GetNode(node_id)
            variable = Kratos.KratosGlobals.GetVariable(variable_name)
            if node.IsFixed(variable):
                reactions[variable_name] = reactions.get(variable_name, 0.0) - value
            else:
                residual_square += value * value
        return math.sqrt(residual_square), reactions

    def free_residual_norm(self):
        return self.force_balance()[0]

    def step(self, label, *, tolerance=1e-8, max_iterations=40):
        if self.failed:
            raise RuntimeError(
                "Runner invalidated by a failed step; reconstruct from initial inputs"
            )
        self.failed = True
        if max_iterations <= 0:
            raise RuntimeError("Native step failed to converge: positive iteration budget required")
        if max_iterations != 40:
            raise ValueError("Configure a supported native iteration budget before initialization")
        self.solver.AddDofs()
        self._sync_connections()
        self._sync_loads()
        if not self.initialized:
            self.solver.Initialize()
            self.solver.convergence_criterion.SetActualizeRHSFlag(True)
            self.initialized = True
        self.solver.Check()
        self.time = self.solver.AdvanceInTime(self.time)
        self.solver.InitializeSolutionStep()
        self.solver.Predict()
        converged = self.solver.SolveSolutionStep()
        if not converged:
            raise RuntimeError(f"Native sparse step {label!r} failed to converge")
        for node in self.active.Nodes:
            for variable in (*MECHANICAL_DOFS, Kratos.WATER_PRESSURE):
                if not math.isfinite(node.GetSolutionStepValue(variable)):
                    raise RuntimeError("Non-finite native solution")
        for slave_key, (master_key, offset) in self.connection_offsets.items():
            variable = Kratos.KratosGlobals.GetVariable(slave_key[1])
            slave_value = self.part.GetNode(slave_key[0]).GetSolutionStepValue(variable)
            master_value = self.part.GetNode(master_key[0]).GetSolutionStepValue(variable)
            if abs(slave_value - master_value - offset) > 1e-12:
                raise RuntimeError("Native installation constraint lost its reference offset")
        residual_norm, reactions = self.force_balance()
        if not math.isfinite(residual_norm) or residual_norm > tolerance:
            raise RuntimeError(
                f"Native sparse equilibrium check failed: {residual_norm}; iterations={self.part.ProcessInfo[Kratos.NL_ITERATION_NUMBER]}"
            )
        system_matrix = self.solver.solving_strategy.GetSystemMatrix()
        matrix_type = type(system_matrix).__name__
        matrix_size = [system_matrix.Size1(), system_matrix.Size2()]
        self.solver.FinalizeSolutionStep()
        self.failed = False
        return {
            "label": label,
            "iterations": self.part.ProcessInfo[Kratos.NL_ITERATION_NUMBER],
            "residual_norm": residual_norm,
            "reaction_sums": reactions,
            "active_elements": self.active.NumberOfElements(),
            "solver": type(self.solver).__name__,
            "strategy": type(self.solver.solving_strategy).__name__,
            "linear_solver": type(self.solver.linear_solver).__name__,
            "builder": type(self.solver.builder_and_solver).__name__,
            "matrix_type": matrix_type,
            "matrix_size": matrix_size,
            "native_constraints": self.active.NumberOfMasterSlaveConstraints(),
        }
