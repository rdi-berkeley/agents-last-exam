"""Native component controls. Run with the installed Kratos Python interpreter."""

import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import unittest
from unittest.mock import patch


NATIVE_AVAILABLE = importlib.util.find_spec("KratosMultiphysics") is not None
EVIDENCE = {}


def load_native():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1800 * 1024**2,) * 2)
    scripts = Path(__file__).resolve().parents[2] / (
        "tasks/engineering/inner_support_elevation_optimization/scripts"
    )
    sys.path.insert(0, str(scripts))
    backend = os.environ.get("SUPPORT_NATIVE_BACKEND", "sparse")
    if backend not in ("dense", "sparse"):
        raise ValueError("SUPPORT_NATIVE_BACKEND must be dense or sparse")
    source = scripts / (
        "sparse_native_components.py" if backend == "sparse" else "native_components.py"
    )
    specification = importlib.util.spec_from_file_location("support_native_components", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.Kratos.ParallelUtilities.SetNumThreads(1)
    return module


@unittest.skipUnless(NATIVE_AVAILABLE, "Installed Kratos is required; no mocked native pass")
class NativeComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = load_native()
        cls.kratos = cls.native.Kratos
        cls.numpy = cls.native.np
        if os.environ.get("SUPPORT_NATIVE_BACKEND", "sparse") == "sparse":
            dense_solve = patch.object(
                cls.numpy.linalg,
                "solve",
                side_effect=AssertionError("Dense solve called by sparse backend"),
            )
            dense_solve.start()
            cls.addClassCleanup(dense_solve.stop)

    def soil_block(self, *, young=5000.0, cohesion=10.0, porosity=0.3, all_fixed=False):
        runner = self.native.NativeStageModel()
        coordinates = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]
        nodes = []
        for point in coordinates:
            fixed = list(self.native.MECHANICAL_DOFS)
            if not all_fixed and point[2] != 0:
                fixed.remove(self.kratos.DISPLACEMENT_Z)
            nodes.append(runner.node(point, fixed=fixed))
        soil = runner.soil(
            nodes,
            young_kpa=young,
            poisson=0.3,
            unit_weight=18,
            cohesion_kpa=cohesion,
            friction_deg=23,
            dilation_deg=0,
            porosity=porosity,
        )
        return runner, nodes, soil

    def test_wall_weight_bending_and_missing_gravity_controls(self):
        evidence = []
        displacements = []
        for thickness in (0.62, 1.0):
            for mutation in ("source_weight", "zero_density", "zero_gravity"):
                runner = self.native.NativeStageModel()
                nodes = [
                    runner.node(point, fixed=self.native.MECHANICAL_DOFS if point[2] == 0 else ())
                    for point in [(0, 0, 0), (2, 0, 0), (2, 0, 3), (0, 0, 3)]
                ]
                wall = runner.wall(nodes, thickness=thickness)
                if mutation == "zero_density":
                    wall.Properties[self.kratos.DENSITY] = 0.0
                runner.gravity(0.0 if mutation == "zero_gravity" else 1.0)
                if (
                    mutation == "zero_density"
                    and os.environ.get("SUPPORT_NATIVE_BACKEND", "sparse") == "sparse"
                ):
                    with self.assertRaisesRegex(RuntimeError, "DENSITY"):
                        runner.step(mutation)
                    evidence.append(
                        {
                            "thickness": thickness,
                            "control": mutation,
                            "native_material_check_rejected": True,
                        }
                    )
                    continue
                step = runner.step(mutation)
                reaction = step["reaction_sums"]["DISPLACEMENT_Z"]
                expected = 25 * 6 * thickness
                self.assertAlmostEqual(
                    reaction, expected if mutation == "source_weight" else 0, places=9
                )
                evidence.append(
                    {"thickness": thickness, "control": mutation, "reaction_kN": reaction}
                )
                if mutation == "source_weight":
                    for node in nodes[2:]:
                        runner.external[node.Id, "DISPLACEMENT_Y"] = -1.0
                    bending = runner.step("normal load")
                    self.assertAlmostEqual(
                        bending["reaction_sums"]["DISPLACEMENT_Y"], 2.0, places=8
                    )
                    displacements.append(
                        abs(nodes[2].GetSolutionStepValue(self.kratos.DISPLACEMENT_Y))
                    )
        self.assertGreater(displacements[0], displacements[1])
        EVIDENCE["walls"] = {"gravity": evidence, "normal_deflections_m": displacements}

    def test_prescribed_groundwater_effective_stress_and_density(self):
        evidence = []
        for porosity in (0.2, 0.4):
            runner, nodes, soil = self.soil_block(porosity=porosity)
            runner.gravity()
            runner.water_table(1.0)
            wet = runner.step("ground level water")
            wet_displacement = nodes[4].GetSolutionStepValue(self.kratos.DISPLACEMENT_Z)
            wet_stress = runner.effective_stresses(soil.Id).copy()
            runner.water_table(0.0)
            dry = runner.step("drawdown")
            dry_displacement = nodes[4].GetSolutionStepValue(self.kratos.DISPLACEMENT_Z)
            dry_stress = runner.effective_stresses(soil.Id)
            constrained_modulus = 5000 * (1 - 0.3) / ((1 + 0.3) * (1 - 2 * 0.3))
            self.assertAlmostEqual(
                wet_displacement, -(18 - 9.81) / (2 * constrained_modulus), places=12
            )
            self.assertAlmostEqual(dry_displacement, -18 / (2 * constrained_modulus), places=12)
            self.assertGreater(abs(dry_displacement), abs(wet_displacement))
            self.assertLess(float(dry_stress[:, 2].mean()), float(wet_stress[:, 2].mean()))
            self.assertAlmostEqual(wet["reaction_sums"]["DISPLACEMENT_Z"], 18.0, places=10)
            self.assertAlmostEqual(dry["reaction_sums"]["DISPLACEMENT_Z"], 18.0, places=10)
            EVIDENCE["water_scope"] = (
                "Prescribed hydrostatic pressure in U-Pw mechanical equilibrium; no seepage solve"
            )
            evidence.append(
                {"porosity": porosity, "wet_top_m": wet_displacement, "dry_top_m": dry_displacement}
            )
        EVIDENCE["groundwater"] = evidence

    def test_mohr_coulomb_yield_unload_and_retained_history(self):
        evidence = []
        for young in (5000.0, 8000.0):
            runner, nodes, soil = self.soil_block(young=young, cohesion=1.0, all_fixed=True)
            stresses = []
            for shear in (0.0, 0.001, 0.015, 0.0):
                for node in nodes:
                    node.SetSolutionStepValue(
                        self.kratos.DISPLACEMENT,
                        [-0.002 * node.X0 + shear * node.Z0, -0.002 * node.Y0, -0.002 * node.Z0],
                    )
                runner.step("strain control")
                stresses.append(runner.effective_stresses(soil.Id).copy())
            confinement = young * 0.002 / (1 - 2 * 0.3)
            yield_shear = math.cos(math.radians(23)) + confinement * math.sin(math.radians(23))
            self.numpy.testing.assert_allclose(stresses[1][:, 5], young / 2.6 * 0.001, atol=1e-10)
            self.numpy.testing.assert_allclose(stresses[2][:, 5], yield_shear, atol=1e-9)
            self.assertLess(yield_shear, young / 2.6 * 0.015)
            self.assertGreater(float(self.numpy.linalg.norm(stresses[3][:, 5])), 1.0)
            hold = runner.step("hold committed plastic state")
            self.assertLess(hold["residual_norm"], 1e-8)
            self.numpy.testing.assert_allclose(
                runner.effective_stresses(soil.Id), stresses[3], atol=1e-10
            )
            fresh, fresh_nodes, fresh_soil = self.soil_block(
                young=young, cohesion=1.0, all_fixed=True
            )
            for node in fresh_nodes:
                node.SetSolutionStepValue(
                    self.kratos.DISPLACEMENT, [-0.002 * node.X0, -0.002 * node.Y0, -0.002 * node.Z0]
                )
            fresh.step("history reset negative control")
            self.assertLess(
                float(self.numpy.linalg.norm(fresh.effective_stresses(fresh_soil.Id)[:, 5])), 1e-10
            )
            evidence.append(
                {
                    "young_kPa": young,
                    "yield_shear_kPa": float(stresses[2][0, 5]),
                    "unloaded_shear_kPa": float(stresses[3][0, 5]),
                }
            )
        EVIDENCE["plastic_history"] = evidence

    def staged_fixture(self, *, spacing, young, support_height):
        runner = self.native.NativeStageModel()
        grid = {}
        heights = [index * spacing for index in range(round(2 / spacing) + 1)]
        for xpos in (-2, 0, 2):
            for ypos in (0, 1):
                for zpos in heights:
                    fixed = [self.kratos.DISPLACEMENT_Y, *self.native.MECHANICAL_DOFS[3:]]
                    if abs(xpos) == 2 or zpos == 0:
                        fixed.append(self.kratos.DISPLACEMENT_X)
                    if zpos == 0:
                        fixed.append(self.kratos.DISPLACEMENT_Z)
                    grid[xpos, ypos, zpos] = runner.node((xpos, ypos, zpos), fixed=fixed)
        excavated = []
        retained = []
        for xpos in (-2, 0):
            for lower, upper in zip(heights[:-1], heights[1:]):
                nodes = [
                    grid[xpos + offset_x, ypos, zpos]
                    for offset_x, ypos, zpos in [
                        (0, 0, lower),
                        (2, 0, lower),
                        (2, 1, lower),
                        (0, 1, lower),
                        (0, 0, upper),
                        (2, 0, upper),
                        (2, 1, upper),
                        (0, 1, upper),
                    ]
                ]
                soil = runner.soil(
                    nodes,
                    young_kpa=young,
                    poisson=0.3,
                    unit_weight=18,
                    cohesion_kpa=10,
                    friction_deg=23,
                    dilation_deg=0,
                    porosity=0.3,
                )
                (excavated if xpos == -2 and lower >= 1 else retained).append(soil.Id)
        runner.gravity()
        runner.water_table(2)
        steps = [runner.step("soil gravity equilibrium")]
        self.assertAlmostEqual(steps[-1]["reaction_sums"]["DISPLACEMENT_Z"], 144, places=8)
        prior_displacement = self.numpy.array(
            grid[0, 0, 2].GetSolutionStepValue(self.kratos.DISPLACEMENT)
        )
        prior_stress = runner.effective_stresses(retained[-1]).copy()
        faces = [
            [grid[0, 0, lower], grid[0, 1, lower], grid[0, 1, upper], grid[0, 0, upper]]
            for lower, upper in zip(heights[:-1], heights[1:])
        ]
        walls, copies = runner.install_wall(
            faces, thickness=0.62, fixed_rotations=[grid[0, 0, 0].Id, grid[0, 1, 0].Id]
        )
        self.numpy.testing.assert_array_equal(prior_stress, runner.effective_stresses(retained[-1]))
        self.numpy.testing.assert_array_equal(
            prior_displacement, grid[0, 0, 2].GetSolutionStepValue(self.kratos.DISPLACEMENT)
        )
        runner.gravity()
        wall_imbalance = runner.free_residual_norm()
        self.assertGreater(wall_imbalance, 1)
        steps.append(runner.step("wall installation with selfweight"))
        self.assertGreater(steps[-1]["iterations"], 0)
        wall_weight = sum(wall.GetGeometry().Area() * 0.62 * 25 for wall in walls)
        self.assertAlmostEqual(
            steps[-1]["reaction_sums"]["DISPLACEMENT_Z"], 144 + wall_weight, places=7
        )
        runner.water_table(0.5)
        steps.append(runner.step("drawdown"))
        excavation_stress = runner.effective_stresses(excavated[0]).copy()
        runner.excavate(excavated)
        steps.append(runner.step("remove upper soil on excavation side"))
        self.assertAlmostEqual(
            steps[-1]["reaction_sums"]["DISPLACEMENT_Z"], 108 + wall_weight, places=7
        )
        self.numpy.testing.assert_array_equal(
            excavation_stress, runner.effective_stresses(excavated[0])
        )
        self.assertTrue(all(not runner.active.HasElement(identifier) for identifier in excavated))
        master = copies[grid[0, 0, support_height].Id]
        prior = self.numpy.array(master.GetSolutionStepValue(self.kratos.DISPLACEMENT))
        retained_stress = runner.effective_stresses(retained[-1]).copy()
        beam, tip, anchor = runner.install_beam(
            master,
            [-2, 0, support_height],
            area=0.02,
            inertia_y=0.0002,
            inertia_z=0.0002,
            torsion=0.0001,
        )
        self.numpy.testing.assert_array_equal(
            prior, master.GetSolutionStepValue(self.kratos.DISPLACEMENT)
        )
        self.numpy.testing.assert_array_equal(
            retained_stress, runner.effective_stresses(retained[-1])
        )
        self.assertLess(float(self.numpy.linalg.norm(runner.local_system(beam)[1])), 1e-10)
        runner.gravity()
        installation_imbalance = runner.free_residual_norm()
        self.assertGreater(installation_imbalance, 0.1)
        steps.append(runner.step("support installation with selfweight"))
        self.assertGreater(steps[-1]["iterations"], 0)
        beam_weight = beam.GetGeometry().Length() * 0.02 * 25
        self.assertAlmostEqual(
            steps[-1]["reaction_sums"]["DISPLACEMENT_Z"], 108 + wall_weight + beam_weight, places=5
        )
        self.numpy.testing.assert_allclose(
            self.numpy.array(master.GetSolutionStepValue(self.kratos.DISPLACEMENT)) - prior,
            tip.GetSolutionStepValue(self.kratos.DISPLACEMENT),
            atol=1e-12,
        )
        for ypos in (0, 1):
            runner.external[copies[grid[0, ypos, 2].Id].Id, "DISPLACEMENT_X"] = -1.0
        steps.append(runner.step("additional horizontal load"))
        observation = list(grid[0, 0, 2].GetSolutionStepValue(self.kratos.DISPLACEMENT))
        final_stress = runner.effective_stresses(retained[-1]).copy()
        steps.append(runner.step("hold"))
        self.assertLess(steps[-1]["residual_norm"], 1e-8)
        if os.environ.get("SUPPORT_NATIVE_BACKEND", "sparse") == "sparse":
            for step in steps:
                self.assertEqual(step["solver"], "UPwSolver")
                self.assertEqual(step["strategy"], "GeoMechanicsNewtonRaphsonStrategy")
                self.assertEqual(step["linear_solver"], "SparseLUSolver")
                self.assertEqual(step["matrix_type"], "CompressedMatrix")
                self.assertGreater(step["matrix_size"][0], 0)
            self.assertGreater(steps[-1]["native_constraints"], 0)
        self.numpy.testing.assert_allclose(
            observation,
            grid[0, 0, 2].GetSolutionStepValue(self.kratos.DISPLACEMENT),
            atol=1e-13,
            rtol=0,
        )
        self.numpy.testing.assert_allclose(
            final_stress, runner.effective_stresses(retained[-1]), atol=1e-10
        )
        return {
            "spacing_m": spacing,
            "young_kPa": young,
            "support_height_m": support_height,
            "wall_installation_imbalance": wall_imbalance,
            "support_installation_imbalance": installation_imbalance,
            "specified_top_point_displacement_m": observation,
            "steps": steps,
        }

    def test_staged_coupling_mesh_material_and_support_variations(self):
        cases = [
            self.staged_fixture(spacing=spacing, young=young, support_height=height)
            for spacing, young, height in [(1.0, 5000.0, 2.0), (0.5, 8000.0, 1.0)]
        ]
        self.assertFalse(
            self.numpy.allclose(
                cases[0]["specified_top_point_displacement_m"],
                cases[1]["specified_top_point_displacement_m"],
                atol=1e-8,
                rtol=0,
            )
        )
        EVIDENCE["staged_fixture_variations"] = cases

    def test_failed_step_cannot_be_silently_reused(self):
        runner, nodes, soil = self.soil_block()
        runner.gravity()
        with self.assertRaisesRegex(RuntimeError, "failed to converge"):
            runner.step("deliberately insufficient iterations", max_iterations=0)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            runner.step("must not reuse a failed state")


if __name__ == "__main__":
    if not NATIVE_AVAILABLE:
        raise SystemExit("Native Kratos unavailable; this command must not report a skipped pass")
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(NativeComponentTests)
    )
    EVIDENCE.update(
        passed=result.wasSuccessful(),
        tests_run=result.testsRun,
        kratos_version=NativeComponentTests.kratos.KratosGlobals.Kernel.Version(),
        max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
        scope="Native bounded component controls; not source-compliant full five-case replay",
        backend=os.environ.get("SUPPORT_NATIVE_BACKEND", "sparse"),
    )
    print("NATIVE_COMPONENT_EVIDENCE=" + json.dumps(EVIDENCE, allow_nan=False), flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)
