"""Regression tests for the spin-triangle fixture builder and verifier."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tasks.physical_sciences.frustrated_spin_triangle_yang_baxter_holonomy.scripts.build_fixtures import (
    build_fixtures,
)
from tasks.physical_sciences.frustrated_spin_triangle_yang_baxter_holonomy.scripts.verify_outputs import (
    PAULIS,
    evaluate_scenario,
    kron3,
    mat_add,
    mat_comm,
    mat_frob_norm,
    mat_zero,
    noise_coefficients,
    score_submission,
)


class SpinTriangleVerifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.scenarios,
            cls.author_answers,
            cls.tier1_answers,
            cls.tier2_answers,
        ) = build_fixtures()

    def test_author_witness_passes_all_scenarios(self) -> None:
        result = score_submission(self.scenarios, self.author_answers)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.correct, 12)
        self.assertEqual(result.total, 12)

    def test_documented_attacker_baselines_are_stable(self) -> None:
        tier1_result = score_submission(self.scenarios, self.tier1_answers)
        tier2_result = score_submission(self.scenarios, self.tier2_answers)
        self.assertAlmostEqual(tier1_result.score, 0.25, places=12)
        self.assertAlmostEqual(tier1_result.diagnostic_score, 0.2950, places=12)
        self.assertEqual(tier1_result.correct, 3)
        self.assertAlmostEqual(tier2_result.score, 1 / 3, places=12)
        self.assertAlmostEqual(tier2_result.diagnostic_score, 0.3733333333333333, places=12)
        self.assertEqual(tier2_result.correct, 4)

    def test_missing_schedules_score_zero(self) -> None:
        result = score_submission(self.scenarios, {})
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.correct, 0)

    def test_nonfinite_pulses_are_rejected(self) -> None:
        submission = {
            "S1": {
                "frame_calibration": [0.0] * 9,
                "pulses": [
                    {
                        "frame_codes": [0, 0, 0],
                        "bond_drives": [float("nan"), 0.0, 0.0],
                        "duration": 0.25,
                    },
                    {
                        "frame_codes": [0, 0, 0],
                        "bond_drives": [0.0, 0.0, 0.0],
                        "duration": 0.25,
                    },
                ],
            }
        }
        result = score_submission(self.scenarios, submission)
        self.assertEqual(result.score, 0.0)
        self.assertIn("non-finite", result.results["S1"]["message"])

    def test_hardware_constraints_are_enforced(self) -> None:
        for field, value, message in (
            ("duration", 0.05, "duration must equal"),
            ("bond_drives", [0.3, 0.0, 0.0], "out of bounds"),
            ("bond_drives", [0.1, 0.1, 0.0], "at most one bond"),
            ("frame_codes", [0, True, 0], "3 integers"),
            ("frame_codes", [0, 4, 0], "3 integers"),
        ):
            with self.subTest(field=field, value=value):
                answer = copy.deepcopy(self.author_answers["S1"])
                answer["pulses"][0][field] = value
                valid, score, reason = evaluate_scenario(self.scenarios["S1"], answer)
                self.assertFalse(valid)
                self.assertEqual(score, 0.0)
                self.assertIn(message, reason)

    def test_invalid_calibration_is_rejected(self) -> None:
        for value in (None, [0.0] * 8, [float("nan")] * 9, [1e308] * 9):
            with self.subTest(value=value):
                answer = dict(self.author_answers["S1"], frame_calibration=value)
                valid, score, reason = evaluate_scenario(self.scenarios["S1"], answer)
                self.assertFalse(valid)
                self.assertEqual(score, 0.0)
                self.assertIn("frame_calibration", reason)

    def test_independent_per_pulse_amplitudes_are_rejected(self) -> None:
        answer = copy.deepcopy(self.author_answers["S1"])
        pulse = answer["pulses"][0]
        active_bond = next(index for index, drive in enumerate(pulse["bond_drives"]) if drive)
        pulse["bond_drives"][active_bond] *= 0.99
        valid, score, reason = evaluate_scenario(self.scenarios["S1"], answer)
        self.assertFalse(valid)
        self.assertEqual(score, 0.0)
        self.assertIn("shared bond_amplitudes", reason)

    def test_misordered_noise_cancelling_gate_does_not_earn_completion_credit(self) -> None:
        answer = copy.deepcopy(self.author_answers["S5"])
        half = answer["pulses"][:48]
        half = half[1:] + half[:1]
        answer["pulses"] = half + half[::-1]
        result = score_submission({"S5": self.scenarios["S5"]}, {"S5": answer})
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.correct, 0)
        self.assertAlmostEqual(result.diagnostic_score, 0.2)
        self.assertIn("1-F=", result.results["S5"]["message"])

    def test_every_scenario_requires_all_three_bonds(self) -> None:
        answer = copy.deepcopy(self.author_answers["S1"])
        for pulse in answer["pulses"]:
            pulse["bond_drives"][2] = 0.0
        valid, score, reason = evaluate_scenario(self.scenarios["S1"], answer)
        self.assertFalse(valid)
        self.assertEqual(score, 0.0)
        self.assertIn("bond 31 exchange action", reason)

    def test_invalid_shared_amplitudes_are_rejected(self) -> None:
        for value in (None, [0.1, 0.1], [float("nan")] * 3, [-0.1] * 3, [0.3] * 3):
            with self.subTest(value=value):
                answer = dict(self.author_answers["S1"], bond_amplitudes=value)
                valid, score, reason = evaluate_scenario(self.scenarios["S1"], answer)
                self.assertFalse(valid)
                self.assertEqual(score, 0.0)
                self.assertIn("bond_amplitudes", reason)

    def test_noise_tensor_matches_direct_magnus_for_arbitrary_amplitudes(self) -> None:
        x = kron3(PAULIS[1], PAULIS[0], PAULIS[0])
        y = kron3(PAULIS[2], PAULIS[0], PAULIS[0])
        z = kron3(PAULIS[3], PAULIS[0], PAULIS[0])
        sequence = [[x, y], [z, x], [y, z]]
        first, second = noise_coefficients(sequence)
        amplitudes = [0.37, -0.81]
        combined = [mat_add(step[0], step[1], *amplitudes) for step in sequence]
        direct_first = mat_zero()
        direct_second = mat_zero()
        for k, current in enumerate(combined):
            direct_first = mat_add(direct_first, current)
            for earlier in combined[:k]:
                direct_second = mat_add(direct_second, mat_comm(current, earlier), 1.0, 0.5)
        tensor_first = mat_add(first[0], first[1], *amplitudes)
        tensor_second = mat_zero()
        for (a, b), value in second.items():
            tensor_second = mat_add(tensor_second, value, 1.0, amplitudes[a] * amplitudes[b])
        self.assertLess(mat_frob_norm(mat_add(direct_first, tensor_first, 1.0, -1.0)), 1e-12)
        self.assertLess(mat_frob_norm(mat_add(direct_second, tensor_second, 1.0, -1.0)), 1e-12)

    def test_cross_channel_error_is_not_masked_by_zero_diagonal_terms(self) -> None:
        x = kron3(PAULIS[1], PAULIS[0], PAULIS[0])
        y = kron3(PAULIS[2], PAULIS[0], PAULIS[0])
        _, second = noise_coefficients([[x, mat_zero()], [mat_zero(), y]])
        self.assertEqual(mat_frob_norm(second[0, 0]), 0.0)
        self.assertEqual(mat_frob_norm(second[1, 1]), 0.0)
        self.assertGreater(mat_frob_norm(second[0, 1]), 1.0)

    def test_standalone_verifier_uses_adjacent_scenarios_file(self) -> None:
        verifier_path = (
            Path(__file__).resolve().parents[2]
            / "tasks/physical_sciences/frustrated_spin_triangle_yang_baxter_holonomy"
            / "scripts/verify_outputs.py"
        )
        verifier_source = verifier_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="spin_triangle_test_") as tmpdir:
            root = Path(tmpdir)
            (root / "check.py").write_text(verifier_source, encoding="utf-8")
            (root / "scenarios.json").write_text(json.dumps(self.scenarios), encoding="utf-8")
            (root / "answers.json").write_text(json.dumps(self.author_answers), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "check.py"),
                    "--answers",
                    str(root / "answers.json"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            completed.stdout.strip().splitlines()[-1],
            "Score: 1.0000 (12/12 PASS)",
        )


if __name__ == "__main__":
    unittest.main()
