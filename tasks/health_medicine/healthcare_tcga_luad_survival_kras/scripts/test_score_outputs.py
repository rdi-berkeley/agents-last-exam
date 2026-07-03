#!/usr/bin/env python
"""Regression tests for the weighted TCGA-LUAD KRAS evaluator."""

from __future__ import annotations

import csv
import io
import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_outputs import (  # noqa: E402
    REQUIRED_FILES,
    _cox_efron,
    _logrank,
    score_submission,
)

STAGED_BASE = Path(
    "/mnt/data/agenthle/health_medicine/healthcare_tcga_luad_survival_kras/base"
)


@unittest.skipUnless(STAGED_BASE.exists(), "local staged task data is unavailable")
class WeightedEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference_kwargs = {
            "reference_cohort_csv": (
                STAGED_BASE / "reference/reference_outputs/cohort.csv"
            ).read_bytes(),
            "reference_cox_json": (
                STAGED_BASE / "reference/reference_outputs/cox_results.json"
            ).read_bytes(),
            "evaluation_contract_json": (
                STAGED_BASE / "reference/evaluation_contract.json"
            ).read_bytes(),
        }
        cls.canonical = {
            name: (STAGED_BASE / "output_test_pos" / name).read_bytes()
            for name in REQUIRED_FILES
        }

    def score(self, outputs: dict[str, bytes]):
        return score_submission(outputs, **self.reference_kwargs)

    def test_canonical_fixture_passes(self) -> None:
        result = self.score(dict(self.canonical))
        self.assertTrue(result.passed, result.to_dict())
        self.assertGreaterEqual(result.score, 0.95)
        self.assertEqual(result.sections["statistics"], 1.0)

    def test_duplicate_patient_is_hard_failure(self) -> None:
        outputs = dict(self.canonical)
        lines = outputs["cohort.csv"].decode("utf-8").splitlines()
        outputs["cohort.csv"] = ("\n".join(lines + [lines[1]]) + "\n").encode()
        result = self.score(outputs)
        self.assertTrue(result.hard_failure)
        self.assertEqual(result.score, 0.0)
        self.assertIn("duplicate patient_id", result.reasons[0])

    def test_marker_only_r_cannot_pass(self) -> None:
        outputs = dict(self.canonical)
        outputs["analysis.R"] = b"# GDCquery GDCdownload survfit coxph cox.zph\n"
        result = self.score(outputs)
        self.assertFalse(result.passed)
        self.assertLess(result.score, 0.85)
        self.assertLess(result.sections["reproducibility"], 0.5)

    def test_fake_png_is_hard_failure(self) -> None:
        outputs = dict(self.canonical)
        outputs["km_plot.png"] = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10_000
        result = self.score(outputs)
        self.assertTrue(result.hard_failure)
        self.assertEqual(result.score, 0.0)

    def test_fabricated_ph_test_cannot_pass(self) -> None:
        outputs = dict(self.canonical)
        data = json.loads(outputs["cox_results.json"])
        data["ph_test"] = {"invented": "anything", "p_value": 42}
        outputs["cox_results.json"] = json.dumps(data).encode()
        result = self.score(outputs)
        self.assertFalse(result.passed)
        self.assertLess(result.score, 0.85)
        self.assertFalse(result.details["ph_test_valid"])

    def test_internally_consistent_synthetic_expression_cannot_pass(self) -> None:
        outputs = dict(self.canonical)
        reader = csv.DictReader(io.StringIO(outputs["cohort.csv"].decode("utf-8")))
        fieldnames = reader.fieldnames or []
        csv_rows = list(reader)
        midpoint = len(csv_rows) // 2
        for index, row in enumerate(csv_rows):
            row["kras_expression_selected"] = str(1000.0 + index)
            row["kras_group"] = "high" if index >= midpoint else "low"
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
        outputs["cohort.csv"] = buffer.getvalue().encode()

        normalized = [
            {
                "patient_id": row["patient_id"],
                "sample_id": row["sample_id"][:15],
                "file_id": row["file_id"],
                "expression": float(row["kras_expression_selected"]),
                "group": row["kras_group"],
                "time_days": float(row["time_days"]),
                "status": int(float(row["status"])),
                "age": float(row["age_at_diagnosis_years"]),
                "raw_stage": row["raw_stage"],
                "stage_group": row["stage_group"],
            }
            for row in csv_rows
        ]
        statistic, p_value = _logrank(normalized)
        coefficients = _cox_efron(normalized)
        data = json.loads(outputs["cox_results.json"])
        data["cohort_summary"] = {
            "n_patients": len(normalized),
            "n_events": sum(row["status"] for row in normalized),
            "n_kras_high": sum(row["group"] == "high" for row in normalized),
            "n_kras_low": sum(row["group"] == "low" for row in normalized),
            "median_kras_expression": 1000.0 + (len(normalized) - 1) / 2,
        }
        data["log_rank"] = {"test_statistic": statistic, "p_value": p_value}
        data["cox_model"]["coefficients"] = coefficients
        outputs["cox_results.json"] = json.dumps(data).encode()

        result = self.score(outputs)
        self.assertFalse(result.passed)
        self.assertLess(result.score, 0.85)
        self.assertLess(result.details["expression_match_rate"], 0.95)

    def test_representative_real_traces_when_available(self) -> None:
        traces = {
            "cursor_opus47_near_match": Path(
                "/mnt/data/unified_logs/cursor_cli/claude-opus-4-7/"
                "bioinformatics__healthcare_tcga_luad_survival_kras/v0/"
                "20260506_102905/output"
            ),
            "opus48_live_gdc": Path(
                "/mnt/data/exp_logs/ale-claude-opus48-full-sweep/cc_opus48_low/"
                "anthropic-claude-opus-4-8/"
                "health_medicine__healthcare_tcga_luad_survival_kras/v0/"
                "20260626_015224/output"
            ),
        }
        available = {name: path for name, path in traces.items() if path.exists()}
        if not available:
            self.skipTest("representative archived traces are unavailable")
        results = {}
        for name, path in available.items():
            outputs = {
                filename: (path / filename).read_bytes() for filename in REQUIRED_FILES
            }
            results[name] = self.score(outputs)
        if "cursor_opus47_near_match" in results:
            self.assertGreaterEqual(results["cursor_opus47_near_match"].score, 0.95)
        if "opus48_live_gdc" in results:
            self.assertGreaterEqual(results["opus48_live_gdc"].score, 0.75)
            self.assertGreater(results["opus48_live_gdc"].details["patient_f1"], 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)
