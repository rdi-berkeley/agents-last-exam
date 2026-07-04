from __future__ import annotations

import csv
import io
import unittest

from score_crf_sdtm_mapping import OUTPUT_COLUMNS, score_mapping_csv


REFERENCE_ROW = {
    "crf_form": "CONCOMITANT MEDICATIONS - BASELINE (CONMED BSL)",
    "crf_field_label": "What is the medication identifier?",
    "crf_item_or_placeholder": "Sponsor-Defined Identifier",
    "sdtm_dataset": "CM",
    "sdtm_variable": "CMSPID",
    "role": "Identifier",
    "origin": "CRF",
    "mapping_rule": "Map the sponsor-defined medication identifier to CMSPID.",
    "controlled_terms_or_expected_values": "",
    "goes_to_suppqual": "NO",
    "notes": "aCRF page 15 annotates this field to CMSPID.",
}


def render_csv(row: dict[str, str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return output.getvalue()


class ScoreMappingCsvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = render_csv(REFERENCE_ROW)

    def score(self, **updates: str):
        row = {**REFERENCE_ROW, **updates}
        return score_mapping_csv(render_csv(row), self.reference, variant="base")

    def test_reference_passes(self) -> None:
        self.assertEqual(self.score().score, 1.0)

    def test_equivalent_free_text_passes(self) -> None:
        result = self.score(
            mapping_rule="Populate CMSPID directly from the sponsor identifier.",
            notes="The annotated CRF identifies the corresponding source field.",
        )
        self.assertEqual(result.score, 1.0)

    def test_structured_column_mismatch_fails(self) -> None:
        result = self.score(role="Topic")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.mismatches[0]["column"], "role")

    def test_empty_mapping_rule_fails(self) -> None:
        result = self.score(mapping_rule="")
        self.assertEqual(result.score, 0.0)
        self.assertIn("empty mapping_rule", result.errors[0])

    def test_mapping_rule_must_name_target_variable(self) -> None:
        result = self.score(mapping_rule="Map the sponsor identifier to the target field.")
        self.assertEqual(result.score, 0.0)
        self.assertIn("must name target variable", result.errors[0])

    def test_empty_notes_fail(self) -> None:
        result = self.score(notes="")
        self.assertEqual(result.score, 0.0)
        self.assertIn("empty notes", result.errors[0])


if __name__ == "__main__":
    unittest.main()
