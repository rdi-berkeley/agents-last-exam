from __future__ import annotations

import csv
import io
import json
import importlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tasks.health_medicine.crf_sdtm_mapping_4.scripts import score_crf_sdtm_mapping as scorer


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "task-data-hf/extracted/health_medicine/crf_sdtm_mapping_4/base"
ROW = {
    "crf_form": "ADVERSE EVENT REPORT (AE)",
    "crf_field_label": "Adverse Event",
    "crf_item_or_placeholder": "aCRF:11:3",
    "sdtm_dataset": "AE",
    "sdtm_variable": "AETERM",
    "role": "Topic",
    "origin": "CRF/aCRF",
    "mapping_rule": "Map the verbatim adverse event term to AETERM.",
    "controlled_terms_or_expected_values": "Free text",
    "goes_to_suppqual": "NO",
    "notes": "If possible specify diagnosis, not individual symptoms.",
}


def render(rows: list[dict[str, str]], variant: str = "base") -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=scorer.VARIANT_SPECS[variant]["columns"])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def score_row(**updates: str) -> scorer.ScoreResult:
    return scorer.score_mapping_csv(render([{**ROW, **updates}]), render([ROW]), variant="base")


@pytest.mark.parametrize(
    "rule",
    [
        "Copy the verbatim adverse event term to AETERM without recoding it.",
        "Populate AETERM directly from the reported event.",
        "Write the reported diagnosis into `aeterm`.",
        "AETERM receives the original event description, not a coded term.",
        "Use the verbatim term recorded on the adverse event form for AETERM.",
    ],
)
@pytest.mark.parametrize(
    "notes",
    [
        "Prefer a diagnosis over a list of symptoms when available.",
        "Record individual symptoms only when a diagnosis is unavailable.",
        "Use the diagnosis if known; otherwise retain the symptoms.",
    ],
)
def test_equivalent_prose_has_full_strict_credit(rule: str, notes: str) -> None:
    result = score_row(mapping_rule=rule, notes=notes)
    assert result.score == result.strict_score == 1.0, result.to_dict()
    assert not result.mismatches


@pytest.mark.parametrize(
    "column,value",
    [
        ("crf_form", "WRONG FORM"),
        ("crf_field_label", "Start Date Time"),
        ("crf_item_or_placeholder", "[Start Date]"),
        ("sdtm_dataset", "SUPPAE"),
        ("sdtm_variable", "AESTDTC"),
        ("role", "Timing"),
        ("origin", "Assigned"),
        ("controlled_terms_or_expected_values", "YES; NO"),
        ("goes_to_suppqual", "YES"),
    ],
)
def test_every_structured_column_still_counts(column: str, value: str) -> None:
    result = score_row(
        **{column: value}, mapping_rule="Copy the original event description to AETERM."
    )
    assert result.strict_score == 0.0
    if column in {"sdtm_dataset", "sdtm_variable", "goes_to_suppqual"}:
        assert result.score == 0.0
    else:
        assert result.score == round(10 / 11, 4)
        assert result.relaxed_matches[0]["mismatched_columns"] == [column]


@pytest.mark.parametrize(
    "rule",
    [
        "",
        " \n\t",
        "TODO",
        "N/A",
        "not applicable",
        "AETERM",
        "12345",
        "Populate AESTDTC directly from the reported event.",
        "Map the event into aeterm_wrong.",
        "Set PRE_AETERM to the source value and write it into AETERM_POST.",
    ],
)
def test_missing_or_incorrect_rule_does_not_earn_prose_credit(rule: str) -> None:
    result = score_row(mapping_rule=rule)
    assert result.score == round(10 / 11, 4)
    assert result.strict_score == 0.0
    assert result.relaxed_matches[0]["mismatched_columns"] == ["mapping_rule"]


@pytest.mark.parametrize("notes", ["TODO", "TBD", "12345", "!!!"])
def test_missing_or_placeholder_notes_lose_credit(notes: str) -> None:
    result = score_row(notes=notes)
    assert result.score == round(10 / 11, 4)
    assert result.mismatches[0]["column"] == "notes"


@pytest.mark.parametrize("expected", ["", "The form requests a diagnosis where possible."])
@pytest.mark.parametrize("observed", ["", " \n\t", "Use the recorded diagnosis."])
def test_notes_presence_is_independent_of_gold(expected: str, observed: str) -> None:
    result = scorer.score_mapping_csv(
        render([{**ROW, "notes": observed}]), render([{**ROW, "notes": expected}]), variant="base"
    )
    assert result.score == result.strict_score == 1.0


def test_rules_can_reference_another_variable_as_a_source() -> None:
    reference_row = {
        **ROW,
        "sdtm_variable": "AEENDTC",
        "mapping_rule": "For same-day events, copy the start date to the end date.",
    }
    candidate = {
        **reference_row,
        "mapping_rule": "Set AEENDTC to AESTDTC when the event ends at its start time.",
    }
    result = scorer.score_mapping_csv(render([candidate]), render([reference_row]), variant="base")
    assert result.score == result.strict_score == 1.0


@pytest.mark.parametrize("column", scorer.KEY_COLUMNS)
def test_empty_composite_key_is_structurally_invalid(column: str) -> None:
    result = score_row(**{column: " \t"})
    assert result.score == 0.0
    assert any("empty composite-key" in error for error in result.errors)


@pytest.mark.parametrize("dataset,flag", [("DM", "NO"), ("AE", "Y"), ("SUPPAE", "NO")])
def test_invalid_dataset_or_flag_fails(dataset: str, flag: str) -> None:
    result = score_row(sdtm_dataset=dataset, goes_to_suppqual=flag)
    assert result.score == 0.0
    assert result.errors


@pytest.mark.parametrize(
    "damage", ["empty", "header", "duplicate_header", "short", "extra", "quote"]
)
def test_malformed_csv_cannot_gain_credit(damage: str) -> None:
    table = list(csv.reader(io.StringIO(render([ROW]))))
    if damage == "empty":
        table = table[:1]
    elif damage == "header":
        table[0].reverse()
    elif damage == "duplicate_header":
        table[0][-1] = table[0][0]
    elif damage == "short":
        table[1].pop()
    elif damage == "extra":
        table[1].append("unheaded")
    output = io.StringIO()
    csv.writer(output).writerows(table)
    candidate = output.getvalue()
    if damage == "quote":
        candidate += '"unterminated'
    result = scorer.score_mapping_csv(candidate, render([ROW]), variant="base")
    assert result.score == 0.0
    assert result.errors


def test_duplicate_composite_key_fails() -> None:
    result = scorer.score_mapping_csv(render([ROW, ROW]), render([ROW]), variant="base")
    assert result.score == 0.0
    assert any("duplicate composite key" in error for error in result.errors)


def test_missing_and_extra_rows_reduce_coverage() -> None:
    other = {
        **ROW,
        "sdtm_variable": "AESPID",
        "crf_item_or_placeholder": "aCRF:11:2",
        "mapping_rule": "Map the source identifier to AESPID.",
    }
    for candidate, reference in [
        (render([ROW]), render([ROW, other])),
        (render([ROW, other]), render([ROW])),
    ]:
        result = scorer.score_mapping_csv(candidate, reference, variant="base")
        assert result.score == result.row_coverage == 0.5
        assert result.missing_keys or result.extra_keys


@pytest.mark.parametrize("variable", ["QVAL", "AEWRONG"])
@pytest.mark.parametrize("placeholder", ["QNAM='AECMGIV'", "QNAM='AECMGIV_WRONG'"])
def test_supplemental_fallback_is_limited_to_qval_and_whole_qnam(
    variable: str, placeholder: str
) -> None:
    reference_row = {
        **ROW,
        "sdtm_dataset": "SUPPAE",
        "sdtm_variable": "AECMGIV",
        "goes_to_suppqual": "YES",
        "mapping_rule": "Store the concomitant medication given flag in AECMGIV.",
    }
    candidate = {**reference_row, "sdtm_variable": variable, "crf_item_or_placeholder": placeholder}
    result = scorer.score_mapping_csv(render([candidate]), render([reference_row]), variant="base")
    expected = round(9 / 11, 4) if variable == "QVAL" and placeholder == "QNAM='AECMGIV'" else 0.0
    assert result.score == expected
    assert result.strict_score == 0.0


@pytest.mark.parametrize("flag", ["CRF", "DERIVED", "ASSIGNED"])
def test_dm_variant_preserves_its_flag_contract(flag: str) -> None:
    reference_row = {
        **ROW,
        "sdtm_dataset": "DM",
        "sdtm_variable": "SEX",
        "origin": "CRF",
        "derived_or_assigned": flag,
        "mapping_rule": "Map collected sex into SEX.",
    }
    del reference_row["goes_to_suppqual"]
    candidate = {**reference_row, "mapping_rule": "Populate SEX from the recorded sex."}
    result = scorer.score_mapping_csv(
        render([candidate], "dm"), render([reference_row], "dm"), variant="dm"
    )
    assert result.score == result.strict_score == 1.0
    candidate["derived_or_assigned"] = "YES"
    assert (
        scorer.score_mapping_csv(
            render([candidate], "dm"), render([reference_row], "dm"), variant="dm"
        ).score
        == 0.0
    )
    candidate["derived_or_assigned"] = flag
    candidate["mapping_rule"] = "Map the collected sex directly."
    result = scorer.score_mapping_csv(
        render([candidate], "dm"), render([reference_row], "dm"), variant="dm"
    )
    assert result.score == 1.0


@pytest.fixture
def official_reference() -> str:
    path = DATA / "reference/ae_mapping.csv"
    if not path.exists():
        pytest.skip("CRF4 release data is not installed")
    return path.read_text(encoding="utf-8-sig")


def test_official_reference_and_visible_example_remain_valid(official_reference: str) -> None:
    contract = json.loads((DATA / "input/output_contract.json").read_text())
    assert contract["columns"] == scorer.VARIANT_SPECS["base"]["columns"]
    assert contract["composite_key"] == scorer.KEY_COLUMNS
    assert contract["allowed_sdtm_dataset"] == list(
        scorer.VARIANT_SPECS["base"]["allowed_datasets"]
    )
    example = {key: value for key, value in contract["example_row"].items() if key != "_comment"}
    for reference in (official_reference, render([example])):
        result = scorer.score_mapping_csv(reference, reference, variant="base")
        assert result.score == result.strict_score == 1.0, result.to_dict()


def test_all_official_rows_accept_prose_rewrites_and_whitespace(official_reference: str) -> None:
    rows = list(csv.DictReader(io.StringIO(official_reference)))
    for row in rows:
        row["mapping_rule"] = (
            f"For {row['sdtm_variable']}, apply this mapping: {row['mapping_rule']}"
        )
        row["notes"] = (
            f"Source annotation context: {row['notes']}"
            if row["notes"]
            else "No additional annotation."
        )
        for column in row:
            row[column] = " \n" + row[column].replace(" ", "  ") + "\t "
    result = scorer.score_mapping_csv(
        "\ufeff" + render(rows[::-1]), official_reference, variant="base"
    )
    assert result.score == result.strict_score == 1.0, result.to_dict()
    assert result.compared_rows == 38


@pytest.mark.parametrize("column", ["origin", "role", "controlled_terms_or_expected_values"])
def test_no_official_business_column_is_excluded(official_reference: str, column: str) -> None:
    rows = list(csv.DictReader(io.StringIO(official_reference)))
    for row in rows:
        row[column] = "INCORRECT"
    result = scorer.score_mapping_csv(render(rows), official_reference, variant="base")
    assert result.score == round(10 / 11, 4)
    assert result.strict_score == 0.0
    assert all(match["mismatched_columns"] == [column] for match in result.relaxed_matches)


def test_unknown_variant_fails() -> None:
    result = scorer.score_mapping_csv(render([ROW]), render([ROW]), variant="unknown")
    assert result.score == 0.0
    assert result.errors == ["unknown variant 'unknown'"]


@pytest.mark.parametrize("candidate", ["Y; N", "n;y", ' ["Y", "N"] ', "Y/N", "N/Y"])
def test_encoded_boolean_sets_are_semantically_normalized(candidate: str) -> None:
    reference = {**ROW, "controlled_terms_or_expected_values": "N; Y"}
    result = scorer.score_mapping_csv(
        render([{**reference, "controlled_terms_or_expected_values": candidate}]),
        render([reference]),
        variant="base",
    )
    assert result.score == result.strict_score == 1.0


@pytest.mark.parametrize(
    "candidate",
    [
        "YES; NO",
        "Y",
        "N; Y; UNKNOWN",
        "Y; Y; N",
        "N;;Y",
        '["Y", null]',
        "[1, 0]",
        "{}",
        '["Y", "Y", "N"]',
        '["Y", "N"',
    ],
)
def test_wrong_incomplete_or_malformed_encoded_sets_lose_credit(candidate: str) -> None:
    reference = {**ROW, "controlled_terms_or_expected_values": "N; Y"}
    result = scorer.score_mapping_csv(
        render([{**reference, "controlled_terms_or_expected_values": candidate}]),
        render([reference]),
        variant="base",
    )
    assert result.score == round(10 / 11, 4)


@pytest.mark.parametrize("candidate", ["CRF", "crf", "CRF/aCRF", "aCRF"])
def test_crf_origin_aliases_never_become_assigned(candidate: str) -> None:
    assert score_row(origin=candidate).score == 1.0
    reference = {**ROW, "origin": "Assigned"}
    result = scorer.score_mapping_csv(
        render([{**reference, "origin": candidate}]), render([reference]), variant="base"
    )
    assert result.score == round(10 / 11, 4)


def test_source_locator_sets_and_labels_preserve_identity() -> None:
    reference = {
        **ROW,
        "crf_item_or_placeholder": "aCRF:12:11;aCRF:12:12",
        "crf_field_label": "Medication given? || Non-drug treatment given?",
    }
    candidate = {
        **reference,
        "crf_item_or_placeholder": "ACRF:12:12; acrf:12:11",
        "crf_field_label": "Non-drug treatment given: || MEDICATION GIVEN",
    }
    result = scorer.score_mapping_csv(render([candidate]), render([reference]), variant="base")
    assert result.score == result.strict_score == 1.0
    for wrong in ["aCRF:12:11", "aCRF:12:11;aCRF:12:13", "a brief invented description"]:
        candidate["crf_item_or_placeholder"] = wrong
        result = scorer.score_mapping_csv(render([candidate]), render([reference]), variant="base")
        assert result.score == round(10 / 11, 4)


@pytest.mark.parametrize(
    "label",
    [
        "Serious Adverse Event Number",
        "Serious Adverse Event Number: For Pfizer Use Only",
        "SERIOUS ADVERSE EVENT NUMBER (for Pfizer use only)",
        "Seri ous Adverse Event Number: For\nPfizer Use Only",
    ],
)
def test_verified_administrative_suffix_is_not_part_of_field_identity(label: str) -> None:
    reference = {**ROW, "crf_field_label": "Serious Adverse Event Number: For Pfizer Use Only"}
    candidate = {**reference, "crf_field_label": label}
    for observed, expected in [(candidate, reference), (reference, candidate)]:
        result = scorer.score_mapping_csv(render([observed]), render([expected]), variant="base")
        assert result.score == result.strict_score == 1.0


@pytest.mark.parametrize(
    "label",
    [
        "Adverse Event Number",
        "Serious Adverse Event",
        "Comparison Term: For Pfizer Use Only",
        "Serious Adverse Event Number: For Other Sponsor Use Only",
        "Serious Adverse Event Number: Not For Pfizer Use Only",
        "For Pfizer Use Only: Serious Adverse Event Number",
        "Serious Adverse Event Number: For Pfizer Use Only WRONG",
        "For Pfizer Use Only",
        "",
        "Serious Adverse Event Number || For Pfizer Use Only",
    ],
)
def test_administrative_suffix_does_not_hide_wrong_or_empty_labels(label: str) -> None:
    reference = {**ROW, "crf_field_label": "Serious Adverse Event Number: For Pfizer Use Only"}
    result = scorer.score_mapping_csv(
        render([{**reference, "crf_field_label": label}]), render([reference]), variant="base"
    )
    assert result.score == round(10 / 11, 4)
    assert result.mismatches[0]["column"] == "crf_field_label"


def test_administrative_suffix_normalization_is_label_only_and_preserves_label_sets() -> None:
    reference = {
        **ROW,
        "crf_field_label": "Serious Adverse Event Number: For Pfizer Use Only || AE ID",
    }
    candidate = {**reference, "crf_field_label": "AE ID || Serious Adverse Event Number"}
    assert (
        scorer.score_mapping_csv(render([candidate]), render([reference]), variant="base").score
        == 1.0
    )
    assert score_row(crf_form=ROW["crf_form"] + ": For Pfizer Use Only").score == round(10 / 11, 4)
    assert score_row(
        controlled_terms_or_expected_values="Free text: For Pfizer Use Only"
    ).score == round(10 / 11, 4)
    reference = {**reference, "sdtm_dataset": "DM", "derived_or_assigned": "CRF"}
    del reference["goes_to_suppqual"]
    candidate = {**reference, "crf_field_label": "Serious Adverse Event Number || AE ID"}
    result = scorer.score_mapping_csv(
        render([candidate], "dm"), render([reference], "dm"), variant="dm"
    )
    assert result.score == round(10 / 11, 4)


def test_administrative_suffix_is_supported_by_both_visible_pdfs(official_reference: str) -> None:
    fitz = pytest.importorskip("fitz")
    for filename, page_index in [("annotated_crf.pdf", 11), ("sample_crf.pdf", 2)]:
        with fitz.open(DATA / "input/source_documents" / filename) as document:
            blocks = document[page_index].get_text("blocks")
            question = next(block[4] for block in blocks if "Pfizer Use Only" in block[4])
            compact = "".join(question.split()).replace("Serous", "Serious")
            assert "15.SeriousAdverseEventNumber:ForPfizerUseOnly" in compact
            assert any(
                "[SeriousAdverseEventNumber]" in "".join(block[4].split()) for block in blocks
            )


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("MedDRA lowest level term", "MedDRA LLT"),
        ("MedDRA preferred term", "MedDRA PT"),
        ("MedDRA high level term", "MedDRA HLT"),
        ("MedDRA high level group term", "MedDRA HLGT"),
        ("MedDRA primary system organ class", "MedDRA SOC"),
        ("MedDRA lowest level term code", "MedDRA LLT code"),
        ("MedDRA high level term code", "MedDRA HLT code"),
        ("MedDRA high level group term code", "MedDRA HLGT code"),
        ("ISO 8601 date/time", "ISO 8601 datetime"),
    ],
)
def test_only_finite_verified_type_aliases_match(candidate: str, expected: str) -> None:
    reference = {**ROW, "controlled_terms_or_expected_values": expected}
    result = scorer.score_mapping_csv(
        render([{**reference, "controlled_terms_or_expected_values": candidate}]),
        render([reference]),
        variant="base",
    )
    assert result.score == 1.0
    for wrong in ["MedDRA OTHER", "Some equivalent coding scheme", "SNOMED", "Free text"]:
        result = scorer.score_mapping_csv(
            render([{**reference, "controlled_terms_or_expected_values": wrong}]),
            render([reference]),
            variant="base",
        )
        assert result.score == round(10 / 11, 4)


def test_every_source_derived_target_origin_and_codelist(official_reference: str) -> None:
    namespace = {"o": "http://www.cdisc.org/ns/odm/v1.3", "d": "http://www.cdisc.org/ns/def/v2.0"}
    root = ET.parse(DATA / "input/source_documents/sdtm_define.xml")
    items = {item.get("OID"): item for item in root.findall(".//o:ItemDef", namespace)}
    lists = {item.get("OID"): item for item in root.findall(".//o:CodeList", namespace)}
    rows = list(csv.DictReader(io.StringIO(official_reference)))
    for row in rows:
        variable = row["sdtm_variable"]
        if row["sdtm_dataset"] == "SUPPAE":
            clauses = [
                clause.get("OID")
                for clause in root.findall(".//d:WhereClauseDef", namespace)
                if variable
                in [value.text for value in clause.findall(".//o:CheckValue", namespace)]
            ]
            refs = root.findall('.//d:ValueListDef[@OID="VL.SUPPAE.QVAL"]/o:ItemRef', namespace)
            candidates = [
                items[ref.get("ItemOID")]
                for ref in refs
                if any(
                    clause.get("WhereClauseOID") in clauses
                    for clause in ref.findall("d:WhereClauseRef", namespace)
                )
            ]
            assert len(candidates) == 1
            item = candidates[0]
        else:
            item = items[f"IT.AE.{variable}"]
        origin = item.find("d:Origin", namespace)
        if origin is None:
            assert variable in {"AESTDTC", "AEENDTC"}
            candidates = [
                value
                for key, value in items.items()
                if key.startswith(f"IT.AE.{variable}.AE.AECAT.NE.")
            ]
            assert len(candidates) == 1
            item = candidates[0]
            origin = item.find("d:Origin", namespace)
        assert row["origin"] == origin.get("Type"), variable
        ref = item.find("o:CodeListRef", namespace)
        if ref is not None and ref.get("CodeListOID") != "CL.MedDRA":
            values = {
                entry.get("CodedValue")
                for entry in lists[ref.get("CodeListOID")]
                if entry.get("CodedValue")
            }
            expected = {
                part.strip() for part in row["controlled_terms_or_expected_values"].split(";")
            }
            if variable == "AECAT":
                assert expected == {"ADVERSE EVENT"} <= values
            elif variable == "AEACN":
                assert expected == {"DRUG WITHDRAWN", "NOT APPLICABLE"} <= values
            else:
                assert expected == values, variable


def test_confirmed_neighboring_annotation_and_source_corrections(official_reference: str) -> None:
    rows = {row["sdtm_variable"]: row for row in csv.DictReader(io.StringIO(official_reference))}
    assert rows["AECONTRT"]["crf_item_or_placeholder"] == "aCRF:12:11;aCRF:12:12"
    assert rows["AECONTRT"]["controlled_terms_or_expected_values"] == "N; Y"
    assert set(rows["AERELNST"]["controlled_terms_or_expected_values"].split("; ")) == {
        "CONCOMITANT DRUG TREATMENT",
        "CONCOMITANT NON-DRUG TREATMENT",
        "OTHER",
    }
    assert rows["AEREFID"]["crf_item_or_placeholder"] == "aCRF:12:15"
    assert rows["AEREFID"]["controlled_terms_or_expected_values"] == "Free text"
    assert "not submit" not in rows["AEREFID"]["mapping_rule"].lower()
    fitz = pytest.importorskip("fitz")
    with fitz.open(DATA / "input/source_documents/annotated_crf.pdf") as document:
        blocks = document[11].get_text("blocks")
        reference_id = next(block for block in blocks if block[4].strip() == "AEREFID")
        not_submitted = next(block for block in blocks if block[4].strip() == "NOT SUBMITTED")
        comparison = next(block for block in blocks if "16. Comparison Term" in block[4])
        assert reference_id[3] < comparison[1]
        assert comparison[1] < not_submitted[1] < comparison[3]


def test_runtime_uses_own_scorer_and_publishes_revision() -> None:
    main = importlib.import_module("tasks.health_medicine.crf_sdtm_mapping_4.main")
    assert main.score_mapping_csv is scorer.score_mapping_csv
    description = main._config_from_variant(main.VARIANTS[0]).task_description
    assert "20260907-source-verified" in description
    assert "Notes are optional" in description
    assert "must name their target variable" in description
    card = json.loads(
        (ROOT / "tasks/health_medicine/crf_sdtm_mapping_4/task_card.json").read_text()
    )
    assert "20260907-source-verified" in card["taskPrompt"]
    assert "Proportional CSV evaluation" in card["evaluation"]


def test_no_reuse_of_one_supplemental_row_for_multiple_targets() -> None:
    first = {
        **ROW,
        "sdtm_dataset": "SUPPAE",
        "sdtm_variable": "AECMGIV",
        "goes_to_suppqual": "YES",
        "mapping_rule": "Store the response in AECMGIV.",
    }
    second = {**first, "sdtm_variable": "AENDGIV", "mapping_rule": "Store the response in AENDGIV."}
    candidate = {
        **first,
        "sdtm_variable": "QVAL",
        "crf_item_or_placeholder": "AECMGIV AENDGIV",
        "mapping_rule": "Store AECMGIV and AENDGIV in QVAL.",
    }
    result = scorer.score_mapping_csv(render([candidate]), render([first, second]), variant="base")
    assert result.row_coverage == 0.5
    assert result.score == round(0.5 * 9 / 11, 4)


def test_staged_revision_has_reproducible_hashes_and_preserved_sources(
    official_reference: str, tmp_path: Path
) -> None:
    import hashlib

    from tasks.health_medicine.crf_sdtm_mapping_4.scripts.build_fair_contract import build

    workspace = ROOT.parent / "task-fairness-gpt6-20260907"
    before = workspace / "data-before/health_medicine/crf_sdtm_mapping_4/base"
    override = workspace / "data-overrides/health_medicine/crf_sdtm_mapping_4/base"
    if not before.exists():
        pytest.skip("Original release backup is not installed")
    build(before, tmp_path)
    manifest = json.loads((DATA / "reference/source_manifest.json").read_text())
    for relative, expected in manifest["active_revision"]["original_file_hashes"].items():
        assert hashlib.sha256((before / relative).read_bytes()).hexdigest() == expected
    for relative in [
        "input/output_contract.json",
        "input/task_brief.md",
        "reference/ae_mapping.csv",
        "reference/source_manifest.json",
    ]:
        assert (
            (DATA / relative).read_bytes()
            == (override / relative).read_bytes()
            == (tmp_path / relative).read_bytes()
        )
    for relative, expected in manifest["active_revision"]["generated_file_hashes"].items():
        assert hashlib.sha256((DATA / relative).read_bytes()).hexdigest() == expected
    for filename, source in manifest["raw_files"].items():
        relative = (
            f"reference/{filename}"
            if filename.endswith(".csv")
            else f"input/source_documents/{filename}"
        )
        assert hashlib.sha256((before / relative).read_bytes()).hexdigest() == source["sha256"]
        if filename.endswith(".csv"):
            assert (DATA / relative).read_bytes() != (before / relative).read_bytes()
        else:
            assert (DATA / relative).read_bytes() == (before / relative).read_bytes()
