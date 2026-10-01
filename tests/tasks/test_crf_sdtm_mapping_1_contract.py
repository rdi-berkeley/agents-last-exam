import csv
import importlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tasks.health_medicine.crf_sdtm_mapping_1.scripts.cm_contract import (
    CM_CONTRACT_TEXT,
    cm_source_fields,
)
from tasks.health_medicine.crf_sdtm_mapping_1.scripts.score_crf_sdtm_mapping import (
    CM_SOURCE_ALIASES,
    OUTPUT_COLUMNS,
    score_mapping_csv,
)


ROW = {
    "crf_form": "CONCOMITANT MEDICATIONS - BASELINE (CONMED BSL)",
    "crf_field_label": "Category",
    "crf_item_or_placeholder": "Category for Medication",
    "sdtm_dataset": "CM",
    "sdtm_variable": "CMCAT",
    "role": "Grouping Qualifier",
    "origin": "Assigned",
    "mapping_rule": "Assign the category to CMCAT from the form.",
    "controlled_terms_or_expected_values": "GENERAL CONCOMITANT MEDICATIONS",
    "goes_to_suppqual": "NO",
    "notes": "The Category field on aCRF page 15 maps to CMCAT; ItemDef origin is Assigned.",
}


def render(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"crf_field_label": '"Category:"'},
        {"crf_field_label": "CATEGORY [hidden]"},
        {"crf_form": "C4591001: CONCOMITANT MEDICATIONS - BASELINE (CONMED BSL) Repeating Form"},
        {"crf_field_label": "aCRF:15:2", "crf_item_or_placeholder": "aCRF:15:2"},
        {"crf_field_label": "aCRF : 15 : 2"},
        {"crf_item_or_placeholder": "[Category for Medication]"},
        {"role": "grouping    qualifier", "origin": "ASSIGNED"},
        {"controlled_terms_or_expected_values": "'general concomitant medications'"},
        {"controlled_terms_or_expected_values": '["GENERAL CONCOMITANT MEDICATIONS"]'},
        {"mapping_rule": "Populate CMCAT using the form's printed category value."},
    ],
)
def test_source_and_representation_equivalence(updates):
    assert score_mapping_csv(render([{**ROW, **updates}]), render([ROW]), variant="base").score == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"crf_field_label": "Medication"},
        {"crf_field_label": "aCRF:15:4"},
        {"crf_item_or_placeholder": "aCRF:16:2"},
        {"crf_item_or_placeholder": "aCRF:15:3"},
        {"crf_form": "CONCOMITANT MEDICATIONS - PROHIBITED (PROHIB CM)"},
        {"origin": "CRF"},
        {"origin": "Derived"},
        {"role": "Topic"},
        {"sdtm_dataset": "SUPPCM"},
        {"sdtm_variable": "CMSCAT"},
        {"goes_to_suppqual": "YES"},
        {"controlled_terms_or_expected_values": "VACCINATIONS"},
        {"controlled_terms_or_expected_values": "[]"},
        {"controlled_terms_or_expected_values": "[1]"},
        {"controlled_terms_or_expected_values": '["GENERAL CONCOMITANT MEDICATIONS", null]'},
        {"controlled_terms_or_expected_values": '{"a":"GENERAL CONCOMITANT MEDICATIONS"}'},
        {"controlled_terms_or_expected_values": "GENERAL CONCOMITANT MEDICATIONS;"},
        {
            "controlled_terms_or_expected_values": "GENERAL CONCOMITANT MEDICATIONS; general concomitant medications"
        },
        {
            "controlled_terms_or_expected_values": '["GENERAL CONCOMITANT MEDICATIONS", "general concomitant medications"]'
        },
        {"controlled_terms_or_expected_values": '["GENERAL CONCOMITANT MEDICATIONS"'},
        {"mapping_rule": "This is CMCATX, a different variable."},
        {"notes": ""},
    ],
)
def test_semantic_errors_still_fail_binary(updates):
    assert score_mapping_csv(render([{**ROW, **updates}]), render([ROW]), variant="base").score == 0


@pytest.mark.parametrize(
    "value",
    ["B; A", "'B'; 'A'", '"B"; "A"', '["b", "a"]', " b ; a "],
)
def test_term_sets_are_order_independent(value):
    reference = {**ROW, "controlled_terms_or_expected_values": "A; B"}
    agent = {**reference, "controlled_terms_or_expected_values": value}
    assert score_mapping_csv(render([agent]), render([reference]), variant="base").score == 1


@pytest.mark.parametrize("reference_value", ["CL.UNIT", "CL.FREQ", "CL.ROUTE", "CL.WHO DDE"])
def test_code_list_identity_preserved(reference_value):
    reference = {**ROW, "controlled_terms_or_expected_values": reference_value}
    for value, expected in [
        (reference_value, 1),
        (f"'{reference_value}'", 1),
        (reference_value.lower(), 0),
        ("Unit terms", 0),
    ]:
        candidate = {**reference, "controlled_terms_or_expected_values": value}
        assert (
            score_mapping_csv(render([candidate]), render([reference]), variant="base").score
            == expected
        )


@pytest.mark.parametrize(
    "date", ["ISO 8601 datetime", "ISO 8601 date/time", "'ISO 8601 date/datetime'"]
)
def test_date_representation(date):
    reference = {**ROW, "controlled_terms_or_expected_values": "ISO 8601 date/datetime"}
    candidate = {**reference, "controlled_terms_or_expected_values": date}
    assert score_mapping_csv(render([candidate]), render([reference]), variant="base").score == 1


def test_source_locator_covers_every_distinct_field_without_collisions():
    fields = list(cm_source_fields())
    assert len(CM_SOURCE_ALIASES) == 2 * len(fields)
    assert len({source["locator"] for source in fields}) == len(fields)
    for source in fields:
        reference = {
            **ROW,
            "crf_form": source["form"],
            "crf_field_label": source["label"],
            "crf_item_or_placeholder": source["placeholder"],
        }
        candidate = {
            **reference,
            "crf_field_label": source["locator"],
            "crf_item_or_placeholder": source["locator"],
        }
        assert (
            score_mapping_csv(render([candidate]), render([reference]), variant="base").score == 1
        )


def test_row_order_bom_quoting_and_duplicate_detection():
    another = {**ROW, "sdtm_variable": "CMSCAT", "mapping_rule": "Assign CMSCAT from the header."}
    reference = render([ROW, another])
    assert (
        score_mapping_csv("\ufeff" + render([another, ROW]), reference, variant="base").score == 1
    )
    duplicate = {**ROW, "crf_field_label": "aCRF:15:2"}
    result = score_mapping_csv(render([ROW, duplicate]), reference, variant="base")
    assert result.score == 0
    assert any("duplicate composite key" in message for message in result.errors)
    assert score_mapping_csv(render([ROW]), reference, variant="base").score == 0


@pytest.mark.parametrize("bad", ["missing", "extra", "unclosed_quote", "header"])
def test_malformed_csv_rejected(bad):
    raw = render([ROW])
    if bad == "missing":
        raw = raw.rsplit(",", 1)[0] + "\n"
    elif bad == "extra":
        raw = raw.rstrip("\n") + ",extra\n"
    elif bad == "unclosed_quote":
        raw = raw.split("\n", 1)[0] + '\n"unterminated\n'
    else:
        raw = raw.replace("crf_form,crf_field_label", "crf_field_label,crf_form", 1)
    assert score_mapping_csv(raw, render([ROW]), variant="base").score == 0


def test_main_uses_own_scorer_after_crf4_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tasks.health_medicine.crf_sdtm_mapping_4.main; import tasks.health_medicine.crf_sdtm_mapping_1.main as task; print(task.score_mapping_csv.__module__)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().endswith("crf_sdtm_mapping_1.scripts.score_crf_sdtm_mapping")


def test_standalone_scorer_outside_repository(tmp_path):
    reference = tmp_path / "reference.csv"
    reference.write_text(render([ROW]))
    module = importlib.import_module(
        "tasks.health_medicine.crf_sdtm_mapping_1.scripts.score_crf_sdtm_mapping"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-E",
            str(Path(module.__file__)),
            "--agent",
            str(reference),
            "--reference",
            str(reference),
            "--variant",
            "base",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["score"] == 1


def test_public_base_contract_does_not_change_other_variants():
    task = importlib.import_module("tasks.health_medicine.crf_sdtm_mapping_1.main")
    base = task._config_from_variant(task.VARIANTS[0]).task_description
    assert CM_CONTRACT_TEXT in base
    assert "Copy CRF field labels and annotated item/placeholder text verbatim" not in base
    for spec in task.VARIANTS[1:]:
        other = task._config_from_variant(spec).task_description
        assert CM_CONTRACT_TEXT not in other
        assert "Copy CRF field labels and annotated item/placeholder text verbatim" in other
    assert json.loads(json.dumps({"contract": CM_CONTRACT_TEXT}))["contract"] == CM_CONTRACT_TEXT
