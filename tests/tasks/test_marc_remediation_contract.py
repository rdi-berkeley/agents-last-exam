from __future__ import annotations

import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_ROOT = (
    ROOT
    / "task-data-hf/extracted/education_info/marc_remediation_folio_overlay/base/reference/evaluator"
)


@pytest.fixture
def evaluator():
    if not EVALUATOR_ROOT.exists():
        pytest.skip("MARC release data is not installed")
    spec = importlib.util.spec_from_file_location(
        "marc_contract_evaluator", EVALUATOR_ROOT / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["public_case", "hidden_river_maple", "hidden_harbor_cedar"])
def output_pair(request, tmp_path):
    if request.param == "public_case":
        reference = EVALUATOR_ROOT / "reference_outputs/public_case"
    else:
        reference = (
            EVALUATOR_ROOT / "evaluator_only/hidden_cases" / request.param / "reference_outputs"
        )
    if not reference.exists():
        pytest.skip("MARC release data is not installed")
    output = tmp_path / "output"
    shutil.copytree(reference, output)
    return output, reference


def test_reference_artifacts_pass(evaluator, output_pair):
    output, reference = output_pair
    score, checks = evaluator.compare_outputs(output, reference)
    assert score == 80, checks


def test_equivalent_formatting_and_free_prose_pass(evaluator, output_pair):
    output, reference = output_pair
    xml = output / "remediated_records.xml"
    tree = evaluator.ET.parse(xml)
    for record in tree.getroot():
        record[:] = sorted(record, key=lambda child: child.attrib.get("tag", ""), reverse=True)
        for field in record:
            if field.tag.endswith("datafield"):
                field[:] = sorted(
                    field, key=lambda child: child.attrib.get("code", ""), reverse=True
                )
    tree.write(xml, encoding="utf-8", xml_declaration=True)
    csv_path = output / "overlay_decisions.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["reason"] = "Applied the documented identifier precedence and cataloging rules."
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(reversed(rows))
    summary = output / "remediation_summary.md"
    summary.write_text(
        summary.read_text() + "\nIndependent handoff wording, not the reference paragraph.\n"
    )
    score, checks = evaluator.compare_outputs(output, reference)
    assert score == 80, checks


@pytest.mark.parametrize(
    "mutation,loss",
    [("owner", 25), ("title", 25), ("item", 25), ("rda", 25), ("reason", 15), ("summary", 5)],
)
def test_meaningful_errors_still_lose_credit(evaluator, output_pair, mutation, loss):
    output, reference = output_pair
    if mutation == "owner":
        path = output / "remediated_records.xml"
        tree = evaluator.ET.parse(path)
        field = tree.find(f".//{{{evaluator.NS}}}controlfield[@tag='003']")
        field.text = "WrongOwner"
        tree.write(path)
    elif mutation in {"title", "item", "rda"}:
        path = output / "folio_import_plan.json"
        plan = json.loads(path.read_text())
        if mutation == "title":
            plan["records"][0]["instance"]["title"] = "Wrong title"
        elif mutation == "item":
            plan["records"][0]["item"].pop("status")
        else:
            plan["records"][0]["srs"].pop("rdaConverted")
        path.write_text(json.dumps(plan))
    elif mutation == "reason":
        path = output / "overlay_decisions.csv"
        rows = evaluator.load_csv(path)
        rows[0]["reason"] = ""
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        (output / "remediation_summary.md").write_text("No meaningful handoff.")
    score, checks = evaluator.compare_outputs(output, reference)
    assert score == 80 - loss, checks


def test_local_identifier_owner_is_preserved(evaluator):
    root = EVALUATOR_ROOT / "evaluator_only/hidden_cases/hidden_harbor_cedar"
    values = []
    for path in [
        root / "inputs/legacy_records.xml",
        root / "reference_outputs/remediated_records.xml",
    ]:
        tree = evaluator.ET.parse(path)
        record = next(
            record
            for record in tree.getroot()
            if record.find(f"{{{evaluator.NS}}}controlfield[@tag='001']").text == "CEDAR202"
        )
        values.append(record.find(f"{{{evaluator.NS}}}controlfield[@tag='003']").text)
    assert values == ["LOCAL", "LOCAL"]
