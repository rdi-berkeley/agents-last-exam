from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[2]
TASK = "education_info/marc_remediation_folio_overlay"
DATA = ROOT / "task-data-hf/extracted" / TASK / "base"
NS = "{http://www.loc.gov/MARC21/slim}"


@pytest.fixture
def evaluator():
    if not DATA.exists():
        pytest.skip("MARC release data is not installed")
    spec = importlib.util.spec_from_file_location(
        "marc_post_validation", DATA / "reference/evaluator/evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["public_case", "hidden_river_maple", "hidden_harbor_cedar"])
def case(request, evaluator):
    if request.param == "public_case":
        return (
            DATA / "input/starter_project/input/public_case",
            DATA / "reference/evaluator/reference_outputs/public_case",
        )
    base = DATA / "reference/evaluator/evaluator_only/hidden_cases" / request.param
    return base / "inputs", base / "reference_outputs"


def test_every_reference_record_follows_published_carrier_precedence(case):
    case_input, reference = case
    source = list(ET.parse(case_input / "legacy_records.xml").getroot())
    with (reference / "overlay_decisions.csv").open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    active = []
    for record, decision in zip(source, decisions, strict=True):
        assert record.find(f"{NS}controlfield[@tag='001']").text == decision["record_id"]
        if decision["action"] != "suppress_duplicate":
            active.append(record)
    actual = list(ET.parse(reference / "remediated_records.xml").getroot())
    assert len(active) == len(actual)
    conflicts = []
    for original, expected in zip(active, actual, strict=True):
        online_location = any(
            (field.text or "").strip().upper() == "ONLINE"
            for field in original.findall(f"{NS}datafield[@tag='949']/{NS}subfield[@code='a']")
        )
        online_description = any(
            "online resource" in (field.text or "").lower()
            for field in original.findall(f"{NS}datafield[@tag='300']/{NS}subfield[@code='a']")
        )
        if online_location and not online_description:
            conflicts.append(original.find(f"{NS}controlfield[@tag='001']").text)
        online = online_location or online_description
        fields = [
            ("336", "text", "txt", "rdacontent"),
            ("337", "computer" if online else "unmediated", "c" if online else "n", "rdamedia"),
            (
                "338",
                "online resource" if online else "volume",
                "cr" if online else "nc",
                "rdacarrier",
            ),
        ]
        for tag, term, code, vocabulary in fields:
            field = expected.find(f"{NS}datafield[@tag='{tag}']")
            assert {subfield.get("code"): subfield.text for subfield in field} == {
                "a": term,
                "b": code,
                "2": vocabulary,
            }
    assert len(conflicts) == (14 if case_input.name == "public_case" else 0)
    print(
        json.dumps(
            {
                "marc_reference": case_input.parent.name
                if case_input.name == "inputs"
                else case_input.name,
                "active_records": len(active),
                "conflicts": conflicts,
            }
        )
    )


def test_staged_case_policies_publish_one_consistent_rule():
    release = ROOT.parent / "release-staging/ale-task-fixes-20260913/post-validation-20260918"
    overlay = release / "data-overrides" / TASK / "base"
    if not overlay.exists():
        pytest.skip("MARC post-validation overlay is not installed")
    policy = (ROOT / "tasks" / TASK / "cataloging_policy.md").read_bytes()
    policies = list(overlay.rglob("cataloging_policy.md"))
    assert len(policies) == 3
    assert all(path.read_bytes() == policy for path in policies)
    assert b"even when `300$a`" in policy
    assert b"$2 rdacontent" in policy
    contract = (overlay / "input/starter_project/OUTPUT_CONTRACT.md").read_text()
    assert "`949$a=ONLINE` takes precedence" in contract


@pytest.mark.parametrize("label,expected", [("kimi_after", 55), ("codex_after_repair", 80)])
def test_actual_cli_replay_and_bounded_rda_controls(evaluator, case, tmp_path, label, expected):
    release = ROOT.parent / "release-staging/ale-task-fixes-20260913"
    selection = release / "validation-queue-20260917/comparison.json"
    if not selection.exists():
        pytest.skip("Selected validation artifacts are not installed")
    row = next(row for row in json.loads(selection.read_text())["rows"] if row["task"] == TASK)
    run = Path(row[label]["run_json"])
    assert hashlib.sha256(run.read_bytes()).hexdigest() == row[label]["sha256"]
    original = run.parent / "output/submission"
    submission = tmp_path / "submission"
    (submission / "scripts").mkdir(parents=True)
    for relative in ("scripts/remediate_catalog.py", "catalog_remediation_lib.py"):
        shutil.copyfile(original / relative, submission / relative)
    case_input, reference = case
    output = tmp_path / "actual"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            str(submission / "scripts/remediate_catalog.py"),
            "--case-dir",
            str(case_input),
            "--output-dir",
            str(output),
        ],
        cwd=submission,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    score, checks = evaluator.compare_outputs(output, reference)
    assert score == expected, checks
    if case_input.name == "public_case":
        for filename in (
            "remediated_records.xml",
            "overlay_decisions.csv",
            "folio_import_plan.json",
            "qa_report.json",
        ):
            assert (output / filename).read_bytes() == (
                original / "outputs/public_case" / filename
            ).read_bytes()
    if label == "codex_after_repair":
        print(
            json.dumps(
                {
                    "marc": label,
                    "case": case_input.parent.name
                    if case_input.name == "inputs"
                    else case_input.name,
                    "original_case_points": score,
                }
            )
        )
        return
    xml_path = output / "remediated_records.xml"
    tree = ET.parse(xml_path)
    for record in tree.getroot():
        for tag, vocabulary in (("336", "rdacontent"), ("337", "rdamedia"), ("338", "rdacarrier")):
            field = record.find(f"{NS}datafield[@tag='{tag}']")
            assert field.find(f"{NS}subfield[@code='2']") is None
            ET.SubElement(field, f"{NS}subfield", {"code": "2"}).text = vocabulary
    tree.write(xml_path)
    assert evaluator.compare_outputs(output, reference)[0] == (
        55 if case_input.name == "public_case" else 80
    )
    for record in tree.getroot():
        online = any(
            (field.text or "").strip().upper() == "ONLINE"
            for field in record.findall(f"{NS}datafield[@tag='949']/{NS}subfield[@code='a']")
        )
        if online:
            for tag, term, code in (("337", "computer", "c"), ("338", "online resource", "cr")):
                field = record.find(f"{NS}datafield[@tag='{tag}']")
                field.find(f"{NS}subfield[@code='a']").text = term
                field.find(f"{NS}subfield[@code='b']").text = code
    tree.write(xml_path)
    assert evaluator.compare_outputs(output, reference)[0] == 80
    tree.find(f".//{NS}datafield[@tag='338']/{NS}subfield[@code='2']").text = "incorrect-vocabulary"
    tree.write(xml_path)
    assert evaluator.compare_outputs(output, reference)[0] == 55
    print(
        json.dumps(
            {
                "marc": label,
                "case": case_input.parent.name if case_input.name == "inputs" else case_input.name,
                "original_case_points": score,
                "both_corrections": 80,
                "wrong_vocabulary_control": 55,
            }
        )
    )
