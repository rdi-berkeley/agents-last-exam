import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from lxml import etree
import pytest

from tasks.business_finance.bpmn_category_governance_restructuring_l3.scripts import (
    evaluate_L3 as evaluator,
    score_output_bundle as scorer,
)


@pytest.fixture
def compliance_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    process = etree.Element(f"{{{evaluator.BPMN_NS}}}process", id=evaluator.MODIFIED_PROCESS_KEY)
    lanes = etree.SubElement(process, f"{{{evaluator.BPMN_NS}}}laneSet", id="governance_lanes")
    for identity in ["logistics_lane", "peer_owner_lane", ""]:
        etree.SubElement(lanes, f"{{{evaluator.BPMN_NS}}}lane", id=identity)
    extensions = etree.SubElement(process, f"{{{evaluator.BPMN_NS}}}extensionElements")
    etree.SubElement(
        extensions, f"{{{evaluator.FLOWABLE_NS}}}formProperty", id="extension_property"
    )
    etree.SubElement(extensions, f"{{{evaluator.FLOWABLE_NS}}}laneSet", id="extension_lane_set")
    graph = evaluator.BPMNGraph(process)
    rules = {
        rule: {
            "priority": "Structural",
            "enforcing_elements": [{"element": evaluator.MODIFIED_PROCESS_KEY}],
        }
        for rule in evaluator.REQUIRED_RULES
    }
    modifications = {
        name: {"elements_added": [evaluator.MODIFIED_PROCESS_KEY]}
        for name in evaluator.REQUIRED_MODIFICATIONS
    }
    (tmp_path / "business_rules_compliance.json").write_text(json.dumps({"rules": rules}))
    (tmp_path / "structural_changes.json").write_text(json.dumps({"modifications": modifications}))
    decisions = "\n".join(
        f"## Modification {index}\n"
        "chosen_approach: Distinct roles.\n"
        "rejected_alternatives: Monolithic owner role; unbounded exception loop.\n"
        "rationale: Applied rule_2.\n"
        "trade-off: Additional handoffs.\n"
        for index in range(1, 19)
    )
    (tmp_path / "design_decisions.md").write_text(decisions)
    return tmp_path, graph


def compliance_result(bundle):
    directory, graph = bundle
    return evaluator.check_compliance(
        str(directory / "structural_changes.json"),
        str(directory / "business_rules_compliance.json"),
        graph,
        {},
    )


@pytest.mark.parametrize("schema", ["elements_added", "elements", "nested"])
@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ("logistics_lane", True),
        ("peer_owner_lane", True),
        ("missing_lane", False),
        ("logistics_lane_typo", False),
        ("logistics_lane-typo", False),
        ("logistics_lane.typo", False),
        ("", False),
        ("governance_lanes", True),
        ("governance_lanes_typo", False),
        ("governance_lanes.typo", False),
        ("missing_lane_set", False),
        ("extension_property", False),
        ("extension_lane_set", False),
    ],
)
def test_c07_real_lane_binding_across_structured_schemas(
    compliance_bundle, schema, identity, expected
):
    directory, _ = compliance_bundle
    path = directory / "structural_changes.json"
    payload = json.loads(path.read_text())
    entry = {schema: [identity]} if schema != "nested" else {"changes": {"target": identity}}
    payload["modifications"]["modification_1"] = entry
    path.write_text(json.dumps(payload))
    result = compliance_result(compliance_bundle)
    assert result["c07_modification_to_bpmn_binding"] is expected
    if not expected:
        assert result["c07_issues"] == [
            {"modification": "modification_1", "reason": "no_existing_bpmn_id_referenced"}
        ]


@pytest.mark.parametrize("schema", ["structured", "free_text"])
@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ("governance_lanes", True),
        ("governance_lanes_typo", False),
        ("governance_lanes-typo", False),
        ("governance_lanes.typo", False),
        ("missing_lane_set", False),
        ("", False),
        ("extension_property", False),
        ("extension_lane_set", False),
    ],
)
def test_c05_lane_set_requires_exact_bpmn_namespace_binding(
    compliance_bundle, schema, identity, expected
):
    directory, _ = compliance_bundle
    path = directory / "business_rules_compliance.json"
    payload = json.loads(path.read_text())
    entry = {"priority": "Structural"}
    if schema == "structured":
        entry["enforcing_elements"] = [{"element": identity}]
    else:
        entry["evidence"] = f"Preserved {identity}." if identity else ""
    payload["rules"]["rule_1"] = entry
    path.write_text(json.dumps(payload))
    result = compliance_result(compliance_bundle)
    assert result["c05_rule_to_bpmn_binding"] is expected
    if not expected:
        assert [issue["rule"] for issue in result["c05_issues"]] == ["rule_1"]


@pytest.mark.parametrize(
    "damage",
    [
        "removed_lane_still_present",
        "removed_lane_set_still_present",
        "empty",
        "prose_only",
        "missing",
    ],
)
def test_c07_preserves_invalid_modification_rejection(compliance_bundle, damage):
    directory, _ = compliance_bundle
    path = directory / "structural_changes.json"
    payload = json.loads(path.read_text())
    if damage == "removed_lane_still_present":
        entry = {"elements_added": ["peer_owner_lane"], "elements_removed": ["logistics_lane"]}
    elif damage == "removed_lane_set_still_present":
        entry = {"elements_added": ["peer_owner_lane"], "elements_removed": ["governance_lanes"]}
    elif damage == "empty":
        entry = {"elements_added": []}
    elif damage == "prose_only":
        entry = {"description": "Preserved logistics_lane"}
    else:
        entry = None
    if entry is None:
        del payload["modifications"]["modification_1"]
    else:
        payload["modifications"]["modification_1"] = entry
    path.write_text(json.dumps(payload))
    result = compliance_result(compliance_bundle)
    assert not result["c07_modification_to_bpmn_binding"]
    if damage.startswith("removed_"):
        assert result["c07_issues"] == [
            {"modification": "modification_1", "reason": "elements_removed_still_present"}
        ]


@pytest.mark.parametrize(
    "alternatives",
    [
        "Monolithic owner role; unbounded exception loop.",
        "Monolithic owner role\nunbounded exception loop.",
        "\n- Monolithic owner role.\n- Unbounded exception loop.",
        "\n1. Monolithic owner role.\n2. Unbounded exception loop.",
        "\n1) Monolithic owner role.; 2) Unbounded exception loop.",
        "Monolithic owner role.;\n* Unbounded exception loop.",
        "Monolithic owner role;; Unbounded exception loop.;",
    ],
)
def test_c11_accepts_two_distinct_alternatives_as_lines_bullets_or_semicolons(
    compliance_bundle, alternatives
):
    directory, _ = compliance_bundle
    path = directory / "design_decisions.md"
    text = path.read_text().replace(
        "Monolithic owner role; unbounded exception loop.", alternatives
    )
    path.write_text(text)
    assert compliance_result(compliance_bundle)["c11_design_decisions_audit"]


@pytest.mark.parametrize(
    "alternatives",
    [
        "",
        ";; ;",
        "...; !!!",
        "Monolithic owner role;",
        ";Monolithic owner role;;",
        "Monolithic owner role; Monolithic owner role",
        "Monolithic owner role.; Monolithic owner role!!!",
        "\n- Monolithic owner role.\n- Monolithic owner role?",
        "1. Monolithic owner role; 2. Monolithic owner role.",
        "Distinct roles.; Monolithic owner role",
        "Distinct roles!!!; Monolithic owner role",
        "Monolithic owner role\nMonolithic   owner role.",
    ],
)
def test_c11_rejects_empty_single_duplicate_and_chosen_alternatives(
    compliance_bundle, alternatives
):
    directory, _ = compliance_bundle
    path = directory / "design_decisions.md"
    path.write_text(
        path.read_text().replace("Monolithic owner role; unbounded exception loop.", alternatives)
    )
    assert not compliance_result(compliance_bundle)["c11_design_decisions_audit"]


@pytest.mark.parametrize(
    "damage", ["chosen_empty", "rule_invalid", "rule_empty", "trade_empty", "missing_section"]
)
def test_c11_keeps_all_other_required_fields_and_eighteen_sections(compliance_bundle, damage):
    directory, _ = compliance_bundle
    path = directory / "design_decisions.md"
    text = path.read_text()
    replacements = {
        "chosen_empty": ("chosen_approach: Distinct roles.", "chosen_approach:"),
        "rule_invalid": ("rationale: Applied rule_2.", "rationale: Applied rule_999."),
        "rule_empty": ("rationale: Applied rule_2.", "rationale:"),
        "trade_empty": ("trade-off: Additional handoffs.", "trade-off:"),
    }
    if damage == "missing_section":
        text = text[: text.index("## Modification 18")]
    else:
        text = text.replace(*replacements[damage])
    path.write_text(text)
    assert not compliance_result(compliance_bundle)["c11_design_decisions_audit"]


@pytest.fixture
def authentic_evidence():
    directory = os.environ.get("BPMN_FORMAT_EVIDENCE")
    if not directory:
        pytest.skip("Set BPMN_FORMAT_EVIDENCE for exact024545 candidate and native controls")
    return Path(directory)


@pytest.mark.parametrize("label", ["actual_candidate", "reference", "prior_positive"])
def test_actual_bundle_and_native_control_receipts_preserve_scores(
    authentic_evidence, tmp_path, label
):
    if label == "actual_candidate":
        source = authentic_evidence / "captured-task/output"
        runtime_path = next(
            (authentic_evidence / "evaluator-evidence").glob("bpmn-evaluator-*/result.json")
        )
    else:
        repair = authentic_evidence.parent
        source = (
            repair / "captured-task/reference"
            if label == "reference"
            else repair.parent / "fresh-run/diagnosis/candidate"
        )
        control = (
            "final-reference-entrypoint"
            if label == "reference"
            else "final-prior-positive-entrypoint"
        )
        runtime_path = repair / "trusted-native-controls" / control / "guest-result.json"
    runtime = json.loads(runtime_path.read_text())
    assert (
        hashlib.sha256((source / "modified_process.bpmn20.xml").read_bytes()).hexdigest()
        == runtime["bpmn_sha256"]
    )
    bundle = tmp_path / label
    shutil.copytree(source, bundle)
    report = scorer.score_output_bundle(
        bpmn_path=bundle / "modified_process.bpmn20.xml",
        structural_path=bundle / "structural_changes.json",
        rules_path=bundle / "business_rules_compliance.json",
        results_path=bundle / "test_results.json",
        scenario_path=authentic_evidence
        / "captured-task/input/starter_project/test_scenarios_L3.json",
        trusted_runtime=runtime,
    )
    assert report["score"] == (0.0 if label == "actual_candidate" else 1.0)
    if label != "actual_candidate":
        return
    before = json.loads((authentic_evidence / "complete-scorer-replay.json").read_text())
    expected = copy.deepcopy(before["report"])
    compliance = expected["sections"]["C_compliance"]
    assert {"rule": "rule_1", "reason": "no_existing_bpmn_id_referenced"} in compliance[
        "c05_issues"
    ]
    compliance["c05_issues"] = [row for row in compliance["c05_issues"] if row["rule"] != "rule_1"]
    assert not compliance["c11_design_decisions_audit"]
    compliance["c11_design_decisions_audit"] = True
    del compliance["c11_issues"]
    lane_modifications = {f"modification_{index}" for index in range(1, 7)}
    assert lane_modifications.issubset({row["modification"] for row in compliance["c07_issues"]})
    compliance["c07_issues"] = [
        row for row in compliance["c07_issues"] if row["modification"] not in lane_modifications
    ]
    for result in [expected, report["report"]]:
        details = result["sections"]["A_structural"]["details"]
        details["a25_peer_tasks"] = sorted(details["a25_peer_tasks"])
    assert expected == report["report"]
    assert report["runtime"] == before["runtime"]
