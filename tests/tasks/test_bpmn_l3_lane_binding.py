import json
import os
import shutil
from pathlib import Path

import pytest
from lxml import etree

from tasks.business_finance.bpmn_category_governance_restructuring_l3.scripts import (
    evaluate_L3 as evaluator,
    score_output_bundle as wrapper,
)


@pytest.mark.parametrize("schema", ["structured", "free_text"])
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
    ],
)
def test_c05_lane_ids_require_real_nonempty_binding(tmp_path, monkeypatch, schema, identity, expected):
    monkeypatch.chdir(tmp_path)
    process = etree.Element("process", id=evaluator.MODIFIED_PROCESS_KEY)
    lane_set = etree.SubElement(process, "laneSet", id="governance_lanes")
    for lane_id in ["logistics_lane", "peer_owner_lane", ""]:
        etree.SubElement(lane_set, "lane", id=lane_id)
    graph = evaluator.BPMNGraph(process)
    rules = {
        rule_id: {"priority": "Structural", "enforcing_elements": [{"element": evaluator.MODIFIED_PROCESS_KEY}]}
        for rule_id in evaluator.REQUIRED_RULES
    }
    entry = {"priority": "Structural"}
    if schema == "structured":
        entry["enforcing_elements"] = [{"element": identity}]
    else:
        entry["evidence"] = f"Preserved {identity}." if identity else ""
    rules["rule_8"] = entry
    rules_path = tmp_path / "business_rules_compliance.json"
    structural_path = tmp_path / "structural_changes.json"
    rules_path.write_text(json.dumps({"rules": rules}))
    structural_path.write_text(json.dumps({"modifications": {}}))
    result = evaluator.check_compliance(str(structural_path), str(rules_path), graph, {})
    assert result["c05_rule_to_bpmn_binding"] is expected
    if not expected:
        assert [issue["rule"] for issue in result["c05_issues"]] == ["rule_8"]


@pytest.fixture
def authentic_audit():
    value = os.environ.get("BPMN_C05_EVIDENCE")
    if not value:
        pytest.skip("Set BPMN_C05_EVIDENCE for the final-source authentic bundle")
    return Path(value)


@pytest.mark.parametrize("label", ["actual_candidate", "retained_reference"])
def test_c05_authentic_bundle_changes_only_lane_binding(authentic_audit, tmp_path, label):
    before = json.loads((authentic_audit / "complete-scorer-replay.json").read_text())[label]
    if label == "actual_candidate":
        manifest = json.loads((authentic_audit / "source-run-manifest.json").read_text())
        original = Path(manifest["leaf"]) / "output"
    else:
        original = authentic_audit / "captured-task/reference"
    bundle = tmp_path / label
    shutil.copytree(original, bundle)
    after = wrapper.score_output_bundle(
        bpmn_path=bundle / "modified_process.bpmn20.xml",
        structural_path=bundle / "structural_changes.json",
        rules_path=bundle / "business_rules_compliance.json",
        results_path=bundle / "test_results.json",
        scenario_path=authentic_audit / "captured-task/input/starter_project/test_scenarios_L3.json",
        static_only=True,
    )
    assert after["score"] == before["score"] == (0.0 if label == "actual_candidate" else 1.0)
    expected_report = before["report"]
    compliance = expected_report["sections"]["C_compliance"]
    if label == "actual_candidate":
        assert compliance["c05_rule_to_bpmn_binding"] is False
        compliance["c05_rule_to_bpmn_binding"] = True
        del compliance["c05_issues"]
    for report in [after["report"], expected_report]:
        details = report["sections"]["A_structural"]["details"]
        details["a25_peer_tasks"] = sorted(details["a25_peer_tasks"])
    assert after["report"] == expected_report
