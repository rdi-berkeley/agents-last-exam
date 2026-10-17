from types import SimpleNamespace

import pytest
from lxml import etree

from tasks.business_finance.bpmn_category_governance_restructuring_l3.scripts import (
    evaluate_L3 as evaluator,
    runtime_validation as runtime,
)


def escalation_graph(senior_count=3, total=10, senior_ids=("senior_lead_expedited_decision",)):
    process = etree.Element("process", id=evaluator.MODIFIED_PROCESS_KEY)
    etree.SubElement(process, "exclusiveGateway", id="routing")
    etree.SubElement(process, "userTask", id="ordinary")
    for identity in senior_ids:
        task = etree.SubElement(process, "userTask", id=identity)
        task.set(f"{{{evaluator.FLOWABLE_NS}}}assignee", "${zSeniorLead}")
    for index in range(total):
        target = senior_ids[index % len(senior_ids)] if index < senior_count else "ordinary"
        etree.SubElement(
            process, "sequenceFlow", id=f"flow_{index}", sourceRef="routing", targetRef=target
        )
    return process


def senior_checks(process):
    graph = evaluator.BPMNGraph(process)
    return evaluator.check_anti_gaming(graph, evaluator.discover_nodes(graph))


@pytest.mark.parametrize(
    ("senior_count", "passed"), [(0, True), (2, True), (3, True), (4, False), (10, False)]
)
def test_direct_senior_cap_including_exact_thirty_percent(senior_count, passed):
    checks = senior_checks(escalation_graph(senior_count))
    assert checks["d03_senior_flows"] == senior_count
    assert checks["d03_total_decision_flows"] == 10
    assert checks["d03_senior_escalation_ratio_le_30"] is passed


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_later_committee_and_critical_tier_do_not_reclassify_upstream_flows(depth):
    process = escalation_graph(1, 10)
    previous = "ordinary"
    for index in range(depth - 1):
        identity = f"review_{index}"
        etree.SubElement(process, "userTask", id=identity)
        etree.SubElement(process, "sequenceFlow", sourceRef=previous, targetRef=identity)
        previous = identity
    etree.SubElement(
        process, "sequenceFlow", sourceRef=previous, targetRef="senior_lead_expedited_decision"
    )
    checks = senior_checks(process)
    assert checks["d03_senior_flows"] == 1
    assert checks["d03_total_decision_flows"] == 10
    assert checks["d03_senior_escalation_ratio_le_30"]


def test_conditional_critical_branch_and_boundary_timer_are_not_upstream_escalations():
    process = escalation_graph(0, 3)
    etree.SubElement(process, "exclusiveGateway", id="critical_tier")
    etree.SubElement(process, "endEvent", id="main_end")
    etree.SubElement(process, "boundaryEvent", id="timer", attachedToRef="ordinary")
    for source, target in [
        ("ordinary", "critical_tier"),
        ("critical_tier", "senior_lead_expedited_decision"),
        ("critical_tier", "main_end"),
        ("timer", "senior_lead_expedited_decision"),
    ]:
        etree.SubElement(process, "sequenceFlow", sourceRef=source, targetRef=target)
    checks = senior_checks(process)
    assert checks["d03_senior_flows"] == 1
    assert checks["d03_total_decision_flows"] == 5
    assert checks["d03_senior_escalation_ratio_le_30"]


def test_all_direct_senior_assignees_count_not_only_discovered_task():
    checks = senior_checks(
        escalation_graph(4, senior_ids=("senior_lead_expedited_decision", "second_senior"))
    )
    assert checks["d03_senior_flows"] == 4
    assert not checks["d03_senior_escalation_ratio_le_30"]


def test_canonical_task_name_does_not_override_actual_assignee():
    process = escalation_graph(4)
    process.find("userTask[@id='senior_lead_expedited_decision']").set(
        f"{{{evaluator.FLOWABLE_NS}}}assignee", "${interimCommitteeChair}"
    )
    assert senior_checks(process)["d03_senior_flows"] == 0


@pytest.mark.parametrize(
    "kind, identity",
    [
        ("exclusiveGateway", "gw_exception_exists"),
        ("exclusiveGateway", "gw_exception_join"),
        ("inclusiveGateway", "inclusive"),
        ("parallelGateway", "parallel"),
    ],
)
def test_rule11_denominator_only_new_exclusive_gateway_outgoing_flows(kind, identity):
    process = escalation_graph(4)
    etree.SubElement(process, kind, id=identity)
    for index in range(10):
        etree.SubElement(
            process, "sequenceFlow", id=f"extra_{index}", sourceRef=identity, targetRef="ordinary"
        )
    checks = senior_checks(process)
    assert checks["d03_total_decision_flows"] == 10
    assert not checks["d03_senior_escalation_ratio_le_30"]


@pytest.mark.parametrize(("senior_count", "passed"), [(3, True), (4, False)])
def test_g02_reuses_direct_count_and_remains_mandatory(tmp_path, monkeypatch, senior_count, passed):
    monkeypatch.chdir(tmp_path)
    process = escalation_graph(senior_count)
    graph = evaluator.BPMNGraph(process)
    binding = SimpleNamespace(graph=graph, discovered=evaluator.discover_nodes(graph))
    probes = runtime.probe_results(
        binding,
        [
            {
                "id": "G02",
                "category": "anti_gaming",
                "is_structural_check": True,
                "validation_focus": "senior_lead_centralization",
            }
        ],
    )
    assert probes[0]["pass"] is passed
    summary = evaluator.check_test_results(
        "unused",
        {
            "total_scenarios": 60,
            "passed_scenarios": 59 + int(passed),
            "scenarios": probes,
        },
    )
    assert summary["meets_threshold"] is passed
