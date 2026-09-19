import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from tasks.business_finance.bpmn_category_governance_restructuring_l3.scripts import (
    runtime_validation as runtime,
)
from tasks.business_finance.bpmn_category_governance_restructuring_l3.scripts import (
    score_output_bundle as scorer,
)


@pytest.fixture
def manifest():
    scenarios = [
        {
            "id": f"N{index:02d}",
            "category": "normal",
            "input_variables": {},
            "task_outputs": {},
            "expected_end_event": "main_end",
            "should_complete": True,
        }
        for index in range(55)
    ]
    scenarios += [
        {
            "id": f"G{index}",
            "category": "anti_gaming",
            "is_structural_check": True,
            "validation_focus": focus,
            "expected_end_event": "main_end",
        }
        for index, focus in enumerate(runtime.PROBE_CHECKS, 1)
    ]
    return {"scenarios": scenarios}


def test_manifest_is_fifty_five_runtime_and_five_probes(manifest):
    rows = runtime.validate_manifest(manifest)
    assert len([row for row in rows if row.get("is_structural_check") is not True]) == 55
    broken = copy.deepcopy(manifest)
    broken["scenarios"][-1]["validation_focus"] = "invented_probe"
    with pytest.raises(runtime.EvaluationError):
        runtime.validate_manifest(broken)


@pytest.mark.parametrize(
    ("failed", "failed_probe", "expected"), [(3, False, True), (4, False, False), (1, True, False)]
)
def test_runtime_threshold_is_57_of_60_and_every_probe(manifest, failed, failed_probe, expected):
    scenarios = runtime.validate_manifest(manifest)
    rows = [
        {"scenario_id": row["id"], "category": row["category"], "pass": True} for row in scenarios
    ]
    selected = rows[-failed:] if failed_probe else rows[:failed]
    for row in selected:
        row["pass"] = False
    assert runtime.summarize_results(rows, scenarios)["meets_threshold"] is expected


@pytest.mark.parametrize("damage", ["duplicate", "category", "string_pass", "missing"])
def test_runtime_rejects_inconsistent_trusted_results(manifest, damage):
    scenarios = runtime.validate_manifest(manifest)
    rows = [
        {"scenario_id": row["id"], "category": row["category"], "pass": True} for row in scenarios
    ]
    if damage == "duplicate":
        rows[-1] = rows[0]
    elif damage == "category":
        rows[0]["category"] = "anti_gaming"
    elif damage == "string_pass":
        rows[0]["pass"] = "true"
    else:
        rows.pop()
    with pytest.raises(runtime.EvaluationError):
        runtime.summarize_results(rows, scenarios)


def test_alias_and_form_variable_binding_do_not_need_reference_task_id():
    xml = b"""<process id="zMBCategoryGovernance_modified_L3" xmlns:flowable="http://flowable.org/bpmn">
      <userTask id="independent_chair" name="Temporary Unified Decision" flowable:assignee="${customChair}">
        <extensionElements><flowable:formProperty id="chair_answer" variable="out_unified_decision" type="string" writable="true"/></extensionElements>
      </userTask></process>"""
    binding = runtime.Bindings(xml)
    scenario = {
        "task_outputs": {"temporary_unified_decision": {"out_unified_decision": "rejected"}}
    }
    assert binding.task_roles["independent_chair"] == "temporary_unified_decision"
    assert binding.initial_values(scenario)["customChair"] == "admin"
    assert binding.task_values("independent_chair", 1, scenario) == {
        "chair_answer": "rejected",
        "out_unified_decision": "rejected",
    }


def test_form_prefix_aliases_preserve_fixture_outcomes_and_isolation():
    xml = b"""<process id="zMBCategoryGovernance_modified_L3" xmlns:flowable="http://flowable.org/bpmn">
      <userTask id="team_joint_mature_plan_preparation"><extensionElements>
        <flowable:formProperty id="out_inventoryOk" type="boolean" writable="true"/>
        <flowable:formProperty id="out_jointPlanAttempt" type="long" writable="true"/>
      </extensionElements></userTask></process>"""
    binding = runtime.Bindings(xml)
    scenario = {
        "input_variables": {"peerOwnerAvailable": False},
        "task_outputs": {
            "team_joint_mature_plan_preparation": [{"inventoryOk": False}, {"inventoryOk": True}]
        },
    }
    original = copy.deepcopy(scenario)
    first = binding.task_values("team_joint_mature_plan_preparation", 1, scenario)
    second = binding.task_values("team_joint_mature_plan_preparation", 2, scenario)
    assert first["out_inventoryOk"] is False and second["out_inventoryOk"] is True
    assert first["out_jointPlanAttempt"] == 1 and second["out_jointPlanAttempt"] == 2
    assert binding.initial_values(scenario)["peerOwnerAvailable"] is False
    assert "peerOwnerAvailable" not in binding.initial_values({})
    assert scenario == original
    with pytest.raises(runtime.CandidateFailure):
        binding.task_values("team_joint_mature_plan_preparation", 3, scenario)


def test_unbound_new_routing_decision_is_not_given_favorable_value():
    xml = b"""<process id="zMBCategoryGovernance_modified_L3" xmlns:flowable="http://flowable.org/bpmn">
      <userTask id="campaign_closeout"><extensionElements><flowable:formProperty id="out_cycle_closed" type="boolean" writable="true"/></extensionElements></userTask>
      <exclusiveGateway id="closed_gate"/><endEvent id="main_end"/>
      <sequenceFlow id="to_gate" sourceRef="campaign_closeout" targetRef="closed_gate"/>
      <sequenceFlow id="to_end" sourceRef="closed_gate" targetRef="main_end"><conditionExpression>${out_cycle_closed == true}</conditionExpression></sequenceFlow>
    </process>"""
    binding = runtime.Bindings(xml)
    assert binding.task_values("campaign_closeout", 1, {}) == {}
    assert binding.unbound_routing_outputs("campaign_closeout", {}) == ["out_cycle_closed"]


def test_normal_completion_defaults_do_not_inspect_candidate_branch_conditions():
    xml = b"""<process id="zMBCategoryGovernance_modified_L3" xmlns:flowable="http://flowable.org/bpmn">
      <userTask id="team_executes_plan"><extensionElements>
        <flowable:formProperty id="custom_report" variable="out_execution_report" type="string" writable="true"/>
        <flowable:formProperty id="custom_retry" variable="out_execution_needs_replan" type="boolean" writable="true"/>
      </extensionElements></userTask>
      <userTask id="archive_record" name="Campaign cadence closeout"><extensionElements>
        <flowable:formProperty id="custom_ack" variable="out_cycle_closed" type="boolean" writable="true"/>
      </extensionElements></userTask>
    </process>"""
    binding = runtime.Bindings(xml)
    assert binding.task_values("team_executes_plan", 1, {}) == {
        "custom_report": "completed",
        "out_execution_report": "completed",
        "custom_retry": False,
        "out_execution_needs_replan": False,
    }
    assert binding.task_values("archive_record", 1, {}) == {
        "custom_ack": True,
        "out_cycle_closed": True,
    }
    adverse = {
        "task_outputs": {
            "team_executes_plan": {"out_execution_needs_replan": True},
            "archive_record": {"out_cycle_closed": False},
        }
    }
    assert binding.task_values("team_executes_plan", 1, adverse)["custom_retry"] is True
    assert binding.task_values("archive_record", 1, adverse)["custom_ack"] is False


@pytest.mark.parametrize("reassessed", [True, False])
def test_compliance_latch_follows_explicit_reassessment_outcome(reassessed):
    xml = b"""<process id="zMBCategoryGovernance_modified_L3" xmlns:flowable="http://flowable.org/bpmn">
      <userTask id="brand_risk_reassessment_task"><extensionElements>
        <flowable:formProperty id="out_complianceHandled" type="boolean" writable="true"/>
      </extensionElements></userTask></process>"""
    binding = runtime.Bindings(xml)
    scenario = {
        "task_outputs": {"brand_risk_reassessment_task": {"brandRiskReassessed": reassessed}}
    }
    assert (
        binding.task_values("brand_risk_reassessment_task", 1, scenario)["out_complianceHandled"]
        is reassessed
    )


def test_flowable_pagination_keeps_all_rows():
    api = runtime.Flowable("http://unused/")
    api.request = Mock(
        side_effect=[
            {"total": 3, "data": [{"id": "first"}, {"id": "second"}]},
            {"total": 3, "data": [{"id": "third"}]},
        ]
    )
    assert len(api.all("history")) == 3
    assert api.request.call_args_list[1].kwargs["params"]["start"] == 2


def test_engine_transport_error_is_evaluator_error():
    api = runtime.Flowable("http://unused/")
    api.session.request = Mock(side_effect=requests.ConnectionError("engine down"))
    with pytest.raises(runtime.EvaluationError, match="transport unavailable"):
        api.request("POST", "runtime/process-instances", candidate=True, json={})


def test_engine_http500_is_not_evidence_of_candidate_fault():
    api = runtime.Flowable("http://unused/")
    api.session.request = Mock(return_value=Mock(status_code=500, text="engine exception"))
    with pytest.raises(runtime.EvaluationError, match="HTTP500"):
        api.request("POST", "runtime/process-instances", candidate=True, json={})
    assert api.session.request.call_count == 1


def test_repeated_crisis_checks_each_visit_without_retroactive_failure():
    discovered = {
        "brand_risk_reassessment_task": "risk",
        "unified_decision_task": "decide",
        "compliance_subprocess": "crisis",
    }
    activities = [
        {"activityId": "crisis", "startTime": "01"},
        {"activityId": "crisis", "startTime": "05"},
    ]
    tasks = [
        {"taskDefinitionKey": "risk", "startTime": "02", "endTime": "03"},
        {"taskDefinitionKey": "decide", "startTime": "04"},
        {"taskDefinitionKey": "risk", "startTime": "06", "endTime": "07"},
        {"taskDefinitionKey": "decide", "startTime": "08"},
    ]
    assert runtime.crisis_precedes_decisions(tasks, activities, discovered)
    tasks[-1]["startTime"] = "06"
    assert not runtime.crisis_precedes_decisions(tasks, activities, discovered)


def test_global_budget_exhaustion_is_evaluator_error():
    api = Mock()
    api.request.side_effect = [{"id": "instance"}, {"endTime": None}]
    api.all.return_value = []
    binding = Mock()
    binding.initial_values.return_value = {}
    result = runtime.run_case(
        api,
        binding,
        {"process_definition_id": "definition", "tenant_id": "fresh"},
        {"id": "N01", "category": "normal"},
        global_deadline=0,
    )
    assert "global runtime budget" in result["evaluation_error"]
    assert "failure" not in result


@pytest.mark.parametrize(
    ("exception", "bound", "candidate_failure"),
    [
        (
            "No outgoing sequence flow of element 'route' could be selected for continuing the process",
            True,
            True,
        ),
        (
            "No outgoing sequence flow of element 'route' could be selected for continuing the process",
            False,
            False,
        ),
        ("Internal database exception", True, False),
        (
            "No outgoing sequence flow of element 'unknown_node' could be selected for continuing the process",
            True,
            False,
        ),
    ],
)
def test_only_recognized_bound_routing_http500_is_candidate_failure(
    exception, bound, candidate_failure
):
    xml = b"""<process id="zMBCategoryGovernance_modified_L3">
      <userTask id="work"/><exclusiveGateway id="route"/><endEvent id="main_end"/>
      <sequenceFlow id="next" sourceRef="work" targetRef="route"/>
      <sequenceFlow id="done" sourceRef="route" targetRef="main_end"><conditionExpression>${choice == true}</conditionExpression></sequenceFlow>
    </process>"""
    binding = runtime.Bindings(xml)
    api = Mock()

    def request(method, path, **kwargs):
        if path == "runtime/process-instances":
            return {"id": "instance"}
        if path == "runtime/tasks/task":
            raise runtime.FlowableResponseError("HTTP500", 500, {"exception": exception})
        if path == "runtime/process-instances/instance/variables":
            return [{"name": "choice", "value": False}] if bound else []
        return {"endTime": None}

    api.request.side_effect = request
    api.all.side_effect = lambda path, **kwargs: (
        [{"id": "task", "taskDefinitionKey": "work"}] if path == "runtime/tasks" else []
    )
    result = runtime.run_case(
        api,
        binding,
        {"process_definition_id": "definition", "tenant_id": "fresh"},
        {"id": "N01", "category": "normal"},
    )
    assert ("failure" in result) is candidate_failure
    assert ("evaluation_error" in result) is not candidate_failure


def test_evaluator_error_rows_cannot_be_turned_into_candidate_zero(manifest):
    scenarios = runtime.validate_manifest(manifest)
    rows = [
        {"scenario_id": row["id"], "category": row["category"], "pass": False} for row in scenarios
    ]
    rows[0]["evaluation_error"] = "Missing fixture output"
    with pytest.raises(runtime.EvaluationError, match="evaluator error"):
        runtime.summarize_results(rows, scenarios)


def test_deployment_resource_hash_is_verified():
    api = runtime.Flowable("http://unused/")
    api.request = Mock(
        side_effect=[{"id": "deployment"}, {"contentUrl": "http://unused/process-api/content"}]
    )
    api.all = Mock(
        return_value=[
            {
                "id": "definition",
                "key": runtime.topology.MODIFIED_PROCESS_KEY,
                "tenantId": "fresh",
                "resource": "http://unused/process-api/resource",
            }
        ]
    )
    response = Mock(content=b"different XML")
    api.session.get = Mock(return_value=response)
    with pytest.raises(runtime.EvaluationError, match="hash differs"):
        api.deploy(b"actual XML", "fresh")


def test_engine_timer_is_moved_and_boundary_history_required():
    api = runtime.Flowable("http://unused/")
    api.all = Mock(
        side_effect=[
            [
                {"id": "timer-job", "elementId": "actual_boundary"},
                {"id": "unrelated-timer", "elementId": "another_boundary"},
            ],
            [{"activityId": "actual_boundary", "endTime": "2026-09-13T00:00:00Z"}],
        ]
    )
    api.request = Mock(return_value={})
    receipt = api.accelerate_timer("process", "actual_boundary", float("inf"))
    api.request.assert_called_once_with(
        "POST", "management/timer-jobs/timer-job", json={"action": "move"}
    )
    assert receipt["boundary"] == "actual_boundary"


def test_production_wrapper_will_not_accept_candidate_flags_alone(tmp_path):
    with pytest.raises(RuntimeError, match="Evaluator-owned runtime results required"):
        scorer.score_output_bundle(
            bpmn_path=tmp_path / "modified.xml",
            structural_path=tmp_path / "structural.json",
            rules_path=tmp_path / "rules.json",
            results_path=tmp_path / "results.json",
            scenario_path=tmp_path / "scenarios.json",
        )


@pytest.fixture
def authentic_native():
    location = os.environ.get("BPMN_C05_EVIDENCE")
    if not location:
        pytest.skip("Set BPMN_C05_EVIDENCE for hash-bound native control receipts")
    root = Path(location)
    receipt = root / "trusted-native-controls/reference-entrypoint"
    if not (receipt / "guest-result.json").exists():
        pytest.skip("Native production-entrypoint reference receipt not yet captured")
    result = json.loads((receipt / "guest-result.json").read_text())
    xml = root / "captured-task/reference/modified_process.bpmn20.xml"
    assert hashlib.sha256(xml.read_bytes()).hexdigest() == result["bpmn_sha256"]
    return root, result


@pytest.mark.parametrize("probe_failure", [False, True])
def test_native_reference_and_false_probe_override_submitted_flags(
    authentic_native, tmp_path, probe_failure
):
    root, result = authentic_native
    assert result["passed_scenarios"] == 60
    assert len([row for row in result["scenarios"] if row["category"] != "anti_gaming"]) == 55
    if probe_failure:
        result = copy.deepcopy(result)
        next(row for row in result["scenarios"] if row["scenario_id"] == "G02")["pass"] = False
        manifest = json.loads(
            (root / "captured-task/input/starter_project/test_scenarios_L3.json").read_text()
        )
        result.update(
            runtime.summarize_results(result["scenarios"], runtime.validate_manifest(manifest))
        )
    bundle = tmp_path / "reference"
    shutil.copytree(root / "captured-task/reference", bundle)
    report = scorer.score_output_bundle(
        bpmn_path=bundle / "modified_process.bpmn20.xml",
        structural_path=bundle / "structural_changes.json",
        rules_path=bundle / "business_rules_compliance.json",
        results_path=bundle / "test_results.json",
        scenario_path=root / "captured-task/input/starter_project/test_scenarios_L3.json",
        trusted_runtime=result,
    )
    assert report["score"] == (0.0 if probe_failure else 1.0)
    assert report["diagnostic_only"] is False


def test_actual_current_native_probe_rejects_fabricated_g02(authentic_native):
    root, _ = authentic_native
    evidence = json.loads((root / "trusted-native-controls/initial/result.json").read_text())
    current = evidence["controls"]["actual_current"]
    claimed = json.loads((root / "captured-task/output/test_results.json").read_text())
    assert next(row for row in claimed["scenarios"] if row["scenario_id"] == "G02")["pass"] is True
    assert next(row for row in current["probes"] if row["scenario_id"] == "G02")["pass"] is False
    assert (
        current["sha256"]
        == hashlib.sha256(
            (root / "captured-task/output/modified_process.bpmn20.xml").read_bytes()
        ).hexdigest()
    )
