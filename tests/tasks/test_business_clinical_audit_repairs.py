import json
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
import pytest

from tasks.business_finance.bpmn_category_governance_restructuring_l3.scripts import (
    runtime_validation as runtime,
)
from tasks.business_finance.saas_onepager_brand_refresh_instance_1 import main as onepager
from tasks.business_finance.saas_onepager_brand_refresh_instance_1.scripts import (
    score_onepager_v2 as visual,
)
from tasks.health_medicine.prostate_imrt_matrad_reproduction.scripts import evaluate as prostate
from tasks.utils import evaluation


CRISIS_XML = b"""<process id="zMBCategoryGovernance_modified_L3"
    xmlns:flowable="http://flowable.org/bpmn">
  <userTask id="work"/>
  <exclusiveGateway id="route"/>
  <userTask id="brand_risk_reassessment_task"><extensionElements>
    <flowable:formProperty id="out_complianceHandled" variable="complianceHandled"
      type="boolean" writable="true"/>
  </extensionElements></userTask>
  <sequenceFlow id="next" sourceRef="work" targetRef="route"/>
  <sequenceFlow id="crisis" sourceRef="route" targetRef="brand_risk_reassessment_task">
    <conditionExpression>${complianceCrisisDetected == true &amp;&amp; !complianceHandled}</conditionExpression>
  </sequenceFlow>
</process>"""
CRISIS_EXPRESSION = "${complianceCrisisDetected == true && !complianceHandled}"


def failing_api(exception, *, start=False, status=500, engine_error=None):
    api = Mock()

    def request(method, path, **kwargs):
        if path == "management/engine":
            if engine_error is not None:
                raise engine_error
            return {"version": "6.5.0"}
        if path == "runtime/process-instances" and not start:
            return {"id": "instance"}
        if path in {"runtime/process-instances", "runtime/tasks/task"}:
            raise runtime.FlowableResponseError("HTTP failure", status, {"exception": exception})
        return {"endTime": None}

    api.request.side_effect = request
    api.all.side_effect = lambda path, **kwargs: (
        [{"id": "task", "taskDefinitionKey": "work"}] if path == "runtime/tasks" else []
    )
    return api


def crisis_result(api, binding=None):
    return runtime.run_case(
        api,
        binding or runtime.Bindings(CRISIS_XML),
        {"process_definition_id": "definition", "tenant_id": "isolated"},
        {"id": "C01", "category": "crisis", "input_variables": {"complianceCrisisDetected": True}},
    )


@pytest.mark.parametrize("start", [False, True])
def test_candidate_unknown_property_is_a_scored_failure_without_injected_state(start):
    binding = runtime.Bindings(CRISIS_XML)
    assert "complianceHandled" not in binding.initial_values({})
    api = failing_api("Unknown property used in expression: " + CRISIS_EXPRESSION, start=start)
    result = crisis_result(api, binding)
    assert result["pass"] is False and "failure" in result
    assert "evaluation_error" not in result
    assert result["expression_failure_evidence"]["candidate_expression"] == CRISIS_EXPRESSION
    for call in api.request.call_args_list:
        variables = call.kwargs.get("json", {}).get("variables", [])
        assert not any(row["name"] == "complianceHandled" for row in variables)
    assert any(call.args == ("GET", "management/engine") for call in api.request.call_args_list)
    scenarios = [
        {"id": f"case-{index}", "category": "crisis" if index < 55 else "anti_gaming"}
        for index in range(60)
    ]
    results = [
        {**result, "scenario_id": row["id"], "category": row["category"], "pass": index >= 55}
        for index, row in enumerate(scenarios)
    ]
    assert runtime.summarize_results(results, scenarios)["meets_threshold"] is False


@pytest.mark.parametrize(
    "exception,status",
    [
        ("Unknown property used in expression: ${unrelated_engine_setting}", 500),
        ("Internal database exception", 500),
        ("Unknown property used in expression: " + CRISIS_EXPRESSION, 503),
        (None, 500),
    ],
)
def test_unattributed_or_engine_errors_remain_infrastructure_errors(exception, status):
    result = crisis_result(failing_api(exception, status=status))
    assert "evaluation_error" in result and "failure" not in result
    assert result["pass"] is False


def test_failed_engine_health_check_never_becomes_candidate_failure():
    api = failing_api(
        "Unknown property used in expression: " + CRISIS_EXPRESSION,
        engine_error=runtime.EvaluationError("engine unavailable"),
    )
    with pytest.raises(runtime.EvaluationError, match="engine unavailable"):
        crisis_result(api)


def test_missing_declared_fixture_outcome_remains_an_evaluator_error():
    xml = CRISIS_XML.replace(
        b'<userTask id="work"/>',
        b'<userTask id="work"><extensionElements><flowable:formProperty '
        b'id="out_complianceHandled" variable="complianceHandled" type="boolean" '
        b'writable="true"/></extensionElements></userTask>',
    )
    result = crisis_result(failing_api("unused"), runtime.Bindings(xml))
    assert "no declared outcome" in result["evaluation_error"]
    assert "failure" not in result


@pytest.fixture
def reported_dvh():
    rows = []
    for key, spec in prostate.DVH_CONSTRAINTS.items():
        if key == "Bowel_Dmax":
            continue
        structure, metric = key.rsplit("_", 1)
        units = "%" if spec["metric"] == "V_pct" else "Gy"
        rows.append(f"| {structure} | {metric} | {spec['limit']} {units} | PASS |")
    return "## DVH\n\n| Structure | Metric | Value | Result |\n|---|---|---|---|\n" + "\n".join(
        rows
    )


@pytest.mark.parametrize(
    "summary",
    [
        "seven coplanar 6 MV fields at 0, 51, 102, 153, 204, 255,\n306 degrees; 5 mm bixels.",
        "7 coplanar 6 MV beams (gantry 0/51/102/153/204/255/306 deg, collimator 0).",
        "7 coplanar 6 MV IMRT fields. Gantry angles: 0, 51, 102, 153, 204, 255, 306 degrees.",
    ],
)
def test_actual_beam_summary_styles_and_result_column_are_recognized(summary, reported_dvh):
    assert prostate.report_section_coverage(summary + "\n\n" + reported_dvh) == {
        "beam summary",
        "dvh",
        "constraint compliance",
    }


@pytest.mark.parametrize(
    "summary",
    [
        "seven coplanar fields at 0, 51, 102, 153, 204, 255, 306 degrees.",
        "seven coplanar 6 MV fields at 0, 51, 102, 153, 204, 255 degrees.",
        "seven coplanar 6 MV fields at 0, 51, 102, 153, 204, 255, 306, 350 degrees.",
        "coplanar 6 MV fields at 0, 51, 102, 153, 204, 255, 306 degrees.",
    ],
)
def test_implicit_beam_summary_still_requires_count_energy_and_all_angles(summary):
    assert "beam summary" not in prostate.report_section_coverage(summary)


@pytest.mark.parametrize("damage", ["missing_metric", "missing_result", "nonfinite", "code_block"])
def test_result_table_still_requires_complete_quantitative_content(reported_dvh, damage):
    if damage == "missing_metric":
        report = reported_dvh.rsplit("\n", 1)[0]
    elif damage == "missing_result":
        report = reported_dvh.replace("| PASS |", "| |")
    elif damage == "nonfinite":
        report = reported_dvh.replace("52.5 Gy", "nan Gy")
    else:
        report = "```markdown\n" + reported_dvh + "\n```"
    assert "constraint compliance" not in prostate.report_section_coverage(report)


def test_onepager_entry_point_promises_installed_impress_and_editable_pptx():
    config = onepager.OnePagerBrandRefreshConfig()
    card = json.loads(Path(onepager.__file__).with_name("task_card.json").read_text())
    assert config.impress_executable == r"C:\Program Files\LibreOffice\program\simpress.exe"
    assert config.to_metadata()["impress_executable"] == config.impress_executable
    assert card["software"] == ["LibreOffice Impress"]
    for prompt in (config.task_description, card["taskPrompt"]):
        assert config.impress_executable in prompt
        assert "OpenPowerPoint.bat" not in prompt
        assert "meaningfully editable single-slide PPTX" in prompt
        assert "edited_onepager.pptx" in prompt and "edited_onepager.png" in prompt


def test_visual_requests_reconcile_only_unrequested_copy_and_keep_spacing(monkeypatch, tmp_path):
    calls = []

    def judge_text(**kwargs):
        calls.append(kwargs)
        return "NO"

    monkeypatch.setattr(evaluation, "llm_multimodal_text_sync", judge_text)
    monkeypatch.setattr(evaluation, "resolve_llm_judge_model", lambda **kwargs: "test-judge")
    judge = visual.SharedVlmJudge()
    submission = tmp_path / "submission.png"
    reference = tmp_path / "reference.png"
    Image.new("RGB", (200, 400), "navy").save(submission)
    Image.new("RGB", (200, 400), "white").save(reference)
    regions = {name: (0, 0, 200, 40) for name in visual.REGION_RUBRIC}
    score, details = visual.vlm_region_score(judge, submission, reference, regions)
    global_details = visual.vlm_global_score(judge, submission, reference)
    assert score == 0 and not any(details.values()) and not any(global_details.values())
    assert len(calls) == 12
    for call in calls:
        prompt = call["content"][0]["text"]
        assert "most spacing relationships" in prompt
        assert "asset presence alone does not excuse" in prompt
        for text in (
            "Operations playbook",
            "Manual work saved per store per week",
            "Standardize staff handoffs",
            "Recent performance overview",
            "Workflow completion trend",
            "pilot customer",
            "Task routing and checklists",
            "Bring every store workflow into one decision layer.",
        ):
            assert text in prompt
        assert len([item for item in call["content"] if item["type"] == "image_url"]) == 2
