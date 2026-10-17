import asyncio
import hashlib
import inspect
import itertools
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from lxml import etree

from tasks.business_finance.bpmn_category_governance_restructuring_l3 import main
from tasks.business_finance.bpmn_category_governance_restructuring_l3.scripts import (
    evaluate_L3 as evaluator,
    score_output_bundle as wrapper,
)


def graph_from_edges(edges, parallel=()):
    process = etree.Element("process", id=evaluator.MODIFIED_PROCESS_KEY)
    nodes = set(itertools.chain.from_iterable(edges))
    for identity in sorted(nodes):
        kind = "parallelGateway" if identity in parallel else "exclusiveGateway"
        etree.SubElement(process, kind, id=identity)
    for index, (source, target) in enumerate(edges):
        etree.SubElement(
            process, "sequenceFlow", id=f"flow_{index}", sourceRef=source, targetRef=target
        )
    return evaluator.BPMNGraph(process)


NESTED_EDGES = [
    ("start", "outer_split"),
    ("outer_split", "logistics"),
    ("logistics", "outer_join"),
    ("outer_split", "inner_split"),
    ("inner_split", "review_left"),
    ("inner_split", "review_right"),
    ("review_left", "inner_join"),
    ("review_right", "inner_join"),
    ("inner_join", "execute"),
    ("execute", "outer_join"),
    ("outer_join", "closeout"),
    ("closeout", "end"),
]
NESTED_PARALLEL = {"outer_split", "outer_join", "inner_split", "inner_join"}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("variant", ["simple", "loop", "bypass", "deeper"])
def test_a45_nested_join_closes_inner_split(tmp_path, monkeypatch, reverse, variant):
    monkeypatch.chdir(tmp_path)
    edges = list(NESTED_EDGES)
    parallel = set(NESTED_PARALLEL)
    if variant == "loop":
        edges.extend([("execute", "retry"), ("retry", "inner_split")])
        edges.remove(("outer_split", "inner_split"))
        edges.extend([("outer_split", "retry"), ("retry", "execute")])
    elif variant == "bypass":
        edges.remove(("outer_split", "inner_split"))
        edges.extend([
            ("outer_split", "available"), ("available", "inner_split"),
            ("available", "committee"), ("committee", "execute"),
        ])
    elif variant == "deeper":
        edges.remove(("inner_split", "review_left"))
        edges.extend([
            ("inner_split", "deep_split"), ("deep_split", "deep_left"),
            ("deep_split", "deep_right"), ("deep_left", "deep_join"),
            ("deep_right", "deep_join"), ("deep_join", "review_left"),
        ])
        parallel.update({"deep_split", "deep_join"})
    graph = graph_from_edges(list(reversed(edges)) if reverse else edges, parallel)
    assert evaluator._nested_parallel_join_split(graph, "outer_join") == "outer_split"
    checks, details = evaluator.check_structural(graph)
    assert checks["a45_parallel_join_common_split"], details


@pytest.mark.parametrize(
    "damage", ["exclusive_merge", "unclosed_inner", "exclusive_inputs", "unrelated", "cycle"]
)
def test_a45_nested_fallback_does_not_accept_path_existence_alone(tmp_path, monkeypatch, damage):
    monkeypatch.chdir(tmp_path)
    edges = list(NESTED_EDGES)
    parallel = set(NESTED_PARALLEL)
    if damage == "exclusive_merge":
        parallel.remove("inner_join")
    elif damage == "unclosed_inner":
        edges.remove(("review_right", "inner_join"))
        edges.append(("review_right", "outer_join"))
    elif damage == "exclusive_inputs":
        edges.remove(("inner_split", "review_right"))
        edges.extend([("inner_split", "unjoined_end"), ("review_left", "review_right")])
    elif damage == "unrelated":
        edges.remove(("outer_split", "logistics"))
        edges.append(("unrelated_start", "logistics"))
    else:
        edges.append(("closeout", "logistics"))
    graph = graph_from_edges(edges, parallel)
    assert evaluator._nested_parallel_join_split(graph, "outer_join") is None
    checks, details = evaluator.check_structural(graph)
    assert not checks["a45_parallel_join_common_split"], details


@pytest.mark.parametrize(
    "edges",
    list(
        itertools.permutations(
            [
                ("guard", "policy_applied"),
                ("policy_applied", "guard"),
                ("guard", "main_end"),
            ]
        )
    ),
)
def test_escapable_cycles_are_not_dead_ends_regardless_of_xml_order(edges):
    graph = graph_from_edges(edges)
    assert evaluator._find_non_terminating_paths(graph, "guard", {"main_end"}) == []


@pytest.mark.parametrize(
    "edges",
    [
        [("senior", "dead"), ("senior", "main_end")],
        [("senior", "closed"), ("senior", "main_end"), ("closed", "closed")],
    ],
)
def test_true_dead_end_and_no_exit_cycle_still_fail(edges):
    assert evaluator._find_non_terminating_paths(graph_from_edges(edges), "senior", {"main_end"})


def test_termination_walk_does_not_truncate_long_paths():
    edges = [(f"node_{index}", f"node_{index + 1}") for index in range(100)]
    edges.append(("node_100", "main_end"))
    graph = graph_from_edges(edges)
    assert evaluator._find_non_terminating_paths(graph, "node_0", {"main_end"}) == []
    edges.append(("node_90", "dead"))
    assert evaluator._find_non_terminating_paths(
        graph_from_edges(edges), "node_0", {"main_end"}
    ) == ["dead"]


@pytest.mark.parametrize(
    "identity", ["coordination", "merchant_readiness_coordination", "campaign_cadence_coordination"]
)
def test_explicitly_combined_coordination_covers_both_responsibilities(identity):
    process = etree.Element("process")
    etree.SubElement(
        process,
        "userTask",
        id=identity,
        name="Merchant readiness and campaign cadence coordination",
    )
    found = evaluator.discover_nodes(evaluator.BPMNGraph(process))
    assert found["merchant_coordination_task"] == identity
    assert found["campaign_coordination_task"] == identity


def test_missing_campaign_responsibility_is_not_inferred():
    process = etree.Element("process")
    etree.SubElement(process, "userTask", id="coordination", name="Merchant readiness coordination")
    found = evaluator.discover_nodes(evaluator.BPMNGraph(process))
    assert found["merchant_coordination_task"] == "coordination"
    assert found["campaign_coordination_task"] is None


@pytest.mark.parametrize(
    "name",
    [
        "Merchant readiness",
        "merchant_readiness",
        "Merchant-readiness",
        "Campaign cadence",
        "A/B test",
        "A/B-test",
        "Review based promotion",
        "ad_ratio",
        "Free trial",
        "New merchant",
    ],
)
def test_a50_accepts_equivalent_required_domain_names(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    process = etree.Element("process")
    etree.SubElement(process, "userTask", id="noncanonical", name=name)
    checks, details = evaluator.check_structural(evaluator.BPMNGraph(process))
    assert checks["a50_no_shadow_tasks"], details


def test_a50_still_rejects_uncovered_placeholder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    process = etree.Element("process")
    etree.SubElement(process, "userTask", id="future_work", name="Future placeholder work")
    checks, details = evaluator.check_structural(evaluator.BPMNGraph(process))
    assert not checks["a50_no_shadow_tasks"]
    assert details["a50_shadow_tasks"] == ["future_work"]


@pytest.fixture
def evidence():
    value = os.environ.get("BPMN_CONTRACT_EVIDENCE")
    if not value:
        pytest.skip("Set BPMN_CONTRACT_EVIDENCE for authentic retained-bundle tests")
    return Path(value)


def score_copy(evidence, bundle):
    return wrapper.score_output_bundle(
        bpmn_path=bundle / "modified_process.bpmn20.xml",
        structural_path=bundle / "structural_changes.json",
        rules_path=bundle / "business_rules_compliance.json",
        results_path=bundle / "test_results.json",
        scenario_path=evidence / "before/input/starter_project/test_scenarios_L3.json",
        static_only=True,
    )


@pytest.fixture
def fresh_candidate(evidence, tmp_path):
    audit = json.loads((evidence / "fresh-run/independent-final-audit.json").read_text())
    source = evidence / "fresh-run/diagnosis/candidate"
    assert hashlib.sha256((source / "modified_process.bpmn20.xml").read_bytes()).hexdigest() == (
        "923d52ed1399565005be51fb949390c674ed2df796958db1869d890841e01f76"
    )
    original = Path(next(iter(audit["output_sha256"]))).parent
    for path in source.iterdir():
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["output_sha256"][str(original / path.name)]
    bundle = tmp_path / "fresh-candidate"
    shutil.copytree(source, bundle)
    return bundle


def test_actual_fresh_candidate_changes_only_a45(evidence, fresh_candidate, monkeypatch):
    repaired = score_copy(evidence, fresh_candidate)
    assert repaired["score"] == 1.0
    with monkeypatch.context() as legacy:
        legacy.setattr(evaluator, "_nested_parallel_join_split", lambda *args: None)
        checks, details = evaluator.check_structural(
            evaluator.BPMNGraph(etree.parse(str(fresh_candidate / "modified_process.bpmn20.xml")))
        )
    assert not checks["a45_parallel_join_common_split"]
    assert details["a45_issues"] == [{
        "join": "gw_parallel_join", "reason": "sources_from_different_splits",
        "ancestors": ["gw_parallel_split", "gw_parallel_review_split"],
    }]
    before = json.loads((evidence / "fresh-run/diagnosis/candidate-report.json").read_text())
    before_checks = before["report"]["sections"]["A_structural"]["checks"]
    after_checks = repaired["report"]["sections"]["A_structural"]["checks"]
    assert [key for key in before_checks if before_checks[key] != after_checks[key]] == [
        "a45_parallel_join_common_split"
    ]
    for section in before["report"]["sections"]:
        if section == "A_structural":
            continue
        old_section = dict(before["report"]["sections"][section])
        new_section = dict(repaired["report"]["sections"][section])
        if section == "E_fingerprint":
            assert old_section.pop("novel_qualifying") is False
            assert new_section.pop("novel_qualifying") is True
        assert old_section == new_section


@pytest.mark.parametrize("damage", ["exclusive_merge", "unclosed_inner", "exclusive_inputs", "unrelated"])
def test_fresh_candidate_with_invalid_nested_join_is_negative(evidence, fresh_candidate, damage):
    path = fresh_candidate / "modified_process.bpmn20.xml"
    tree = etree.parse(str(path))
    if damage == "exclusive_merge":
        join = tree.find(".//bpmn:parallelGateway[@id='gw_parallel_review_join']", evaluator.NS)
        join.tag = f"{{{evaluator.BPMN_NS}}}exclusiveGateway"
    else:
        flows = tree.findall(".//bpmn:sequenceFlow", evaluator.NS)
        if damage == "unclosed_inner":
            flow = next(flow for flow in flows if flow.get("sourceRef") == "peer_ad_ratio_task")
            flow.set("targetRef", "gw_parallel_join")
        elif damage == "exclusive_inputs":
            flow = next(flow for flow in flows if flow.get("targetRef") == "peer_review_based_promotion_task")
            flow.set("sourceRef", "gw_ab_test_enabled")
        else:
            flow = next(flow for flow in flows if flow.get("targetRef") == "logistics_capacity_confirmation")
            flow.set("sourceRef", "startEvent")
    tree.write(str(path), xml_declaration=True, encoding="UTF-8")
    result = score_copy(evidence, fresh_candidate)
    assert result["score"] == 0.0
    assert not result["report"]["sections"]["A_structural"]["checks"]["a45_parallel_join_common_split"]


@pytest.mark.parametrize("label", ["candidate", "reference", "missing"])
def test_a45_production_entrypoint_retained_fresh_guest(evidence, tmp_path, monkeypatch, label):
    port = os.environ.get("BPMN_A45_READ_ONLY_GUEST_PORT")
    if not port:
        pytest.skip("Set BPMN_A45_READ_ONLY_GUEST_PORT for the terminal fresh guest")
    assert port == "38067"
    from cua_bench.computers.remote import RemoteDesktopSession

    root = "/media/user/data/agenthle/business_finance/bpmn_category_governance_restructuring_l3/base"
    directory = "output" if label == "candidate" else label
    metadata = {
        "output_" + key: f"{root}/{directory}/{filename}"
        for key, filename in [
            ("bpmn", "modified_process.bpmn20.xml"),
            ("structural", "structural_changes.json"),
            ("rules", "business_rules_compliance.json"),
            ("results", "test_results.json"),
            ("design_decisions", "design_decisions.md"),
        ]
    }
    metadata["starter_scenarios"] = f"{root}/input/starter_project/test_scenarios_L3.json"
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    async def read_and_score():
        session = RemoteDesktopSession(api_url=f"http://127.0.0.1:{port}", os_type="linux")
        try:
            await session.start()
            assert session._client_only_mode
            pins = {
                name: hashlib.sha256(await session.read_bytes(path)).hexdigest()
                for name, path in metadata.items() if directory != "missing"
            }
            score = await main.evaluate(SimpleNamespace(metadata=metadata), session)
            return score, pins
        finally:
            await session.close()

    score, pins = asyncio.run(read_and_score())
    assert score == ([0.0] if label == "missing" else [1.0])
    source = Path(inspect.getfile(RemoteDesktopSession))
    receipt = {
        "label": label, "score": score, "metadata": metadata, "read_sha256": pins,
        "session_class": "cua_bench.computers.remote.RemoteDesktopSession",
        "session_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "guest_port": port, "guest_reads_only": True, "method_replacement": False,
        "historical_score_rewritten": False,
    }
    (evidence / "a45-followup" / f"production-entrypoint-{label}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


@pytest.mark.parametrize("label", ["reference", "candidate"])
def test_authentic_complete_bundle_positive(evidence, tmp_path, label):
    bundle = tmp_path / "bundle"
    shutil.copytree(evidence / "before" / label, bundle)
    assert score_copy(evidence, bundle)["score"] == 1.0


@pytest.mark.parametrize(
    ("failed_count", "fail_anti_gaming", "expected_score"),
    [
        (3, False, 1.0),
        (4, False, 0.0),
        (1, True, 0.0),
    ],
)
def test_original_95_percent_and_mandatory_anti_gaming_gates_remain(
    evidence, tmp_path, failed_count, fail_anti_gaming, expected_score
):
    bundle = tmp_path / "bundle"
    shutil.copytree(evidence / "before/reference", bundle)
    results_path = bundle / "test_results.json"
    payload = json.loads(results_path.read_text())
    remaining = failed_count
    for row in payload["scenarios"]:
        if remaining and (row["category"] == "anti_gaming") == fail_anti_gaming:
            row["pass"] = False
            remaining -= 1
    assert remaining == 0
    payload["passed_scenarios"] = 60 - failed_count
    payload["pass_rate"] = (60 - failed_count) / 60
    summary = {}
    for row in payload["scenarios"]:
        counts = summary.setdefault(row["category"], {"total": 0, "passed": 0})
        counts["total"] += 1
        counts["passed"] += bool(row["pass"])
    payload["scenarios_summary"] = summary
    results_path.write_text(json.dumps(payload))
    result = score_copy(evidence, bundle)
    (tmp_path / "report.json").write_text(json.dumps(result, indent=2))
    assert result["score"] == expected_score


@pytest.mark.parametrize(
    "damage",
    [
        "wrong_key",
        "shadow_task",
        "dead_policy",
        "missing_campaign",
        "missing_peer_topic",
        "same_execution_decision_role",
        "chosen_empty",
        "duplicate_alternatives",
        "one_alternative",
        "invalid_rule",
        "tradeoff_empty",
        "scenario_failure",
    ],
)
def test_authentic_candidate_with_real_contract_violations_is_negative(evidence, tmp_path, damage):
    bundle = tmp_path / "bundle"
    shutil.copytree(evidence / "before/candidate", bundle)
    path = bundle / "modified_process.bpmn20.xml"
    tree = etree.parse(str(path))
    process = tree.find(".//bpmn:process", evaluator.NS)
    if damage == "wrong_key":
        process.set("id", "wrong_key")
    elif damage == "shadow_task":
        etree.SubElement(
            process, f"{{{evaluator.BPMN_NS}}}userTask", id="uncovered", name="Future work"
        )
    elif damage == "dead_policy":
        flow = tree.find(".//bpmn:sequenceFlow[@sourceRef='ab_test_policy_applied']", evaluator.NS)
        flow.getparent().remove(flow)
    elif damage == "missing_campaign":
        task = tree.find(".//bpmn:userTask[@id='team_coordinates_merchants']", evaluator.NS)
        task.set("name", "Merchant readiness coordination")
        task.find("bpmn:documentation", evaluator.NS).text = "Merchant readiness coordination."
    elif damage == "missing_peer_topic":
        task = tree.find(".//bpmn:userTask[@id='peer_ad_ratio_task']", evaluator.NS)
        task.getparent().remove(task)
    elif damage == "same_execution_decision_role":
        task = tree.find(".//bpmn:userTask[@id='temporary_unified_decision']", evaluator.NS)
        task.set(f"{{{evaluator.FLOWABLE_NS}}}assignee", "${mbTeamLead}")
    elif damage == "scenario_failure":
        results_path = bundle / "test_results.json"
        payload = json.loads(results_path.read_text())
        payload["scenarios"][0]["pass"] = False
        results_path.write_text(json.dumps(payload))
    else:
        decisions = bundle / "design_decisions.md"
        text = decisions.read_text()
        start = text.index("chosen_approach:")
        stop = text.index("## Modification 2:")
        fields = {
            "chosen": "Distinct roles.",
            "alternatives": "- One role.\n- Two roles.",
            "rationale": "rule_2",
            "trade": "Extra handoff.",
        }
        if damage == "chosen_empty":
            fields["chosen"] = ""
        elif damage == "duplicate_alternatives":
            fields["alternatives"] = "- One role.\n- One role."
        elif damage == "one_alternative":
            fields["alternatives"] = "- One role."
        elif damage == "invalid_rule":
            fields["rationale"] = "rule_299"
        elif damage == "tradeoff_empty":
            fields["trade"] = ""
        block = (
            f"chosen_approach: {fields['chosen']}\n\n"
            f"rejected_alternatives:\n{fields['alternatives']}\n\n"
            f"rationale: {fields['rationale']}\n\ntrade-off: {fields['trade']}\n\n"
        )
        decisions.write_text(text[:start] + block + text[stop:])
    tree.write(str(path), xml_declaration=True, encoding="UTF-8")
    result = score_copy(evidence, bundle)
    (tmp_path / "report.json").write_text(json.dumps(result, indent=2))
    assert result["score"] == 0.0, damage


@pytest.mark.parametrize("label", ["reference", "candidate", "missing"])
def test_production_desktop_entrypoint_reads_retained_guest_only(
    evidence, tmp_path, monkeypatch, label
):
    port = os.environ.get("BPMN_READ_ONLY_GUEST_PORT")
    if not port:
        pytest.skip("Set BPMN_READ_ONLY_GUEST_PORT for production retained-guest reads")
    assert port == "38057"
    from cua_bench.computers.remote import RemoteDesktopSession

    root = (
        "/media/user/data/agenthle/business_finance/bpmn_category_governance_restructuring_l3/base"
    )
    directory = "output" if label == "candidate" else label
    metadata = {
        "output_" + key: f"{root}/{directory}/{filename}"
        for key, filename in [
            ("bpmn", "modified_process.bpmn20.xml"),
            ("structural", "structural_changes.json"),
            ("rules", "business_rules_compliance.json"),
            ("results", "test_results.json"),
            ("design_decisions", "design_decisions.md"),
        ]
    }
    metadata["starter_scenarios"] = f"{root}/input/starter_project/test_scenarios_L3.json"
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    async def read_and_score():
        session = RemoteDesktopSession(api_url=f"http://127.0.0.1:{port}", os_type="linux")
        try:
            await session.start()
            assert session._client_only_mode
            return await main.evaluate(SimpleNamespace(metadata=metadata), session)
        finally:
            await session.close()

    score = asyncio.run(read_and_score())
    assert score == ([0.0] if label == "missing" else [1.0])
    source = Path(inspect.getfile(RemoteDesktopSession))
    receipt = {
        "label": label,
        "score": score,
        "metadata": metadata,
        "session_class": "cua_bench.computers.remote.RemoteDesktopSession",
        "session_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "guest_reads_only": True,
        "method_replacement": False,
        "historical_score_rewritten": False,
    }
    (evidence / f"production-entrypoint-{label}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
