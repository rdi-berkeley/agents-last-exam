"""Evaluator-owned Flowable scenario execution and evidence collection."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import re
import shlex
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable

import requests

try:
    from . import evaluate_L3 as topology
except ImportError:
    import evaluate_L3 as topology


class EvaluationError(RuntimeError):
    pass


class CandidateFailure(RuntimeError):
    pass


class FlowableResponseError(EvaluationError):
    def __init__(self, message: str, status: int, payload: dict):
        super().__init__(message)
        self.status = status
        self.payload = payload


TASK_ROLES = {
    "peer_readiness_check": "peer_readiness_check_task",
    "compliance_signal_check": "compliance_signal_check_task",
    "team_joint_mature_plan_preparation": "joint_plan_task",
    "temporary_unified_decision": "unified_decision_task",
    "senior_lead_expedited_decision": "senior_lead_escalation_task",
    "emergency_delist_task": "emergency_delist_task",
    "brand_risk_reassessment_task": "brand_risk_reassessment_task",
}
PROBE_CHECKS = {
    "single_role_absorption": ("d01_single_role_absorption",),
    "senior_lead_centralization": ("d03_senior_escalation_ratio_le_30",),
    "logistics_modification": (
        "a04_anchor_logistics_lane",
        "a05_anchor_logistics_capacity_confirmation",
        "d05_logistics_not_modified",
    ),
    "ab_shutdown": (
        "d04_ab_test_not_shutdown",
        "d04_free_trial_not_shutdown",
        "d04_new_merchant_not_shutdown",
    ),
    "default_flows": ("a33_default_flows_on_exclusive_gateways", "a41_default_flow_exclusivity"),
}


def variable_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^(?:in|out)_", "", name).lower())


def validate_manifest(payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        raise EvaluationError("Evaluator manifest must be an object")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 60:
        raise EvaluationError("Evaluator manifest must contain exactly60 scenarios")
    identities = [row.get("id") for row in scenarios if isinstance(row, dict)]
    if len(identities) != 60 or len(set(identities)) != 60 or not all(identities):
        raise EvaluationError("Evaluator manifest has invalid or duplicate scenario IDs")
    probes = [row for row in scenarios if row.get("is_structural_check") is True]
    if len(probes) != 5 or {row.get("validation_focus") for row in probes} != set(PROBE_CHECKS):
        raise EvaluationError("Evaluator manifest must contain the five declared structural probes")
    for row in scenarios:
        if row.get("expected_end_event") not in {"main_end", "escalation_final_end"}:
            raise EvaluationError(f"Unsupported expected end in scenario {row['id']}")
        if not isinstance(row.get("input_variables", {}), dict) or not isinstance(
            row.get("task_outputs", {}), dict
        ):
            raise EvaluationError(f"Invalid evaluator variables in scenario {row['id']}")
        for outputs in row.get("task_outputs", {}).values():
            if not isinstance(outputs, dict) and not (
                isinstance(outputs, list)
                and outputs
                and all(isinstance(value, dict) for value in outputs)
            ):
                raise EvaluationError(f"Invalid per-visit outputs in scenario {row['id']}")
    return scenarios


class Bindings:
    def __init__(self, xml: bytes):
        try:
            root = topology.LET.fromstring(xml)
        except Exception as error:
            raise CandidateFailure(f"Invalid candidate BPMN XML: {error}") from error
        self.graph = topology.BPMNGraph(root)
        self.discovered = topology.discover_nodes(self.graph)
        self.task_roles = {}
        for canonical, role in TASK_ROLES.items():
            actual = canonical if canonical in self.graph.elements else self.discovered.get(role)
            if not actual:
                matches = [
                    identity
                    for identity in self.graph.elements
                    if self.graph.get_type(identity) == "userTask"
                    and variable_key(canonical)
                    in {
                        variable_key(identity),
                        variable_key(self.graph.get_task_name(identity)),
                    }
                ]
                if len(matches) == 1:
                    actual = matches[0]
                    self.discovered[role] = actual
            if actual:
                self.task_roles[actual] = canonical
        self.properties = {}
        for identity in self.graph.elements:
            if self.graph.get_type(identity) == "userTask":
                element = self.graph.get_element(identity)
                self.properties[identity] = [
                    dict(prop.attrib)
                    for prop in element.iter()
                    if self.graph._local(prop.tag) == "formProperty"
                ]
        self.condition_variables = set()
        for flows in self.graph.out_flows.values():
            for flow in flows:
                self.condition_variables.update(
                    topology.extract_vars_from_condition(flow["condition"])
                )
        self.role_variables = set()
        for identity in self.properties:
            element = self.graph.get_element(identity)
            role = element.get(f"{{{topology.FLOWABLE_NS}}}assignee", "")
            match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", role)
            if match:
                self.role_variables.add(match.group(1))

    def bind_values(self, identity: str, supplied: dict) -> dict:
        values = copy.deepcopy(supplied)
        for prop in self.properties.get(identity, []):
            if prop.get("writable", "true").lower() != "true":
                continue
            keys = {variable_key(prop.get("id", "")), variable_key(prop.get("variable", ""))}
            matches = [name for name in supplied if variable_key(name) in keys]
            if len(matches) > 1 and any(supplied[name] != supplied[matches[0]] for name in matches):
                raise EvaluationError(
                    f"Conflicting fixture aliases for {identity}/{prop.get('id')}"
                )
            if matches:
                value = supplied[matches[0]]
                values[prop["id"]] = value
                binding = prop.get("variable", prop["id"])
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", binding):
                    raise EvaluationError(f"Unsupported writable form binding {identity}/{binding}")
                values[binding] = value
        return values

    def initial_values(self, scenario: dict) -> dict:
        values = {role: "admin" for role in self.role_variables}
        values.update(copy.deepcopy(scenario.get("input_variables", {})))
        for props in self.properties.values():
            for prop in props:
                default = prop.get("default") or prop.get("defaultValue")
                if default is None or prop.get("id", "").startswith("out_"):
                    continue
                if prop.get("type") == "boolean" and default.lower() in {"true", "false"}:
                    value = default.lower() == "true"
                elif prop.get("type") in {"long", "double"}:
                    value = float(default) if prop["type"] == "double" else int(default)
                else:
                    value = default
                values.setdefault(prop.get("variable", prop["id"]), value)
                values.setdefault(re.sub(r"^in_", "", prop["id"]), value)
        return values

    def task_values(self, identity: str, visit: int, scenario: dict) -> dict:
        canonical = self.task_roles.get(identity, identity)
        supplied = scenario.get("task_outputs", {}).get(canonical, {})
        if isinstance(supplied, list):
            if visit > len(supplied):
                raise CandidateFailure(
                    f"{identity} exceeded the scenario's {len(supplied)} planning visits"
                )
            supplied = supplied[visit - 1]
        supplied = copy.deepcopy(supplied)
        if canonical == "team_joint_mature_plan_preparation":
            supplied["jointPlanAttempt"] = visit
        return {
            **self.completion_values(identity, scenario),
            **self.bind_values(identity, supplied),
        }

    def completion_values(self, identity: str, scenario: dict) -> dict:
        canonical = self.task_roles.get(identity, identity)
        defaults = {}
        if identity == "team_executes_plan":
            defaults = {"executionneedsreplan": False, "executionreport": "completed"}
        description = variable_key(
            " ".join(
                [
                    identity,
                    self.graph.get_task_name(identity),
                    self.graph.get_documentation(identity),
                ]
            )
        )
        if "campaigncadence" in description and "closeout" in description:
            defaults["cycleclosed"] = True
        if canonical == "brand_risk_reassessment_task":
            outcomes = scenario.get("task_outputs", {}).get(canonical, {})
            if isinstance(outcomes, dict):
                reassessed = next(
                    (
                        value
                        for name, value in outcomes.items()
                        if variable_key(name) == "brandriskreassessed"
                    ),
                    None,
                )
                if type(reassessed) is bool:
                    defaults["compliancehandled"] = reassessed
        values = {}
        for prop in self.properties.get(identity, []):
            if prop.get("writable", "true").lower() != "true":
                continue
            keys = {variable_key(prop.get("id", "")), variable_key(prop.get("variable", ""))}
            matches = keys.intersection(defaults)
            if matches:
                value = defaults[sorted(matches)[0]]
                values[prop["id"]] = value
                values[prop.get("variable", prop["id"])] = value
        return values

    def unbound_routing_outputs(self, identity: str, values: dict) -> list[str]:
        unbound = []
        element = self.graph.get_element(identity)
        listeners = " ".join(
            prop.get("expression", "")
            for prop in element.iter()
            if self.graph._local(prop.tag) == "executionListener"
        )
        immediate_variables = set()
        pending = [identity]
        visited = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            for flow in self.graph.outgoing_flows(node):
                immediate_variables.update(topology.extract_vars_from_condition(flow["condition"]))
                if self.graph.get_type(flow["target"]) != "userTask":
                    pending.append(flow["target"])
        for prop in self.properties.get(identity, []):
            binding = prop.get("variable", prop.get("id", ""))
            if (
                prop.get("writable", "true").lower() != "true"
                or binding not in immediate_variables
                or binding in values
            ):
                continue
            if re.search(r"setVariable\(\s*['\"]" + re.escape(binding) + r"['\"]", listeners):
                continue
            unbound.append(binding)
        return sorted(unbound)


class Flowable:
    def __init__(self, api_url: str, *, timeout_s: float = 20):
        self.url = api_url.rstrip("/") + "/"
        self.session = requests.Session()
        self.session.auth = ("admin", "test")
        self.timeout_s = timeout_s

    def request(self, method: str, path: str, *, candidate: bool = False, **kwargs):
        try:
            response = self.session.request(
                method, self.url + path.lstrip("/"), timeout=self.timeout_s, **kwargs
            )
        except requests.RequestException as error:
            raise EvaluationError(f"Flowable transport unavailable: {error}") from error
        if response.status_code >= 400:
            message = f"Flowable {method} {path}: HTTP{response.status_code} {response.text[:2000]}"
            if response.status_code >= 500:
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                raise FlowableResponseError(message, response.status_code, payload)
            if candidate and response.status_code in {400, 422}:
                self.request("GET", "management/engine")
                raise CandidateFailure(message)
            raise EvaluationError(message)
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as error:
            raise EvaluationError(f"Invalid Flowable JSON at {path}") from error

    def all(self, path: str, **params) -> list[dict]:
        rows = []
        while True:
            page = self.request("GET", path, params={**params, "start": len(rows), "size": 100})
            if not isinstance(page.get("data"), list) or not isinstance(page.get("total"), int):
                raise EvaluationError(f"Invalid Flowable pagination at {path}")
            rows.extend(page["data"])
            if len(rows) == page["total"]:
                return rows
            if not page["data"] or len(rows) > page["total"]:
                raise EvaluationError(f"Incomplete Flowable pagination at {path}")

    def deploy(self, xml: bytes, tenant: str) -> dict:
        deployment = self.request(
            "POST",
            "repository/deployments",
            candidate=True,
            params={"tenantId": tenant},
            files={"deployment": ("modified_process.bpmn20.xml", xml, "application/xml")},
        )
        definitions = self.all("repository/process-definitions", deploymentId=deployment["id"])
        if (
            len(definitions) != 1
            or definitions[0]["key"] != topology.MODIFIED_PROCESS_KEY
            or definitions[0].get("tenantId") != tenant
        ):
            raise CandidateFailure("Deployment must expose one expected process definition")
        definition = definitions[0]
        resource_path = definition["resource"].split("/process-api/", 1)[-1]
        resource = self.request("GET", resource_path)
        content_path = resource["contentUrl"].split("/process-api/", 1)[-1]
        try:
            response = self.session.get(self.url + content_path, timeout=self.timeout_s)
            response.raise_for_status()
        except requests.RequestException as error:
            raise EvaluationError(f"Cannot verify deployed resource: {error}") from error
        digest = hashlib.sha256(xml).hexdigest()
        if hashlib.sha256(response.content).hexdigest() != digest:
            raise EvaluationError("Deployed resource hash differs from evaluator candidate bytes")
        return {
            "deployment_id": deployment["id"],
            "process_definition_id": definition["id"],
            "tenant_id": tenant,
            "bpmn_sha256": digest,
        }

    def accelerate_timer(self, identity: str, boundary: str, deadline: float) -> dict:
        timers = self.all("management/timer-jobs", processInstanceId=identity)
        timers = [job for job in timers if job.get("elementId") == boundary]
        if len(timers) != 1:
            raise CandidateFailure(f"Expected one waiting decision timer, found {len(timers)}")
        job = timers[0]
        moved = self.request("POST", f"management/timer-jobs/{job['id']}", json={"action": "move"})
        timer_deadline = min(deadline, time.monotonic() + 10)
        while time.monotonic() < timer_deadline:
            history = self.all("history/historic-activity-instances", processInstanceId=identity)
            if any(row.get("activityId") == boundary and row.get("endTime") for row in history):
                return {"timer_job": job, "management_result": moved, "boundary": boundary}
            jobs = self.all("management/deadletter-jobs", processInstanceId=identity)
            if jobs:
                raise CandidateFailure(f"Timer execution failed: {jobs}")
            time.sleep(0.2)
        raise EvaluationError("Moved engine timer produced no boundary history before deadline")


def probe_results(bindings: Bindings, scenarios: list[dict]) -> list[dict]:
    structural, _ = topology.check_structural(bindings.graph)
    checks = {**structural, **topology.check_anti_gaming(bindings.graph, bindings.discovered)}
    results = []
    for scenario in scenarios:
        if scenario.get("is_structural_check") is not True:
            continue
        names = PROBE_CHECKS[scenario["validation_focus"]]
        evidence = {name: checks[name] for name in names}
        results.append(
            {
                "scenario_id": scenario["id"],
                "category": scenario["category"],
                "pass": all(evidence.values()),
                "validation": "trusted_structural_probe",
                "checks": evidence,
            }
        )
    return results


def summarize_results(results: list[dict], scenarios: list[dict]) -> dict:
    if not isinstance(results, list) or any(not isinstance(row, dict) for row in results):
        raise EvaluationError("Trusted runtime results must be objects")
    if any(row.get("evaluation_error") for row in results):
        raise EvaluationError("Trusted runtime contains an evaluator error, not candidate outcomes")
    expected = {row["id"]: row["category"] for row in scenarios}
    actual = {row.get("scenario_id"): row.get("category") for row in results}
    if (
        len(results) != 60
        or actual != expected
        or any(type(row.get("pass")) is not bool for row in results)
    ):
        raise EvaluationError(
            "Trusted runtime result IDs/categories/boolean outcomes do not match the manifest"
        )
    passed = sum(row["pass"] for row in results)
    probes = [row for row in results if row["category"] == "anti_gaming"]
    return {
        "total_scenarios": 60,
        "passed_scenarios": passed,
        "pass_rate": passed / 60,
        "meets_threshold": passed >= 57 and len(probes) == 5 and all(row["pass"] for row in probes),
    }


def crisis_precedes_decisions(tasks: list[dict], activities: list[dict], discovered: dict) -> bool:
    risks = [
        row
        for row in tasks
        if row["taskDefinitionKey"] == discovered.get("brand_risk_reassessment_task")
        and row.get("endTime")
        and not row.get("deleteReason")
    ]
    cycles = [
        row for row in activities if row["activityId"] == discovered.get("compliance_subprocess")
    ]
    decisions = [
        row for row in tasks if row["taskDefinitionKey"] == discovered.get("unified_decision_task")
    ]
    if not risks or not cycles:
        return False
    for decision in decisions:
        preceding = [cycle for cycle in cycles if cycle["startTime"] <= decision["startTime"]]
        if not preceding:
            return False
        cycle = max(preceding, key=lambda row: row["startTime"])
        if not any(
            cycle["startTime"] <= risk["startTime"] <= risk["endTime"] <= decision["startTime"]
            for risk in risks
        ):
            return False
    return True


def run_case(
    api: Flowable,
    binding: Bindings,
    deployment: dict,
    scenario: dict,
    *,
    timeout_s: float = 30,
    global_deadline: float = float("inf"),
) -> dict:
    result = {
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "pass": False,
        "trace": [],
    }
    deadline = time.monotonic() + timeout_s
    visits = Counter()
    process_id = None
    try:
        process = api.request(
            "POST",
            "runtime/process-instances",
            candidate=True,
            json={
                "processDefinitionId": deployment["process_definition_id"],
                "businessKey": f"{deployment['tenant_id']}:{scenario['id']}",
                "variables": [
                    {"name": name, "value": value}
                    for name, value in binding.initial_values(scenario).items()
                ],
            },
        )
        process_id = process["id"]
        result["process_instance_id"] = process_id
        timer_fired = False
        for _ in range(150):
            if time.monotonic() >= global_deadline:
                raise EvaluationError("Trusted evaluator suite exhausted its global runtime budget")
            if time.monotonic() >= deadline:
                raise CandidateFailure("Scenario exceeded the bounded runtime deadline")
            tasks = api.all("runtime/tasks", processInstanceId=process_id)
            if not tasks:
                break
            task = sorted(tasks, key=lambda row: (row.get("createTime", ""), row["id"]))[0]
            identity = task["taskDefinitionKey"]
            visits[identity] += 1
            values = binding.task_values(identity, visits[identity], scenario)
            if (
                identity == binding.discovered.get("unified_decision_task")
                and scenario.get("input_variables", {}).get("timer_expedited")
                and not timer_fired
            ):
                metadata = {
                    name: value
                    for name, value in values.items()
                    if variable_key(name) != "unifieddecision"
                }
                if metadata:
                    api.request(
                        "PUT",
                        f"runtime/process-instances/{process_id}/variables",
                        json=[{"name": name, "value": value} for name, value in metadata.items()],
                    )
                boundary = binding.graph.find_boundary_event_attached_to(identity)
                if not boundary:
                    raise CandidateFailure("Timer scenario has no decision boundary")
                result["trace"].append(
                    {
                        "action": "engine_timer",
                        **api.accelerate_timer(
                            process_id, boundary, min(deadline, global_deadline)
                        ),
                    }
                )
                timer_fired = True
                continue
            unbound = binding.unbound_routing_outputs(identity, values)
            if unbound:
                raise EvaluationError(
                    f"Evaluator fixture has no declared outcome for legal task {identity}: {unbound}; refusing to invent routing decisions"
                )
            try:
                api.request(
                    "POST",
                    f"runtime/tasks/{task['id']}",
                    candidate=True,
                    json={
                        "action": "complete",
                        "variables": [
                            {"name": name, "value": value} for name, value in values.items()
                        ],
                    },
                )
            except FlowableResponseError as error:
                exception = error.payload.get("exception", "")
                match = re.fullmatch(
                    "No outgoing sequence flow of element '([^']+)' could be selected for continuing the process",
                    exception,
                )
                if error.status != 500 or not match or match.group(1) not in binding.graph.elements:
                    raise
                variables = api.request("GET", f"runtime/process-instances/{process_id}/variables")
                if not isinstance(variables, list):
                    raise EvaluationError("Invalid Flowable process-variable response") from error
                known = {row["name"] for row in variables}.union(values)
                required = set()
                for flow in binding.graph.outgoing_flows(match.group(1)):
                    required.update(topology.extract_vars_from_condition(flow["condition"]))
                if not required.issubset(known):
                    raise EvaluationError(
                        f"Routing error has unbound inputs: {sorted(required - known)}"
                    ) from error
                result["routing_failure_evidence"] = {
                    "response": error.payload,
                    "element": match.group(1),
                    "required_variables": sorted(required),
                    "bound_variables": sorted(known),
                }
                raise CandidateFailure(str(error)) from error
            result["trace"].append(
                {
                    "action": "complete",
                    "task_id": task["id"],
                    "task_definition_key": identity,
                    "visit": visits[identity],
                    "variables": values,
                    "completion_defaults": binding.completion_values(identity, scenario),
                }
            )
        else:
            raise CandidateFailure("Scenario exceeded the bounded task-completion count")
        history = api.request("GET", f"history/historic-process-instances/{process_id}")
        activities = api.all("history/historic-activity-instances", processInstanceId=process_id)
        tasks = api.all("history/historic-task-instances", processInstanceId=process_id)
        ends = {
            row["activityId"]
            for row in activities
            if row.get("activityType") == "endEvent" and row.get("endTime")
        }
        completed = Counter(
            row["taskDefinitionKey"]
            for row in tasks
            if row.get("endTime") and not row.get("deleteReason")
        )
        assertions = {
            "expected_definition": history.get("processDefinitionId")
            == deployment["process_definition_id"],
            "completed": bool(history.get("endTime"))
            if scenario.get("should_complete", True)
            else True,
            "not_deleted": history.get("deleteReason")
            in {None, "terminate end event (escalation_final_end)"},
            "expected_end": scenario["expected_end_event"] in ends,
            "draft_completed": completed["team_drafts_initial_plan"] > 0,
            "execute_completed": scenario["expected_end_event"] != "main_end"
            or completed["team_executes_plan"] > 0,
            "bounded_plan_visits": completed[binding.discovered.get("joint_plan_task")] <= 3,
            "actual_timer_fired": not scenario.get("input_variables", {}).get("timer_expedited")
            or timer_fired,
        }
        if scenario.get("input_variables", {}).get("complianceCrisisDetected"):
            assertions["risk_before_decision"] = crisis_precedes_decisions(
                tasks, activities, binding.discovered
            )
        if scenario.get("input_variables", {}).get("peerOwnerAvailable") is False:
            peer_tasks = {
                identity
                for identity in binding.properties
                if "peerCategoryOwner" in binding.graph.get_assignee(identity)
            }
            assertions["unavailable_peer_bypassed"] = not any(
                completed[identity] for identity in peer_tasks
            )
        result.update(
            {
                "pass": all(assertions.values()),
                "assertions": assertions,
                "history": history,
                "activities": activities,
                "tasks": tasks,
            }
        )
    except CandidateFailure as error:
        result["failure"] = str(error)
    except EvaluationError as error:
        result["evaluation_error"] = str(error)
    finally:
        if process_id:
            result["process_instance_id"] = process_id
            if "history" not in result:
                result["history"] = api.request(
                    "GET", f"history/historic-process-instances/{process_id}"
                )
                result["activities"] = api.all(
                    "history/historic-activity-instances", processInstanceId=process_id
                )
                result["tasks"] = api.all(
                    "history/historic-task-instances", processInstanceId=process_id
                )
    if time.monotonic() >= global_deadline:
        result["evaluation_error"] = "Trusted evaluator suite exhausted its global runtime budget"
        result["pass"] = False
    return result


def run_suite(
    xml: bytes,
    manifest: dict,
    *,
    api_url: str,
    progress: Callable[[dict], None] = lambda row: None,
    timeout_s: float = 600,
) -> dict:
    scenarios = validate_manifest(manifest)
    started = time.monotonic()
    binding = Bindings(xml)
    api = Flowable(api_url)
    engine = api.request("GET", "management/engine")
    tenant = "ale-eval-" + uuid.uuid4().hex
    deployment = api.deploy(xml, tenant)
    results = probe_results(binding, scenarios)
    progress({"phase": "deployed", "deployment": deployment, "engine": engine})
    for row in results:
        progress(row)
    for scenario in scenarios:
        if scenario.get("is_structural_check") is True:
            continue
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise EvaluationError("Trusted evaluator suite exceeded its deadline")
        progress({"phase": "scenario_started", "scenario_id": scenario["id"]})
        try:
            result = run_case(
                api, binding, deployment, scenario, global_deadline=started + timeout_s
            )
        except EvaluationError as error:
            progress(
                {"phase": "evaluation_error", "scenario_id": scenario["id"], "error": str(error)}
            )
            raise
        results.append(result)
        progress(result)
        if result.get("evaluation_error"):
            raise EvaluationError(result["evaluation_error"])
    results.sort(
        key=lambda row: next(
            index
            for index, scenario in enumerate(scenarios)
            if scenario["id"] == row["scenario_id"]
        )
    )
    return {
        "deployment": deployment,
        "engine": engine,
        "duration_s": time.monotonic() - started,
        "bpmn_sha256": hashlib.sha256(xml).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        **summarize_results(results, scenarios),
        "scenarios": results,
        "authority": "evaluator_owned_flowable",
    }


async def evaluate_remote(
    session, xml: bytes, manifest: dict, progress: Callable[[dict], None]
) -> dict:
    validate_manifest(manifest)
    directory = "/home/user/.ale-audit/bpmn-evaluator-" + uuid.uuid4().hex
    await session.run_command("mkdir -p " + shlex.quote(directory + "/starter_project"), check=True)
    for name in ["runtime_validation.py", "evaluate_L3.py"]:
        await session.write_bytes(
            directory + "/" + name, (Path(__file__).parent / name).read_bytes()
        )
    await session.write_bytes(directory + "/modified_process.bpmn20.xml", xml)
    await session.write_bytes(
        directory + "/starter_project/test_scenarios_L3.json", json.dumps(manifest).encode()
    )
    command = [
        "python3",
        "-B",
        directory + "/runtime_validation.py",
        "--bpmn",
        directory + "/modified_process.bpmn20.xml",
        "--scenarios",
        directory + "/starter_project/test_scenarios_L3.json",
        "--output",
        directory + "/result.json",
        "--progress",
        directory + "/progress.jsonl",
    ]
    launch = (
        "import subprocess; "
        f"stream=open({directory + '/driver.log'!r},'xb'); "
        f"process=subprocess.Popen({command!r},cwd={directory!r},stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True); "
        "print(process.pid)"
    )
    receipt = await session.run_command("python3 -B -c " + shlex.quote(launch), check=True)
    progress({"phase": "runtime_launched", "directory": directory, "launch_receipt": receipt})
    deadline = time.monotonic() + 660
    seen = 0
    while True:
        if await session.file_exists(directory + "/progress.jsonl"):
            content = (await session.read_bytes(directory + "/progress.jsonl")).decode()
            lines = content.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines.pop()
            for line in lines[seen:]:
                progress(json.loads(line))
            seen = len(lines)
        if await session.file_exists(directory + "/result.json"):
            result = json.loads(await session.read_bytes(directory + "/result.json"))
            progress({"phase": "runtime_finished", "directory": directory, "result": result})
            if result.get("evaluation_error"):
                if result.get("error_type") == "CandidateFailure":
                    return {
                        "candidate_failure": result["evaluation_error"],
                        "authority": "evaluator_owned_flowable",
                        "bpmn_sha256": hashlib.sha256(xml).hexdigest(),
                        "manifest_sha256": hashlib.sha256(
                            json.dumps(manifest, sort_keys=True).encode()
                        ).hexdigest(),
                    }
                raise EvaluationError(result["evaluation_error"])
            return result
        if time.monotonic() >= deadline:
            raise EvaluationError(
                f"Runtime receipt missing after deadline; inspect {directory}; do not relaunch this job"
            )
        await asyncio.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpmn", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--api-url", default="http://localhost:8080/flowable-task/process-api")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    with Path(args.progress).open("x") as stream:

        def progress(row):
            stream.write(json.dumps(row) + "\n")
            stream.flush()

        try:
            result = run_suite(
                Path(args.bpmn).read_bytes(),
                json.loads(Path(args.scenarios).read_text()),
                api_url=args.api_url,
                progress=progress,
                timeout_s=args.timeout,
            )
        except Exception as error:
            result = {"evaluation_error": str(error), "error_type": type(error).__name__}
        Path(args.output).write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
