#!/usr/bin/env python3
"""
Run the 60 Case 2 L3 test scenarios against a live Flowable 6.5.0 instance.
Produces test_results.json for evaluation.

Uses anchor-based detection: original element IDs are known; new elements
added by the agent are detected by exclusion from the original set.

Usage:
  python run_tests.py \\
    --scenarios starter_project/test_scenarios_L3.json \\
    --process-key zMBCategoryGovernance_modified_L3 \\
    --output test_results.json \\
    --api-url http://localhost:8080/flowable-task/process-api \\
    --user admin --password test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_API_URL = "http://localhost:8080/flowable-task/process-api"
DEFAULT_PROCESS_KEY = "zMBCategoryGovernance_modified_L3"
ORIGINAL_PROCESS_KEY = "zMBCategoryGovernance_original"
DEFAULT_USER = "admin"
DEFAULT_PASSWORD = "test"

# Anchored elements from the original process
ORIGINAL_ELEMENTS = {
    "startEvent",
    "team_drafts_initial_plan",
    "team_executes_plan",
    "logistics_capacity_confirmation",
}

# End events
MAIN_END = "main_end"
TERMINAL_ESC_END = "escalation_final_end"

# Role variables (set at process start to "admin")
ROLE_VARS = [
    {"name": "mbTeamLead", "value": "admin"},
    {"name": "mbMerchantOpsLead", "value": "admin"},
    {"name": "mbBusinessAnalyst", "value": "admin"},
    {"name": "peerCategoryOwner_Beauty", "value": "admin"},
    {"name": "peerCategoryOwner_Apparel", "value": "admin"},
    {"name": "logisticsCoordinator", "value": "admin"},
    {"name": "zSeniorLead", "value": "admin"},
    {"name": "complianceOfficer", "value": "admin"},
    {"name": "interimCommitteeChair", "value": "admin"},
    {"name": "teamInternalCommitteeChair", "value": "admin"},
]

# Default decision variables (initialized to prevent "Unknown property" runtime errors)
DEFAULT_DECISION_VARS = [
    {"name": "peerOwnerAvailable", "value": True},
    {"name": "complianceCrisisDetected", "value": False},
    {"name": "inventoryOk", "value": True},
    {"name": "marginOk", "value": True},
    {"name": "aovOk", "value": True},
    {"name": "salesOk", "value": True},
    {"name": "brandRiskOk", "value": True},
    {"name": "platformRiskOk", "value": True},
    {"name": "escalationTier", "value": "minor"},
    {"name": "out_unified_decision", "value": "approved"},
    {"name": "jointPlanAttempt", "value": 0},
    {"name": "brandRiskReassessed", "value": False},
]

# Tasks whose completion auto-increments a counter
AUTO_INCREMENT_VARS = {
    "team_joint_mature_plan_preparation": "jointPlanAttempt",
}

# Globals (updated by CLI)
API_URL = DEFAULT_API_URL
PROCESS_KEY = DEFAULT_PROCESS_KEY
AUTH: Optional[HTTPBasicAuth] = None


# ============================================================================
# HTTP HELPERS
# ============================================================================


def api_get(path: str, retries: int = 2) -> Dict:
    url = f"{API_URL}/{path}"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, auth=AUTH, timeout=30)
            if r.status_code >= 500 and attempt < retries:
                time.sleep(1)
                continue
            if r.status_code >= 400:
                return {
                    "error": True,
                    "status": r.status_code,
                    "message": r.text[:500],
                }
            if not r.text.strip():
                return {"ok": True, "status": r.status_code}
            return r.json()
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(1)
                continue
            return {"error": True, "status": 0, "message": str(e)}
    return {"error": True, "status": 0, "message": "max retries exceeded"}


def api_post(path: str, data: Dict, retries: int = 2) -> Dict:
    url = f"{API_URL}/{path}"
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=data, auth=AUTH, timeout=30)
            if r.status_code >= 500 and attempt < retries:
                time.sleep(1)
                continue
            if r.status_code >= 400:
                return {
                    "error": True,
                    "status": r.status_code,
                    "message": r.text[:500],
                }
            if not r.text.strip():
                return {"ok": True, "status": r.status_code}
            return r.json()
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(1)
                continue
            return {"error": True, "status": 0, "message": str(e)}
    return {"error": True, "status": 0, "message": "max retries exceeded"}


# ============================================================================
# PROCESS OPERATIONS
# ============================================================================


def start_process(input_variables: Optional[Dict] = None) -> Dict:
    """Start a new process instance with role + default decision vars."""
    all_vars = list(ROLE_VARS) + list(DEFAULT_DECISION_VARS)
    if input_variables:
        existing_names = {v["name"] for v in all_vars}
        for k, v in input_variables.items():
            if k in existing_names:
                for entry in all_vars:
                    if entry["name"] == k:
                        entry["value"] = v
                        break
            else:
                all_vars.append({"name": k, "value": v})
    return api_post("runtime/process-instances", {
        "processDefinitionKey": PROCESS_KEY,
        "variables": all_vars,
    })


def build_task_output_map(task_outputs: Dict) -> Dict:
    """Normalize task_outputs into per-task-key round-indexed dicts."""
    result: Dict = {}
    for task_key, outputs in task_outputs.items():
        if isinstance(outputs, list):
            result[task_key] = {i + 1: out for i, out in enumerate(outputs)}
        elif isinstance(outputs, dict):
            result[task_key] = {"_default": outputs}
        else:
            continue
    return result


def complete_all_tasks(
    proc_id: str,
    task_output_map: Optional[Dict] = None,
    max_steps: int = 30,
) -> Dict:
    """Complete all pending tasks for a process instance."""
    task_counts: Dict[str, int] = {}
    errors: List[Dict] = []
    steps = 0
    for step_idx in range(max_steps):
        steps = step_idx + 1
        tasks = api_get(f"runtime/tasks?processInstanceId={proc_id}&size=50")
        if tasks.get("error"):
            errors.append({
                "step": steps, "phase": "fetch_tasks",
                "status": tasks.get("status"),
                "message": tasks.get("message", "fetch failed"),
            })
            break
        if not tasks.get("data"):
            break
        for task in tasks["data"]:
            task_key = task.get("taskDefinitionKey", "")
            task_counts[task_key] = task_counts.get(task_key, 0) + 1
            count = task_counts[task_key]

            vars_to_set: Dict = {}
            # Auto-increment counter
            if task_key in AUTO_INCREMENT_VARS:
                vars_to_set[AUTO_INCREMENT_VARS[task_key]] = count

            # Scenario outputs
            if task_output_map and task_key in task_output_map:
                by_round = task_output_map[task_key]
                if count in by_round:
                    vars_to_set.update(by_round[count])
                elif "_default" in by_round:
                    vars_to_set.update(by_round["_default"])

            completion = {"action": "complete"}
            if vars_to_set:
                completion["variables"] = [
                    {"name": k, "value": v} for k, v in vars_to_set.items()
                ]
            resp = api_post(f"runtime/tasks/{task['id']}", completion)
            if resp.get("error"):
                errors.append({
                    "step": steps, "phase": "complete_task",
                    "task_id": task.get("id"),
                    "task_definition_key": task_key,
                    "status": resp.get("status"),
                    "message": resp.get("message", "task completion failed"),
                })
                return {"ok": False, "steps_executed": steps, "errors": errors}
        time.sleep(0.15)
    return {"ok": len(errors) == 0, "steps_executed": steps, "errors": errors}


def get_activity_history(proc_id: str) -> List[str]:
    resp = api_get(
        f"history/historic-activity-instances?processInstanceId={proc_id}"
        f"&orderBy=startTime&order=asc&size=200"
    )
    if "data" not in resp:
        return []
    seen = set()
    activities = []
    for a in resp["data"]:
        aid = a.get("activityId", "")
        atype = a.get("activityType", "")
        if atype in (
            "userTask", "exclusiveGateway", "inclusiveGateway",
            "parallelGateway", "startEvent", "endEvent",
            "subProcess", "boundaryEvent",
        ):
            if aid and aid not in seen:
                activities.append(aid)
                seen.add(aid)
    return activities


def check_process_ended(proc_id: str) -> bool:
    resp = api_get(f"history/historic-process-instances/{proc_id}")
    return resp.get("endTime") is not None


# ============================================================================
# SCENARIO VALIDATION
# ============================================================================


def validate_scenario(
    scenario: Dict,
    actual_path: List[str],
    ended: bool,
) -> Dict:
    """Validate the actual runtime path against the scenario's expectations."""
    category = scenario.get("category", "")
    expected_end_event = scenario.get("expected_end_event", MAIN_END)
    should_complete = scenario.get("should_complete", True)

    result = {
        "category": category,
        "expected_end_event": expected_end_event,
        "ended": ended,
        "has_anchor_draft": "team_drafts_initial_plan" in actual_path,
        "has_anchor_execute": "team_executes_plan" in actual_path,
        "has_logistics": "logistics_capacity_confirmation" in actual_path,
        "reached_main_end": MAIN_END in actual_path,
        "reached_escalation_end": TERMINAL_ESC_END in actual_path,
    }

    # Anchor checks
    anchor_ok = "team_drafts_initial_plan" in actual_path
    if category not in ("anti_gaming",):
        if expected_end_event == "main_end":
            anchor_ok = anchor_ok and "team_executes_plan" in actual_path
            anchor_ok = anchor_ok and MAIN_END in actual_path
        elif expected_end_event == "escalation_final_end":
            # escalation-final doesn't require team_executes_plan
            anchor_ok = anchor_ok and TERMINAL_ESC_END in actual_path

    # Category-specific extras
    extras_ok = True
    if category == "compliance_crisis":
        extras_ok = "emergency_delist_task" in actual_path or "compliance_subprocess" in actual_path
    elif category == "peer_unavailable":
        # peer-unavailable should still complete; could bypass peer tasks
        extras_ok = True
    elif category == "escalation_timeout":
        extras_ok = True  # timer firing not easily triggered in tests
    elif category == "bounded_loop":
        # Bounded loop should exhaust and reach escalation_final_end (for failing variants)
        if expected_end_event == "escalation_final_end":
            extras_ok = TERMINAL_ESC_END in actual_path
    elif category == "anti_gaming":
        # Structural check — pass if BPMN deployable
        return {**result, "pass": True, "note": "Structural check handled by evaluate_L3.py"}

    result["pass"] = anchor_ok and extras_ok and (ended if should_complete else True)
    if not result["pass"]:
        if not anchor_ok:
            result["anchor_error"] = "Anchor or expected end event not reached"
        if not extras_ok:
            result["category_error"] = f"Category {category} extras not satisfied"
    return result


def run_scenario(scenario: Dict) -> Dict:
    sid = scenario["id"]
    result: Dict = {
        "scenario_id": sid,
        "description": scenario.get("description", ""),
        "category": scenario.get("category", ""),
    }

    # Structural-only scenarios
    if scenario.get("is_structural_check"):
        result["pass"] = True
        result["validation"] = "Structural check handled by evaluate_L3.py"
        result["validation_focus"] = scenario.get("validation_focus", "")
        return result

    input_vars = scenario.get("input_variables", {})
    task_outputs = scenario.get("task_outputs", {})
    task_output_map = build_task_output_map(task_outputs) if task_outputs else None

    proc = start_process(input_vars)
    if proc.get("error"):
        result["pass"] = False
        result["error"] = proc.get("message", "failed to start process")
        return result

    proc_id = proc.get("id", "")
    result["process_instance_id"] = proc_id

    completion = complete_all_tasks(proc_id, task_output_map=task_output_map)
    actual_path = get_activity_history(proc_id)
    ended = check_process_ended(proc_id)

    result["actual_path"] = actual_path
    result["ended"] = ended
    result["steps_executed"] = completion.get("steps_executed", 0)
    if not completion.get("ok"):
        result["completion_error"] = completion.get("errors", [])

    val = validate_scenario(scenario, actual_path, ended)
    result.update(val)
    return result


# ============================================================================
# MAIN
# ============================================================================


def main():
    global API_URL, PROCESS_KEY, AUTH
    parser = argparse.ArgumentParser(
        description="Run Case 2 L3 test scenarios against Flowable 6.5.0"
    )
    parser.add_argument("--scenarios", required=True, help="Test scenarios JSON file")
    parser.add_argument("--process-key", default=DEFAULT_PROCESS_KEY,
                        help="Process definition key to start")
    parser.add_argument("--output", default="test_results.json", help="Output path")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Flowable REST API base URL")
    parser.add_argument("--user", default=DEFAULT_USER, help="Flowable user")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="Flowable password")
    args = parser.parse_args()

    API_URL = args.api_url
    PROCESS_KEY = args.process_key
    AUTH = HTTPBasicAuth(args.user, args.password)

    with open(args.scenarios) as f:
        data = json.load(f)
    scenarios = data.get("scenarios", [])
    total = len(scenarios)

    print(f"Running {total} test scenarios against {API_URL}")
    print(f"Process key: {PROCESS_KEY}")
    print("=" * 60)

    results: List[Dict] = []
    passed = 0
    for scenario in scenarios:
        sid = scenario["id"]
        desc = scenario.get("description", "")[:60]
        print(f"  {sid}: {desc}...", end=" ", flush=True)
        try:
            r = run_scenario(scenario)
        except Exception as e:
            r = {
                "scenario_id": sid,
                "description": scenario.get("description", ""),
                "category": scenario.get("category", ""),
                "pass": False,
                "error": f"runner_exception: {e}",
            }
        results.append(r)
        status = "PASS" if r.get("pass") else "FAIL"
        if r.get("pass"):
            passed += 1
        print(status)
        if not r.get("pass"):
            if "error" in r:
                print(f"       Error: {r['error'][:120]}")
            elif r.get("anchor_error"):
                print(f"       Anchor: {r['anchor_error']}")
            elif r.get("category_error"):
                print(f"       Category: {r['category_error']}")

    print("=" * 60)
    pct = (passed / total * 100) if total else 0
    print(f"Results: {passed}/{total} passed ({pct:.0f}%)")
    threshold = int(total * 0.9)
    meets = passed >= threshold
    print(f"Threshold: >= {threshold}/{total} (90%) - {'PASS' if meets else 'FAIL'}")

    output = {
        "api_url": API_URL,
        "process_definition_key": PROCESS_KEY,
        "total_scenarios": total,
        "passed_scenarios": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "meets_threshold": meets,
        "scenarios": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    sys.exit(0 if meets else 1)


if __name__ == "__main__":
    main()
