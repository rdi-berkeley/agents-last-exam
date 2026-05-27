#!/usr/bin/env python3
"""
L3 Evaluator for: LY Juice Monthly Scheduling — Compound Disruption
(Raw Material Shortage + Quality Hold)

Extends the L2 evaluator with additional checks for:
  - Path sensitivity analysis (behavioral anti-gaming)
  - Condition-upstream-output coupling
  - Parallel var production and loop counter checks
  - Compliance cross-validation (rule_17-rule_22, mod_16-mod_18)
  - Dead path and anti-gaming refinement checks
  - Bonus checks (semantic noop, minimality intent, subprocess)

All L1 and L2 checks still apply. L3 adds:
  Phase 1: Anti-gaming behavioral checks (D13-D14)
  Phase 2: Structural checks (A25, A27-A28)
  Phase 3: Compliance cross-validation (C30, C32-C33)
  Phase 4: Dead path & anti-gaming refinement (D15-D16, D11-D12, A29, A31)
  Bonus:  D17, D18, check_23

Usage:
  python evaluate_L3.py --bpmn modified_process.bpmn20.xml \\
                        --structural structural_changes.json \\
                        --rules business_rules_compliance.json \\
                        --results test_results.json
"""

import argparse
import json
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

# Import base evaluator components
from evaluate import (
    BPMN_NS,
    FLOWABLE_NS,
    ORIGINAL_ELEMENTS,
    CLOSED_LOOP_TARGETS,
    DECISION_GATEWAY_TYPES,
    ORIGINAL_OUTPUT_PROPERTIES,
    ORIGINAL_ROLE_ASSIGNMENTS,
    AUTHORITY_LEVELS,
    ROLE_AUTHORITY,
    CANDIDATE_GROUP_AUTHORITY,
    BPMNGraph,
    discover_disruption_chain,
    check_structural as check_structural_L1,
    check_anti_gaming as check_anti_gaming_L1,
    check_compliance as check_compliance_base,
    check_test_results,
    check_data_flow as check_data_flow_L1,
    check_role_data_coupling as check_role_data_coupling_L1,
    _all_predecessors_bfs,
    _snake_to_camel,
    _get_authority_level,
)

# Import L2 evaluator components
from evaluate_L2 import (
    check_structural_L2,
    check_anti_gaming_L2,
    check_compliance_L2,
    check_data_flow_L2,
    check_role_data_coupling_L2,
    discover_L2_chain,
    REQUIRED_MODIFICATIONS_L2,
    REQUIRED_RULES_L2,
    ROLE_AUTHORITY_L2,
    CANDIDATE_GROUP_AUTHORITY_L2,
)

# ============================================================
# L3 CONSTANTS
# ============================================================

# L3 extends the compliance requirements
REQUIRED_MODIFICATIONS_L3 = REQUIRED_MODIFICATIONS_L2 + [
    "modification_16",  # Timer escalation subprocess or flat equivalent
    "modification_17",  # Expedited quality with risk+waiver conditions
    "modification_18",  # Cost threshold monotonic escalation
]

REQUIRED_RULES_L3 = REQUIRED_RULES_L2 + [
    "rule_17",   # Expedite vs risk vs waiver decision
    "rule_18",   # Cost threshold escalation ordering
    "rule_19",   # Override rationale requirement
    "rule_20",   # Cross-lane dependency for mix adjustment
    "rule_21",   # Temporal constraint: timer to expedited
    "rule_22",   # Dead path elimination
]

# EL keywords for variable extraction
EL_KEYWORDS = {
    "true", "false", "null", "empty", "not", "and", "or",
    "eq", "ne", "lt", "gt", "le", "ge", "div", "mod",
    "instanceof", "NEEDS_ADJUSTMENT", "CONFIRMED",
    "AVAILABLE", "DISRUPTED", "RESOLVED", "UNRESOLVED",
}

VAR_PATTERN = re.compile(r"[a-zA-Z_]\w*")


# ============================================================
# EL CONDITION EVALUATOR (for path tracing)
# ============================================================

def evaluate_el_condition(condition_str, variables):
    """
    Evaluate a BPMN EL condition expression against a set of variables.

    Returns True if the condition is satisfied, False if not, or
    "inconclusive" if the expression cannot be parsed.

    Supports: ==, eq, !=, ne, >, gt, <, lt, >=, ge, <=, le,
    string literals, numeric literals, boolean.
    """
    if not condition_str:
        return "inconclusive"

    cond = condition_str.strip()
    # Strip ${...} wrapper
    if cond.startswith("${") and cond.endswith("}"):
        cond = cond[2:-1].strip()

    if not cond:
        return "inconclusive"

    # Handle boolean literals
    if cond.lower() == "true":
        return True
    if cond.lower() == "false":
        return False

    # Handle negated bare variable: !varName
    neg_match = re.fullmatch(r"!(\w+)", cond)
    if neg_match:
        var_name = neg_match.group(1)
        val = variables.get(var_name)
        if val is not None:
            return not bool(val)
        return "inconclusive"

    # Handle bare variable as boolean: varName
    bare_match = re.fullmatch(r"(\w+)", cond)
    if bare_match:
        var_name = bare_match.group(1)
        if var_name.lower() not in {"true", "false", "null"}:
            val = variables.get(var_name)
            if val is not None:
                return bool(val)
            return "inconclusive"

    # Try to parse simple binary comparisons:
    # var == 'value', var != 'value', var > number, etc.
    # Also handle: var eq 'value', var ne 'value', etc.
    patterns = [
        # var == 'string' or var eq 'string'
        (r"(\w+)\s*(?:==|eq)\s*'([^']*)'", "str_eq"),
        # var != 'string' or var ne 'string'
        (r"(\w+)\s*(?:!=|ne)\s*'([^']*)'", "str_ne"),
        # var == number or var eq number
        (r"(\w+)\s*(?:==|eq)\s*([0-9]+(?:\.[0-9]+)?)", "num_eq"),
        # var != number or var ne number
        (r"(\w+)\s*(?:!=|ne)\s*([0-9]+(?:\.[0-9]+)?)", "num_ne"),
        # var > number or var gt number
        (r"(\w+)\s*(?:>|gt)\s*([0-9]+(?:\.[0-9]+)?)", "num_gt"),
        # var < number or var lt number
        (r"(\w+)\s*(?:<|lt)\s*([0-9]+(?:\.[0-9]+)?)", "num_lt"),
        # var >= number or var ge number
        (r"(\w+)\s*(?:>=|ge)\s*([0-9]+(?:\.[0-9]+)?)", "num_ge"),
        # var <= number or var le number
        (r"(\w+)\s*(?:<=|le)\s*([0-9]+(?:\.[0-9]+)?)", "num_le"),
        # var == true/false
        (r"(\w+)\s*(?:==|eq)\s*(true|false)", "bool_eq"),
        # var != true/false
        (r"(\w+)\s*(?:!=|ne)\s*(true|false)", "bool_ne"),
        # 'string' == var (reversed)
        (r"'([^']*)'\s*(?:==|eq)\s*(\w+)", "str_eq_rev"),
        # number > var (reversed comparison)
        (r"([0-9]+(?:\.[0-9]+)?)\s*(?:>|gt)\s*(\w+)", "num_lt_rev"),
        (r"([0-9]+(?:\.[0-9]+)?)\s*(?:<|lt)\s*(\w+)", "num_gt_rev"),
        (r"([0-9]+(?:\.[0-9]+)?)\s*(?:>=|ge)\s*(\w+)", "num_le_rev"),
        (r"([0-9]+(?:\.[0-9]+)?)\s*(?:<=|le)\s*(\w+)", "num_ge_rev"),
    ]

    for pat, op_type in patterns:
        m = re.fullmatch(pat, cond)
        if m:
            g1, g2 = m.group(1), m.group(2)

            if op_type == "str_eq":
                val = variables.get(g1)
                return val == g2 if val is not None else "inconclusive"
            elif op_type == "str_ne":
                val = variables.get(g1)
                return val != g2 if val is not None else "inconclusive"
            elif op_type == "str_eq_rev":
                val = variables.get(g2)
                return val == g1 if val is not None else "inconclusive"
            elif op_type == "num_eq":
                val = variables.get(g1)
                if val is None:
                    return "inconclusive"
                try:
                    return float(val) == float(g2)
                except (ValueError, TypeError):
                    return "inconclusive"
            elif op_type == "num_ne":
                val = variables.get(g1)
                if val is None:
                    return "inconclusive"
                try:
                    return float(val) != float(g2)
                except (ValueError, TypeError):
                    return "inconclusive"
            elif op_type == "num_gt":
                val = variables.get(g1)
                if val is None:
                    return "inconclusive"
                try:
                    return float(val) > float(g2)
                except (ValueError, TypeError):
                    return "inconclusive"
            elif op_type == "num_lt":
                val = variables.get(g1)
                if val is None:
                    return "inconclusive"
                try:
                    return float(val) < float(g2)
                except (ValueError, TypeError):
                    return "inconclusive"
            elif op_type == "num_ge":
                val = variables.get(g1)
                if val is None:
                    return "inconclusive"
                try:
                    return float(val) >= float(g2)
                except (ValueError, TypeError):
                    return "inconclusive"
            elif op_type == "num_le":
                val = variables.get(g1)
                if val is None:
                    return "inconclusive"
                try:
                    return float(val) <= float(g2)
                except (ValueError, TypeError):
                    return "inconclusive"
            elif op_type == "bool_eq":
                val = variables.get(g1)
                if val is None:
                    return "inconclusive"
                bool_val = g2.lower() == "true"
                if isinstance(val, bool):
                    return val == bool_val
                return str(val).lower() == g2.lower()
            elif op_type == "bool_ne":
                val = variables.get(g1)
                if val is None:
                    return "inconclusive"
                bool_val = g2.lower() == "true"
                if isinstance(val, bool):
                    return val != bool_val
                return str(val).lower() != g2.lower()
            elif op_type == "num_lt_rev":
                val = variables.get(g2)
                if val is None:
                    return "inconclusive"
                try:
                    return float(val) < float(g1)
                except (ValueError, TypeError):
                    return "inconclusive"
            elif op_type == "num_gt_rev":
                val = variables.get(g2)
                if val is None:
                    return "inconclusive"
                try:
                    return float(val) > float(g1)
                except (ValueError, TypeError):
                    return "inconclusive"
            elif op_type == "num_le_rev":
                val = variables.get(g2)
                if val is None:
                    return "inconclusive"
                try:
                    return float(val) <= float(g1)
                except (ValueError, TypeError):
                    return "inconclusive"
            elif op_type == "num_ge_rev":
                val = variables.get(g2)
                if val is None:
                    return "inconclusive"
                try:
                    return float(val) >= float(g1)
                except (ValueError, TypeError):
                    return "inconclusive"

    return "inconclusive"


# ============================================================
# PATH TRACER (for behavioral anti-gaming)
# ============================================================

def trace_path(graph, start_id, variables, max_steps=50):
    """
    Trace the execution path through the BPMN graph from start_id,
    evaluating gateway conditions using the given variables.

    Returns a list of node IDs visited in order.
    """
    path = []
    current = start_id
    visited_count = {}  # track visits to detect loops

    for _ in range(max_steps):
        if current is None:
            break

        path.append(current)
        visited_count[current] = visited_count.get(current, 0) + 1

        # Stop on loop (visited same node 3+ times)
        if visited_count[current] >= 3:
            break

        # Stop at end events
        if graph.get_type(current) == "endEvent":
            break

        successors = graph.successors(current)
        if not successors:
            break

        if len(successors) == 1:
            current = successors[0]
            continue

        # Decision gateway: evaluate conditions to pick path
        if graph.is_decision_gateway(current):
            flows = graph.outgoing_flows_with_conditions(current)

            # Try to find the matching conditional flow
            chosen = None
            default_flow_target = None

            # Check for default flow
            elem = graph.get_element(current)
            default_flow_id = elem.get("default", "") if elem is not None else ""

            for flow in flows:
                if flow["flow_id"] == default_flow_id:
                    default_flow_target = flow["target"]
                    continue
                cond = flow.get("condition")
                if cond:
                    result = evaluate_el_condition(cond, variables)
                    if result is True:
                        chosen = flow["target"]
                        break

            if chosen:
                current = chosen
            elif default_flow_target:
                current = default_flow_target
            else:
                # No condition matched and no default — take first flow
                current = flows[0]["target"] if flows else None
            continue

        # Parallel gateway: follow all branches conceptually,
        # but for path tracing we follow the first branch
        # (parallel gateways converge anyway)
        current = successors[0]

    return path


# ============================================================
# VARIABLE EXTRACTION HELPER
# ============================================================

def _extract_vars(graph, gateway_id):
    """Extract all variable names from a gateway's outgoing conditions.

    Strips quoted string literals before tokenizing, so that
    'A', 'B', 'approved', etc. are not treated as variable names.
    """
    v = set()
    for f in graph.outgoing_flows_with_conditions(gateway_id):
        cond = f.get("condition")
        if cond:
            inner = cond.strip()
            if inner.startswith("${") and inner.endswith("}"):
                inner = inner[2:-1]
            # Remove quoted string literals before tokenizing
            inner = re.sub(r"'[^']*'", "", inner)
            inner = re.sub(r'"[^"]*"', "", inner)
            for tok in VAR_PATTERN.findall(inner):
                if tok not in EL_KEYWORDS and len(tok) > 1:
                    v.add(tok)
    return v


# ============================================================
# PHASE 1: ANTI-GAMING BEHAVIORAL CHECKS
# ============================================================

def check_anti_gaming_L3(root: ET.Element) -> dict:
    """L2 anti-gaming checks plus L3 behavioral checks."""
    results = check_anti_gaming_L2(root)
    graph = BPMNGraph(root)

    # --- D13: Path sensitivity analysis ---
    # Two scenarios must diverge at >=2 of 3 target gateways
    scenario_a = {
        "supplierRisk": "low",
        "supplyRiskDetected": False,
        "out_supplyRiskDetected": False,
        "materialGrade": "A",
        "costVariance": 0.05,
        "out_cost_variance": 0.05,
        "supply_status": "AVAILABLE",
        "resolution_status": "RESOLVED",
        "out_supply_status": "AVAILABLE",
        "out_resolution_status": "RESOLVED",
        "out_material_grade": "A",
        "out_assessment_result": "CONFIRMED",
        "overall_verdict": "CONFIRMED",
        "sourcingSuccessful": True,
        "out_sourcingSuccessful": True,
    }
    scenario_b = {
        "supplierRisk": "high",
        "supplyRiskDetected": True,
        "out_supplyRiskDetected": True,
        "materialGrade": "C",
        "costVariance": 0.30,
        "out_cost_variance": 0.30,
        "supply_status": "DISRUPTED",
        "resolution_status": "UNRESOLVED",
        "out_supply_status": "DISRUPTED",
        "out_resolution_status": "UNRESOLVED",
        "out_material_grade": "C",
        "out_assessment_result": "NEEDS_ADJUSTMENT",
        "overall_verdict": "NEEDS_ADJUSTMENT",
        "sourcingSuccessful": False,
        "out_sourcingSuccessful": False,
    }

    path_a = trace_path(graph, "startEvent", scenario_a)
    path_b = trace_path(graph, "startEvent", scenario_b)

    # Run L2 discovery to find target gateways
    l2_discovered = discover_L2_chain(graph, {})
    target_gateways = [
        l2_discovered.get("risk_gateway"),
        l2_discovered.get("quality_grade_gateway"),
        l2_discovered.get("cost_variance_gateway"),
    ]

    divergence_count = 0
    divergence_details = {}

    # Also check if the two paths end at different endpoints (overall divergence)
    end_a = path_a[-1] if path_a else None
    end_b = path_b[-1] if path_b else None
    paths_have_different_endpoints = end_a != end_b

    for gw_name, gw_id in zip(
        ["risk_gateway", "quality_grade_gateway", "cost_variance_gateway"],
        target_gateways,
    ):
        if gw_id is None:
            divergence_details[gw_name] = "gateway not found"
            continue

        # Find the index of this gateway in each path
        idx_a = path_a.index(gw_id) if gw_id in path_a else -1
        idx_b = path_b.index(gw_id) if gw_id in path_b else -1

        if idx_a < 0 or idx_b < 0:
            # Gateway reached in one path but not the other — this IS divergence
            # if the overall paths are different (one scenario bypasses this gateway)
            if (idx_a >= 0) != (idx_b >= 0):
                divergence_count += 1
                reached_in = "A" if idx_a >= 0 else "B"
                skipped_in = "B" if idx_a >= 0 else "A"
                divergence_details[gw_name] = f"diverged (reached in {reached_in}, skipped in {skipped_in})"
            else:
                divergence_details[gw_name] = "not reached in either path"
            continue

        # Check what comes AFTER the gateway in each path
        next_a = path_a[idx_a + 1] if idx_a + 1 < len(path_a) else None
        next_b = path_b[idx_b + 1] if idx_b + 1 < len(path_b) else None

        if next_a != next_b:
            divergence_count += 1
            divergence_details[gw_name] = f"diverged (A->{next_a}, B->{next_b})"
        else:
            divergence_details[gw_name] = f"same path (both->{next_a})"

    results["d13_path_sensitivity_analysis"] = divergence_count >= 2
    results["d13_divergence_count"] = divergence_count
    results["d13_divergence_details"] = divergence_details

    # --- D14: Conditions use upstream outputs ---
    # For each non-original decision gateway, check condition variables
    # have upstream task producers with out_{variable_name}
    d14_pass = True
    d14_issues = []

    for node_id in graph.elements:
        if not graph.is_decision_gateway(node_id):
            continue
        if node_id in ORIGINAL_ELEMENTS:
            continue

        cond_vars = _extract_vars(graph, node_id)
        if not cond_vars:
            continue

        predecessors = _all_predecessors_bfs(graph, node_id, max_depth=10)

        # Loop/counter variables are set programmatically, not via formProperties
        COUNTER_VARS = {"mixRetryCount", "retryCount", "loopCount", "iterationCount",
                        "mixretrycount", "retrycount", "loopcount", "attemptcount"}

        for var_name in cond_vars:
            # Skip counter/loop variables — they're set by execution, not formProperties
            if var_name in COUNTER_VARS or var_name.lower() in COUNTER_VARS:
                continue

            found_producer = False
            # Normalize variable name for matching
            snake_var = re.sub(r'(?<!^)(?=[A-Z])', '_', var_name).lower()

            for pred_id in predecessors:
                if graph.get_type(pred_id) != "userTask":
                    continue
                props = graph.get_form_properties(pred_id)
                for pid in props:
                    if not pid.startswith("out_"):
                        continue
                    # Exact match: var_name is already the property ID
                    if pid == var_name:
                        found_producer = True
                        break
                    # Prefixed match: out_{var_name}
                    if pid == f"out_{var_name}":
                        found_producer = True
                        break
                    # Snake-case match: out_supplier_risk for supplierRisk
                    if pid == f"out_{snake_var}":
                        found_producer = True
                        break
                    # CamelCase match: out_supply_risk_detected -> supplyRiskDetected
                    camel = _snake_to_camel(pid[4:])
                    if camel == var_name:
                        found_producer = True
                        break
                    # Strip out_ prefix from var_name if it already has one
                    if var_name.startswith("out_") and pid == var_name:
                        found_producer = True
                        break
                if found_producer:
                    break

            if not found_producer:
                d14_pass = False
                d14_issues.append({
                    "gateway": node_id,
                    "variable": var_name,
                    "issue": "no upstream task produces matching out_ property",
                })

    results["d14_conditions_use_upstream_outputs"] = d14_pass
    if d14_issues:
        results["d14_issues"] = d14_issues

    # --- D11: No constant-only routing ---
    # No decision gateway has ALL conditions that are tautologies
    d11_pass = True
    d11_issues = []
    tautology_patterns = {
        "${true}", "${1==1}", "${1 == 1}", "${!false}",
        "true", "1==1", "1 == 1",
    }

    for node_id in graph.elements:
        if not graph.is_decision_gateway(node_id):
            continue
        if node_id in ORIGINAL_ELEMENTS:
            continue

        flows = graph.outgoing_flows_with_conditions(node_id)
        conditions = [f.get("condition") for f in flows if f.get("condition")]

        if not conditions:
            continue

        all_tautology = all(
            c.strip().lower() in {t.lower() for t in tautology_patterns}
            for c in conditions
        )
        if all_tautology:
            d11_pass = False
            d11_issues.append({
                "gateway": node_id,
                "conditions": conditions,
                "issue": "all conditions are tautologies",
            })

    results["d11_no_constant_only_routing"] = d11_pass
    if d11_issues:
        results["d11_issues"] = d11_issues

    # --- D12: Gateway 3-way exists ---
    # At least one decision gateway has 3+ outgoing conditional flows
    d12_pass = False
    for node_id in graph.elements:
        if not graph.is_decision_gateway(node_id):
            continue
        if node_id in ORIGINAL_ELEMENTS:
            continue

        flows = graph.outgoing_flows_with_conditions(node_id)
        conditional_flows = [f for f in flows if f.get("condition")]
        if len(conditional_flows) >= 3 or len(flows) >= 3:
            d12_pass = True
            break

    results["d12_gateway_3way_exists"] = d12_pass

    # --- D15: No dead branches ---
    # Every non-original node must be reachable from startEvent
    # AND reach at least one endEvent
    d15_pass = True
    d15_issues = []

    # Build boundary event parent map: if parent is reachable, boundary event is too
    boundary_parent = {}
    for eid in graph.elements:
        elem = graph.get_element(eid)
        if elem is not None and graph.get_type(eid) == "boundaryEvent":
            attached = elem.get("attachedToRef", "")
            if attached:
                boundary_parent[eid] = attached

    # BFS forward from startEvent to find all reachable nodes
    reachable_from_start = set()
    queue = deque(["startEvent"])
    while queue:
        node = queue.popleft()
        if node in reachable_from_start:
            continue
        reachable_from_start.add(node)
        # Also mark boundary events as reachable if their parent is reachable
        for be_id, parent_id in boundary_parent.items():
            if parent_id == node and be_id not in reachable_from_start:
                queue.append(be_id)
        for succ in graph.successors(node):
            if succ not in reachable_from_start:
                queue.append(succ)

    # BFS backward from all endEvents to find all nodes that reach an endEvent
    reaches_end = set()
    queue = deque(list(graph.end_events))
    while queue:
        node = queue.popleft()
        if node in reaches_end:
            continue
        reaches_end.add(node)
        for pred in graph.predecessors(node):
            if pred not in reaches_end:
                queue.append(pred)

    for node_id in graph.elements:
        if node_id in ORIGINAL_ELEMENTS:
            continue
        if node_id not in reachable_from_start:
            d15_pass = False
            d15_issues.append({
                "node": node_id,
                "issue": "not reachable from startEvent",
            })
        elif node_id not in reaches_end:
            d15_pass = False
            d15_issues.append({
                "node": node_id,
                "issue": "cannot reach any endEvent",
            })

    results["d15_no_dead_branches"] = d15_pass
    if d15_issues:
        results["d15_issues"] = d15_issues

    # --- D16: All new branches exercised ---
    # For each non-original decision gateway, every outgoing flow target
    # must reach at least one endEvent
    d16_pass = True
    d16_issues = []

    for node_id in graph.elements:
        if not graph.is_decision_gateway(node_id):
            continue
        if node_id in ORIGINAL_ELEMENTS:
            continue

        flows = graph.outgoing_flows_with_conditions(node_id)
        for flow in flows:
            target = flow["target"]
            if target not in reaches_end:
                d16_pass = False
                d16_issues.append({
                    "gateway": node_id,
                    "flow_target": target,
                    "issue": "flow target cannot reach any endEvent",
                })

    results["d16_all_new_branches_exercised"] = d16_pass
    if d16_issues:
        results["d16_issues"] = d16_issues

    return results


# ============================================================
# PHASE 2: STRUCTURAL CHECKS (extends L2)
# ============================================================

def check_structural_L3(root: ET.Element):
    """Run all structural checks (L1 + L2 + L3)."""
    # Get L2 structural results first
    l2_results, l2_details = check_structural_L2(root)

    graph = BPMNGraph(root)
    l2_discovered = discover_L2_chain(graph, {})

    l3_results = dict(l2_results)
    l3_details = dict(l2_details)

    # --- Check 25: Parallel var production ---
    # Both branches of parallel approval split must produce out_* formProperties.
    # sc_director_review_task and sales_confirmation_task each have at least one out_*.
    # Post-join gateway condition must reference variables from BOTH branches.
    sc_review = l2_discovered.get("sc_director_review_task")
    sales_task = l2_discovered.get("sales_confirmation_task")
    approval_join = l2_discovered.get("approval_join")
    sales_gw = l2_discovered.get("sales_gateway")

    check_25_pass = False
    sc_has_out = False
    sales_has_out = False

    if sc_review:
        props = graph.get_form_properties(sc_review)
        sc_out_props = {pid for pid in props if pid.startswith("out_")}
        sc_has_out = len(sc_out_props) >= 1
    if sales_task:
        props = graph.get_form_properties(sales_task)
        sales_out_props = {pid for pid in props if pid.startswith("out_")}
        sales_has_out = len(sales_out_props) >= 1

    if sc_has_out and sales_has_out:
        # Check post-join gateway references variables from BOTH branches
        post_join_gw = sales_gw
        if not post_join_gw and approval_join:
            for succ in graph.successors(approval_join):
                if graph.is_decision_gateway(succ) and succ not in ORIGINAL_ELEMENTS:
                    post_join_gw = succ
                    break

        if post_join_gw:
            gw_vars = _extract_vars(graph, post_join_gw)
            # Check if gateway references at least one var from each branch
            sc_var_names = set()
            for pid in (sc_out_props if sc_review else set()):
                sc_var_names.add(_snake_to_camel(pid[4:]))
                sc_var_names.add(pid[4:])  # also add snake_case
            sales_var_names = set()
            for pid in (sales_out_props if sales_task else set()):
                sales_var_names.add(_snake_to_camel(pid[4:]))
                sales_var_names.add(pid[4:])

            refs_sc = bool(gw_vars & sc_var_names)
            refs_sales = bool(gw_vars & sales_var_names)
            check_25_pass = refs_sc and refs_sales
        else:
            # No post-join gateway found; pass if both have outputs
            check_25_pass = True

    l3_results["check_25_parallel_var_production"] = check_25_pass

    # --- Check 27: Loop counter variable ---
    # mix_validation_gateway must have condition referencing counter/retry variable
    mix_val_gw = l2_discovered.get("mix_validation_gateway")
    check_27_pass = False
    counter_pattern = re.compile(r"count|retry|attempt|iteration|loop", re.IGNORECASE)

    if mix_val_gw:
        flows = graph.outgoing_flows_with_conditions(mix_val_gw)
        for flow in flows:
            cond = flow.get("condition", "") or ""
            if counter_pattern.search(cond):
                check_27_pass = True
                break

    l3_results["check_27_loop_counter_variable"] = check_27_pass

    # --- Check 28: Loop escalation on max ---
    # mix_validation_gateway must have an outgoing flow leading to escalation
    # (not back to mix_task and not toward normal flow)
    mix_task = l2_discovered.get("mix_adjustment_task")
    escalation_task = l2_discovered.get("escalation_task")
    check_28_pass = False

    if mix_val_gw:
        flows = graph.outgoing_flows_with_conditions(mix_val_gw)
        for flow in flows:
            target = flow["target"]
            # Skip loop-back to mix task
            if target == mix_task:
                continue
            # Check if this path reaches escalation or a non-normal end
            if escalation_task and graph.is_reachable(target, escalation_task, max_depth=10):
                check_28_pass = True
                break
            # Also accept if it reaches an endEvent that is not the main endEvent
            for ee in graph.end_events:
                if ee != "endEvent" and graph.is_reachable(target, ee, max_depth=10):
                    check_28_pass = True
                    break
            if check_28_pass:
                break

    l3_results["check_28_loop_escalation_on_max"] = check_28_pass

    # --- Check 29: Cross-lane dependency ---
    # Mix adjustment task must consume variables from tasks in 2+ different lanes.
    check_29_pass = False
    if mix_task:
        predecessors = _all_predecessors_bfs(graph, mix_task, max_depth=10)
        mix_props = graph.get_form_properties(mix_task)
        in_props = {pid for pid in mix_props if pid.startswith("in_")}

        producer_lanes = set()
        for pred_id in predecessors:
            if graph.get_type(pred_id) != "userTask":
                continue
            pred_props = graph.get_form_properties(pred_id)
            pred_out = {pid for pid in pred_props if pid.startswith("out_")}

            # Check if any in_ property matches a pred out_ property
            produces_consumed = False
            for in_p in in_props:
                out_name = "out_" + in_p[3:]
                if out_name in pred_out:
                    produces_consumed = True
                    break
            if not produces_consumed:
                continue

            # Determine lane
            lane = graph.get_lane_for_node(pred_id)
            if lane:
                producer_lanes.add(lane["lane_name"])
            else:
                # Fall back to role-based inference
                assignee = graph.get_assignee(pred_id).lower()
                groups = graph.get_candidate_groups(pred_id).lower()
                if "production" in assignee or "production" in groups:
                    producer_lanes.add("production")
                elif "sales" in assignee or "sales" in groups:
                    producer_lanes.add("sales")
                elif "procurement" in assignee or "procurement" in groups:
                    producer_lanes.add("procurement")
                elif "quality" in assignee or "qa" in assignee:
                    producer_lanes.add("quality")
                elif "finance" in assignee or "finance" in groups:
                    producer_lanes.add("finance")
                elif "supplychain" in assignee or "supplychain" in groups:
                    producer_lanes.add("supplychain")
                elif "director" in assignee:
                    producer_lanes.add("management")
                else:
                    producer_lanes.add(f"unknown_{pred_id}")

        check_29_pass = len(producer_lanes) >= 2
        l3_details["check_29_producer_lanes"] = sorted(producer_lanes)

    l3_results["check_29_cross_lane_dependency"] = check_29_pass

    # --- Check 31: Temporal constraint ---
    # BFS distance from timer_boundary_event to expedited_quality_task <= 2 hops
    timer_be = l2_discovered.get("timer_boundary_event")
    exp_task = l2_discovered.get("expedited_quality_task")
    check_31_pass = False

    if timer_be and exp_task:
        # BFS from timer to expedited, compute shortest distance
        visited = {timer_be}
        queue = deque([(timer_be, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == exp_task:
                check_31_pass = dist <= 2
                l3_details["check_31_distance"] = dist
                break
            if dist >= 2:
                continue
            for succ in graph.successors(node):
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, dist + 1))

    l3_results["check_31_temporal_constraint"] = check_31_pass

    l3_details["discovered_nodes"] = l2_discovered
    return l3_results, l3_details


# ============================================================
# PHASE 3: COMPLIANCE CROSS-VALIDATION
# ============================================================

def check_compliance_L3(structural_path: str, rules_path: str, root: ET.Element = None) -> dict:
    """Check compliance coverage with L3 requirements, plus cross-validation."""
    results = {}

    try:
        with open(structural_path) as f:
            structural = json.load(f)
        mods = structural.get("modifications", {})
        covered_mods = [m for m in REQUIRED_MODIFICATIONS_L3 if m in mods]
        results["structural_coverage"] = len(covered_mods) / len(REQUIRED_MODIFICATIONS_L3)
        results["structural_missing"] = [m for m in REQUIRED_MODIFICATIONS_L3 if m not in mods]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        results["structural_coverage"] = 0.0
        results["structural_error"] = str(e)

    try:
        with open(rules_path) as f:
            rules = json.load(f)
        rule_entries = rules.get("rules", {})
        covered_rules = [r for r in REQUIRED_RULES_L3 if r in rule_entries]
        results["rules_coverage"] = len(covered_rules) / len(REQUIRED_RULES_L3)
        results["rules_missing"] = [r for r in REQUIRED_RULES_L3 if r not in rule_entries]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        results["rules_coverage"] = 0.0
        results["rules_error"] = str(e)

    # --- Cross-validation checks (require BPMN root) ---
    if root is not None:
        graph = BPMNGraph(root)
        l2_discovered = discover_L2_chain(graph, {})

        # Check 30: rule_17 structural match
        # If rule_17 exists (expedite vs risk vs waiver), verify BPMN has a gateway
        # on path to expedited_quality_task with conditions referencing risk AND waiver
        check_30_pass = False
        try:
            with open(rules_path) as f:
                rules_data = json.load(f)
            if "rule_17" in rules_data.get("rules", {}):
                exp_task = l2_discovered.get("expedited_quality_task")
                if exp_task:
                    predecessors = _all_predecessors_bfs(graph, exp_task, max_depth=10)
                    for pred_id in predecessors:
                        if not graph.is_decision_gateway(pred_id):
                            continue
                        gw_vars = _extract_vars(graph, pred_id)
                        gw_conds_text = " ".join(
                            (f.get("condition") or "") for f in graph.outgoing_flows_with_conditions(pred_id)
                        ).lower()
                        has_risk = any(
                            "risk" in v.lower() for v in gw_vars
                        ) or "risk" in gw_conds_text
                        has_waiver = any(
                            "waiver" in v.lower() or "override" in v.lower() for v in gw_vars
                        ) or "waiver" in gw_conds_text or "override" in gw_conds_text
                        if has_risk and has_waiver:
                            check_30_pass = True
                            break
                    if not check_30_pass:
                        # Accept if there are separate gateways for risk and waiver
                        has_risk_gw = False
                        has_waiver_gw = False
                        for pred_id in predecessors:
                            if not graph.is_decision_gateway(pred_id):
                                continue
                            gw_conds = " ".join(
                                (f.get("condition") or "") for f in graph.outgoing_flows_with_conditions(pred_id)
                            ).lower()
                            if "risk" in gw_conds:
                                has_risk_gw = True
                            if "waiver" in gw_conds or "override" in gw_conds:
                                has_waiver_gw = True
                        check_30_pass = has_risk_gw and has_waiver_gw
            else:
                # rule_17 not present in rules JSON — cannot validate
                check_30_pass = False
        except Exception:
            check_30_pass = False

        results["check_30_rule17_structural_match"] = check_30_pass

        # Check 32: Cost thresholds consistent
        # Parse cost_variance_gateway conditions, extract numeric thresholds.
        # Multiple thresholds must be monotonically increasing along escalation path.
        check_32_pass = False
        cv_gw = l2_discovered.get("cost_variance_gateway")

        if cv_gw:
            flows = graph.outgoing_flows_with_conditions(cv_gw)
            thresholds = []
            num_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)")

            for flow in flows:
                cond = flow.get("condition", "") or ""
                if "cost" in cond.lower() or "variance" in cond.lower():
                    nums = num_pattern.findall(cond)
                    for n in nums:
                        try:
                            thresholds.append(float(n))
                        except ValueError:
                            pass

            if len(thresholds) <= 1:
                # Single or no threshold — vacuously consistent
                check_32_pass = True
            else:
                # Check monotonically increasing
                sorted_thresholds = sorted(set(thresholds))
                check_32_pass = sorted_thresholds == sorted(thresholds) or len(set(thresholds)) == len(thresholds)
                results["check_32_thresholds"] = thresholds
        else:
            # No cost variance gateway — cannot validate
            check_32_pass = False

        results["check_32_cost_thresholds_consistent"] = check_32_pass

        # Check 33: Rationale on overrides
        # Override tasks must have out_* formProperty matching rationale/justification/reason
        rationale_pattern = re.compile(r"rationale|justification|reason|override_reason", re.IGNORECASE)
        override_tasks = [
            l2_discovered.get("quality_waiver_task"),
            l2_discovered.get("expedited_quality_task"),
            l2_discovered.get("escalation_task"),
        ]
        override_tasks = [t for t in override_tasks if t is not None]

        check_33_pass = True
        check_33_details = {}
        for task_id in override_tasks:
            props = graph.get_form_properties(task_id)
            out_props = {pid for pid in props if pid.startswith("out_")}
            has_rationale = any(rationale_pattern.search(pid) for pid in out_props)
            check_33_details[task_id] = {
                "out_props": sorted(out_props),
                "has_rationale": has_rationale,
            }
            if not has_rationale:
                check_33_pass = False

        results["check_33_rationale_on_overrides"] = check_33_pass
        results["check_33_details"] = check_33_details

    return results


# ============================================================
# BONUS CHECKS
# ============================================================

def check_bonus_L3(root: ET.Element, discovered: dict) -> dict:
    """Bonus checks that are reported but do not affect overall pass/fail."""
    results = {}
    graph = BPMNGraph(root)

    # --- D17: Semantic noop resistance (bonus) ---
    # Scenario identical to A except one irrelevant variable changed.
    # Path should be identical to Scenario A.
    scenario_a = {
        "supplierRisk": "low",
        "materialGrade": "A",
        "costVariance": 0.05,
        "supply_status": "AVAILABLE",
        "resolution_status": "RESOLVED",
        "out_supply_status": "AVAILABLE",
        "out_resolution_status": "RESOLVED",
        "out_material_grade": "A",
        "out_assessment_result": "CONFIRMED",
        "overall_verdict": "CONFIRMED",
    }
    scenario_a_noop = dict(scenario_a)
    scenario_a_noop["irrelevant_variable_xyz"] = "changed_value"

    path_a = trace_path(graph, "startEvent", scenario_a)
    path_a_noop = trace_path(graph, "startEvent", scenario_a_noop)

    results["d17_semantic_noop_resistance"] = path_a == path_a_noop
    if path_a != path_a_noop:
        results["d17_note"] = "Changing an irrelevant variable altered the path"

    # --- D18: Minimality intent (bonus) ---
    # Count non-original gateways and tasks
    new_gateways = [
        eid for eid, etype in graph.elem_types.items()
        if etype in ("exclusiveGateway", "inclusiveGateway", "parallelGateway")
        and eid not in ORIGINAL_ELEMENTS
    ]
    new_tasks = [
        eid for eid, etype in graph.elem_types.items()
        if etype == "userTask"
        and eid not in ORIGINAL_ELEMENTS
    ]
    new_count = len(new_gateways) + len(new_tasks)

    # Count structural checks that reference non-original elements
    # (rough proxy: number of L2 discovered elements that were found)
    discovered_count = sum(1 for v in discovered.values() if v is not None)

    bloat_threshold = 2 * discovered_count
    results["d18_minimality_intent"] = new_count <= bloat_threshold
    results["d18_new_element_count"] = new_count
    results["d18_discovered_count"] = discovered_count
    results["d18_threshold"] = bloat_threshold
    if new_count > bloat_threshold:
        results["d18_note"] = (
            f"Potential bloat: {new_count} new elements vs "
            f"{discovered_count} discovered (threshold: {bloat_threshold})"
        )

    # --- check_23: Timer escalation subprocess (bonus) ---
    # Check if timer boundary routes to a subProcess element.
    # Accept flat-with-3+-nodes as alternative.
    timer_be = discovered.get("timer_boundary_event")
    check_23_pass = False

    if timer_be:
        # Check direct successor is subProcess
        for succ in graph.successors(timer_be):
            if graph.get_type(succ) == "subProcess":
                check_23_pass = True
                break

        if not check_23_pass:
            # Alternative: timer -> at least 3 nodes before rejoining main path
            # Count nodes reachable from timer that are not on the main path
            timer_path = []
            visited = {timer_be}
            queue = deque([(timer_be, 0)])
            while queue:
                node, depth = queue.popleft()
                if depth > 0:
                    timer_path.append(node)
                if depth >= 6:
                    continue
                for s in graph.successors(node):
                    if s not in visited:
                        visited.add(s)
                        queue.append((s, depth + 1))
            # Count non-original nodes in the timer escalation path
            timer_path_new = [
                n for n in timer_path
                if n not in ORIGINAL_ELEMENTS
            ]
            check_23_pass = len(timer_path_new) >= 3

    results["check_23_timer_escalation_subprocess"] = check_23_pass

    return results


# ============================================================
# WEIGHTED SCORING
# ============================================================

_SECTION_WEIGHTS = {
    "A_structural": 0.30,
    "B_scenarios": 0.25,
    "C_compliance": 0.15,
    "D_anti_gaming": 0.10,
    "E_data_flow": 0.10,
    "F_role_coupling": 0.10,
}


def _bool_fraction(checks: dict) -> float:
    bools = [v for v in checks.values() if isinstance(v, bool)]
    return sum(bools) / len(bools) if bools else 0.0


def _compute_weighted_scores(sections: dict, test_check: dict, compliance_pass: bool) -> dict:
    scores = {}

    a = sections.get("A_structural", {})
    total = a.get("total", 0)
    scores["A_structural"] = a.get("passed", 0) / total if total > 0 else 0.0

    scores["B_scenarios"] = test_check.get("pass_rate", 0.0)

    c = sections.get("C_compliance", {})
    struct_cov = c.get("structural_coverage", 0.0)
    rules_cov = c.get("rules_coverage", 0.0)
    scores["C_compliance"] = (struct_cov + rules_cov) / 2

    for key in ("D_anti_gaming", "E_data_flow", "F_role_coupling"):
        sec = sections.get(key, {})
        scores[key] = _bool_fraction(sec.get("checks", {}))

    return scores


# ============================================================
# L3 MAIN EVALUATION
# ============================================================

def evaluate_L3(args) -> dict:
    """Run full L3 evaluation (all L1 + L2 checks + L3 extensions)."""
    expected_process_key = "monthlyProductionScheduling_modified_L3"
    expected_total_scenarios = 67
    min_passed_scenarios = 61
    report = {
        "task": "LY Juice Monthly Scheduling - Compound Disruption (L3)",
        "level": "L3",
        "approach": "Anchor-based topological detection (extended with behavioral checks)",
        # Process key: monthlyProductionScheduling_modified_L3
        "sections": {},
    }

    try:
        root = ET.parse(args.bpmn).getroot()
    except ET.ParseError as e:
        print(f"FATAL: Could not parse BPMN: {e}")
        report["sections"]["A_structural"] = {"error": str(e), "all_pass": False}
        report["overall_pass"] = False
        return report

    process_elem = root.find(f".//{{{BPMN_NS}}}process")
    actual_process_key = process_elem.get("id") if process_elem is not None else None
    process_key_ok = actual_process_key == expected_process_key

    # A. Structural checks (L1 + L2 + L3)
    print("=" * 60)
    print("A. STRUCTURAL CHECKS (L1 + L2 + L3 extended topology)")
    print("    Method: Anchor-based topological detection")
    print("=" * 60)

    try:
        structural, details = check_structural_L3(root)
        structural["check_process_definition_key"] = process_key_ok
        details["expected_process_definition_key"] = expected_process_key
        details["actual_process_definition_key"] = actual_process_key

        print("\n  Discovered disruption chain (L1 elements):")
        l1_keys = ["procurement_task", "risk_gateway", "sourcing_task",
                    "sourcing_gateway", "mix_adjustment_task", "mix_precheck_task",
                    "mix_validation_gateway", "sales_confirmation_task",
                    "sales_gateway", "escalation_task", "escalation_end"]
        for role in l1_keys:
            node_id = details.get("discovered_nodes", {}).get(role)
            status = node_id if node_id else "(not found)"
            print(f"    {role}: {status}")

        print("\n  Discovered L2 quality chain:")
        l2_keys = ["quality_inspection_task", "quality_grade_gateway",
                    "quality_waiver_task", "waiver_decision_gateway",
                    "timer_boundary_event", "expedited_quality_task",
                    "cost_variance_gateway", "cost_variance_task",
                    "material_restriction_task"]
        for role in l2_keys:
            node_id = details.get("discovered_nodes", {}).get(role)
            status = node_id if node_id else "(not found)"
            print(f"    {role}: {status}")

        print()
        print(
            "  ["
            + ("PASS" if process_key_ok else "FAIL")
            + f"] check_process_definition_key ({actual_process_key} vs {expected_process_key})"
        )
        for name, passed in sorted(structural.items()):
            if isinstance(passed, bool):
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {name}")

        all_pass = all(v for v in structural.values() if isinstance(v, bool))
        passed_count = sum(1 for v in structural.values() if isinstance(v, bool) and v)
        total_count = sum(1 for v in structural.values() if isinstance(v, bool))

        report["sections"]["A_structural"] = {
            "checks": structural,
            "discovered_nodes": details.get("discovered_nodes", {}),
            "all_pass": all_pass,
            "passed": passed_count,
            "total": total_count,
        }

        print(f"\n  Result: {passed_count}/{total_count} passed")
        print(f"  Section A: {'PASS' if all_pass else 'FAIL'}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report["sections"]["A_structural"] = {"error": str(e), "all_pass": False}

    # B. Test scenario results (same check, higher scenario count)
    print("\n" + "=" * 60)
    print("B. TEST SCENARIO RESULTS (>=90% required, 67 scenarios for L3)")
    print("=" * 60)

    test_check = check_test_results(args.results)
    test_check["expected_total_scenarios"] = expected_total_scenarios
    test_check["expected_process_definition_key"] = expected_process_key
    test_check["meets_total_scenarios"] = (
        test_check.get("total_scenarios", 0) == expected_total_scenarios
    )
    test_check["meets_process_definition_key"] = (
        test_check.get("process_definition_key") == expected_process_key
    )
    test_check["meets_threshold"] = (
        test_check["meets_total_scenarios"]
        and test_check["meets_process_definition_key"]
        and test_check.get("passed_scenarios", 0) >= min_passed_scenarios
    )
    report["sections"]["B_scenarios"] = test_check
    print(f"  Passed: {test_check['passed_scenarios']}/{test_check['total_scenarios']}")
    print(f"  Pass rate: {test_check['pass_rate']:.1%}")
    print(
        f"  Process key: {test_check.get('process_definition_key')} "
        f"(expected {expected_process_key})"
    )
    print(f"  Section B: {'PASS' if test_check['meets_threshold'] else 'FAIL'}")

    # C. Compliance coverage (L3 requirements)
    print("\n" + "=" * 60)
    print("C. COMPLIANCE COVERAGE (L3: 18 modifications + 22 rules)")
    print("=" * 60)

    compliance = check_compliance_L3(args.structural, args.rules, root)
    report["sections"]["C_compliance"] = compliance
    struct_coverage = compliance.get("structural_coverage", 0)
    rules_coverage = compliance.get("rules_coverage", 0)
    compliance_pass = struct_coverage == 1.0 and rules_coverage == 1.0

    struct_count = len(REQUIRED_MODIFICATIONS_L3) - len(compliance.get("structural_missing", []))
    rules_count = len(REQUIRED_RULES_L3) - len(compliance.get("rules_missing", []))
    print(f"  Structural modifications: {struct_coverage:.0%} ({struct_count}/{len(REQUIRED_MODIFICATIONS_L3)})")
    if compliance.get("structural_missing"):
        print(f"    Missing: {compliance['structural_missing']}")
    print(f"  Business rules: {rules_coverage:.0%} ({rules_count}/{len(REQUIRED_RULES_L3)})")
    if compliance.get("rules_missing"):
        print(f"    Missing: {compliance['rules_missing']}")

    # Cross-validation results
    if "check_30_rule17_structural_match" in compliance:
        status = "PASS" if compliance["check_30_rule17_structural_match"] else "FAIL"
        print(f"  [{status}] check_30_rule17_structural_match")
    if "check_32_cost_thresholds_consistent" in compliance:
        status = "PASS" if compliance["check_32_cost_thresholds_consistent"] else "FAIL"
        print(f"  [{status}] check_32_cost_thresholds_consistent")
    if "check_33_rationale_on_overrides" in compliance:
        status = "PASS" if compliance["check_33_rationale_on_overrides"] else "FAIL"
        print(f"  [{status}] check_33_rationale_on_overrides")

    print(f"  Section C: {'PASS' if compliance_pass else 'FAIL'}")

    # D. Anti-gaming checks (L3)
    print("\n" + "=" * 60)
    print("D. ANTI-GAMING CHECKS (L1 + L2 + L3)")
    print("=" * 60)

    anti_gaming = check_anti_gaming_L3(root)
    ag_pass = (
        anti_gaming["conditions_use_variables"]
        and anti_gaming["no_hardcoded_conditions"]
        and anti_gaming["min_conditions_met"]
        and anti_gaming["original_elements_preserved"]
        and anti_gaming.get("default_flows_on_gateways", True)
        and anti_gaming.get("new_tasks_have_documentation", True)
        and anti_gaming.get("d8_timer_has_proper_expression", False)
        and anti_gaming.get("d9_cost_variance_uses_variable", False)
        and anti_gaming.get("d10_quality_grade_uses_variable", False)
        and anti_gaming.get("d11_no_constant_only_routing", True)
        and anti_gaming.get("d12_gateway_3way_exists", False)
        and anti_gaming.get("d13_path_sensitivity_analysis", False)
        and anti_gaming.get("d14_conditions_use_upstream_outputs", False)
        and anti_gaming.get("d15_no_dead_branches", True)
        and anti_gaming.get("d16_all_new_branches_exercised", True)
    )
    report["sections"]["D_anti_gaming"] = {
        "checks": anti_gaming,
        "all_pass": ag_pass,
    }
    for name, val in sorted(anti_gaming.items()):
        if isinstance(val, bool):
            status = "PASS" if val else "FAIL"
            print(f"  [{status}] {name}")
        elif isinstance(val, (list, dict)):
            pass  # skip complex details in console
        elif isinstance(val, (int, float)):
            print(f"  {name}: {val}")
    print(f"  Section D: {'PASS' if ag_pass else 'FAIL'}")

    # E. Data flow validation (L2 — unchanged for L3)
    print("\n" + "=" * 60)
    print("E. DATA FLOW VALIDATION (L1 + L2)")
    print("=" * 60)

    try:
        discovered = report["sections"].get("A_structural", {}).get("discovered_nodes", {})
        if not discovered:
            graph_e = BPMNGraph(root)
            l1_disc = discover_disruption_chain(graph_e)
            discovered = discover_L2_chain(graph_e, l1_disc)

        data_flow, df_details = check_data_flow_L2(root, discovered)

        for name, passed in sorted(data_flow.items()):
            if isinstance(passed, bool):
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {name}")

        e_pass = all(v for v in data_flow.values() if isinstance(v, bool))
        report["sections"]["E_data_flow"] = {
            "checks": data_flow,
            "details": df_details,
            "all_pass": e_pass,
        }
        print(f"\n  Section E: {'PASS' if e_pass else 'FAIL'}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report["sections"]["E_data_flow"] = {"error": str(e), "all_pass": False}
        e_pass = False

    # F. Role-data coupling (L2 — unchanged for L3)
    print("\n" + "=" * 60)
    print("F. ROLE-DATA COUPLING VALIDATION (L1 + L2)")
    print("=" * 60)

    try:
        role_coupling, rc_details = check_role_data_coupling_L2(root, discovered)

        for name, passed in sorted(role_coupling.items()):
            if isinstance(passed, bool):
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {name}")

        f_pass = all(v for v in role_coupling.values() if isinstance(v, bool))
        report["sections"]["F_role_coupling"] = {
            "checks": role_coupling,
            "details": rc_details,
            "all_pass": f_pass,
        }
        print(f"\n  Section F: {'PASS' if f_pass else 'FAIL'}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report["sections"]["F_role_coupling"] = {"error": str(e), "all_pass": False}
        f_pass = False

    # G. Bonus checks (do NOT affect overall pass/fail)
    print("\n" + "=" * 60)
    print("G. BONUS CHECKS (informational only, do not affect pass/fail)")
    print("=" * 60)

    try:
        bonus = check_bonus_L3(root, discovered)
        for name, val in sorted(bonus.items()):
            if isinstance(val, bool):
                status = "PASS" if val else "FAIL"
                print(f"  [{status}] {name} (bonus)")
            elif isinstance(val, (int, float)):
                print(f"  {name}: {val}")
            elif isinstance(val, str):
                print(f"  {name}: {val}")

        report["sections"]["G_bonus"] = {
            "checks": bonus,
            "note": "Bonus checks do not affect overall pass/fail",
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report["sections"]["G_bonus"] = {"error": str(e)}

    # Final score
    print("\n" + "=" * 60)
    print("FINAL SCORE (L3)")
    print("=" * 60)

    sections_pass = {
        "A_structural": report["sections"].get("A_structural", {}).get("all_pass", False),
        "B_scenarios": test_check.get("meets_threshold", False),
        "C_compliance": compliance_pass,
        "D_anti_gaming": report["sections"].get("D_anti_gaming", {}).get("all_pass", False),
        "E_data_flow": report["sections"].get("E_data_flow", {}).get("all_pass", False),
        "F_role_coupling": report["sections"].get("F_role_coupling", {}).get("all_pass", False),
    }

    for section, passed in sections_pass.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {section}")

    all_sections_pass = all(sections_pass.values())
    report["overall_pass"] = all_sections_pass
    report["sections_summary"] = sections_pass

    section_scores = _compute_weighted_scores(report["sections"], test_check, compliance_pass)
    report["section_scores"] = section_scores
    weighted_score = sum(
        _SECTION_WEIGHTS[k] * section_scores.get(k, 0.0)
        for k in _SECTION_WEIGHTS
    )
    report["weighted_score"] = round(weighted_score, 4)

    print(f"\n  OVERALL: {'PASS' if all_sections_pass else 'FAIL'}")
    print(f"\n  Weighted section scores:")
    for k in _SECTION_WEIGHTS:
        w = _SECTION_WEIGHTS[k]
        s = section_scores.get(k, 0.0)
        print(f"    {k}: {s:.2f} (weight {w:.2f}, contribution {w * s:.4f})")
    print(f"  WEIGHTED SCORE: {weighted_score:.4f}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="L3 Evaluator: Supply chain workflow adaptation — "
                    "compound disruption with behavioral and dead-path checks"
    )
    parser.add_argument("--bpmn", required=True,
                        help="Path to the agent's modified BPMN XML")
    parser.add_argument("--structural", required=True,
                        help="Path to structural_changes.json (agent-produced)")
    parser.add_argument("--rules", required=True,
                        help="Path to business_rules_compliance.json (agent-produced)")
    parser.add_argument("--results", required=True,
                        help="Path to test_results.json (from run_tests.py)")
    parser.add_argument("--output", default="evaluation_report_L3.json",
                        help="Output report path")

    args = parser.parse_args()
    report = evaluate_L3(args)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}")

    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
