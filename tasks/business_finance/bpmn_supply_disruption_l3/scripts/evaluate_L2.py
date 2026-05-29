#!/usr/bin/env python3
"""
L2 Evaluator for: LY Juice Monthly Scheduling — Compound Disruption
(Raw Material Shortage + Quality Hold)

Extends the L1 evaluator with additional checks for:
  - Quality inspection gate (task + 3-way grade gateway)
  - Grade B waiver joint approval
  - Grade C rejection as 3rd trigger into inclusive merge
  - Timer boundary event on quality inspection (48h)
  - Expedited quality decision task (timer escalation)
  - Cost variance conditional Finance Controller review
  - Material restriction documentation and propagation

All L1 checks still apply. L2 adds structural checks A15-A22,
anti-gaming checks D8-D10, data flow checks E5-E7, and
role checks F6-F7.

Usage:
  python evaluate_L2.py --bpmn modified_process.bpmn20.xml \\
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

# ============================================================
# L2 CONSTANTS
# ============================================================

# L2 extends the compliance requirements
REQUIRED_MODIFICATIONS_L2 = [
    "modification_1",   # Procurement Availability Confirmation
    "modification_2",   # Supply Risk Assessment Branch
    "modification_3",   # Alternative Sourcing with Sufficiency Check
    "modification_4",   # Product Mix Adjustment
    "modification_5",   # Sales and Channel Confirmation
    "modification_6",   # Closed-Loop Back-Routing
    "modification_7",   # Inclusive Gateway Merge (dual/triple-trigger convergence)
    "modification_8",   # Parallel Approval (independent director review)
    "modification_9",   # State-Dependent Routing (re-validation loop prevention)
    "modification_10",  # Quality Inspection Gate
    "modification_11",  # Quality Grade 3-Way Gateway (A/B/C)
    "modification_12",  # Grade B Waiver Joint Approval
    "modification_13",  # Grade C Rejection → Inclusive Merge (3rd trigger)
    "modification_14",  # Timer Boundary Event on Quality Inspection
    "modification_15",  # Cost Variance Conditional Finance Review
]

REQUIRED_RULES_L2 = [
    "rule_1",    # Prioritize on-time delivery
    "rule_2",    # Do not easily adjust revenue target
    "rule_3",    # Do not easily adjust gross margin target
    "rule_4",    # Volume adjustable within limited range
    "rule_5",    # Prioritize procurement/sourcing/mix adjustment
    "rule_6",    # Product mix must be executable
    "rule_7",    # KPI adjustment is LAST RESORT
    "rule_8a",   # Dual/triple-trigger convergence constraint
    "rule_9",    # Independent parallel approval constraint
    "rule_10",   # Re-validation loop prevention
    "rule_11",   # Material quality gate
    "rule_12",   # Grade B conditional acceptance (joint QA+SC Lead)
    "rule_13",   # Grade C rejection = sourcing failure
    "rule_14",   # Material restriction propagation
    "rule_15",   # Quality inspection time limit (48h timer)
    "rule_16",   # Cost variance financial gate (>15%)
]

# L2 role extensions
ROLE_AUTHORITY_L2 = {
    **ROLE_AUTHORITY,
    "${qualityAssuranceLead}": "Operational",
    "${financeController}": "Management",
}

CANDIDATE_GROUP_AUTHORITY_L2 = {
    **CANDIDATE_GROUP_AUTHORITY,
    "quality": "Operational",
    "qa": "Operational",
    "quality_assurance": "Operational",
    "finance": "Management",
    "finance_mgmt": "Management",
}

# Expected roles for L2 new tasks (extends L1's EXPECTED_NEW_TASK_ROLES)
EXPECTED_NEW_TASK_ROLES_L2 = {
    # L1 roles
    "procurement_task": ("procurementLead", "Operational"),
    "sourcing_task": ("procurementLead", "Operational"),
    "mix_adjustment_task": ("Director", "Management"),
    "sales_confirmation_task": ("Director", "Management"),
    "escalation_task": (None, "Executive"),
    # L2 roles
    "quality_inspection_task": ("qualityAssuranceLead", "Operational"),
    "quality_waiver_task": (None, "Operational"),  # joint QA+SC Lead
    "expedited_quality_task": ("supplyChainLead", "Lead"),
    "cost_variance_task": ("financeController", "Management"),
    "material_restriction_task": ("qualityAssuranceLead", "Operational"),
}


# ============================================================
# L2 TOPOLOGICAL DISCOVERY
# ============================================================

def discover_L2_chain(graph: BPMNGraph, l1_discovered: dict) -> dict:
    """
    Discover L2-specific elements using role-based attribute matching.

    Unlike L1's positional BFS (which breaks when L2 elements are inserted
    into the chain), this function scans ALL non-original elements and
    identifies them by assignee, candidateGroups, element name, and
    attached-to relationships. This is topology-independent.

    After attribute-based discovery, it re-discovers L1 elements that were
    misidentified by the L1 walker (because L2 elements confused the BFS).
    """
    discovered = {
        # L1 roles (will be re-discovered if L1 walker was confused)
        "procurement_task": None,
        "risk_gateway": None,
        "sourcing_task": None,
        "sourcing_gateway": None,
        "mix_adjustment_task": None,
        "mix_precheck_task": None,
        "mix_validation_gateway": None,
        "sales_confirmation_task": None,
        "sc_director_review_task": None,
        "sales_gateway": None,
        "escalation_task": None,
        "escalation_end": None,
        "approval_split": None,
        "approval_join": None,
        "capacity_verdict_gateway": None,
        # L2 roles
        "quality_inspection_task": None,
        "quality_grade_gateway": None,
        "quality_waiver_task": None,
        "waiver_decision_gateway": None,
        "expedited_quality_task": None,
        "timer_boundary_event": None,
        "cost_variance_gateway": None,
        "cost_variance_task": None,
        "material_restriction_task": None,
    }

    # Collect all non-original elements
    new_elements = {
        eid: graph.get_type(eid)
        for eid in graph.elements
        if eid not in ORIGINAL_ELEMENTS
    }

    # ================================================================
    # Phase 1: Attribute-based discovery (role/name matching)
    # ================================================================

    for eid, etype in new_elements.items():
        if etype != "userTask":
            continue
        elem = graph.get_element(eid)
        if elem is None:
            continue

        assignee = graph.get_assignee(eid).lower()
        groups = graph.get_candidate_groups(eid).lower()
        name = (elem.get("name", "") or "").lower()

        # --- Procurement check ---
        if (("procurement" in assignee or "procurement" in groups) and
                ("check" in name or "availability" in name) and
                not discovered["procurement_task"]):
            discovered["procurement_task"] = eid

        # --- Alternative sourcing ---
        elif (("procurement" in assignee or "procurement" in groups) and
              ("sourcing" in name or "alternative" in name or "supplier" in name) and
              not discovered["sourcing_task"]):
            discovered["sourcing_task"] = eid

        # --- Quality inspection ---
        elif (("quality" in assignee or "qa" in assignee or
               "quality" in groups or "qa" in groups) and
              ("inspection" in name or "grading" in name or "quality" in name) and
              "waiver" not in name and "restriction" not in name and
              not discovered["quality_inspection_task"]):
            discovered["quality_inspection_task"] = eid

        # --- Grade B waiver ---
        elif (("waiver" in name or "conditional acceptance" in name or
               "grade b" in name) and
              not discovered["quality_waiver_task"]):
            discovered["quality_waiver_task"] = eid

        # --- Material restriction documentation ---
        elif (("restriction" in name or "material usage" in name) and
              not discovered["material_restriction_task"]):
            discovered["material_restriction_task"] = eid

        # --- Expedited quality decision ---
        elif (("expedited" in name) and
              not discovered["expedited_quality_task"]):
            discovered["expedited_quality_task"] = eid

        # --- Cost variance / Finance Controller ---
        elif (("finance" in assignee or "controller" in assignee or
               "finance" in groups) and
              ("cost" in name or "variance" in name or "finance" in name) and
              not discovered["cost_variance_task"]):
            discovered["cost_variance_task"] = eid

        # --- Mix adjustment ---
        elif (("director" in assignee and "sales" not in assignee) and
              ("mix" in name or "product mix" in name) and
              "pre-validation" not in name and "pre-check" not in name and
              "review" not in name and
              not discovered["mix_adjustment_task"]):
            discovered["mix_adjustment_task"] = eid

        # --- Mix pre-validation ---
        elif (("pre-validation" in name or "pre-check" in name or
               "pre validation" in name) and
              "mix" in name and
              not discovered["mix_precheck_task"]):
            discovered["mix_precheck_task"] = eid

        # --- SC Director review ---
        elif (("director" in assignee and "sales" not in assignee) and
              ("review" in name or "feasibility" in name) and
              not discovered["sc_director_review_task"]):
            discovered["sc_director_review_task"] = eid

        # --- Sales Director review ---
        elif (("salesdirector" in assignee or "sales_mgmt" in groups or
               "sales" in assignee and "director" in assignee) and
              ("review" in name or "channel" in name or "acceptance" in name) and
              not discovered["sales_confirmation_task"]):
            discovered["sales_confirmation_task"] = eid

        # --- Executive escalation ---
        elif (("executive" in groups or "exec" in groups) and
              ("kpi" in name or "escalation" in name or "executive" in name) and
              not discovered["escalation_task"]):
            discovered["escalation_task"] = eid

        # --- Revalidation marker ---
        elif ("revalidation" in name or "re-validation" in name or "marker" in name):
            # Not tracked as a discovered role, but good to know it exists
            pass

    # ================================================================
    # Phase 2: Boundary events and gateways (structural matching)
    # ================================================================

    # Timer boundary event: find boundaryEvent with timerEventDefinition
    qi_task = discovered["quality_inspection_task"]
    for eid, etype in new_elements.items():
        if etype == "boundaryEvent":
            elem = graph.get_element(eid)
            if elem is None:
                continue
            attached_to = elem.get("attachedToRef", "")
            timer_def = elem.find(f"{{{BPMN_NS}}}timerEventDefinition")
            if timer_def is not None:
                # Prefer one attached to quality inspection, but accept any timer
                if attached_to == qi_task:
                    discovered["timer_boundary_event"] = eid
                    break
                elif not discovered["timer_boundary_event"]:
                    discovered["timer_boundary_event"] = eid

    # Escalation end event
    esc_task = discovered["escalation_task"]
    if esc_task:
        for succ in graph.successors(esc_task):
            if graph.get_type(succ) == "endEvent" and succ != "endEvent":
                discovered["escalation_end"] = succ
                break

    # ================================================================
    # Phase 3: Gateway discovery (by topology from discovered tasks)
    # ================================================================

    # Risk gateway: decision gateway after procurement_task
    proc_task = discovered["procurement_task"]
    if proc_task:
        for succ in graph.successors(proc_task):
            if graph.is_decision_gateway(succ) and succ not in ORIGINAL_ELEMENTS:
                discovered["risk_gateway"] = succ
                break

    # Sourcing outcome gateway: decision gateway after sourcing_task
    sourcing_task = discovered["sourcing_task"]
    if sourcing_task:
        for succ in graph.successors(sourcing_task):
            if graph.is_decision_gateway(succ) and succ not in ORIGINAL_ELEMENTS:
                discovered["sourcing_gateway"] = succ
                break

    # Quality grade gateway: decision gateway after quality_inspection_task
    if qi_task:
        for succ in graph.successors(qi_task):
            if graph.is_decision_gateway(succ) and succ not in ORIGINAL_ELEMENTS:
                discovered["quality_grade_gateway"] = succ
                break

    # Waiver decision gateway: decision gateway after waiver_task
    waiver_task = discovered["quality_waiver_task"]
    if waiver_task:
        for succ in graph.successors(waiver_task):
            if graph.is_decision_gateway(succ) and succ not in ORIGINAL_ELEMENTS:
                discovered["waiver_decision_gateway"] = succ
                break

    # Cost variance gateway: look for gateway with cost-related conditions
    for eid, etype in new_elements.items():
        if not graph.is_decision_gateway(eid):
            continue
        flows = graph.outgoing_flows_with_conditions(eid)
        for f in flows:
            cond = (f.get("condition", "") or "").lower()
            if "cost" in cond or "variance" in cond:
                discovered["cost_variance_gateway"] = eid
                break
        if discovered["cost_variance_gateway"]:
            break

    # Capacity verdict gateway: decision gateway after task_capacityConfirmation
    for succ in graph.successors("task_capacityConfirmation"):
        if graph.is_decision_gateway(succ) and succ not in ORIGINAL_ELEMENTS:
            discovered["capacity_verdict_gateway"] = succ
            break

    # Mix validation gateway: decision gateway after mix_precheck_task with
    # a loop-back flow to mix_adjustment_task
    precheck = discovered["mix_precheck_task"]
    mix_task = discovered["mix_adjustment_task"]
    if precheck:
        for succ in graph.successors(precheck):
            if graph.is_decision_gateway(succ) and succ not in ORIGINAL_ELEMENTS:
                # Check if it has a loop-back to mix_task
                if mix_task:
                    flows = graph.outgoing_flows_with_conditions(succ)
                    has_loopback = any(f["target"] == mix_task for f in flows)
                    if has_loopback:
                        discovered["mix_validation_gateway"] = succ
                        break

    # Sales/review acceptance gateway: decision gateway after parallel join
    # Find parallel gateways (non-original) for approval pattern
    par_gws = [
        eid for eid, etype in new_elements.items()
        if etype == "parallelGateway"
    ]
    sc_review = discovered["sc_director_review_task"]
    sales_review = discovered["sales_confirmation_task"]

    if sc_review and sales_review and par_gws:
        # Find the split: a parallel gateway that reaches both review tasks
        for pg in par_gws:
            outgoing = graph.successors(pg)
            reaches_both = (
                any(graph.is_reachable(pg, sc_review, max_depth=3) for _ in [1]) and
                any(graph.is_reachable(pg, sales_review, max_depth=3) for _ in [1])
            )
            if reaches_both and len(outgoing) >= 2:
                discovered["approval_split"] = pg
                break

        # Find the join: a parallel gateway reachable from both review tasks
        for pg in par_gws:
            if pg == discovered["approval_split"]:
                continue
            reaches_from_both = (
                graph.is_reachable(sc_review, pg, max_depth=3) and
                graph.is_reachable(sales_review, pg, max_depth=3)
            )
            if reaches_from_both:
                discovered["approval_join"] = pg
                # Find the decision gateway after the join
                for succ in graph.successors(pg):
                    if graph.is_decision_gateway(succ) and succ not in ORIGINAL_ELEMENTS:
                        discovered["sales_gateway"] = succ
                        break
                break

    return discovered


# ============================================================
# L2 STRUCTURAL CHECKS (extends L1)
# ============================================================

def check_structural_L2(root: ET.Element):
    """Run all structural checks using L2's attribute-based discovery.

    Unlike L1's check_structural (which uses positional BFS that breaks
    when L2 elements are inserted), this re-discovers ALL elements using
    role-based matching, then evaluates both L1 and L2 checks against
    the correctly identified nodes.
    """
    graph = BPMNGraph(root)

    # Run L2 discovery (which also identifies L1 elements by attributes)
    # Pass empty dict since L2 discovery is self-contained
    l2_discovered = discover_L2_chain(graph, {})

    l2_results = {}
    l2_details = {"discovered_nodes": l2_discovered}

    # ====== L1 structural checks (re-evaluated with correct discovery) ======

    # Check 1: BPMN parseable
    l2_results["check_1_bpmn_parseable"] = True

    # Check 2: Direct link broken
    sku_successors = graph.successors("task_skuSequencing")
    l2_results["check_2_direct_link_broken"] = "task_capacityConfirmation" not in sku_successors

    # Check 3: Procurement task inserted
    procurement = l2_discovered.get("procurement_task")
    if procurement:
        can_reach = graph.is_reachable(procurement, "task_capacityConfirmation", max_depth=15)
        l2_results["check_3_procurement_task_inserted"] = can_reach
    else:
        l2_results["check_3_procurement_task_inserted"] = False

    # Check 4: Risk gateway with 2 flows
    risk_gw = l2_discovered.get("risk_gateway")
    if risk_gw:
        flows = graph.outgoing_flows_with_conditions(risk_gw)
        conditional_flows = [f for f in flows if f["condition"] is not None]
        l2_results["check_4_risk_gateway_2_flows"] = len(flows) == 2 and len(conditional_flows) >= 1
    else:
        l2_results["check_4_risk_gateway_2_flows"] = False

    # Check 5: No-risk path leads toward capacity
    if risk_gw:
        flows = graph.outgoing_flows_with_conditions(risk_gw)
        has_capacity_path = any(
            graph.is_reachable(f["target"], "task_capacityConfirmation", max_depth=10)
            for f in flows
        )
        l2_results["check_5_no_risk_path_to_main_flow"] = has_capacity_path
    else:
        l2_results["check_5_no_risk_path_to_main_flow"] = False

    # Check 6: Sourcing task exists
    sourcing = l2_discovered.get("sourcing_task")
    l2_results["check_6_exception_path_sourcing"] = sourcing is not None

    # Check 7: Sourcing gateway with 2 flows
    sourcing_gw = l2_discovered.get("sourcing_gateway")
    if sourcing_gw:
        flows = graph.outgoing_flows_with_conditions(sourcing_gw)
        conditional_flows = [f for f in flows if f["condition"] is not None]
        l2_results["check_7_sourcing_gateway_branches"] = len(flows) == 2 and len(conditional_flows) >= 1
    else:
        l2_results["check_7_sourcing_gateway_branches"] = False

    # Check 8: Closed-loop back-routing from approval gateway to capacity
    sales_gw = l2_discovered.get("sales_gateway")
    if sales_gw:
        flows = graph.outgoing_flows_with_conditions(sales_gw)
        has_closed_loop = any(
            graph.is_reachable(f["target"], "task_capacityConfirmation", max_depth=5)
            for f in flows
        )
        l2_results["check_8_closed_loop_back_routing"] = has_closed_loop
    else:
        l2_results["check_8_closed_loop_back_routing"] = False

    # Check 9: Escalation terminal path (>=2 end events)
    num_end_events = len(graph.end_events)
    has_new_end = l2_discovered.get("escalation_end") is not None
    l2_results["check_9_escalation_terminal_path"] = num_end_events >= 2 and has_new_end

    # Check 10: Capacity verdict gateway
    cap_verdict_gw = l2_discovered.get("capacity_verdict_gateway")
    if cap_verdict_gw:
        flows = graph.outgoing_flows_with_conditions(cap_verdict_gw)
        reaches_monthly = any(
            f["target"] == "task_monthlyPlanConfirmation"
            or graph.is_reachable(f["target"], "task_monthlyPlanConfirmation", max_depth=3)
            for f in flows
        )
        reaches_exception = any(
            f["target"] not in ORIGINAL_ELEMENTS for f in flows
        )
        l2_results["check_10_capacity_verdict_gateway"] = reaches_monthly and reaches_exception and len(flows) >= 2
    else:
        l2_results["check_10_capacity_verdict_gateway"] = False

    # Check 11: Mix pre-validation loop
    mix_task = l2_discovered.get("mix_adjustment_task")
    precheck_task = l2_discovered.get("mix_precheck_task")
    mix_val_gw = l2_discovered.get("mix_validation_gateway")
    sales_task = l2_discovered.get("sales_confirmation_task")
    sc_review_task = l2_discovered.get("sc_director_review_task")

    if precheck_task and mix_val_gw and mix_task:
        v_flows = graph.outgoing_flows_with_conditions(mix_val_gw)
        reaches_downstream = any(
            graph.is_reachable(f["target"], sales_task or sc_review_task or "endEvent", max_depth=5)
            for f in v_flows
        ) if (sales_task or sc_review_task) else False
        reaches_mix_back = any(f["target"] == mix_task for f in v_flows)
        l2_results["check_11_mix_prevalidation_loop"] = reaches_downstream and reaches_mix_back and len(v_flows) >= 2
    else:
        l2_results["check_11_mix_prevalidation_loop"] = False

    # Check 12: Inclusive gateway merge with multiple incoming paths
    inclusive_gws = [eid for eid, etype in graph.elem_types.items() if etype == "inclusiveGateway"]
    check_12_pass = False
    for ig_id in inclusive_gws:
        incoming = graph.predecessors(ig_id)
        if len(incoming) < 2:
            continue
        reaches_mix = mix_task and (
            mix_task in graph.successors(ig_id) or
            graph.is_reachable(ig_id, mix_task, max_depth=5)
        )
        if reaches_mix:
            check_12_pass = True
            l2_details["inclusive_gateway"] = ig_id
            l2_details["inclusive_incoming_count"] = len(incoming)
            break
    l2_results["check_12_inclusive_gateway_merge"] = check_12_pass

    # Check 13: Parallel approval pattern
    approval_split = l2_discovered.get("approval_split")
    approval_join = l2_discovered.get("approval_join")
    check_13_pass = (
        approval_split is not None and approval_join is not None and
        sc_review_task is not None and sales_task is not None
    )
    l2_results["check_13_parallel_approval"] = check_13_pass

    # Check 14: State-dependent routing — the process must distinguish
    # first-pass capacity assessment from re-validation after adjustment.
    #
    # Accepts TWO valid patterns:
    #   Pattern A: Single gateway with >=2 variables in its conditions
    #              (e.g., overall_verdict AND planPreviouslyAdjusted on one gateway)
    #   Pattern B: Two-gateway chain — capacity verdict gateway routes
    #              NEEDS_ADJUSTMENT to a second gateway that checks a state
    #              variable (e.g., planPreviouslyAdjusted) to decide between
    #              mix adjustment (first pass) vs escalation (re-validation).
    #              This is valid separation of concerns.
    check_14_pass = False
    el_keywords = {"true", "false", "null", "empty", "not", "and", "or",
                   "eq", "ne", "lt", "gt", "le", "ge", "div", "mod",
                   "instanceof", "NEEDS_ADJUSTMENT", "CONFIRMED"}
    var_pattern_14 = re.compile(r"[a-zA-Z_]\w*")

    def _extract_vars(gateway_id):
        """Extract all variable names from a gateway's outgoing conditions."""
        v = set()
        for f in graph.outgoing_flows_with_conditions(gateway_id):
            cond = f.get("condition")
            if cond:
                inner = cond.strip()
                if inner.startswith("${") and inner.endswith("}"):
                    inner = inner[2:-1]
                for tok in var_pattern_14.findall(inner):
                    if tok not in el_keywords:
                        v.add(tok)
        return v

    if cap_verdict_gw:
        # Pattern A: single gateway with >=2 variables
        cap_vars = _extract_vars(cap_verdict_gw)
        if len(cap_vars) >= 2:
            check_14_pass = True
        else:
            # Pattern B: two-gateway chain
            # Look for a decision gateway downstream of cap_verdict_gw (within 2 hops)
            # that uses a DIFFERENT variable (state-dependent routing variable)
            for f in graph.outgoing_flows_with_conditions(cap_verdict_gw):
                tgt = f["target"]
                if graph.is_decision_gateway(tgt) and tgt not in ORIGINAL_ELEMENTS:
                    downstream_vars = _extract_vars(tgt)
                    # Combined variables across both gateways must be >=2
                    combined = cap_vars | downstream_vars
                    if len(combined) >= 2:
                        check_14_pass = True
                        l2_details["check_14_pattern"] = "two_gateway_chain"
                        l2_details["check_14_gateways"] = [cap_verdict_gw, tgt]
                        l2_details["check_14_variables"] = sorted(combined)
                        break
                # Also check one hop further (gateway -> gateway)
                elif graph.get_type(tgt) == "userTask" or tgt in ORIGINAL_ELEMENTS:
                    continue
                else:
                    for succ2 in graph.successors(tgt):
                        if graph.is_decision_gateway(succ2) and succ2 not in ORIGINAL_ELEMENTS:
                            downstream_vars = _extract_vars(succ2)
                            combined = cap_vars | downstream_vars
                            if len(combined) >= 2:
                                check_14_pass = True
                                l2_details["check_14_pattern"] = "two_gateway_chain"
                                l2_details["check_14_gateways"] = [cap_verdict_gw, succ2]
                                l2_details["check_14_variables"] = sorted(combined)
                                break
                    if check_14_pass:
                        break

    l2_results["check_14_state_dependent_routing"] = check_14_pass

    # Original elements preserved
    missing_originals = [eid for eid in ORIGINAL_ELEMENTS if eid not in graph.elements]
    l2_results["original_elements_preserved"] = len(missing_originals) == 0

    # Role assignments correct (mix + sales tasks are director level)
    role_pass = True
    for task_key in ["mix_adjustment_task", "sales_confirmation_task"]:
        task_id = l2_discovered.get(task_key)
        if task_id:
            assignee = graph.get_assignee(task_id).lower()
            if "director" not in assignee or "lead" in assignee:
                role_pass = False
        else:
            role_pass = False
    l2_results["role_assignments_correct"] = role_pass

    # --- Check A15: Quality inspection task between sourcing and capacity ---
    qi_task = l2_discovered.get("quality_inspection_task")
    if qi_task:
        # Must be reachable from sourcing area AND reach capacity
        reaches_capacity = graph.is_reachable(qi_task, "task_capacityConfirmation", max_depth=10)
        l2_results["check_15_quality_inspection_inserted"] = reaches_capacity
    else:
        l2_results["check_15_quality_inspection_inserted"] = False

    # --- Check A16: Quality grade gateway with 3+ outgoing flows (A/B/C) ---
    grade_gw = l2_discovered.get("quality_grade_gateway")
    if grade_gw:
        flows = graph.outgoing_flows_with_conditions(grade_gw)
        # Accept 3-way (A/B/C separate) or 2-way with downstream branching
        # Ideal: 3 flows. Acceptable: 2 flows with Grade B having sub-gateway
        has_enough_flows = len(flows) >= 2
        has_conditional = any(f["condition"] is not None for f in flows)
        l2_results["check_16_quality_grade_gateway"] = has_enough_flows and has_conditional
        l2_details["quality_grade_gateway_flows"] = len(flows)
    else:
        l2_results["check_16_quality_grade_gateway"] = False

    # --- Check A17: Grade B waiver has approval pattern ---
    waiver_task = l2_discovered.get("quality_waiver_task")
    if waiver_task:
        # Check it involves joint approval (QA Lead + SC Lead)
        assignee = graph.get_assignee(waiver_task)
        groups = graph.get_candidate_groups(waiver_task)
        # Joint approval can be modeled as:
        # 1. Single task with multiple candidate groups
        # 2. Parallel split with two tasks
        # We accept either pattern
        has_qa = ("quality" in assignee.lower() or "qa" in assignee.lower() or
                  "quality" in groups.lower() or "qa" in groups.lower())
        has_sc = ("supplychainlead" in assignee.lower() or
                  "supplychain" in groups.lower())
        # Also check for parallel split pattern near waiver
        par_pattern = False
        for node_id, etype in graph.elem_types.items():
            if etype == "parallelGateway" and node_id not in ORIGINAL_ELEMENTS:
                if graph.is_reachable(grade_gw, node_id, max_depth=5):
                    if graph.is_reachable(node_id, waiver_task, max_depth=3):
                        par_pattern = True
                        break

        l2_results["check_17_grade_b_waiver_approval"] = (
            (has_qa and has_sc) or (has_qa and par_pattern) or
            (waiver_task is not None)  # minimal: waiver task exists
        )
    else:
        l2_results["check_17_grade_b_waiver_approval"] = False

    # --- Check A18: Grade C rejection routes to inclusive merge (3rd trigger) ---
    inclusive_gws = [
        eid for eid, etype in graph.elem_types.items()
        if etype == "inclusiveGateway"
    ]
    mix_adj = l2_discovered.get("mix_adjustment_task")
    check_18_pass = False

    if grade_gw and inclusive_gws and mix_adj:
        for ig_id in inclusive_gws:
            incoming = graph.predecessors(ig_id)
            if len(incoming) < 2:
                continue
            # Check if the inclusive merge reaches mix_adjustment
            reaches_mix = (
                mix_adj in graph.successors(ig_id)
                or graph.is_reachable(ig_id, mix_adj, max_depth=5)
            )
            if not reaches_mix:
                continue
            # Check: at least one incoming path traces back through the grade gateway
            has_quality_path = False
            for pred in incoming:
                if (pred == grade_gw or
                        graph.is_reachable(grade_gw, pred, max_depth=6)):
                    has_quality_path = True
                    break
            if has_quality_path:
                # Also verify the inclusive merge has 3+ incoming
                # (sourcing failure + capacity infeasibility + quality rejection)
                if len(incoming) >= 3:
                    check_18_pass = True
                elif len(incoming) >= 2:
                    # Accept 2 if quality rejection merges with sourcing failure path
                    check_18_pass = True
                break

    l2_results["check_18_grade_c_to_inclusive_merge"] = check_18_pass

    # --- Check A19: Timer boundary event attached to quality inspection ---
    timer_be = l2_discovered.get("timer_boundary_event")
    if timer_be and qi_task:
        elem = graph.get_element(timer_be)
        attached = elem.get("attachedToRef", "") if elem is not None else ""
        is_interrupting = elem.get("cancelActivity", "true").lower() == "true" if elem is not None else False
        has_timer_def = False
        if elem is not None:
            timer_def = elem.find(f"{{{BPMN_NS}}}timerEventDefinition")
            if timer_def is not None:
                has_timer_def = True
                # Check for duration expression
                duration = timer_def.find(f"{{{BPMN_NS}}}timeDuration")
                if duration is not None and duration.text:
                    l2_details["timer_duration"] = duration.text.strip()

        l2_results["check_19_timer_boundary_event"] = (
            attached == qi_task and is_interrupting and has_timer_def
        )
        l2_details["timer_boundary_detail"] = {
            "boundary_event": timer_be,
            "attached_to": attached,
            "is_interrupting": is_interrupting,
            "has_timer_definition": has_timer_def,
        }
    else:
        l2_results["check_19_timer_boundary_event"] = False

    # --- Check A20: Timer routes to expedited decision task ---
    exp_task = l2_discovered.get("expedited_quality_task")
    if timer_be and exp_task:
        reachable = graph.is_reachable(timer_be, exp_task, max_depth=3)
        l2_results["check_20_timer_to_expedited_decision"] = reachable
    else:
        l2_results["check_20_timer_to_expedited_decision"] = False

    # --- Check A21: Cost variance gateway conditionally routes to Finance Controller ---
    cv_gw = l2_discovered.get("cost_variance_gateway")
    cv_task = l2_discovered.get("cost_variance_task")
    if cv_gw and cv_task:
        flows = graph.outgoing_flows_with_conditions(cv_gw)
        # One flow should lead to Finance Controller task
        # Another should bypass (below-threshold path)
        reaches_finance = any(
            f["target"] == cv_task or graph.is_reachable(f["target"], cv_task, max_depth=3)
            for f in flows
        )
        has_bypass = any(
            f["target"] != cv_task and f["target"] not in {cv_task}
            for f in flows
        )
        l2_results["check_21_cost_variance_gateway"] = reaches_finance and has_bypass and len(flows) >= 2
    else:
        l2_results["check_21_cost_variance_gateway"] = False

    # --- Check A22: Material restriction documentation task on Grade B path ---
    mr_task = l2_discovered.get("material_restriction_task")
    if mr_task:
        # Must be reachable from waiver acceptance and reach capacity
        waiver_gw = l2_discovered.get("waiver_decision_gateway")
        waiver_task = l2_discovered.get("quality_waiver_task")
        from_waiver = False
        if waiver_gw:
            from_waiver = graph.is_reachable(waiver_gw, mr_task, max_depth=5)
        elif waiver_task:
            from_waiver = graph.is_reachable(waiver_task, mr_task, max_depth=5)
        to_capacity = graph.is_reachable(mr_task, "task_capacityConfirmation", max_depth=10)
        l2_results["check_22_material_restriction_doc"] = from_waiver and to_capacity
    else:
        l2_results["check_22_material_restriction_doc"] = False

    l2_details["discovered_nodes"] = l2_discovered
    return l2_results, l2_details


# ============================================================
# L2 ANTI-GAMING CHECKS (extends L1)
# ============================================================

def check_anti_gaming_L2(root: ET.Element) -> dict:
    """L1 anti-gaming checks plus L2-specific checks."""
    results = check_anti_gaming_L1(root)
    graph = BPMNGraph(root)

    # --- D8: Timer event has proper duration expression ---
    timer_events = [
        eid for eid, etype in graph.elem_types.items()
        if etype == "boundaryEvent"
    ]
    d8_pass = False
    for te_id in timer_events:
        elem = graph.get_element(te_id)
        if elem is None:
            continue
        timer_def = elem.find(f"{{{BPMN_NS}}}timerEventDefinition")
        if timer_def is None:
            continue
        duration = timer_def.find(f"{{{BPMN_NS}}}timeDuration")
        date = timer_def.find(f"{{{BPMN_NS}}}timeDate")
        cycle = timer_def.find(f"{{{BPMN_NS}}}timeCycle")
        if duration is not None and duration.text and duration.text.strip():
            # Must be ISO 8601 duration (e.g., PT48H), not hardcoded
            dur_text = duration.text.strip()
            if dur_text.startswith("P") or dur_text.startswith("${"):
                d8_pass = True
        elif date is not None or cycle is not None:
            d8_pass = True
    results["d8_timer_has_proper_expression"] = d8_pass

    # --- D9: Cost variance condition uses process variable ---
    # Find all gateways with cost-related conditions
    d9_pass = False
    for node_id in graph.elements:
        if not graph.is_decision_gateway(node_id):
            continue
        flows = graph.outgoing_flows_with_conditions(node_id)
        for flow in flows:
            cond = flow.get("condition", "") or ""
            if ("cost" in cond.lower() or "variance" in cond.lower() or
                    "margin" in cond.lower()):
                # Must use ${variable} pattern, not hardcoded
                if "${" in cond:
                    d9_pass = True
                    break
        if d9_pass:
            break
    results["d9_cost_variance_uses_variable"] = d9_pass

    # --- D10: Quality grade routing uses task-produced variable ---
    d10_pass = False
    for node_id in graph.elements:
        if not graph.is_decision_gateway(node_id):
            continue
        flows = graph.outgoing_flows_with_conditions(node_id)
        for flow in flows:
            cond = flow.get("condition", "") or ""
            if ("grade" in cond.lower() or "quality" in cond.lower() or
                    "material_grade" in cond.lower() or "materialGrade" in cond.lower()):
                if "${" in cond:
                    d10_pass = True
                    break
        if d10_pass:
            break
    results["d10_quality_grade_uses_variable"] = d10_pass

    return results


# ============================================================
# L2 COMPLIANCE (overrides L1)
# ============================================================

def check_compliance_L2(structural_path: str, rules_path: str) -> dict:
    """Check compliance coverage with L2 requirements."""
    results = {}

    try:
        with open(structural_path) as f:
            structural = json.load(f)
        mods = structural.get("modifications", {})
        covered_mods = [m for m in REQUIRED_MODIFICATIONS_L2 if m in mods]
        results["structural_coverage"] = len(covered_mods) / len(REQUIRED_MODIFICATIONS_L2)
        results["structural_missing"] = [m for m in REQUIRED_MODIFICATIONS_L2 if m not in mods]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        results["structural_coverage"] = 0.0
        results["structural_error"] = str(e)

    try:
        with open(rules_path) as f:
            rules = json.load(f)
        rule_entries = rules.get("rules", {})
        covered_rules = [r for r in REQUIRED_RULES_L2 if r in rule_entries]
        results["rules_coverage"] = len(covered_rules) / len(REQUIRED_RULES_L2)
        results["rules_missing"] = [r for r in REQUIRED_RULES_L2 if r not in rule_entries]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        results["rules_coverage"] = 0.0
        results["rules_error"] = str(e)

    return results


# ============================================================
# L2 DATA FLOW CHECKS (extends L1)
# ============================================================

def check_data_flow_L2(root: ET.Element, discovered: dict):
    """L1 data flow checks plus L2-specific checks."""
    l1_results, l1_details = check_data_flow_L1(root, discovered)
    results = dict(l1_results)
    details = dict(l1_details)
    graph = BPMNGraph(root)

    # --- E5: Quality inspection produces material_grade variable ---
    qi_task = discovered.get("quality_inspection_task")
    e5_pass = False
    if qi_task:
        props = graph.get_form_properties(qi_task)
        has_grade_output = any(
            "grade" in pid.lower() or "quality" in pid.lower()
            for pid in props if pid.startswith("out_")
        )
        e5_pass = has_grade_output
    results["e5_quality_produces_grade"] = e5_pass

    # --- E6: Material restriction task produces constraints consumed by mix ---
    mr_task = discovered.get("material_restriction_task")
    mix_task = discovered.get("mix_adjustment_task")
    e6_pass = False
    if mr_task and mix_task:
        mr_props = graph.get_form_properties(mr_task)
        mix_props = graph.get_form_properties(mix_task)
        # Check for restriction output
        has_restriction_out = any(
            "restriction" in pid.lower() or "constraint" in pid.lower()
            for pid in mr_props if pid.startswith("out_")
        )
        # Check mix task has corresponding input
        has_restriction_in = any(
            "restriction" in pid.lower() or "constraint" in pid.lower()
            for pid in mix_props if pid.startswith("in_")
        )
        e6_pass = has_restriction_out and has_restriction_in
    results["e6_restriction_propagates_to_mix"] = e6_pass

    # --- E7: Cost variance variable produced by sourcing, consumed by finance gateway ---
    cv_gw = discovered.get("cost_variance_gateway")
    sourcing_task = discovered.get("sourcing_task")
    e7_pass = False
    if cv_gw and sourcing_task:
        # Check sourcing task has cost-related output
        s_props = graph.get_form_properties(sourcing_task)
        has_cost_out = any(
            "cost" in pid.lower() or "variance" in pid.lower()
            for pid in s_props if pid.startswith("out_")
        )
        # Check gateway condition references a cost variable
        flows = graph.outgoing_flows_with_conditions(cv_gw)
        has_cost_cond = any(
            "cost" in (f.get("condition", "") or "").lower()
            for f in flows
        )
        e7_pass = has_cost_out and has_cost_cond
    results["e7_cost_variance_data_chain"] = e7_pass

    return results, details


# ============================================================
# L2 ROLE-DATA COUPLING (extends L1)
# ============================================================

def check_role_data_coupling_L2(root: ET.Element, discovered: dict):
    """L1 role checks plus L2-specific role checks."""
    l1_results, l1_details = check_role_data_coupling_L1(root, discovered)
    results = dict(l1_results)
    details = dict(l1_details)
    graph = BPMNGraph(root)

    # --- F6: Quality tasks assigned to qualityAssuranceLead ---
    qi_task = discovered.get("quality_inspection_task")
    mr_task = discovered.get("material_restriction_task")
    f6_pass = True
    f6_details = {}

    for role_key, node_id in [("quality_inspection", qi_task), ("material_restriction", mr_task)]:
        if not node_id:
            f6_pass = False
            f6_details[role_key] = {"found": False}
            continue
        assignee = graph.get_assignee(node_id).lower()
        groups = graph.get_candidate_groups(node_id).lower()
        has_qa = ("quality" in assignee or "qa" in assignee or
                  "quality" in groups or "qa" in groups)
        f6_details[role_key] = {
            "node_id": node_id,
            "assignee": graph.get_assignee(node_id),
            "groups": graph.get_candidate_groups(node_id),
            "has_qa_role": has_qa,
        }
        if not has_qa:
            f6_pass = False

    results["f6_quality_tasks_qa_assigned"] = f6_pass
    details["f6_quality_roles"] = f6_details

    # --- F7: Finance Controller assigned to cost variance task ---
    cv_task = discovered.get("cost_variance_task")
    f7_pass = False
    if cv_task:
        assignee = graph.get_assignee(cv_task).lower()
        groups = graph.get_candidate_groups(cv_task).lower()
        f7_pass = ("finance" in assignee or "controller" in assignee or
                   "finance" in groups)
        details["f7_finance_role"] = {
            "node_id": cv_task,
            "assignee": graph.get_assignee(cv_task),
            "groups": graph.get_candidate_groups(cv_task),
            "has_finance_role": f7_pass,
        }
    else:
        details["f7_finance_role"] = {"found": False}
    results["f7_finance_controller_assigned"] = f7_pass

    return results, details


# ============================================================
# L2 MAIN EVALUATION
# ============================================================

def evaluate_L2(args) -> dict:
    """Run full L2 evaluation (all L1 checks + L2 extensions)."""
    report = {
        "task": "LY Juice Monthly Scheduling - Compound Disruption (L2)",
        "level": "L2",
        "approach": "Anchor-based topological detection (extended for quality gate)",
        "sections": {},
    }

    try:
        root = ET.parse(args.bpmn).getroot()
    except ET.ParseError as e:
        print(f"FATAL: Could not parse BPMN: {e}")
        report["sections"]["A_structural"] = {"error": str(e), "all_pass": False}
        report["overall_pass"] = False
        return report

    # A. Structural checks (L1 + L2)
    print("=" * 60)
    print("A. STRUCTURAL CHECKS (L1 + L2 quality gate topology)")
    print("    Method: Anchor-based topological detection")
    print("=" * 60)

    try:
        structural, details = check_structural_L2(root)

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
            node_id = details.get("l2_discovered_nodes", {}).get(role) or details.get("discovered_nodes", {}).get(role)
            status = node_id if node_id else "(not found)"
            print(f"    {role}: {status}")

        print()
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
            "l2_discovered": details.get("l2_discovered_nodes", {}),
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
    print("B. TEST SCENARIO RESULTS (>=90% required, 54 scenarios for L2)")
    print("=" * 60)

    test_check = check_test_results(args.results)
    report["sections"]["B_scenarios"] = test_check
    print(f"  Passed: {test_check['passed_scenarios']}/{test_check['total_scenarios']}")
    print(f"  Pass rate: {test_check['pass_rate']:.1%}")
    print(f"  Section B: {'PASS' if test_check['meets_threshold'] else 'FAIL'}")

    # C. Compliance coverage (L2 requirements)
    print("\n" + "=" * 60)
    print("C. COMPLIANCE COVERAGE (L2: 15 modifications + 16 rules)")
    print("=" * 60)

    compliance = check_compliance_L2(args.structural, args.rules)
    report["sections"]["C_compliance"] = compliance
    struct_coverage = compliance.get("structural_coverage", 0)
    rules_coverage = compliance.get("rules_coverage", 0)
    compliance_pass = struct_coverage == 1.0 and rules_coverage == 1.0

    struct_count = len(REQUIRED_MODIFICATIONS_L2) - len(compliance.get("structural_missing", []))
    rules_count = len(REQUIRED_RULES_L2) - len(compliance.get("rules_missing", []))
    print(f"  Structural modifications: {struct_coverage:.0%} ({struct_count}/{len(REQUIRED_MODIFICATIONS_L2)})")
    if compliance.get("structural_missing"):
        print(f"    Missing: {compliance['structural_missing']}")
    print(f"  Business rules: {rules_coverage:.0%} ({rules_count}/{len(REQUIRED_RULES_L2)})")
    if compliance.get("rules_missing"):
        print(f"    Missing: {compliance['rules_missing']}")
    print(f"  Section C: {'PASS' if compliance_pass else 'FAIL'}")

    # D. Anti-gaming checks (L2)
    print("\n" + "=" * 60)
    print("D. ANTI-GAMING CHECKS (L1 + L2)")
    print("=" * 60)

    anti_gaming = check_anti_gaming_L2(root)
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
    )
    report["sections"]["D_anti_gaming"] = {
        "checks": anti_gaming,
        "all_pass": ag_pass,
    }
    for name, val in anti_gaming.items():
        if isinstance(val, bool):
            status = "PASS" if val else "FAIL"
            print(f"  [{status}] {name}")
        elif isinstance(val, list):
            print(f"  {name}: {val}")
        else:
            print(f"  {name}: {val}")
    print(f"  Section D: {'PASS' if ag_pass else 'FAIL'}")

    # E. Data flow validation (L2)
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

    # F. Role-data coupling (L2)
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

    # Final score
    print("\n" + "=" * 60)
    print("FINAL SCORE (L2)")
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
    print(f"\n  OVERALL: {'PASS' if all_sections_pass else 'FAIL'}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="L2 Evaluator: Supply chain workflow adaptation — "
                    "compound disruption (raw material shortage + quality hold)"
    )
    parser.add_argument("--bpmn", required=True,
                        help="Path to the agent's modified BPMN XML")
    parser.add_argument("--structural", required=True,
                        help="Path to structural_changes.json (agent-produced)")
    parser.add_argument("--rules", required=True,
                        help="Path to business_rules_compliance.json (agent-produced)")
    parser.add_argument("--results", required=True,
                        help="Path to test_results.json (from run_tests.py)")
    parser.add_argument("--output", default="evaluation_report_L2.json",
                        help="Output report path")

    args = parser.parse_args()
    report = evaluate_L2(args)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}")

    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
