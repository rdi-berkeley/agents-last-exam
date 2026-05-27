#!/usr/bin/env python3
"""
Evaluator for: LY Juice Monthly Scheduling — Workflow Adaptation Under Supply Disruption

Uses anchor-based topological detection (Approach B):
  - Original elements from the input BPMN are "anchored" (exact ID matching)
  - New elements added by the agent are detected by topology relative to anchors
  - Agent may rename new elements freely; evaluator checks structure, not names

Checks:
  A. 16 structural checks via flow-graph topology (all must pass)
  B. 24 test scenario results (>=22/24 pass rate)
  C. Compliance coverage (9 modifications + 10 rules = 100%)
  D. Anti-gaming checks
  E. Data flow validation (form property I/O chains between tasks)
  F. Role-data coupling (correct role assignment per authority level)

Usage:
  python evaluate.py --bpmn modified_process.bpmn20.xml \
                     --structural structural_changes.json \
                     --rules business_rules_compliance.json \
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

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
FLOWABLE_NS = "http://flowable.org/bpmn"

# ============================================================
# ANCHORED ELEMENTS — from case1-original-process.bpmn20.xml
# These IDs are known because the agent starts from the original
# input file and has no reason to rename existing elements.
# ============================================================

ORIGINAL_ELEMENTS = {
    "startEvent",
    "task_salesDemand",
    "task_productionSchedule",
    "task_skuSequencing",
    "task_capacityConfirmation",
    "task_monthlyPlanConfirmation",
    "task_procurementExecution",
    "task_factoryExecution",
    "gw_parallel_split",
    "gw_parallel_join",
    "endEvent",
}

# Gateway types that serve as decision points (exclusive or inclusive
# are semantically equivalent when exactly 2 mutually exclusive flows exist)
DECISION_GATEWAY_TYPES = {"exclusiveGateway", "inclusiveGateway"}

# Valid closed-loop targets — the exception handling path may route back
# to either capacity confirmation (for re-assessment with adjusted plan)
# or monthly plan confirmation (if capacity is implicitly re-confirmed).
# Both interpretations are valid per the source document Section VI item 6.
CLOSED_LOOP_TARGETS = {"task_capacityConfirmation", "task_monthlyPlanConfirmation"}

REQUIRED_MODIFICATIONS = [
    "modification_1",  # Procurement Availability Confirmation
    "modification_2",  # Supply Risk Assessment Branch
    "modification_3",  # Alternative Sourcing with Sufficiency Check
    "modification_4",  # Product Mix Adjustment
    "modification_5",  # Sales and Channel Confirmation
    "modification_6",  # Closed-Loop Back-Routing
    "modification_7",  # Inclusive Gateway Merge (dual-trigger convergence)
    "modification_8",  # Parallel Approval (independent director review)
    "modification_9",  # State-Dependent Routing (re-validation loop prevention)
]

REQUIRED_RULES = [
    "rule_1",   # Prioritize on-time delivery
    "rule_2",   # Do not easily adjust revenue target
    "rule_3",   # Do not easily adjust gross margin target
    "rule_4",   # Volume adjustable within limited range
    "rule_5",   # Prioritize procurement/sourcing/mix adjustment
    "rule_6",   # Product mix must be executable
    "rule_7",   # KPI adjustment is LAST RESORT
    "rule_8a",  # Dual-trigger convergence constraint
    "rule_9",   # Independent parallel approval constraint
    "rule_10",  # Re-validation loop prevention
]

# ============================================================
# ORIGINAL FORM PROPERTIES — data contracts from the original
# process that must be preserved in the modified version.
# Maps: task_id -> set of formProperty IDs
# ============================================================

# Output properties only — these are consumed by downstream tasks and MUST
# be preserved. Input properties may legitimately change when flow topology
# is restructured (e.g., convergence points gain new upstream paths).
ORIGINAL_OUTPUT_PROPERTIES = {
    "task_salesDemand": {
        "out_sku_demand", "out_delivery_rhythm", "out_market_constraints",
    },
    "task_productionSchedule": {
        "out_production_schedule", "out_warehouse_rhythm",
    },
    "task_skuSequencing": {
        "out_line_schedule", "out_changeover_plan",
    },
    "task_capacityConfirmation": {
        "out_capacity_status", "out_labor_status", "out_material_status",
        "out_cost_feasibility", "out_overall_verdict",
    },
    "task_monthlyPlanConfirmation": {
        "out_plan_id", "out_plan_status", "out_final_production_schedule",
        "out_final_sku_sequence", "out_material_orders",
    },
    "task_procurementExecution": {
        "out_purchase_orders",
    },
    # task_factoryExecution has no formProperties in the original
}

# ============================================================
# ORIGINAL ROLE ASSIGNMENTS — from the original process.
# Each entry: (flowable:assignee, flowable:candidateGroups)
# ============================================================

ORIGINAL_ROLE_ASSIGNMENTS = {
    "task_salesDemand": ("${salesLead}", "sales"),
    "task_productionSchedule": ("${productionLead}", "production"),
    "task_skuSequencing": ("${productionLead}", "production"),
    "task_capacityConfirmation": ("${supplyChainLead}", "supplychain"),
    "task_monthlyPlanConfirmation": ("${supplyChainLead}", "supplychain"),
    "task_procurementExecution": ("${procurementLead}", "procurement"),
    "task_factoryExecution": ("${productionLead}", "production"),
}

# ============================================================
# AUTHORITY LEVELS — for escalation monotonicity checking
# ============================================================

AUTHORITY_LEVELS = {
    "Unknown": 0,       # unrecognized role — flags issues in F2/F3
    "Operational": 1,   # productionLead, procurementLead
    "Lead": 2,          # salesLead, supplyChainLead
    "Management": 3,    # supplyChainDirector, salesDirector
    "Executive": 4,     # executive group
}

# Maps role variable names and candidate groups to authority levels
ROLE_AUTHORITY = {
    "${salesLead}": "Lead",
    "${productionLead}": "Operational",
    "${supplyChainLead}": "Lead",
    "${procurementLead}": "Operational",
    "${supplyChainDirector}": "Management",
    "${salesDirector}": "Management",
}

CANDIDATE_GROUP_AUTHORITY = {
    "sales": "Lead",
    "production": "Operational",
    "supplychain": "Lead",
    "procurement": "Operational",
    "supplychain_mgmt": "Management",
    "sales_mgmt": "Management",
    "executive": "Executive",
}

# Expected roles for new tasks discovered by topology
# (role_keyword_in_assignee, min_authority_level)
EXPECTED_NEW_TASK_ROLES = {
    "procurement_task": ("procurementLead", "Operational"),
    "sourcing_task": ("procurementLead", "Operational"),
    "mix_adjustment_task": ("Director", "Management"),
    "sales_confirmation_task": ("Director", "Management"),
    "escalation_task": (None, "Executive"),  # assignee may vary, but group must be executive
}


# ============================================================
# BPMN GRAPH BUILDER
# ============================================================

class BPMNGraph:
    """Builds a directed graph from BPMN XML for topological analysis."""

    def __init__(self, root: ET.Element):
        self.root = root
        self.forward = {}   # node_id -> [target_ids]
        self.reverse = {}   # node_id -> [source_ids]
        self.elements = {}  # node_id -> ET.Element
        self.elem_types = {}  # node_id -> tag (e.g. "userTask", "exclusiveGateway")
        self.flows = {}     # flow_id -> (sourceRef, targetRef, conditionExpr or None)
        self.end_events = set()

        self._build(root)

    def _build(self, root):
        """Parse all BPMN elements and sequence flows.

        Uses recursive XPath (.//), so elements and flows inside
        subProcess containers are also included in the graph. Additional
        synthetic edges bridge subProcess boundaries: an incoming flow
        targeting a subProcess is extended to the subProcess's internal
        startEvent, and the subProcess's internal endEvents are extended
        to the subProcess's outgoing flow targets.
        """
        for tag in ["startEvent", "endEvent", "userTask", "serviceTask",
                     "exclusiveGateway", "parallelGateway", "inclusiveGateway",
                     "callActivity", "subProcess", "task",
                     "intermediateCatchEvent", "intermediateThrowEvent",
                     "boundaryEvent"]:
            for elem in root.findall(f".//{{{BPMN_NS}}}{tag}"):
                eid = elem.get("id")
                if eid:
                    self.elements[eid] = elem
                    self.elem_types[eid] = tag
                    self.forward.setdefault(eid, [])
                    self.reverse.setdefault(eid, [])
                    if tag == "endEvent":
                        self.end_events.add(eid)

        for flow in root.findall(f".//{{{BPMN_NS}}}sequenceFlow"):
            fid = flow.get("id", "")
            src = flow.get("sourceRef", "")
            tgt = flow.get("targetRef", "")
            cond_el = flow.find(f"{{{BPMN_NS}}}conditionExpression")
            cond = cond_el.text.strip() if cond_el is not None and cond_el.text else None

            self.flows[fid] = (src, tgt, cond)
            self.forward.setdefault(src, []).append(tgt)
            self.reverse.setdefault(tgt, []).append(src)

        # Bridge subProcess boundaries with synthetic edges so BFS
        # can traverse into and out of subProcesses seamlessly.
        for sp in root.findall(f".//{{{BPMN_NS}}}subProcess"):
            sp_id = sp.get("id")
            if not sp_id:
                continue

            # Find internal startEvents — bridge: subProcess -> internal startEvent
            for se in sp.findall(f"{{{BPMN_NS}}}startEvent"):
                se_id = se.get("id")
                if se_id:
                    self.forward.setdefault(sp_id, []).append(se_id)
                    self.reverse.setdefault(se_id, []).append(sp_id)

            # Find internal endEvents — bridge: internal endEvent -> subProcess outgoing targets
            internal_ends = []
            for ee in sp.findall(f"{{{BPMN_NS}}}endEvent"):
                ee_id = ee.get("id")
                if ee_id:
                    internal_ends.append(ee_id)

            # For each outgoing flow from the subProcess, connect internal
            # endEvents to the outgoing target
            for tgt in self.forward.get(sp_id, []):
                if tgt == sp_id:
                    continue
                for ee_id in internal_ends:
                    if tgt not in self.forward.get(ee_id, []):
                        self.forward.setdefault(ee_id, []).append(tgt)
                        self.reverse.setdefault(tgt, []).append(ee_id)

    def successors(self, node_id):
        """Direct successors of a node."""
        return self.forward.get(node_id, [])

    def predecessors(self, node_id):
        """Direct predecessors of a node."""
        return self.reverse.get(node_id, [])

    def get_type(self, node_id):
        """Get BPMN element type (tag name)."""
        return self.elem_types.get(node_id, "")

    def is_decision_gateway(self, node_id):
        """Check if node is a decision gateway (exclusive or inclusive)."""
        return self.get_type(node_id) in DECISION_GATEWAY_TYPES

    def get_element(self, node_id):
        """Get the XML element for a node."""
        return self.elements.get(node_id)

    def get_assignee(self, node_id):
        """Get flowable:assignee attribute for a node."""
        elem = self.elements.get(node_id)
        if elem is None:
            return ""
        val = elem.get(f"{{{FLOWABLE_NS}}}assignee", "")
        if not val:
            val = elem.get("flowable:assignee", "")
        return val

    def get_candidate_groups(self, node_id):
        """Get flowable:candidateGroups attribute for a node."""
        elem = self.elements.get(node_id)
        if elem is None:
            return ""
        val = elem.get(f"{{{FLOWABLE_NS}}}candidateGroups", "")
        if not val:
            val = elem.get("flowable:candidateGroups", "")
        return val

    def get_form_properties(self, node_id):
        """Get all formProperty IDs and their [IN]/[OUT] direction for a node."""
        elem = self.elements.get(node_id)
        if elem is None:
            return {}
        props = {}
        # Look in extensionElements -> formProperty (use .// to find nested ones too,
        # e.g., inside <flowable:formData> wrappers from some BPMN authoring tools)
        for ext in elem.findall(f"{{{BPMN_NS}}}extensionElements"):
            for fp in ext.findall(f".//{{{FLOWABLE_NS}}}formProperty"):
                fp_id = fp.get("id", "")
                if fp_id:
                    props[fp_id] = {
                        "name": fp.get("name", ""),
                        "type": fp.get("type", ""),
                        "required": fp.get("required", "false"),
                        "writable": fp.get("writable", ""),
                        "readable": fp.get("readable", ""),
                    }
        return props

    def get_lane_for_node(self, node_id):
        """Find which lane a node belongs to, if any."""
        for lane_set in self.root.findall(f".//{{{BPMN_NS}}}laneSet"):
            for lane in lane_set.findall(f"{{{BPMN_NS}}}lane"):
                for ref in lane.findall(f"{{{BPMN_NS}}}flowNodeRef"):
                    if ref.text and ref.text.strip() == node_id:
                        return {
                            "lane_id": lane.get("id", ""),
                            "lane_name": lane.get("name", ""),
                        }
        return None

    def outgoing_flows_with_conditions(self, node_id):
        """Get all outgoing flows from a node with their conditions."""
        result = []
        for fid, (src, tgt, cond) in self.flows.items():
            if src == node_id:
                result.append({"flow_id": fid, "target": tgt, "condition": cond})
        return result

    def is_reachable(self, from_id, to_id, max_depth=20):
        """BFS check if to_id is reachable from from_id within max_depth hops."""
        visited = {from_id}
        queue = deque([(from_id, 0)])
        while queue:
            node, depth = queue.popleft()
            if node == to_id:
                return True
            if depth >= max_depth:
                continue
            for succ in self.successors(node):
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, depth + 1))
        return False

    def find_new_node_by_bfs(self, start_id, target_type, max_depth=4,
                             must_reach=None, visited_global=None):
        """
        BFS from start_id to find the first NEW node (not in ORIGINAL_ELEMENTS)
        of the given type within max_depth hops.

        Args:
            start_id: Node to start searching from
            target_type: Element type to find ("userTask", or "decision_gateway"
                         for exclusive/inclusive)
            max_depth: Maximum hops to search
            must_reach: If set, the found node must be able to reach this ID.
                        Can be a string (single target) or a set/list of IDs
                        (any one must be reachable).
            visited_global: Global visited set to avoid re-discovering nodes
        """
        if visited_global is None:
            visited_global = set()

        # Normalize must_reach to a set for uniform handling
        if must_reach is None:
            reach_targets = None
        elif isinstance(must_reach, str):
            reach_targets = {must_reach}
        else:
            reach_targets = set(must_reach)

        visited = {start_id}
        queue = deque([(start_id, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth > 0 and node not in ORIGINAL_ELEMENTS and node not in visited_global:
                type_match = False
                if target_type == "decision_gateway":
                    type_match = self.is_decision_gateway(node)
                else:
                    type_match = self.get_type(node) == target_type
                if type_match:
                    if reach_targets is None or any(
                        self.is_reachable(node, t, max_depth=10) for t in reach_targets
                    ):
                        return node
            if depth >= max_depth:
                continue
            for succ in self.successors(node):
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, depth + 1))
        return None


# ============================================================
# TOPOLOGICAL WALK: Discover new elements by position
# ============================================================

def discover_disruption_chain(graph: BPMNGraph):
    """
    Walk the flow graph from task_skuSequencing to discover the
    disruption handling chain using bounded BFS at each step.

    Uses a global visited set to prevent discovering the same node
    for multiple roles.
    """
    discovered = {
        "procurement_task": None,
        "risk_gateway": None,
        "sourcing_task": None,
        "sourcing_gateway": None,
        "mix_adjustment_task": None,
        "mix_precheck_task": None,
        "mix_validation_gateway": None,
        "sales_confirmation_task": None,
        "sales_gateway": None,
        "escalation_task": None,
        "escalation_end": None,
    }

    # Global visited set: once a node is assigned a role, it cannot be reused
    visited = set()

    # Step 1: Find new userTask reachable from task_skuSequencing that can reach
    # a closed-loop target (the procurement availability check)
    proc = graph.find_new_node_by_bfs(
        "task_skuSequencing", "userTask", max_depth=4,
        must_reach=CLOSED_LOOP_TARGETS, visited_global=visited
    )
    if proc:
        discovered["procurement_task"] = proc
        visited.add(proc)

    # Step 2: Find decision gateway after procurement task
    start = discovered["procurement_task"] or "task_skuSequencing"
    risk_gw = graph.find_new_node_by_bfs(
        start, "decision_gateway", max_depth=4, visited_global=visited
    )
    if risk_gw:
        discovered["risk_gateway"] = risk_gw
        visited.add(risk_gw)

    # Step 3: From risk gateway, find the exception-path userTask
    # (the flow that does NOT go to a closed-loop target)
    if discovered["risk_gateway"]:
        flows = graph.outgoing_flows_with_conditions(discovered["risk_gateway"])
        for flow in flows:
            tgt = flow["target"]
            if tgt not in CLOSED_LOOP_TARGETS and tgt not in ORIGINAL_ELEMENTS:
                # This branch leads to exception handling — find first new userTask
                sourcing = graph.find_new_node_by_bfs(
                    discovered["risk_gateway"], "userTask", max_depth=4,
                    visited_global=visited
                )
                if sourcing:
                    discovered["sourcing_task"] = sourcing
                    visited.add(sourcing)
                break

    # Step 4: Find sourcing result gateway after sourcing task
    if discovered["sourcing_task"]:
        src_gw = graph.find_new_node_by_bfs(
            discovered["sourcing_task"], "decision_gateway", max_depth=4,
            visited_global=visited
        )
        if src_gw:
            discovered["sourcing_gateway"] = src_gw
            visited.add(src_gw)

    # Step 5: From sourcing gateway, find mix adjustment task
    # (the flow that does NOT go to a closed-loop target)
    if discovered["sourcing_gateway"]:
        flows = graph.outgoing_flows_with_conditions(discovered["sourcing_gateway"])
        for flow in flows:
            tgt = flow["target"]
            if tgt not in CLOSED_LOOP_TARGETS and tgt not in ORIGINAL_ELEMENTS:
                mix = graph.find_new_node_by_bfs(
                    discovered["sourcing_gateway"], "userTask", max_depth=4,
                    visited_global=visited
                )
                if mix:
                    discovered["mix_adjustment_task"] = mix
                    visited.add(mix)
                break

    # Step 6a: Find mix capacity pre-check task and validation gateway
    # (Rule 6 requires pre-validation before presenting to Sales)
    #
    # Strategy: look for a decision gateway BETWEEN mix adjustment and any
    # downstream userTask, where that gateway has a DIRECT flow back to the
    # mix adjustment task. If such a gateway exists, the userTask before it
    # is the pre-check, and the userTask after it (on the non-loop branch)
    # is the sales confirmation.
    #
    # This avoids misidentifying the sales task as a pre-check via long
    # indirect paths through the capacity closed-loop.
    mix_id = discovered["mix_adjustment_task"]
    precheck = None
    mix_val_gw = None

    if mix_id:
        # Search for a gateway reachable from mix that has a DIRECT flow
        # back to mix (loop-back)
        visited_search = set(visited)
        queue = deque([(mix_id, 0)])
        visited_bfs = {mix_id}
        candidate_tasks = []  # new userTasks found before any loop-back gw

        while queue:
            node, depth = queue.popleft()
            # Tight depth limit: pre-check + validation gateway = 2-3 hops from mix
            if depth > 3:
                continue
            if depth > 0 and node not in ORIGINAL_ELEMENTS and node not in visited:
                ntype = graph.get_type(node)
                if ntype == "userTask":
                    candidate_tasks.append(node)
                elif graph.is_decision_gateway(node):
                    # Check for DIRECT flow back to mix_id
                    for f in graph.outgoing_flows_with_conditions(node):
                        if f["target"] == mix_id:
                            # Found the loop-back gateway!
                            mix_val_gw = node
                            break
                    if mix_val_gw:
                        break
            for succ in graph.successors(node):
                if succ not in visited_bfs:
                    visited_bfs.add(succ)
                    queue.append((succ, depth + 1))

        if mix_val_gw and candidate_tasks:
            # The last candidate task before the loop-back gateway is the pre-check
            # (typically there's just one new task between mix and the gateway)
            precheck = candidate_tasks[-1]
            discovered["mix_precheck_task"] = precheck
            discovered["mix_validation_gateway"] = mix_val_gw
            visited.add(precheck)
            visited.add(mix_val_gw)

    # Step 6c: After mix validation gateway (or pre-check), look for
    # the parallel approval pattern or fall back to single sales task.
    #
    # New topology:
    #   mix_validation_gw -> gw_approvalSplit (parallel) -> {task_scDirectorMixReview, task_salesChannelConfirmation}
    #                     -> gw_approvalJoin (parallel) -> gw_approvalDecision (exclusive)
    #                     -> escalation_task (path to new endEvent)
    #                     -> task_setRevalidationFlag (path back to capacity)
    #
    # We first try to discover the parallel approval pattern. If no parallel
    # split is found, fall back to the old single-task discovery.

    approval_search_start = (
        discovered["mix_validation_gateway"]
        or discovered["mix_precheck_task"]
        or discovered["mix_adjustment_task"]
    )

    parallel_approval_found = False
    if approval_search_start:
        # Look for a new parallelGateway (split) after the validation gateway
        par_splits = [
            eid for eid, etype in graph.elem_types.items()
            if etype == "parallelGateway" and eid not in ORIGINAL_ELEMENTS
        ]
        approval_split = None
        for ps_id in par_splits:
            if graph.is_reachable(approval_search_start, ps_id, max_depth=4):
                outgoing = graph.successors(ps_id)
                if len(outgoing) >= 2:
                    approval_split = ps_id
                    break

        if approval_split:
            # Found parallel approval split — discover both branch tasks
            branch_tasks = []
            for branch_target in graph.successors(approval_split):
                # BFS from each branch to find the first new userTask
                bfs_q = deque([(branch_target, 0)])
                bfs_visited = {approval_split}
                while bfs_q:
                    n, d = bfs_q.popleft()
                    if d > 3:
                        continue
                    if n in bfs_visited:
                        continue
                    bfs_visited.add(n)
                    if graph.get_type(n) == "userTask" and n not in ORIGINAL_ELEMENTS and n not in visited:
                        branch_tasks.append(n)
                        break
                    for s in graph.successors(n):
                        if s not in bfs_visited:
                            bfs_q.append((s, d + 1))

            if len(branch_tasks) >= 2:
                parallel_approval_found = True
                discovered["approval_split"] = approval_split
                visited.add(approval_split)

                # Identify which is the sales confirmation task — look for
                # salesDirector in assignee or sales_mgmt in candidateGroups
                sales_task_found = None
                sc_review_found = None
                for bt in branch_tasks:
                    assignee = graph.get_assignee(bt).lower()
                    groups = graph.get_candidate_groups(bt).lower()
                    if "salesdirector" in assignee or "sales_mgmt" in groups:
                        sales_task_found = bt
                    else:
                        sc_review_found = bt

                # If we can't distinguish by role, use the first and second
                if not sales_task_found and len(branch_tasks) >= 2:
                    sales_task_found = branch_tasks[1]
                    sc_review_found = branch_tasks[0]
                elif not sc_review_found and len(branch_tasks) >= 2:
                    sc_review_found = branch_tasks[0]

                discovered["sales_confirmation_task"] = sales_task_found
                discovered["sc_director_review_task"] = sc_review_found
                if sales_task_found:
                    visited.add(sales_task_found)
                if sc_review_found:
                    visited.add(sc_review_found)

                # Find the parallel join — a new parallelGateway reachable from both branches
                approval_join = None
                for pj_id in par_splits:
                    if pj_id == approval_split:
                        continue
                    if pj_id in visited:
                        continue
                    # Must be reachable from both branch tasks
                    both_reach = all(
                        graph.is_reachable(bt, pj_id, max_depth=5)
                        for bt in branch_tasks
                    )
                    if both_reach:
                        approval_join = pj_id
                        break

                if approval_join:
                    discovered["approval_join"] = approval_join
                    visited.add(approval_join)

                    # Find the approval decision gateway (exclusive) after the join
                    decision_gw = graph.find_new_node_by_bfs(
                        approval_join, "decision_gateway", max_depth=3,
                        visited_global=visited
                    )
                    if decision_gw:
                        discovered["sales_gateway"] = decision_gw
                        visited.add(decision_gw)

    if not parallel_approval_found and approval_search_start:
        # Fallback: old single-task discovery
        sales = graph.find_new_node_by_bfs(
            approval_search_start, "userTask", max_depth=4,
            visited_global=visited
        )
        if sales:
            discovered["sales_confirmation_task"] = sales
            visited.add(sales)

        # Find sales acceptance gateway after sales confirmation
        if discovered["sales_confirmation_task"]:
            sales_gw = graph.find_new_node_by_bfs(
                discovered["sales_confirmation_task"], "decision_gateway", max_depth=4,
                visited_global=visited
            )
            if sales_gw:
                discovered["sales_gateway"] = sales_gw
                visited.add(sales_gw)

    # Step 8: From the approval decision / sales gateway, find escalation task
    # The escalation task is on the SHORTEST path to a NON-original endEvent.
    # We must distinguish the escalation path (short, direct to endEvent) from
    # the revalidation path (loops back through capacity, which can also
    # eventually reach the escalation endEvent via a long path).
    if discovered["sales_gateway"]:
        decision_gw = discovered["sales_gateway"]
        flows = graph.outgoing_flows_with_conditions(decision_gw)

        # For each outgoing flow, compute the shortest distance to any
        # non-original endEvent. The path with the shortest distance is
        # the escalation path.
        best_target = None
        best_dist = 999

        for flow in flows:
            tgt = flow["target"]
            # BFS from tgt to find shortest distance to any new endEvent
            bfs_q = deque([(tgt, 1)])
            bfs_visited = {decision_gw}
            min_dist = 999
            while bfs_q:
                n, d = bfs_q.popleft()
                if d >= best_dist:
                    continue
                if n in bfs_visited:
                    continue
                bfs_visited.add(n)
                if graph.get_type(n) == "endEvent" and n != "endEvent":
                    min_dist = d
                    break
                for s in graph.successors(n):
                    if s not in bfs_visited:
                        bfs_q.append((s, d + 1))
            if min_dist < best_dist:
                best_dist = min_dist
                best_target = tgt

        if best_target:
            if graph.get_type(best_target) == "userTask" and best_target not in visited:
                discovered["escalation_task"] = best_target
                visited.add(best_target)
            elif graph.get_type(best_target) == "endEvent":
                discovered["escalation_end"] = best_target
            else:
                # BFS from the target to find escalation task
                esc = graph.find_new_node_by_bfs(
                    best_target, "userTask", max_depth=4,
                    visited_global=visited
                )
                if esc:
                    discovered["escalation_task"] = esc
                    visited.add(esc)

    # Step 9: From escalation task, find terminal end event
    if discovered["escalation_task"] and not discovered["escalation_end"]:
        for succ in graph.successors(discovered["escalation_task"]):
            if graph.get_type(succ) == "endEvent":
                discovered["escalation_end"] = succ
                break
        # Also try BFS if not direct
        if not discovered["escalation_end"]:
            for ee in graph.end_events:
                if ee != "endEvent" and graph.is_reachable(
                    discovered["escalation_task"], ee, max_depth=5
                ):
                    discovered["escalation_end"] = ee
                    break

    return discovered


# ============================================================
# SECTION A: STRUCTURAL CHECKS
# ============================================================

def check_structural(root: ET.Element):
    """
    Run structural checks using anchor-based topological detection.
    Returns (results_dict, details_dict).
    """
    results = {}
    details = {}

    graph = BPMNGraph(root)
    discovered = discover_disruption_chain(graph)
    details["discovered_nodes"] = {k: v for k, v in discovered.items()}

    # Check 1: BPMN is parseable (if we got here, it parsed)
    results["check_1_bpmn_parseable"] = True

    # Check 2: Direct link broken — task_skuSequencing no longer flows
    #           directly to task_capacityConfirmation
    sku_successors = graph.successors("task_skuSequencing")
    direct_link_broken = "task_capacityConfirmation" not in sku_successors
    results["check_2_direct_link_broken"] = direct_link_broken

    # Check 3: New procurement task inserted between SKU and main flow
    procurement = discovered["procurement_task"]
    if procurement:
        can_reach_main = any(
            graph.is_reachable(procurement, t, max_depth=10)
            for t in CLOSED_LOOP_TARGETS
        )
        results["check_3_procurement_task_inserted"] = can_reach_main
    else:
        results["check_3_procurement_task_inserted"] = False

    # Check 4: Risk assessment gateway with 2 outgoing flows
    # Accept: 2 conditional, or 1 conditional + 1 default (len(flows)==2 and >= 1 conditional)
    risk_gw = discovered["risk_gateway"]
    if risk_gw:
        flows = graph.outgoing_flows_with_conditions(risk_gw)
        conditional_flows = [f for f in flows if f["condition"] is not None]
        results["check_4_risk_gateway_2_flows"] = (
            len(flows) == 2 and len(conditional_flows) >= 1
        )
        details["risk_gateway_flows"] = [
            {"target": f["target"], "condition": f["condition"]} for f in flows
        ]
    else:
        results["check_4_risk_gateway_2_flows"] = False

    # Check 5: No-risk path leads to a closed-loop target
    # (task_capacityConfirmation or task_monthlyPlanConfirmation)
    if risk_gw:
        flows = graph.outgoing_flows_with_conditions(risk_gw)
        has_closed_loop_target = any(
            f["target"] in CLOSED_LOOP_TARGETS
            or any(graph.is_reachable(f["target"], t, max_depth=3) for t in CLOSED_LOOP_TARGETS)
            for f in flows
        )
        results["check_5_no_risk_path_to_main_flow"] = has_closed_loop_target
    else:
        results["check_5_no_risk_path_to_main_flow"] = False

    # Check 6: Exception path exists — sourcing task found
    sourcing = discovered["sourcing_task"]
    results["check_6_exception_path_sourcing"] = sourcing is not None

    # Check 7: Sourcing result gateway with 2 flows
    # (one path to closed-loop target, one to further exception)
    sourcing_gw = discovered["sourcing_gateway"]
    if sourcing_gw:
        flows = graph.outgoing_flows_with_conditions(sourcing_gw)
        conditional_flows = [f for f in flows if f["condition"] is not None]
        has_main_flow_path = any(
            f["target"] in CLOSED_LOOP_TARGETS
            or any(graph.is_reachable(f["target"], t, max_depth=3) for t in CLOSED_LOOP_TARGETS)
            for f in flows
        )
        has_exception_path = any(
            f["target"] not in ORIGINAL_ELEMENTS for f in flows
        )
        results["check_7_sourcing_gateway_branches"] = (
            len(flows) == 2
            and len(conditional_flows) >= 1
            and has_main_flow_path
            and has_exception_path
        )
    else:
        results["check_7_sourcing_gateway_branches"] = False

    # Check 8: Closed-loop back-routing — from sales acceptance gateway
    #           one flow leads (directly or indirectly) to a closed-loop target
    #           (task_capacityConfirmation or task_monthlyPlanConfirmation)
    sales_gw = discovered["sales_gateway"]
    if sales_gw:
        flows = graph.outgoing_flows_with_conditions(sales_gw)
        has_closed_loop = any(
            f["target"] in CLOSED_LOOP_TARGETS
            or any(graph.is_reachable(f["target"], t, max_depth=3) for t in CLOSED_LOOP_TARGETS)
            for f in flows
        )
        results["check_8_closed_loop_back_routing"] = has_closed_loop
    else:
        results["check_8_closed_loop_back_routing"] = False

    # Check 9: Escalation terminal path — >= 2 end events
    #           AND a path from exception chain to a non-original endEvent
    num_end_events = len(graph.end_events)
    has_new_end = discovered["escalation_end"] is not None
    results["check_9_escalation_terminal_path"] = num_end_events >= 2 and has_new_end

    # Check 11: Mix capacity pre-validation loop — after mix adjustment,
    #            there must be a pre-check task + validation gateway BEFORE
    #            sales confirmation. The validation gateway must have a loop-back
    #            flow to the mix adjustment task (for revision on failure).
    mix_task = discovered["mix_adjustment_task"]
    precheck_task = discovered["mix_precheck_task"]
    mix_val_gw = discovered["mix_validation_gateway"]
    sales_task = discovered["sales_confirmation_task"]

    if precheck_task and mix_val_gw and mix_task:
        # Verify: mix_val_gw has one flow to sales_task (or reachable to sales)
        # AND one flow back to mix_task (or reachable to mix — loop-back)
        v_flows = graph.outgoing_flows_with_conditions(mix_val_gw)
        reaches_sales = any(
            f["target"] == sales_task
            or graph.is_reachable(f["target"], sales_task, max_depth=3)
            for f in v_flows
        ) if sales_task else False
        reaches_mix_back = any(
            f["target"] == mix_task
            or graph.is_reachable(f["target"], mix_task, max_depth=3)
            for f in v_flows
        )
        results["check_11_mix_prevalidation_loop"] = (
            reaches_sales and reaches_mix_back and len(v_flows) >= 2
        )
    else:
        results["check_11_mix_prevalidation_loop"] = False
        details["mix_prevalidation_note"] = (
            f"Missing: precheck_task={precheck_task}, "
            f"mix_val_gw={mix_val_gw}, mix_task={mix_task}"
        )

    # Check 10: Capacity verdict gateway — after task_capacityConfirmation,
    #            there must be a decision gateway before task_monthlyPlanConfirmation.
    #            This enforces Business Rule 8 (capacity re-validation after adjustment).
    #            Direct link from capacity to monthly plan = skipping re-validation routing.
    cap_successors = graph.successors("task_capacityConfirmation")
    direct_to_monthly = "task_monthlyPlanConfirmation" in cap_successors
    # Must NOT have direct link; must have an intermediate decision gateway
    if not direct_to_monthly:
        # BFS from task_capacityConfirmation: find the first decision gateway
        # that can reach both task_monthlyPlanConfirmation AND the mix adjustment path
        verdict_gw = None
        queue = deque([(nid, 0) for nid in cap_successors])
        visited_v = {"task_capacityConfirmation"}
        while queue:
            node, depth = queue.popleft()
            if depth > 3:
                continue
            if graph.is_decision_gateway(node) and node not in ORIGINAL_ELEMENTS:
                # Check: one outgoing flow reaches monthly plan, another doesn't
                v_flows = graph.outgoing_flows_with_conditions(node)
                reaches_monthly = any(
                    f["target"] == "task_monthlyPlanConfirmation"
                    or graph.is_reachable(f["target"], "task_monthlyPlanConfirmation", max_depth=3)
                    for f in v_flows
                )
                # At least one flow leads to a non-monthly path (mix adjustment or exception)
                mix_task_id = discovered.get("mix_adjustment_task")
                reaches_exception = any(
                    f["target"] not in ORIGINAL_ELEMENTS
                    or (mix_task_id and graph.is_reachable(f["target"], mix_task_id, max_depth=5))
                    for f in v_flows
                )
                if reaches_monthly and reaches_exception and len(v_flows) >= 2:
                    verdict_gw = node
                    break
            if node not in visited_v:
                visited_v.add(node)
                for succ in graph.successors(node):
                    if succ not in visited_v:
                        queue.append((succ, depth + 1))
        results["check_10_capacity_verdict_gateway"] = verdict_gw is not None
        if verdict_gw:
            details["capacity_verdict_gateway"] = verdict_gw
            discovered["capacity_verdict_gateway"] = verdict_gw
    else:
        results["check_10_capacity_verdict_gateway"] = False
        details["capacity_verdict_note"] = (
            "Direct flow from task_capacityConfirmation to task_monthlyPlanConfirmation "
            "detected — Business Rule 8 requires a verdict gateway for re-validation routing"
        )

    # ------------------------------------------------------------------
    # Check 12: Inclusive Gateway Merge — the mix_adjustment_task must be
    # reachable from an inclusiveGateway that has >=2 incoming flows from
    # two different paths (sourcing outcome exception + capacity verdict
    # NEEDS_ADJUSTMENT).
    # ------------------------------------------------------------------
    inclusive_gws = [
        eid for eid, etype in graph.elem_types.items()
        if etype == "inclusiveGateway"
    ]
    check_12_pass = False
    check_12_detail = None

    mix_adj = discovered["mix_adjustment_task"]
    for ig_id in inclusive_gws:
        incoming = graph.predecessors(ig_id)
        if len(incoming) < 2:
            continue
        # Check outgoing leads to mix_adjustment_task (direct or BFS)
        reaches_mix = (
            mix_adj is not None
            and (mix_adj in graph.successors(ig_id)
                 or graph.is_reachable(ig_id, mix_adj, max_depth=5))
        )
        if not reaches_mix:
            continue

        # Verify: one incoming path traces back to a sourcing outcome gateway,
        # another traces back to a capacity verdict gateway.
        sourcing_gw_id = discovered.get("sourcing_gateway")
        # The capacity verdict gateway may have been stored during check_10
        # or we fall back to discovering it from task_capacityConfirmation
        cap_verdict_gw = details.get("capacity_verdict_gateway") or discovered.get("capacity_verdict_gateway")

        has_sourcing_path = False
        has_capacity_path = False
        for pred in incoming:
            # Trace backward from pred to see if sourcing_gw or cap_verdict_gw is upstream
            if sourcing_gw_id and (
                pred == sourcing_gw_id
                or graph.is_reachable(sourcing_gw_id, pred, max_depth=5)
            ):
                has_sourcing_path = True
            if cap_verdict_gw and (
                pred == cap_verdict_gw
                or graph.is_reachable(cap_verdict_gw, pred, max_depth=5)
            ):
                has_capacity_path = True

        if has_sourcing_path and has_capacity_path:
            check_12_pass = True
            check_12_detail = {
                "inclusive_gateway": ig_id,
                "incoming_count": len(incoming),
                "has_sourcing_path": True,
                "has_capacity_path": True,
            }
            break

    results["check_12_inclusive_gateway_merge"] = check_12_pass
    if check_12_detail:
        details["check_12_detail"] = check_12_detail
    else:
        details["check_12_note"] = (
            f"No inclusiveGateway found with >=2 incoming flows leading to "
            f"mix_adjustment_task ({mix_adj}). "
            f"Inclusive gateways in model: {inclusive_gws}"
        )

    # ------------------------------------------------------------------
    # Check 13: Parallel Approval Pattern — between the mix pre-check
    # validation result and the closed-loop back-routing there must be a
    # parallel approval pattern: either (A) two parallelGateway elements
    # (split+join) with exactly 2 userTasks between them assigned to
    # different director-level roles, or (B) a single userTask with
    # multiInstanceLoopCharacteristics (isSequential="false").
    # ------------------------------------------------------------------
    check_13_pass = False
    check_13_detail = None

    # Find parallel gateway pairs: BFS forward from mix_validation_gateway
    # (or mix_precheck_task) looking for parallelGateway split.
    par_search_start = (
        discovered.get("mix_validation_gateway")
        or discovered.get("mix_precheck_task")
        or discovered.get("mix_adjustment_task")
    )

    if par_search_start:
        # Option A: find parallelGateway split
        par_splits = [
            eid for eid, etype in graph.elem_types.items()
            if etype == "parallelGateway" and eid not in ORIGINAL_ELEMENTS
        ]
        for ps_id in par_splits:
            # Must be reachable from par_search_start
            if not graph.is_reachable(par_search_start, ps_id, max_depth=8):
                continue
            outgoing = graph.successors(ps_id)
            if len(outgoing) < 2:
                continue
            # Find matching join: a parallelGateway reachable from all branches
            # Collect userTasks on each branch before the join
            branch_tasks = []
            join_candidate = None
            for branch_target in outgoing:
                # BFS from branch_target, collect userTasks until parallelGateway
                bfs_q = deque([(branch_target, 0)])
                bfs_visited = {ps_id}
                branch_user_tasks = []
                while bfs_q:
                    n, d = bfs_q.popleft()
                    if d > 5:
                        continue
                    if n in bfs_visited:
                        continue
                    bfs_visited.add(n)
                    ntype = graph.get_type(n)
                    if ntype == "userTask" and n not in ORIGINAL_ELEMENTS:
                        branch_user_tasks.append(n)
                    elif ntype == "parallelGateway" and n != ps_id and n not in ORIGINAL_ELEMENTS:
                        join_candidate = n
                        break
                    for s in graph.successors(n):
                        if s not in bfs_visited:
                            bfs_q.append((s, d + 1))
                branch_tasks.extend(branch_user_tasks)

            if join_candidate and len(branch_tasks) == 2:
                # Check the two tasks have different director-level roles
                roles_found = set()
                for bt in branch_tasks:
                    assignee = graph.get_assignee(bt).lower()
                    groups = graph.get_candidate_groups(bt).lower()
                    if "director" in assignee or "mgmt" in groups or "management" in groups:
                        roles_found.add(assignee)

                if len(roles_found) >= 2:
                    check_13_pass = True
                    check_13_detail = {
                        "pattern": "parallel_gateway_pair",
                        "split": ps_id,
                        "join": join_candidate,
                        "user_tasks": branch_tasks,
                        "roles": list(roles_found),
                    }
                    break

        # Option B: multiInstanceLoopCharacteristics (isSequential="false")
        if not check_13_pass and par_search_start:
            for node_id in graph.elements:
                if graph.get_type(node_id) != "userTask":
                    continue
                if node_id in ORIGINAL_ELEMENTS:
                    continue
                elem = graph.get_element(node_id)
                if elem is None:
                    continue
                mi = elem.find(f"{{{BPMN_NS}}}multiInstanceLoopCharacteristics")
                if mi is not None and mi.get("isSequential", "true").lower() == "false":
                    if graph.is_reachable(par_search_start, node_id, max_depth=8):
                        check_13_pass = True
                        check_13_detail = {
                            "pattern": "multi_instance_parallel",
                            "task": node_id,
                        }
                        break

    results["check_13_parallel_approval"] = check_13_pass
    if check_13_detail:
        details["check_13_detail"] = check_13_detail
    else:
        details["check_13_note"] = (
            "No parallel approval pattern found (neither parallelGateway pair "
            "with 2 director-level userTasks nor multiInstance parallel task)"
        )

    # ------------------------------------------------------------------
    # Check 14: State-Dependent Routing — the capacity verdict gateway
    # must have outgoing conditions referencing >=2 distinct variables
    # (e.g. overall_verdict AND isRevalidation), ensuring different
    # behaviour on first-pass vs. re-validation.
    # ------------------------------------------------------------------
    check_14_pass = False
    check_14_detail = None

    cap_verdict_gw = details.get("capacity_verdict_gateway") or discovered.get("capacity_verdict_gateway")
    if cap_verdict_gw:
        flows_cv = graph.outgoing_flows_with_conditions(cap_verdict_gw)
        # Extract ALL variable references from EL expressions.
        # Variables appear as identifiers that are not EL keywords/literals.
        # Strategy: find all word tokens, then exclude known literals and operators.
        el_keywords = {"true", "false", "null", "empty", "not", "and", "or",
                       "eq", "ne", "lt", "gt", "le", "ge", "div", "mod",
                       "instanceof", "NEEDS_ADJUSTMENT", "CONFIRMED"}
        var_pattern_14_all = re.compile(r"[a-zA-Z_]\w*")
        all_vars = set()
        for f in flows_cv:
            cond = f.get("condition")
            if cond:
                # Strip ${...} wrapper
                inner = cond.strip()
                if inner.startswith("${") and inner.endswith("}"):
                    inner = inner[2:-1]
                tokens = var_pattern_14_all.findall(inner)
                for tok in tokens:
                    if tok not in el_keywords and not tok.startswith("'"):
                        all_vars.add(tok)
        if len(all_vars) >= 2:
            check_14_pass = True
            check_14_detail = {
                "capacity_verdict_gateway": cap_verdict_gw,
                "condition_variables": sorted(all_vars),
                "variable_count": len(all_vars),
            }

    results["check_14_state_dependent_routing"] = check_14_pass
    if check_14_detail:
        details["check_14_detail"] = check_14_detail
    else:
        details["check_14_note"] = (
            f"Capacity verdict gateway ({cap_verdict_gw}) has conditions "
            f"referencing fewer than 2 distinct variables"
        )

    # Check original elements preserved
    missing_originals = [eid for eid in ORIGINAL_ELEMENTS if eid not in graph.elements]
    details["missing_original_elements"] = missing_originals
    results["original_elements_preserved"] = len(missing_originals) == 0

    # Role assignment checks on discovered new nodes
    role_checks = {}
    mix_task = discovered["mix_adjustment_task"]
    if mix_task:
        assignee = graph.get_assignee(mix_task)
        assignee_lower = assignee.lower()
        is_director = "director" in assignee_lower
        is_not_lead = "lead" not in assignee_lower
        role_checks["mix_task_assignee"] = assignee
        role_checks["mix_task_is_director_level"] = is_director and is_not_lead
    else:
        role_checks["mix_task_is_director_level"] = False

    sales_task = discovered["sales_confirmation_task"]
    if sales_task:
        assignee = graph.get_assignee(sales_task)
        assignee_lower = assignee.lower()
        is_director = "director" in assignee_lower
        is_not_lead = "lead" not in assignee_lower
        role_checks["sales_task_assignee"] = assignee
        role_checks["sales_task_is_director_level"] = is_director and is_not_lead
    else:
        role_checks["sales_task_is_director_level"] = False

    details["role_checks"] = role_checks

    # Include role check in structural results
    role_pass = (
        role_checks.get("mix_task_is_director_level", False)
        and role_checks.get("sales_task_is_director_level", False)
    )
    results["role_assignments_correct"] = role_pass

    return results, details


# ============================================================
# SECTION D: ANTI-GAMING CHECKS
# ============================================================

def check_anti_gaming(root: ET.Element) -> dict:
    """Check for hardcoded conditions and other gaming patterns."""
    results = {}
    graph = BPMNGraph(root)

    # Collect all conditions from sequence flows
    conditions = []
    for fid, (src, tgt, cond) in graph.flows.items():
        if cond:
            conditions.append(cond)

    # Check conditions use process variables (${...} pattern)
    # Allow complex EL expressions: ${expr}, including spaces and operators
    variable_pattern = re.compile(r"^\$\{.+\}$", re.DOTALL)
    uses_variables = all(
        variable_pattern.search(c.strip()) for c in conditions
    ) if conditions else False
    hardcoded = any(c.strip() in ("true", "false", "1", "0") for c in conditions)

    results["conditions_use_variables"] = uses_variables and not hardcoded
    results["no_hardcoded_conditions"] = not hardcoded
    results["condition_count"] = len(conditions)
    # Need at least 3 conditional flows (3 gateways; some may use default flows)
    results["min_conditions_met"] = len(conditions) >= 3

    # Check original elements are preserved
    missing = [eid for eid in ORIGINAL_ELEMENTS if eid not in graph.elements]
    results["original_elements_preserved"] = len(missing) == 0
    if missing:
        results["missing_original_elements"] = missing

    # D2: Default flows on exclusive gateways
    # BPMN 2.0 best practice: each exclusive gateway should designate one
    # outgoing flow as the default (gateway element has default="flowId" and
    # the default flow has NO conditionExpression). This ensures robustness
    # when no condition matches.
    new_exclusive_gws = [
        eid for eid, etype in graph.elem_types.items()
        if etype == "exclusiveGateway"
        and eid not in ORIGINAL_ELEMENTS
        and len(graph.successors(eid)) >= 2  # Only check decision gateways
    ]
    gws_with_default = 0
    gws_missing_default = []
    for gw_id in new_exclusive_gws:
        elem = graph.get_element(gw_id)
        default_flow = elem.get("default", "") if elem is not None else ""
        if default_flow and default_flow in graph.flows:
            # Verify the default flow has no conditionExpression
            _, _, cond = graph.flows[default_flow]
            if cond is None or cond.strip() == "":
                gws_with_default += 1
            else:
                gws_missing_default.append(gw_id)
        else:
            gws_missing_default.append(gw_id)

    if new_exclusive_gws:
        results["default_flows_on_gateways"] = gws_with_default == len(new_exclusive_gws)
    else:
        results["default_flows_on_gateways"] = True  # No new gateways = vacuously true
    if gws_missing_default:
        results["gateways_missing_default"] = gws_missing_default

    # D3: New userTasks must have <documentation> child elements
    # Professional BPM practice: every task should document its purpose.
    new_tasks = [
        eid for eid, etype in graph.elem_types.items()
        if etype == "userTask" and eid not in ORIGINAL_ELEMENTS
    ]
    tasks_with_docs = 0
    tasks_missing_docs = []
    for task_id in new_tasks:
        elem = graph.get_element(task_id)
        if elem is not None:
            doc = elem.find(f"{{{BPMN_NS}}}documentation")
            if doc is not None and doc.text and len(doc.text.strip()) > 10:
                tasks_with_docs += 1
            else:
                tasks_missing_docs.append(task_id)
        else:
            tasks_missing_docs.append(task_id)

    if new_tasks:
        results["new_tasks_have_documentation"] = tasks_with_docs == len(new_tasks)
    else:
        results["new_tasks_have_documentation"] = True
    if tasks_missing_docs:
        results["tasks_missing_documentation"] = tasks_missing_docs

    return results


# ============================================================
# SECTION C: COMPLIANCE COVERAGE
# ============================================================

def check_compliance(structural_path: str, rules_path: str) -> dict:
    """Check compliance coverage from agent-produced JSON files."""
    results = {}

    try:
        with open(structural_path) as f:
            structural = json.load(f)
        mods = structural.get("modifications", {})
        covered_mods = [m for m in REQUIRED_MODIFICATIONS if m in mods]
        results["structural_coverage"] = len(covered_mods) / len(REQUIRED_MODIFICATIONS)
        results["structural_missing"] = [m for m in REQUIRED_MODIFICATIONS if m not in mods]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        results["structural_coverage"] = 0.0
        results["structural_error"] = str(e)

    try:
        with open(rules_path) as f:
            rules = json.load(f)
        rule_entries = rules.get("rules", {})
        covered_rules = [r for r in REQUIRED_RULES if r in rule_entries]
        results["rules_coverage"] = len(covered_rules) / len(REQUIRED_RULES)
        results["rules_missing"] = [r for r in REQUIRED_RULES if r not in rule_entries]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        results["rules_coverage"] = 0.0
        results["rules_error"] = str(e)

    return results


# ============================================================
# SECTION B: TEST SCENARIO RESULTS
# ============================================================

def _scenario_passed(s: dict) -> bool:
    if "pass" in s:
        return bool(s["pass"])
    if "passed" in s:
        return bool(s["passed"])
    status = s.get("status", "")
    if isinstance(status, str):
        return status.lower() in ("pass", "passed", "success")
    return False


def check_test_results(results_path: str) -> dict:
    """Check test scenario pass rate."""
    results = {}
    try:
        with open(results_path) as f:
            test_results = json.load(f)
        scenarios = test_results.get("scenarios", test_results.get("results", []))
        total = len(scenarios)
        passed = sum(1 for s in scenarios if _scenario_passed(s))
        results["process_definition_key"] = test_results.get("process_definition_key")
        results["total_scenarios"] = total
        results["passed_scenarios"] = passed
        results["pass_rate"] = passed / total if total > 0 else 0.0
        results["meets_threshold"] = passed >= int(total * 0.9) and total >= 20
    except (FileNotFoundError, json.JSONDecodeError) as e:
        results["process_definition_key"] = None
        results["total_scenarios"] = 0
        results["passed_scenarios"] = 0
        results["pass_rate"] = 0.0
        results["meets_threshold"] = False
        results["error"] = str(e)
    return results


# ============================================================
# SECTION E: DATA FLOW VALIDATION
# ============================================================

def check_data_flow(root: ET.Element, discovered: dict) -> dict:
    """
    Validate data flow through the process:
    E1: Original formProperties preserved on original tasks
    E2: New tasks have [IN] and [OUT] form properties
    E3: Producer-consumer chain coherence
    E4: Gateway condition variables match upstream task outputs
    """
    results = {}
    details = {}
    graph = BPMNGraph(root)

    # --- E1: Original OUTPUT formProperties preserved ---
    # Only output properties are checked — inputs may legitimately change
    # when flow topology is restructured (convergence points, new paths).
    e1_missing = {}
    e1_all_preserved = True
    for task_id, expected_outputs in ORIGINAL_OUTPUT_PROPERTIES.items():
        actual_props = set(graph.get_form_properties(task_id).keys())
        missing = expected_outputs - actual_props
        if missing:
            e1_missing[task_id] = sorted(missing)
            e1_all_preserved = False
    results["e1_original_output_properties_preserved"] = e1_all_preserved
    details["e1_missing_output_properties"] = e1_missing

    # --- E2: New tasks have [IN] and [OUT] form properties ---
    e2_results = {}
    new_task_roles = ["procurement_task", "sourcing_task", "mix_adjustment_task",
                      "sales_confirmation_task", "escalation_task"]
    for role in new_task_roles:
        node_id = discovered.get(role)
        if not node_id:
            continue
        props = graph.get_form_properties(node_id)
        has_in = any(pid.startswith("in_") for pid in props)
        has_out = any(pid.startswith("out_") for pid in props)
        # Escalation task may not have typed outputs (it's a terminal path)
        if role == "escalation_task":
            e2_results[role] = {"node_id": node_id, "property_count": len(props), "ok": True}
        else:
            e2_results[role] = {
                "node_id": node_id,
                "has_in": has_in,
                "has_out": has_out,
                "property_count": len(props),
                "ok": has_in and has_out,
            }
    e2_pass = all(r["ok"] for r in e2_results.values())
    results["e2_new_tasks_have_io_properties"] = e2_pass
    details["e2_new_task_properties"] = e2_results

    # --- E3: Producer-consumer chain coherence ---
    # Build a map of all OUT properties -> producing task ID
    out_registry = {}  # out_field_name -> task_id
    for node_id in graph.elements:
        if graph.get_type(node_id) != "userTask":
            continue
        props = graph.get_form_properties(node_id)
        for pid in props:
            if pid.startswith("out_"):
                out_registry[pid] = node_id

    # Build boundary event parent map for reachability through attached events
    boundary_parent_map = {}  # boundary_event_id -> attached_task_id
    for eid in graph.elements:
        elem = graph.get_element(eid)
        if elem is not None and graph.get_type(eid) == "boundaryEvent":
            attached = elem.get("attachedToRef", "")
            if attached:
                boundary_parent_map[eid] = attached

    def _is_upstream(producer_id, consumer_id):
        """Check if producer is upstream of consumer, including boundary event paths."""
        if graph.is_reachable(producer_id, consumer_id, max_depth=30):
            return True
        # Check boundary event path: producer -> (boundary event attached to producer) -> consumer
        for be_id, parent_id in boundary_parent_map.items():
            if parent_id == producer_id:
                if graph.is_reachable(be_id, consumer_id, max_depth=30):
                    return True
        return False

    # For each IN property, find corresponding OUT property upstream
    broken_chains = []
    for node_id in graph.elements:
        if graph.get_type(node_id) != "userTask":
            continue
        props = graph.get_form_properties(node_id)
        for pid in props:
            if not pid.startswith("in_"):
                continue
            # Derive the expected OUT property name
            out_name = "out_" + pid[3:]
            if out_name in out_registry:
                producer = out_registry[out_name]
                # Verify producer is upstream (reachable via reverse graph or boundary event)
                if not _is_upstream(producer, node_id):
                    broken_chains.append({
                        "consumer_task": node_id,
                        "in_property": pid,
                        "producer_task": producer,
                        "issue": "producer not upstream of consumer in flow graph",
                    })
            else:
                # IN property with no matching OUT anywhere — could be a process
                # start variable (e.g., in_revenue_target set at process start)
                # or a variable set via process-level mechanisms (not task formProperty)
                # Only flag if not a plausible start/process-level variable
                process_level_inputs = {
                    "in_supply_status", "in_sourcing_result", "in_material_grade",
                    "in_revenue_target", "in_demand_data",
                }
                if node_id != "task_salesDemand" and pid not in process_level_inputs:
                    broken_chains.append({
                        "consumer_task": node_id,
                        "in_property": pid,
                        "issue": f"no matching {out_name} found on any upstream task",
                    })

    results["e3_producer_consumer_chains_intact"] = len(broken_chains) == 0
    details["e3_broken_chains"] = broken_chains

    # --- E4: Gateway condition variables match upstream outputs ---
    e4_issues = []
    # Extract condition variables from all gateway outgoing flows
    var_pattern = re.compile(r"\$\{!?(\w+)")
    for node_id in graph.elements:
        if not graph.is_decision_gateway(node_id):
            continue
        flows = graph.outgoing_flows_with_conditions(node_id)
        for flow in flows:
            cond = flow.get("condition")
            if not cond:
                continue
            # Extract ALL variable names from the condition (handles compound expressions)
            var_names = var_pattern.findall(cond)
            if not var_names:
                continue

            predecessors = _all_predecessors_bfs(graph, node_id, max_depth=15)
            for var_name in var_names:
                # Skip EL keywords, short tokens, and counter/loop variables
                if var_name.lower() in {"true", "false", "null", "empty",
                                        "not", "and", "or", "eq", "ne",
                                        "lt", "gt", "le", "ge"} or len(var_name) <= 1:
                    continue
                # Counter/loop variables are set programmatically, not via formProperties
                if var_name in {"mixRetryCount", "retryCount", "loopCount",
                                "iterationCount", "isRevalidation"}:
                    continue
                # Find the corresponding out_ property on an upstream task
                # Convention: variable 'supplyRiskDetected' -> out_supply_risk_detected
                # or the variable might be set directly by a form property output
                found_producer = False
                for pred_id in predecessors:
                    if graph.get_type(pred_id) != "userTask":
                        continue
                    props = graph.get_form_properties(pred_id)
                    for pid in props:
                        if not pid.startswith("out_"):
                            continue
                        # Direct match: variable is already out_ prefixed (e.g., out_material_grade)
                        if pid == var_name:
                            found_producer = True
                            break
                        # Normalize: out_supply_risk_detected -> supplyRiskDetected
                        camel = _snake_to_camel(pid[4:])
                        if camel == var_name:
                            found_producer = True
                            break
                        # Also check direct match (out_supplyRiskDetected)
                        if pid == f"out_{var_name}":
                            found_producer = True
                            break
                    if found_producer:
                        break

                if not found_producer:
                    e4_issues.append({
                        "gateway": node_id,
                        "condition": cond,
                        "variable": var_name,
                        "issue": "no upstream task produces a matching output property",
                    })

    results["e4_gateway_variables_have_producers"] = len(e4_issues) == 0
    details["e4_gateway_variable_issues"] = e4_issues

    return results, details


def _all_predecessors_bfs(graph: BPMNGraph, node_id: str, max_depth: int = 5):
    """BFS backward to find all predecessor node IDs within max_depth."""
    visited = {node_id}  # seed with start node to avoid self-loops
    result = set()
    queue = deque([(node_id, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for pred in graph.predecessors(node):
            if pred not in visited:
                visited.add(pred)
                result.add(pred)
                queue.append((pred, depth + 1))
    return result


def _snake_to_camel(snake: str) -> str:
    """Convert snake_case to camelCase: 'supply_risk_detected' -> 'supplyRiskDetected'."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


# ============================================================
# SECTION F: ROLE-DATA COUPLING VALIDATION
# ============================================================

def check_role_data_coupling(root: ET.Element, discovered: dict) -> dict:
    """
    Validate role assignments are correct for the data path:
    F1: Original task assignees preserved
    F2: New task assignees match org hierarchy
    F3: Authority escalation monotonicity
    F4: Lane-task consistency
    F5: No new task missing both assignee and candidateGroups
    """
    results = {}
    details = {}
    graph = BPMNGraph(root)

    # --- F1: Original task assignees preserved ---
    f1_issues = {}
    for task_id, (expected_assignee, expected_group) in ORIGINAL_ROLE_ASSIGNMENTS.items():
        actual_assignee = graph.get_assignee(task_id)
        actual_groups = graph.get_candidate_groups(task_id)
        issues = []
        if actual_assignee != expected_assignee:
            issues.append(f"assignee: expected '{expected_assignee}', got '{actual_assignee}'")
        if expected_group and expected_group not in actual_groups:
            issues.append(f"candidateGroups: expected '{expected_group}' in '{actual_groups}'")
        if issues:
            f1_issues[task_id] = issues
    results["f1_original_assignees_preserved"] = len(f1_issues) == 0
    details["f1_assignee_issues"] = f1_issues

    # --- F2: New task assignees match org hierarchy ---
    f2_results = {}
    for role, (expected_keyword, expected_level) in EXPECTED_NEW_TASK_ROLES.items():
        node_id = discovered.get(role)
        if not node_id:
            f2_results[role] = {"found": False, "ok": False}
            continue

        assignee = graph.get_assignee(node_id)
        groups = graph.get_candidate_groups(node_id)
        assignee_lower = assignee.lower()

        if role == "escalation_task":
            # Escalation: must have executive-level candidateGroups
            has_exec = "executive" in groups.lower() or "exec" in groups.lower()
            actual_level = _get_authority_level(assignee, groups)
            f2_results[role] = {
                "node_id": node_id,
                "assignee": assignee,
                "candidateGroups": groups,
                "has_executive_group": has_exec,
                "actual_level": actual_level,
                "expected_level": "Executive",
                "ok": has_exec,
            }
        elif expected_keyword:
            keyword_lower = expected_keyword.lower()
            # Check assignee contains expected keyword
            if keyword_lower == "director":
                keyword_match = "director" in assignee_lower and "lead" not in assignee_lower
            else:
                keyword_match = keyword_lower in assignee_lower
            # Check authority level
            actual_level = _get_authority_level(assignee, groups)
            level_ok = AUTHORITY_LEVELS.get(actual_level, 0) >= AUTHORITY_LEVELS.get(expected_level, 0)
            f2_results[role] = {
                "node_id": node_id,
                "assignee": assignee,
                "candidateGroups": groups,
                "keyword_match": keyword_match,
                "actual_level": actual_level,
                "expected_level": expected_level,
                "level_ok": level_ok,
                "ok": keyword_match and level_ok,
            }
        else:
            f2_results[role] = {"node_id": node_id, "ok": True}

    f2_pass = all(r["ok"] for r in f2_results.values())
    results["f2_new_task_roles_correct"] = f2_pass
    details["f2_new_task_roles"] = f2_results

    # --- F3: Authority escalation monotonicity ---
    # Walk the exception chain and verify authority levels don't decrease
    chain_roles = ["procurement_task", "sourcing_task", "mix_adjustment_task",
                   "sales_confirmation_task", "escalation_task"]
    escalation_chain = []
    f3_violations = []
    prev_level = 0
    for role in chain_roles:
        node_id = discovered.get(role)
        if not node_id:
            continue
        assignee = graph.get_assignee(node_id)
        groups = graph.get_candidate_groups(node_id)
        level_name = _get_authority_level(assignee, groups)
        level_num = AUTHORITY_LEVELS.get(level_name, 0)
        escalation_chain.append({
            "role": role, "node_id": node_id,
            "assignee": assignee, "level": level_name, "level_num": level_num
        })
        if level_num < prev_level:
            f3_violations.append({
                "role": role,
                "node_id": node_id,
                "level": level_name,
                "issue": f"authority level decreased from {prev_level} to {level_num}",
            })
        prev_level = max(prev_level, level_num)

    results["f3_authority_escalation_monotonic"] = len(f3_violations) == 0
    details["f3_escalation_chain"] = escalation_chain
    details["f3_violations"] = f3_violations

    # --- F4: Lane-task consistency ---
    # Check that each task's lane matches its role assignment
    f4_issues = []
    lane_role_mapping = {
        "sales": ["salesLead", "salesDirector"],
        "production": ["productionLead"],
        "procurement": ["procurementLead"],
        "supplychain": ["supplyChainLead", "supplyChainDirector"],
        "escalation": ["executive"],
        "execution": ["productionLead"],  # factory execution lane
        "factory": ["productionLead"],
    }
    # Check all tasks (both original and new)
    for node_id in graph.elements:
        if graph.get_type(node_id) != "userTask":
            continue
        lane = graph.get_lane_for_node(node_id)
        if not lane:
            continue
        assignee = graph.get_assignee(node_id)
        lane_name_lower = lane["lane_name"].lower()

        # Find which lane category this belongs to
        for lane_key, valid_roles in lane_role_mapping.items():
            if lane_key in lane_name_lower:
                # Check if assignee matches one of the valid roles
                assignee_matches = any(
                    role.lower() in assignee.lower() for role in valid_roles
                ) if assignee else True  # no assignee = can't check
                if not assignee_matches and assignee:
                    f4_issues.append({
                        "task": node_id,
                        "lane": lane["lane_name"],
                        "assignee": assignee,
                        "expected_roles": valid_roles,
                        "issue": f"task in '{lane['lane_name']}' lane but assigned to '{assignee}'",
                    })
                break

    results["f4_lane_task_consistency"] = len(f4_issues) == 0
    details["f4_lane_issues"] = f4_issues

    # --- F5: No new task missing both assignee and candidateGroups ---
    f5_missing = []
    for role in ["procurement_task", "sourcing_task", "mix_adjustment_task",
                 "sales_confirmation_task", "escalation_task"]:
        node_id = discovered.get(role)
        if not node_id:
            continue
        assignee = graph.get_assignee(node_id)
        groups = graph.get_candidate_groups(node_id)
        if not assignee and not groups:
            f5_missing.append({
                "role": role,
                "node_id": node_id,
                "issue": "task has neither assignee nor candidateGroups — no one can claim it",
            })
    results["f5_all_new_tasks_have_role"] = len(f5_missing) == 0
    details["f5_missing_roles"] = f5_missing

    return results, details


def _get_authority_level(assignee: str, candidate_groups: str) -> str:
    """Determine the highest authority level from assignee and candidateGroups.
    Returns 'Unknown' if neither assignee nor candidateGroups are recognized."""
    max_level = "Unknown"
    max_num = 0

    # Check assignee
    if assignee in ROLE_AUTHORITY:
        level = ROLE_AUTHORITY[assignee]
        if AUTHORITY_LEVELS[level] > max_num:
            max_level = level
            max_num = AUTHORITY_LEVELS[level]
    elif assignee and "director" in assignee.lower():
        if AUTHORITY_LEVELS["Management"] > max_num:
            max_level = "Management"
            max_num = AUTHORITY_LEVELS["Management"]

    # Check candidate groups
    if candidate_groups:
        for group in candidate_groups.split(","):
            group = group.strip()
            if group in CANDIDATE_GROUP_AUTHORITY:
                level = CANDIDATE_GROUP_AUTHORITY[group]
                if AUTHORITY_LEVELS[level] > max_num:
                    max_level = level
                    max_num = AUTHORITY_LEVELS[level]
            elif "exec" in group.lower():
                if AUTHORITY_LEVELS["Executive"] > max_num:
                    max_level = "Executive"
                    max_num = AUTHORITY_LEVELS["Executive"]

    return max_level


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(args) -> dict:
    """Run full evaluation."""
    report = {
        "task": "LY Juice Monthly Scheduling - Workflow Adaptation",
        "approach": "Approach B — Anchor-based topological detection",
        "sections": {},
    }

    # Parse BPMN once, reuse for sections A and D
    try:
        root = ET.parse(args.bpmn).getroot()
    except ET.ParseError as e:
        print(f"FATAL: Could not parse BPMN: {e}")
        report["sections"]["A_structural"] = {"error": str(e), "all_pass": False}
        report["sections"]["D_anti_gaming"] = {"error": str(e), "all_pass": False}
        report["overall_pass"] = False
        return report

    # A. Structural checks
    print("=" * 60)
    print("A. STRUCTURAL CHECKS (topology + role verification)")
    print("    Method: Anchor-based topological detection")
    print("=" * 60)

    try:
        structural, details = check_structural(root)

        print("\n  Discovered disruption chain (new elements by topology):")
        for role, node_id in details.get("discovered_nodes", {}).items():
            status = node_id if node_id else "(not found)"
            print(f"    {role}: {status}")

        print()
        for name, passed in structural.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")

        role_checks = details.get("role_checks", {})
        print("\n  Role assignment verification (on discovered nodes):")
        for name, val in role_checks.items():
            if isinstance(val, bool):
                status = "PASS" if val else "FAIL"
                print(f"    [{status}] {name}")
            else:
                print(f"    {name}: {val}")

        all_pass = all(structural.values())
        report["sections"]["A_structural"] = {
            "checks": structural,
            "discovered_nodes": details.get("discovered_nodes", {}),
            "role_checks": role_checks,
            "missing_original_elements": details.get("missing_original_elements", []),
            "all_pass": all_pass,
            "passed": sum(1 for v in structural.values() if v),
            "total": len(structural),
        }

        print(f"\n  Result: {sum(1 for v in structural.values() if v)}/{len(structural)} passed")
        print(f"  Section A: {'PASS' if all_pass else 'FAIL'}")

    except Exception as e:
        print(f"  ERROR: {e}")
        report["sections"]["A_structural"] = {"error": str(e), "all_pass": False}

    # B. Test scenario results
    print("\n" + "=" * 60)
    print("B. TEST SCENARIO RESULTS (>=90% required)")
    print("=" * 60)

    test_check = check_test_results(args.results)
    report["sections"]["B_scenarios"] = test_check
    print(f"  Passed: {test_check['passed_scenarios']}/{test_check['total_scenarios']}")
    print(f"  Pass rate: {test_check['pass_rate']:.1%}")
    print(f"  Section B: {'PASS' if test_check['meets_threshold'] else 'FAIL'}")

    # C. Compliance coverage
    print("\n" + "=" * 60)
    print("C. COMPLIANCE COVERAGE (100% required)")
    print("=" * 60)

    compliance = check_compliance(args.structural, args.rules)
    report["sections"]["C_compliance"] = compliance
    struct_coverage = compliance.get("structural_coverage", 0)
    rules_coverage = compliance.get("rules_coverage", 0)
    compliance_pass = struct_coverage == 1.0 and rules_coverage == 1.0

    struct_count = len(REQUIRED_MODIFICATIONS) - len(compliance.get("structural_missing", []))
    rules_count = len(REQUIRED_RULES) - len(compliance.get("rules_missing", []))
    print(f"  Structural modifications: {struct_coverage:.0%} ({struct_count}/{len(REQUIRED_MODIFICATIONS)})")
    if compliance.get("structural_missing"):
        print(f"    Missing: {compliance['structural_missing']}")
    print(f"  Business rules: {rules_coverage:.0%} ({rules_count}/{len(REQUIRED_RULES)})")
    if compliance.get("rules_missing"):
        print(f"    Missing: {compliance['rules_missing']}")
    print(f"  Section C: {'PASS' if compliance_pass else 'FAIL'}")

    # D. Anti-gaming checks
    print("\n" + "=" * 60)
    print("D. ANTI-GAMING CHECKS")
    print("=" * 60)

    anti_gaming = check_anti_gaming(root)
    ag_pass = (
        anti_gaming["conditions_use_variables"]
        and anti_gaming["no_hardcoded_conditions"]
        and anti_gaming["min_conditions_met"]
        and anti_gaming["original_elements_preserved"]
        and anti_gaming.get("default_flows_on_gateways", True)
        and anti_gaming.get("new_tasks_have_documentation", True)
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

    # E. Data flow validation
    print("\n" + "=" * 60)
    print("E. DATA FLOW VALIDATION")
    print("    Method: Form property I/O chain analysis")
    print("=" * 60)

    try:
        # Reuse discovered nodes from section A
        discovered = report["sections"].get("A_structural", {}).get("discovered_nodes", {})
        if not discovered:
            # Re-discover if A wasn't run or failed early
            graph_e = BPMNGraph(root)
            discovered = discover_disruption_chain(graph_e)

        data_flow, df_details = check_data_flow(root, discovered)

        print("\n  E1: Original output form properties preserved")
        if df_details.get("e1_missing_output_properties"):
            for task_id, missing in df_details["e1_missing_output_properties"].items():
                print(f"    MISSING on {task_id}: {missing}")
        status = "PASS" if data_flow["e1_original_output_properties_preserved"] else "FAIL"
        print(f"  [{status}] e1_original_output_properties_preserved")

        print("\n  E2: New tasks have [IN]/[OUT] form properties")
        for role, info in df_details.get("e2_new_task_properties", {}).items():
            node = info.get("node_id", "?")
            ok = info.get("ok", False)
            status = "PASS" if ok else "FAIL"
            count = info.get("property_count", 0)
            print(f"    [{status}] {role} ({node}): {count} properties")
        status = "PASS" if data_flow["e2_new_tasks_have_io_properties"] else "FAIL"
        print(f"  [{status}] e2_new_tasks_have_io_properties")

        print("\n  E3: Producer-consumer chain coherence")
        if df_details.get("e3_broken_chains"):
            for bc in df_details["e3_broken_chains"]:
                print(f"    BROKEN: {bc['consumer_task']}.{bc['in_property']} — {bc['issue']}")
        status = "PASS" if data_flow["e3_producer_consumer_chains_intact"] else "FAIL"
        print(f"  [{status}] e3_producer_consumer_chains_intact")

        print("\n  E4: Gateway condition variables have upstream producers")
        if df_details.get("e4_gateway_variable_issues"):
            for issue in df_details["e4_gateway_variable_issues"]:
                print(f"    ISSUE: {issue['gateway']} uses ${{{issue['variable']}}} — {issue['issue']}")
        status = "PASS" if data_flow["e4_gateway_variables_have_producers"] else "FAIL"
        print(f"  [{status}] e4_gateway_variables_have_producers")

        e_pass = all(data_flow.values())
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

    # F. Role-data coupling validation
    print("\n" + "=" * 60)
    print("F. ROLE-DATA COUPLING VALIDATION")
    print("    Method: Assignee + candidateGroup + authority level analysis")
    print("=" * 60)

    try:
        role_coupling, rc_details = check_role_data_coupling(root, discovered)

        print("\n  F1: Original task assignees preserved")
        if rc_details.get("f1_assignee_issues"):
            for task_id, issues in rc_details["f1_assignee_issues"].items():
                for issue in issues:
                    print(f"    ISSUE: {task_id} — {issue}")
        status = "PASS" if role_coupling["f1_original_assignees_preserved"] else "FAIL"
        print(f"  [{status}] f1_original_assignees_preserved")

        print("\n  F2: New task roles match org hierarchy")
        for role, info in rc_details.get("f2_new_task_roles", {}).items():
            node = info.get("node_id", "?")
            ok = info.get("ok", False)
            status = "PASS" if ok else "FAIL"
            assignee = info.get("assignee", "")
            level = info.get("actual_level", "")
            print(f"    [{status}] {role} ({node}): assignee='{assignee}', level={level}")
        status = "PASS" if role_coupling["f2_new_task_roles_correct"] else "FAIL"
        print(f"  [{status}] f2_new_task_roles_correct")

        print("\n  F3: Authority escalation monotonicity")
        for entry in rc_details.get("f3_escalation_chain", []):
            print(f"    {entry['role']}: {entry['assignee']} -> {entry['level']} ({entry['level_num']})")
        if rc_details.get("f3_violations"):
            for v in rc_details["f3_violations"]:
                print(f"    VIOLATION: {v['role']} — {v['issue']}")
        status = "PASS" if role_coupling["f3_authority_escalation_monotonic"] else "FAIL"
        print(f"  [{status}] f3_authority_escalation_monotonic")

        print("\n  F4: Lane-task consistency")
        if rc_details.get("f4_lane_issues"):
            for issue in rc_details["f4_lane_issues"]:
                print(f"    ISSUE: {issue['issue']}")
        status = "PASS" if role_coupling["f4_lane_task_consistency"] else "FAIL"
        print(f"  [{status}] f4_lane_task_consistency")

        print("\n  F5: All new tasks have assignee or candidateGroups")
        if rc_details.get("f5_missing_roles"):
            for m in rc_details["f5_missing_roles"]:
                print(f"    MISSING: {m['role']} ({m['node_id']}) — {m['issue']}")
        status = "PASS" if role_coupling["f5_all_new_tasks_have_role"] else "FAIL"
        print(f"  [{status}] f5_all_new_tasks_have_role")

        f_pass = all(role_coupling.values())
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
    print("FINAL SCORE")
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
    print(f"\n  OVERALL: {'PASS' if all_sections_pass else 'FAIL'}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate supply chain workflow adaptation task "
                    "(Approach B: anchor-based topological detection)"
    )
    parser.add_argument("--bpmn", required=True,
                        help="Path to the agent's modified BPMN XML")
    parser.add_argument("--structural", required=True,
                        help="Path to structural_changes.json (agent-produced)")
    parser.add_argument("--rules", required=True,
                        help="Path to business_rules_compliance.json (agent-produced)")
    parser.add_argument("--results", required=True,
                        help="Path to test_results.json (from run_tests.py)")
    parser.add_argument("--output", default="evaluation_report.json",
                        help="Output report path")

    args = parser.parse_args()
    report = evaluate(args)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}")

    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
