#!/usr/bin/env python3
"""
L3 Evaluator for Case 2: Z Global M&B Category Temporary Governance
Restructuring (Hardened).

Self-contained evaluator (no imports from Case 1). Validates a solver's
modified BPMN 2.0 XML against:

  Section A  Structural topology (30+ checks including anchor preservation,
             multi-factor gateway, timer boundary, bounded loop, sub-process
             presence, closed-loop, parallel review, peer-owner lane whitelist,
             coordination lane, unified decision node, terminal escalation,
             default flows, documentation)

  Section B  Runtime scenario pass rate (60 scenarios, >=90%)

  Section C  Compliance coverage (18 modifications + 22 rules including rule_8a)
             plus cross-validation (timer duration, loop counter, multi-factor
             count, rationale presence)

  Section D  Anti-gaming (8 forbidden-pattern checks)

  Section E  Solution fingerprint match (A/B/C) or all-hard-constraints path

CLI:
  python evaluate_L3.py --bpmn modified_process.bpmn20.xml \\
                        --structural structural_changes.json \\
                        --rules business_rules_compliance.json \\
                        --results test_results.json \\
                        [--output evaluation_report_L3.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from lxml import etree as LET
    _USE_LXML = True
except ImportError:
    import xml.etree.ElementTree as LET  # type: ignore
    _USE_LXML = False


# ============================================================================
# NAMESPACES AND CONSTANTS
# ============================================================================

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
FLOWABLE_NS = "http://flowable.org/bpmn"

NS = {
    "bpmn": BPMN_NS,
    "flowable": FLOWABLE_NS,
}

# Anchored original elements (must be preserved with exact IDs)
ORIGINAL_ELEMENTS: Set[str] = {
    "startEvent",
    "team_drafts_initial_plan",
    "team_executes_plan",
    "logistics_lane",
    "logistics_capacity_confirmation",
}

# Explicitly anchored logistics elements (rule 8 / rule 8a)
LOGISTICS_ANCHORS: Set[str] = {
    "logistics_lane",
    "logistics_capacity_confirmation",
}

# Process definition keys
ORIGINAL_PROCESS_KEY = "zMBCategoryGovernance_original"
MODIFIED_PROCESS_KEY = "zMBCategoryGovernance_modified_L3"
TERMINAL_ESCALATION_END = "escalation_final_end"
MAIN_END = "main_end"

# Required modifications and rules (compliance coverage)
REQUIRED_MODIFICATIONS: List[str] = [f"modification_{i}" for i in range(1, 19)]
REQUIRED_RULES: List[str] = (
    [f"rule_{i}" for i in range(1, 6)]
    + ["rule_6", "rule_6a"]
    + ["rule_7", "rule_8", "rule_8a"]
    + [f"rule_{i}" for i in range(9, 15)]
    + ["rule_14a"]
    + ["rule_15", "rule_16", "rule_16a"]
    + [f"rule_{i}" for i in range(17, 23)]
    + [f"rule_{i}" for i in range(23, 27)]  # L3 structural-rigor rules
    + [f"rule_{i}" for i in range(27, 31)]  # Stakeholder-artifact context-bound rules
)

# Factor keywords used by the multi-factor gateway (Rule 6)
FACTOR_KEYWORDS: Set[str] = {
    "inventory", "margin", "aov", "sales", "brand", "risk", "platform",
}

# Counter variable patterns (Rule 16)
COUNTER_PATTERN = re.compile(r"count|retry|attempt|iteration|loop", re.IGNORECASE)

# Peer-owner whitelist terms (Rule 10)
PEER_WHITELIST_TERMS: Set[str] = {
    "ad_ratio", "ab_test", "review", "new_merchant",
    "standardized", "free_trial", "quota", "promotion",
}

# Roles (from org_hierarchy.json)
ROLE_SENIOR_LEAD = "zSeniorLead"
ROLE_INTERIM_CHAIR = "interimCommitteeChair"
ROLE_TEAM_LEAD = "mbTeamLead"
ROLE_MERCHANT_OPS = "mbMerchantOpsLead"
ROLE_PEER_BEAUTY = "peerCategoryOwner_Beauty"
ROLE_PEER_APPAREL = "peerCategoryOwner_Apparel"
ROLE_COMPLIANCE = "complianceOfficer"
ROLE_LOGISTICS = "logisticsCoordinator"

# EL variable extraction keywords (excluded from variable names)
EL_KEYWORDS: Set[str] = {
    "true", "false", "null", "empty", "not", "and", "or",
    "eq", "ne", "lt", "gt", "le", "ge", "div", "mod",
    "instanceof",
}
VAR_TOKEN_RE = re.compile(r"[a-zA-Z_]\w*")

# Rationale keyword pattern (Rule 17)
RATIONALE_RE = re.compile(r"rationale|justification|reason", re.IGNORECASE)


# ============================================================================
# BPMN GRAPH ABSTRACTION
# ============================================================================


def _qn(tag: str) -> str:
    """Return the qualified (namespaced) name for a BPMN tag."""
    return f"{{{BPMN_NS}}}{tag}"


def _fqn(tag: str) -> str:
    """Return the qualified (namespaced) name for a Flowable tag."""
    return f"{{{FLOWABLE_NS}}}{tag}"


class BPMNGraph:
    """Minimal BPMN graph abstraction over parsed XML.

    Supports node lookup, successors/predecessors by sequenceFlow, element
    type, assignee, candidate groups, formProperty extraction, and lane
    membership.
    """

    def __init__(self, root):
        self.root = root
        self.elements: Dict[str, object] = {}
        self.elem_types: Dict[str, str] = {}
        self.out_flows: Dict[str, List[Dict]] = defaultdict(list)
        self.in_flows: Dict[str, List[Dict]] = defaultdict(list)
        self.lanes: Dict[str, Dict] = {}  # lane_id -> {"name": str, "nodes": set}
        self.end_events: Set[str] = set()
        self.start_events: Set[str] = set()
        self.boundary_events: Dict[str, str] = {}  # be_id -> attached_to
        self.subprocesses: Set[str] = set()
        self.subprocess_contents: Dict[str, Set[str]] = {}
        self._scan()

    def _local(self, tag) -> str:
        """Strip namespace from a tag. Returns "" for non-string tags
        (e.g. lxml.etree.Comment / ProcessingInstruction which carry a
        callable as their ``tag`` attribute)."""
        if not isinstance(tag, str):
            return ""
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    def _scan(self):
        # Find process element
        proc = None
        for elem in self.root.iter():
            if self._local(elem.tag) == "process":
                proc = elem
                break
        if proc is None:
            return
        self.process_element = proc

        # Collect flow nodes (depth-first, including subprocess contents)
        for elem in proc.iter():
            local = self._local(elem.tag)
            eid = elem.get("id")
            if eid and local in (
                "startEvent", "endEvent", "userTask", "serviceTask",
                "scriptTask", "task", "exclusiveGateway", "parallelGateway",
                "inclusiveGateway", "eventBasedGateway", "subProcess",
                "boundaryEvent", "intermediateCatchEvent", "intermediateThrowEvent",
                "callActivity",
            ):
                self.elements[eid] = elem
                self.elem_types[eid] = local
                if local == "endEvent":
                    self.end_events.add(eid)
                elif local == "startEvent":
                    self.start_events.add(eid)
                elif local == "boundaryEvent":
                    attached = elem.get("attachedToRef", "")
                    if attached:
                        self.boundary_events[eid] = attached
                elif local == "subProcess":
                    self.subprocesses.add(eid)
                    contents = set()
                    for sub_elem in elem.iter():
                        sub_local = self._local(sub_elem.tag)
                        sub_id = sub_elem.get("id")
                        if sub_id and sub_id != eid and sub_local in (
                            "userTask", "startEvent", "endEvent",
                            "exclusiveGateway", "parallelGateway",
                            "sequenceFlow", "serviceTask", "task",
                        ):
                            contents.add(sub_id)
                    self.subprocess_contents[eid] = contents

        # Collect sequence flows
        for elem in proc.iter():
            if self._local(elem.tag) == "sequenceFlow":
                fid = elem.get("id")
                src = elem.get("sourceRef", "")
                tgt = elem.get("targetRef", "")
                cond = ""
                for child in elem:
                    if self._local(child.tag) == "conditionExpression":
                        cond = (child.text or "").strip()
                        break
                entry = {
                    "flow_id": fid,
                    "source": src,
                    "target": tgt,
                    "condition": cond,
                }
                self.out_flows[src].append(entry)
                self.in_flows[tgt].append(entry)

        # Collect lanes
        for elem in proc.iter():
            if self._local(elem.tag) == "lane":
                lid = elem.get("id", "")
                lname = elem.get("name", lid)
                nodes = set()
                for child in elem:
                    if self._local(child.tag) == "flowNodeRef":
                        text = (child.text or "").strip()
                        if text:
                            nodes.add(text)
                self.lanes[lid] = {"name": lname, "nodes": nodes}

    def get_type(self, node_id: str) -> str:
        return self.elem_types.get(node_id, "")

    def get_element(self, node_id: str):
        return self.elements.get(node_id)

    def successors(self, node_id: str) -> List[str]:
        return [f["target"] for f in self.out_flows.get(node_id, [])]

    def predecessors(self, node_id: str) -> List[str]:
        return [f["source"] for f in self.in_flows.get(node_id, [])]

    def outgoing_flows(self, node_id: str) -> List[Dict]:
        return self.out_flows.get(node_id, [])

    def is_decision_gateway(self, node_id: str) -> bool:
        return self.get_type(node_id) in ("exclusiveGateway", "inclusiveGateway")

    def is_parallel_gateway(self, node_id: str) -> bool:
        return self.get_type(node_id) == "parallelGateway"

    def get_assignee(self, node_id: str) -> str:
        elem = self.get_element(node_id)
        if elem is None:
            return ""
        raw = elem.get(f"{{{FLOWABLE_NS}}}assignee", "") or elem.get("flowable:assignee", "")
        if raw.startswith("${") and raw.endswith("}"):
            return raw[2:-1].strip()
        return raw

    def get_candidate_groups(self, node_id: str) -> str:
        elem = self.get_element(node_id)
        if elem is None:
            return ""
        return elem.get(f"{{{FLOWABLE_NS}}}candidateGroups", "") or elem.get("flowable:candidateGroups", "")

    def get_task_name(self, node_id: str) -> str:
        elem = self.get_element(node_id)
        if elem is None:
            return ""
        return elem.get("name", "")

    def get_documentation(self, node_id: str) -> str:
        elem = self.get_element(node_id)
        if elem is None:
            return ""
        for child in elem:
            if self._local(child.tag) == "documentation":
                return (child.text or "").strip()
        return ""

    def get_form_properties(self, node_id: str) -> Dict[str, Dict]:
        elem = self.get_element(node_id)
        if elem is None:
            return {}
        props: Dict[str, Dict] = {}
        for child in elem.iter():
            if self._local(child.tag) == "formProperty":
                pid = child.get("id", "")
                if pid:
                    props[pid] = {
                        "name": child.get("name", ""),
                        "type": child.get("type", ""),
                        "required": child.get("required", ""),
                        "writable": child.get("writable", ""),
                        "readable": child.get("readable", ""),
                    }
        return props

    def get_lane_for_node(self, node_id: str) -> Optional[Dict]:
        for lid, ldata in self.lanes.items():
            if node_id in ldata["nodes"]:
                return {"lane_id": lid, "lane_name": ldata["name"]}
        return None

    def is_reachable(self, src: str, tgt: str, max_depth: int = 50) -> bool:
        if src == tgt:
            return True
        visited = {src}
        queue: deque = deque([(src, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth > max_depth:
                continue
            for succ in self.successors(node):
                if succ == tgt:
                    return True
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, depth + 1))
            # Also follow boundary events attached to this node
            for be, attached in self.boundary_events.items():
                if attached == node and be not in visited:
                    visited.add(be)
                    if be == tgt:
                        return True
                    queue.append((be, depth + 1))
        return False

    def find_boundary_event_attached_to(self, node_id: str) -> Optional[str]:
        for be, attached in self.boundary_events.items():
            if attached == node_id:
                return be
        return None

    def process_key(self) -> str:
        if self.process_element is None:
            return ""
        return self.process_element.get("id", "")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def extract_vars_from_condition(cond: str) -> Set[str]:
    """Extract variable names from a BPMN EL condition expression."""
    if not cond:
        return set()
    inner = cond.strip()
    if inner.startswith("${") and inner.endswith("}"):
        inner = inner[2:-1]
    # Remove string literals
    inner = re.sub(r"'[^']*'", "", inner)
    inner = re.sub(r'"[^"]*"', "", inner)
    tokens = VAR_TOKEN_RE.findall(inner)
    return {t for t in tokens if t not in EL_KEYWORDS and len(t) > 1 and not t.isdigit()}


def extract_vars_from_gateway(graph: BPMNGraph, gw_id: str) -> Set[str]:
    """Extract variables from all outgoing conditions of a gateway."""
    v: Set[str] = set()
    for f in graph.outgoing_flows(gw_id):
        cond = f.get("condition", "")
        v |= extract_vars_from_condition(cond)
    return v


def gateway_conditions_text(graph: BPMNGraph, gw_id: str) -> str:
    """Join all outgoing conditions of a gateway."""
    parts = [f.get("condition", "") for f in graph.outgoing_flows(gw_id)]
    return " ".join(p for p in parts if p)


# ============================================================================
# TOPOLOGY DISCOVERY
# ============================================================================


def discover_nodes(graph: BPMNGraph) -> Dict[str, Optional[str]]:
    """Locate the key Solution A nodes by name/ID matching + topology hints.

    Returns a dict keyed by role name -> node_id (or None).
    """
    discovered: Dict[str, Optional[str]] = {
        "joint_plan_task": None,
        "multi_factor_gateway": None,
        "peer_available_gateway": None,
        "compliance_crisis_gateway": None,
        "compliance_subprocess": None,
        "unified_decision_task": None,
        "timer_boundary_event": None,
        "senior_lead_escalation_task": None,
        "escalation_tier_gateway": None,
        "joint_plan_retry_gateway": None,
        "decision_outcome_gateway": None,
        "parallel_review_split": None,
        "parallel_review_join": None,
        "merchant_coordination_task": None,
        "campaign_coordination_task": None,
        "emergency_delist_task": None,
        "brand_risk_reassessment_task": None,
        "peer_readiness_check_task": None,
        "compliance_signal_check_task": None,
    }

    # Prefer ID-based match (reference Solution A uses canonical IDs).
    # Otherwise fall back to name-based heuristics.
    id_to_role = {
        "team_joint_mature_plan_preparation": "joint_plan_task",
        "gw_multi_factor": "multi_factor_gateway",
        "gw_peer_available": "peer_available_gateway",
        "gw_compliance_crisis": "compliance_crisis_gateway",
        "compliance_subprocess": "compliance_subprocess",
        "temporary_unified_decision": "unified_decision_task",
        "timer_48h_decision": "timer_boundary_event",
        "senior_lead_expedited_decision": "senior_lead_escalation_task",
        "gw_escalation_tier": "escalation_tier_gateway",
        "gw_joint_plan_retry": "joint_plan_retry_gateway",
        "gw_decision_outcome": "decision_outcome_gateway",
        "gw_parallel_review_split": "parallel_review_split",
        "gw_parallel_review_join": "parallel_review_join",
        "merchant_readiness_coordination": "merchant_coordination_task",
        "campaign_cadence_coordination": "campaign_coordination_task",
        "emergency_delist_task": "emergency_delist_task",
        "brand_risk_reassessment_task": "brand_risk_reassessment_task",
        "peer_readiness_check": "peer_readiness_check_task",
        "compliance_signal_check": "compliance_signal_check_task",
    }
    for eid, role in id_to_role.items():
        if eid in graph.elements:
            discovered[role] = eid

    # Heuristic fallbacks for unmapped roles
    def _find_by_name_fragment(fragment: str, type_filter: Optional[str] = None) -> Optional[str]:
        fragment = fragment.lower()
        for eid, elem in graph.elements.items():
            if type_filter and graph.get_type(eid) != type_filter:
                continue
            name = graph.get_task_name(eid).lower()
            if fragment in name or fragment in eid.lower():
                return eid
        return None

    if discovered["joint_plan_task"] is None:
        discovered["joint_plan_task"] = _find_by_name_fragment(
            "joint_mature_plan", "userTask"
        ) or _find_by_name_fragment("mature_plan", "userTask")

    if discovered["unified_decision_task"] is None:
        discovered["unified_decision_task"] = _find_by_name_fragment(
            "unified_decision", "userTask"
        ) or _find_by_name_fragment("interim_committee", "userTask")

    if discovered["multi_factor_gateway"] is None:
        # Try to find a gateway whose conditions reference multiple factors
        for eid in graph.elements:
            if not graph.is_decision_gateway(eid):
                continue
            text = gateway_conditions_text(graph, eid).lower()
            factor_hits = sum(1 for f in FACTOR_KEYWORDS if f in text)
            if factor_hits >= 4:
                discovered["multi_factor_gateway"] = eid
                break

    if discovered["timer_boundary_event"] is None:
        for be_id in graph.boundary_events:
            elem = graph.get_element(be_id)
            if elem is None:
                continue
            for child in elem:
                if graph._local(child.tag) == "timerEventDefinition":
                    discovered["timer_boundary_event"] = be_id
                    break
            if discovered["timer_boundary_event"]:
                break

    if discovered["compliance_subprocess"] is None and graph.subprocesses:
        discovered["compliance_subprocess"] = sorted(graph.subprocesses)[0]

    # ------------------------------------------------------------------
    # Topology-based fallbacks for the remaining roles.
    #
    # Rationale: solvers that choose non-canonical IDs (e.g. the codex
    # run named `gw_peer_available` as `peer_availability_gateway`)
    # otherwise cause every downstream Section A / E check that reads
    # `discovered[role]` to short-circuit on `None`, producing spurious
    # failures even for structurally and semantically correct BPMN.
    # We identify each role by *what it does* (variables it switches
    # on, its in/out degree, or assignee / name fragments) instead of
    # only its canonical ID.
    # ------------------------------------------------------------------

    def _claimed_gateways() -> Set[str]:
        return {
            v for v in discovered.values()
            if isinstance(v, str) and graph.is_decision_gateway(v)
        }

    def _find_gateway_with_var(
        var_keywords: List[str],
        require_loop_to: Optional[str] = None,
        exclude_var_keywords: Optional[List[str]] = None,
    ) -> Optional[str]:
        keys = [k.lower() for k in var_keywords]
        excl = [k.lower() for k in (exclude_var_keywords or [])]
        claimed = _claimed_gateways()
        for eid in graph.elements:
            if eid in claimed or not graph.is_decision_gateway(eid):
                continue
            vars_ = {v.lower() for v in extract_vars_from_gateway(graph, eid)}
            if not any(any(k in v for k in keys) for v in vars_):
                continue
            if excl and any(any(k in v for k in excl) for v in vars_):
                continue
            if require_loop_to is not None and not any(
                graph.is_reachable(f.get("target", ""), require_loop_to, max_depth=15)
                for f in graph.outgoing_flows(eid)
            ):
                continue
            return eid
        return None

    def _find_parallel_gateway(splitting: bool) -> Optional[str]:
        claimed = {
            v for v in (
                discovered.get("parallel_review_split"),
                discovered.get("parallel_review_join"),
            )
            if isinstance(v, str)
        }
        for eid in graph.elements:
            if eid in claimed or not graph.is_parallel_gateway(eid):
                continue
            in_degree = len(graph.in_flows.get(eid, []))
            out_degree = len(graph.out_flows.get(eid, []))
            if splitting and out_degree >= 2 and in_degree <= 1:
                return eid
            if not splitting and in_degree >= 2 and out_degree <= 1:
                return eid
        return None

    def _find_task_by_all_fragments(
        fragments: List[str],
        type_filter: str = "userTask",
        exclude: Optional[Set[str]] = None,
    ) -> Optional[str]:
        exclude = exclude or set()
        frags = [f.lower() for f in fragments]
        for eid in graph.elements:
            if eid in exclude or graph.get_type(eid) != type_filter:
                continue
            blob = (
                graph.get_task_name(eid) + " "
                + eid + " "
                + graph.get_documentation(eid)
            ).lower()
            if all(f in blob for f in frags):
                return eid
        return None

    def _find_task_by_assignee(
        needle: str,
        exclude: Optional[Set[str]] = None,
    ) -> Optional[str]:
        exclude = exclude or set()
        needle_l = needle.lower()
        for eid in graph.elements:
            if eid in exclude or graph.get_type(eid) != "userTask":
                continue
            if needle_l in graph.get_assignee(eid).lower():
                return eid
        return None

    # ---- Gateway fallbacks (by switched-on variables) ----
    if discovered["peer_available_gateway"] is None:
        discovered["peer_available_gateway"] = _find_gateway_with_var(["peer"])

    if discovered["compliance_crisis_gateway"] is None:
        discovered["compliance_crisis_gateway"] = _find_gateway_with_var(
            ["compliance", "crisis"]
        )

    if discovered["escalation_tier_gateway"] is None:
        discovered["escalation_tier_gateway"] = _find_gateway_with_var(
            ["tier", "escalat"]
        )

    if discovered["joint_plan_retry_gateway"] is None:
        jp = discovered.get("joint_plan_task")
        # A true retry gateway references ONLY counter variables in its
        # conditions; it must not conflate retry bookkeeping with decision-
        # outcome routing (Rule 16a). We therefore exclude gateways whose
        # conditions mention the decision-outcome variable.
        retry_counter_kws = ["count", "retry", "attempt", "iteration", "loop"]
        retry_excluded_kws = [
            "out_unified_decision", "unified_decision",
            "out_expedited_decision", "expedited_decision",
            "decisionoutcome",
        ]
        if jp:
            discovered["joint_plan_retry_gateway"] = _find_gateway_with_var(
                retry_counter_kws,
                require_loop_to=jp,
                exclude_var_keywords=retry_excluded_kws,
            )
        if discovered["joint_plan_retry_gateway"] is None:
            discovered["joint_plan_retry_gateway"] = _find_gateway_with_var(
                retry_counter_kws,
                exclude_var_keywords=retry_excluded_kws,
            )

    if discovered["decision_outcome_gateway"] is None:
        discovered["decision_outcome_gateway"] = _find_gateway_with_var(
            ["decision", "outcome", "approved", "rejected"]
        )

    # ---- Parallel gateway fallbacks (by degree) ----
    if discovered["parallel_review_split"] is None:
        discovered["parallel_review_split"] = _find_parallel_gateway(splitting=True)
    if discovered["parallel_review_join"] is None:
        discovered["parallel_review_join"] = _find_parallel_gateway(splitting=False)

    # ---- User task fallbacks (assignee + name fragments) ----
    claimed_tasks: Set[str] = {
        v for v in discovered.values()
        if isinstance(v, str) and graph.get_type(v) == "userTask"
    }

    if discovered["senior_lead_escalation_task"] is None:
        discovered["senior_lead_escalation_task"] = (
            _find_task_by_assignee("seniorlead", exclude=claimed_tasks)
            or _find_task_by_assignee("senior_lead", exclude=claimed_tasks)
            or _find_task_by_all_fragments(["senior", "expedited"], exclude=claimed_tasks)
            or _find_task_by_all_fragments(["senior", "escalation"], exclude=claimed_tasks)
            or _find_task_by_all_fragments(["senior", "decision"], exclude=claimed_tasks)
        )
        if discovered["senior_lead_escalation_task"]:
            claimed_tasks.add(discovered["senior_lead_escalation_task"])

    if discovered["merchant_coordination_task"] is None:
        discovered["merchant_coordination_task"] = (
            _find_task_by_all_fragments(["merchant", "readiness"], exclude=claimed_tasks)
            or _find_task_by_all_fragments(["merchant", "coordination"], exclude=claimed_tasks)
        )
        if discovered["merchant_coordination_task"]:
            claimed_tasks.add(discovered["merchant_coordination_task"])

    if discovered["campaign_coordination_task"] is None:
        discovered["campaign_coordination_task"] = (
            _find_task_by_all_fragments(["campaign", "cadence"], exclude=claimed_tasks)
            or _find_task_by_all_fragments(["campaign", "coordination"], exclude=claimed_tasks)
        )
        if discovered["campaign_coordination_task"]:
            claimed_tasks.add(discovered["campaign_coordination_task"])

    if discovered["emergency_delist_task"] is None:
        discovered["emergency_delist_task"] = (
            _find_task_by_all_fragments(["emergency", "delist"], exclude=claimed_tasks)
            or _find_task_by_all_fragments(["delist"], exclude=claimed_tasks)
        )
        if discovered["emergency_delist_task"]:
            claimed_tasks.add(discovered["emergency_delist_task"])

    if discovered["brand_risk_reassessment_task"] is None:
        discovered["brand_risk_reassessment_task"] = (
            _find_task_by_all_fragments(["brand", "risk", "reassess"], exclude=claimed_tasks)
            or _find_task_by_all_fragments(["brand", "risk"], exclude=claimed_tasks)
        )
        if discovered["brand_risk_reassessment_task"]:
            claimed_tasks.add(discovered["brand_risk_reassessment_task"])

    if discovered["peer_readiness_check_task"] is None:
        discovered["peer_readiness_check_task"] = (
            _find_task_by_all_fragments(["peer", "readiness"], exclude=claimed_tasks)
            or _find_task_by_all_fragments(["peer", "check"], exclude=claimed_tasks)
            or _find_task_by_all_fragments(["peer", "available"], exclude=claimed_tasks)
        )
        if discovered["peer_readiness_check_task"]:
            claimed_tasks.add(discovered["peer_readiness_check_task"])

    if discovered["compliance_signal_check_task"] is None:
        discovered["compliance_signal_check_task"] = (
            _find_task_by_all_fragments(["compliance", "signal"], exclude=claimed_tasks)
            or _find_task_by_all_fragments(["compliance", "check"], exclude=claimed_tasks)
        )
        if discovered["compliance_signal_check_task"]:
            claimed_tasks.add(discovered["compliance_signal_check_task"])

    return discovered


# ============================================================================
# SECTION A - STRUCTURAL CHECKS
# ============================================================================


def _reaches_within(
    graph: "BPMNGraph",
    start: str,
    target: str,
    max_depth: int = 2,
) -> bool:
    """Shallow BFS: is `target` reachable from `start` within `max_depth` hops?"""
    if start == target:
        return True
    visited = {start}
    queue: deque = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for succ in graph.successors(node):
            if succ == target:
                return True
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, depth + 1))
    return False


def _nearest_parallel_split_ancestor(
    graph: "BPMNGraph",
    start: str,
    max_depth: int = 25,
) -> Optional[str]:
    """Walk backward from `start`; return the first parallelGateway encountered
    that acts as a split (1 incoming, >=2 outgoing). None if not found.
    Used by a45 to verify each join input traces to a matching split."""
    visited: Set[str] = set()
    queue: deque = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if node in visited or depth > max_depth:
            continue
        visited.add(node)
        if graph.is_parallel_gateway(node):
            inc = len(graph.in_flows.get(node, []))
            out = len(graph.outgoing_flows(node))
            if inc <= 1 and out >= 2:
                return node
        for pred in graph.predecessors(node):
            if pred not in visited:
                queue.append((pred, depth + 1))
    return None


def _count_user_tasks_on_path(
    graph: "BPMNGraph",
    start: str,
    target: str,
    initial_visited: Set[str],
    max_depth: int = 30,
) -> int:
    """Count the minimum number of userTasks on any start->target path,
    treating initial_visited as blocked. Returns 0 if no path OR no userTask
    on any path, the min count otherwise (BFS returns first-found depth)."""
    if start == target:
        return 0
    visited = set(initial_visited)
    queue: deque = deque([(start, 0)])  # (node, user_tasks_so_far)
    best = -1
    while queue:
        node, count = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        step_count = count + (1 if graph.get_type(node) == "userTask" else 0)
        if node == target:
            if best < 0 or step_count < best:
                best = step_count
            continue
        if len(visited) > max_depth * 2:
            continue
        for succ in graph.successors(node):
            if succ not in visited:
                queue.append((succ, step_count))
    return best if best >= 0 else 0


def _find_non_terminating_paths(
    graph: "BPMNGraph",
    start: str,
    terminal_ids: Set[str],
    max_depth: int = 40,
) -> List[str]:
    """Walk forward from start; return list of reachable nodes that have no
    outgoing flow and are NOT in terminal_ids (dead ends). Also flags nodes
    whose outgoing all lead to cycles without ever reaching a terminal."""
    visited: Set[str] = set()
    dead: List[str] = []
    queue: deque = deque([(start, 0)])
    # Forward-reach terminal set
    terminal_reach: Dict[str, bool] = {}

    def reaches_terminal(node: str, depth: int, path: Set[str]) -> bool:
        if node in terminal_ids:
            return True
        if node in terminal_reach:
            return terminal_reach[node]
        if depth > max_depth or node in path:
            return False
        path = path | {node}
        succs = graph.successors(node)
        if not succs:
            terminal_reach[node] = False
            return False
        ok = any(reaches_terminal(s, depth + 1, path) for s in succs)
        terminal_reach[node] = ok
        return ok

    while queue:
        node, depth = queue.popleft()
        if node in visited or depth > max_depth:
            continue
        visited.add(node)
        if node in terminal_ids:
            continue
        if not reaches_terminal(node, 0, set()):
            # This node, once reached, cannot get out to a terminal
            dead.append(node)
            continue
        for succ in graph.successors(node):
            if succ not in visited:
                queue.append((succ, depth + 1))
    return dead


def _reaches_avoiding(
    graph: "BPMNGraph",
    start: str,
    target: str,
    initial_visited: Set[str],
    max_depth: int = 50,
) -> bool:
    """BFS-reach from start to target while treating initial_visited as blocked.

    Used by a37 to verify each split-branch reaches the matching join without
    looping back through the split gateway itself.
    """
    if start == target:
        return True
    visited = set(initial_visited)
    visited.add(start)
    queue: deque = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth > max_depth:
            continue
        for succ in graph.successors(node):
            if succ == target:
                return True
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, depth + 1))
    return False


def check_structural(graph: BPMNGraph) -> Tuple[Dict, Dict]:
    """Section A: 30+ structural checks."""
    results: Dict[str, object] = {}
    details: Dict[str, object] = {}

    discovered = discover_nodes(graph)
    details["discovered_nodes"] = discovered

    # A1: Process key is correct (canonical key, or a reference variant suffix _A/_B/_C)
    pk = graph.process_key()
    results["a01_process_key_fixed"] = (
        pk == MODIFIED_PROCESS_KEY
        or pk in (f"{MODIFIED_PROCESS_KEY}_A", f"{MODIFIED_PROCESS_KEY}_B", f"{MODIFIED_PROCESS_KEY}_C")
    )

    # A2: Anchors preserved
    results["a02_anchor_team_drafts_initial_plan"] = "team_drafts_initial_plan" in graph.elements
    results["a03_anchor_team_executes_plan"] = "team_executes_plan" in graph.elements
    results["a04_anchor_logistics_lane"] = "logistics_lane" in graph.lanes
    results["a05_anchor_logistics_capacity_confirmation"] = (
        "logistics_capacity_confirmation" in graph.elements
    )

    # A6: Main end event distinct from terminal escalation
    results["a06_main_end_event"] = MAIN_END in graph.end_events
    results["a07_terminal_escalation_end_event"] = TERMINAL_ESCALATION_END in graph.end_events

    # A8: Joint plan preparation task exists
    jp = discovered.get("joint_plan_task")
    results["a08_joint_plan_task_exists"] = jp is not None and graph.get_type(jp) == "userTask"

    # A9: Multi-factor gateway with >=4 factors
    mfg = discovered.get("multi_factor_gateway")
    factor_hits = 0
    if mfg:
        text = gateway_conditions_text(graph, mfg).lower()
        factor_hits = sum(1 for f in FACTOR_KEYWORDS if f in text)
    details["a09_factor_hits"] = factor_hits
    results["a09_multi_factor_gateway_has_4plus_factors"] = factor_hits >= 4

    # A10: Temporary unified decision task exists
    uid = discovered.get("unified_decision_task")
    results["a10_unified_decision_task_exists"] = uid is not None and graph.get_type(uid) == "userTask"

    # A11: Timer boundary event with PT48H attached to unified decision (or equivalent)
    tb = discovered.get("timer_boundary_event")
    timer_ok = False
    timer_duration = ""
    if tb:
        elem = graph.get_element(tb)
        if elem is not None:
            for child in elem.iter():
                if graph._local(child.tag) == "timeDuration":
                    timer_duration = (child.text or "").strip()
                    if timer_duration.upper() == "PT48H":
                        timer_ok = True
                        break
    results["a11_timer_boundary_48h"] = timer_ok
    details["a11_timer_duration"] = timer_duration

    # A12: Timer attached to unified decision task (not some random node)
    timer_attached_to = graph.boundary_events.get(tb, "") if tb else ""
    results["a12_timer_attached_to_unified_decision"] = (
        timer_attached_to == uid and uid is not None
    )

    # A13: Compliance sub-process exists with >=2 user tasks
    csp = discovered.get("compliance_subprocess")
    sub_task_count = 0
    if csp:
        for cid in graph.subprocess_contents.get(csp, set()):
            if graph.get_type(cid) == "userTask":
                sub_task_count += 1
    details["a13_subprocess_task_count"] = sub_task_count
    results["a13_compliance_subprocess_ge2_tasks"] = csp is not None and sub_task_count >= 2

    # A14: Peer-unavailable fallback gateway exists
    pag = discovered.get("peer_available_gateway")
    results["a14_peer_available_gateway_exists"] = pag is not None
    # Reference peerOwnerAvailable variable
    peer_refs_var = False
    if pag:
        vars_ = extract_vars_from_gateway(graph, pag)
        peer_refs_var = any("peer" in v.lower() for v in vars_)
    results["a15_peer_available_gateway_variable_based"] = peer_refs_var

    # A16: Bounded joint-plan retry gateway with counter
    jpg = discovered.get("joint_plan_retry_gateway")
    has_counter = False
    if jpg:
        text = gateway_conditions_text(graph, jpg)
        has_counter = bool(COUNTER_PATTERN.search(text))
    results["a16_bounded_loop_counter_variable"] = has_counter

    # A17: Joint-plan retry default flow routes to terminal escalation
    retry_default_to_esc = False
    if jpg:
        elem = graph.get_element(jpg)
        default_flow_id = elem.get("default", "") if elem is not None else ""
        for f in graph.outgoing_flows(jpg):
            if f.get("flow_id") == default_flow_id:
                if graph.is_reachable(f.get("target", ""), TERMINAL_ESCALATION_END, max_depth=10):
                    retry_default_to_esc = True
                break
    results["a17_retry_default_routes_to_terminal_escalation"] = retry_default_to_esc

    # A18: Senior-lead escalation task exists and reaches via bounded path
    sle = discovered.get("senior_lead_escalation_task")
    results["a18_senior_lead_escalation_task_exists"] = sle is not None

    # A19: Escalation tier gateway references escalationTier variable
    etg = discovered.get("escalation_tier_gateway")
    tier_var = False
    if etg:
        vars_ = extract_vars_from_gateway(graph, etg)
        tier_var = any("tier" in v.lower() or "escal" in v.lower() for v in vars_)
    results["a19_escalation_tier_gateway_variable"] = tier_var

    # A20: Closed-loop feedback: from tier gateway default or retry loop back to joint plan
    closed_loop = False
    if etg and jp:
        elem = graph.get_element(etg)
        default_flow_id = elem.get("default", "") if elem is not None else ""
        for f in graph.outgoing_flows(etg):
            if f.get("flow_id") == default_flow_id:
                if graph.is_reachable(f.get("target", ""), jp, max_depth=10):
                    closed_loop = True
                    break
    if not closed_loop and jpg and jp:
        # Retry loop provides closed-loop too
        for f in graph.outgoing_flows(jpg):
            if f.get("target", "") == jp:
                closed_loop = True
                break
    results["a20_closed_loop_to_joint_plan"] = closed_loop

    # A21: Parallel review split and join both present
    prs = discovered.get("parallel_review_split")
    prj = discovered.get("parallel_review_join")
    results["a21_parallel_review_split_exists"] = prs is not None and graph.is_parallel_gateway(prs)
    results["a22_parallel_review_join_exists"] = prj is not None and graph.is_parallel_gateway(prj)

    # A23: Coordination lane with merchant + campaign coordination tasks
    mct = discovered.get("merchant_coordination_task")
    cct = discovered.get("campaign_coordination_task")
    results["a23_merchant_coordination_task"] = mct is not None
    results["a24_campaign_coordination_task"] = cct is not None

    # A25: Peer-owner lane has standardized whitelist tasks
    peer_tasks = []
    for lid, ldata in graph.lanes.items():
        if "peer" in ldata["name"].lower() or "peer" in lid.lower():
            for n in ldata["nodes"]:
                if graph.get_type(n) == "userTask":
                    peer_tasks.append(n)
    peer_whitelist_ok = True
    peer_whitelist_issues = []
    for pt in peer_tasks:
        name = graph.get_task_name(pt).lower() + " " + pt.lower() + " " + graph.get_documentation(pt).lower()
        if not any(term in name for term in PEER_WHITELIST_TERMS):
            peer_whitelist_ok = False
            peer_whitelist_issues.append(pt)
    results["a25_peer_lane_whitelist"] = peer_whitelist_ok
    details["a25_peer_tasks"] = peer_tasks
    if peer_whitelist_issues:
        details["a25_issues"] = peer_whitelist_issues

    # A26: Decision outcome gateway exists
    dog = discovered.get("decision_outcome_gateway")
    results["a26_decision_outcome_gateway"] = dog is not None

    # A27: Compliance crisis gateway routes on variable
    ccg = discovered.get("compliance_crisis_gateway")
    ccg_var = False
    if ccg:
        vars_ = extract_vars_from_gateway(graph, ccg)
        ccg_var = any("compliance" in v.lower() or "crisis" in v.lower() for v in vars_)
    results["a27_compliance_gateway_variable_based"] = ccg_var

    # A28: Compliance sub-process exits back to joint plan (replan)
    replan_ok = False
    if csp and jp:
        replan_ok = graph.is_reachable(csp, jp, max_depth=5)
    results["a28_compliance_subprocess_exits_to_replan"] = replan_ok

    # A29: A/B test / free trial / new-merchant tasks reachable
    ab_reachable = False
    free_trial_reachable = False
    new_merchant_reachable = False
    for eid, _ in graph.elements.items():
        if graph.get_type(eid) != "userTask":
            continue
        name_blob = (graph.get_task_name(eid) + " " + eid).lower()
        if "ab_test" in name_blob or "a_b_test" in name_blob or "ab test" in name_blob or "a/b" in name_blob:
            ab_reachable = True
        if "free_trial" in name_blob or "free trial" in name_blob:
            free_trial_reachable = True
        if "new_merchant" in name_blob or "new merchant" in name_blob:
            new_merchant_reachable = True
    results["a29_ab_test_reachable"] = ab_reachable
    results["a30_free_trial_reachable"] = free_trial_reachable
    results["a31_new_merchant_reachable"] = new_merchant_reachable

    # A32: Documentation on every new user task
    missing_doc: List[str] = []
    for eid, _ in graph.elements.items():
        if graph.get_type(eid) != "userTask":
            continue
        if eid in ORIGINAL_ELEMENTS:
            continue
        if not graph.get_documentation(eid):
            missing_doc.append(eid)
    results["a32_documentation_on_new_tasks"] = len(missing_doc) == 0
    if missing_doc:
        details["a32_missing_documentation"] = missing_doc

    # A33: Default flows on every new exclusive gateway
    missing_defaults: List[str] = []
    for eid in graph.elements:
        if graph.get_type(eid) != "exclusiveGateway":
            continue
        elem = graph.get_element(eid)
        default_flow_id = elem.get("default", "") if elem is not None else ""
        # Only check new gateways (not original)
        if eid in ("gw_exception_exists",):  # original from Case 2 original process
            continue
        if not default_flow_id:
            # An exclusive gateway with no default is a potential runtime error.
            # Only count as failure if the gateway has 2+ outgoing flows.
            if len(graph.outgoing_flows(eid)) >= 2:
                missing_defaults.append(eid)
    results["a33_default_flows_on_exclusive_gateways"] = len(missing_defaults) == 0
    if missing_defaults:
        details["a33_missing_defaults"] = missing_defaults

    # A34 (Rule 6a): multi_factor_gateway must route to >=2 distinct targets.
    # A gateway whose every outgoing flow targets the same node is decorative
    # (the conditions become cosmetic). Reference Solution A routes "all-green"
    # and "any-fail" to different downstream gateways.
    a34_distinct = 0
    if mfg:
        tgts = {f.get("target", "") for f in graph.outgoing_flows(mfg)}
        tgts.discard("")
        a34_distinct = len(tgts)
    details["a34_multi_factor_distinct_targets"] = a34_distinct
    results["a34_multi_factor_gateway_distinct_targets"] = a34_distinct >= 2

    # A35 (Rule 16a): the retry gateway's outgoing conditions must reference
    # only counter variables. Conditions that mix counter + decision-outcome
    # collapse Rule 4 and Rule 16 into one element (forbidden conflation).
    retry_only_counter = False
    if jpg:
        decision_keys = (
            "out_unified_decision", "unified_decision",
            "out_expedited_decision", "expedited_decision",
            "decisionoutcome",
        )
        saw_conditional = False
        clean = True
        for f in graph.outgoing_flows(jpg):
            cond = (f.get("condition") or "").strip()
            if not cond:
                continue
            saw_conditional = True
            cond_lower = cond.lower()
            if any(k in cond_lower for k in decision_keys):
                clean = False
                break
        retry_only_counter = saw_conditional and clean
    results["a35_retry_gateway_dedicated_to_counter"] = retry_only_counter

    # A36 (Rule 14a): compliance re-entry guard. If compliance_signal_check
    # is reachable from team_joint_mature_plan_preparation (i.e. it sits
    # inside the replan loop, cc-style), the compliance-crisis gateway MUST
    # reference a latch variable ("handled", "processed", "latch", "guard",
    # "applied", "complete") to prevent infinite re-entry. Otherwise, if
    # csc is upstream of jp (one-shot pattern), the check passes automatically.
    csc = discovered.get("compliance_signal_check_task")
    ccg = discovered.get("compliance_crisis_gateway")
    compliance_guard_ok = True
    compliance_latch_vars: List[str] = []
    compliance_in_loop = False
    if csc and jp:
        compliance_in_loop = graph.is_reachable(jp, csc, max_depth=20)
        if compliance_in_loop:
            has_latch = False
            if ccg:
                vars_ = {v.lower() for v in extract_vars_from_gateway(graph, ccg)}
                latch_kws = (
                    "handled", "processed", "latch", "guard",
                    "applied", "complete", "done", "resolved",
                )
                for v in vars_:
                    if any(k in v for k in latch_kws):
                        has_latch = True
                        compliance_latch_vars.append(v)
            compliance_guard_ok = has_latch
    details["a36_compliance_in_replan_loop"] = compliance_in_loop
    if compliance_latch_vars:
        details["a36_compliance_latch_vars"] = compliance_latch_vars
    results["a36_compliance_reentry_guard"] = compliance_guard_ok

    # A37: Parallel-review block symmetry. Every outgoing branch from the split
    # must reach the matching join, and split out-degree must equal join in-degree.
    # Catches the common gaming pattern where split fans out to tasks {X,Y} but
    # join waits on {P,Q}: each gateway exists (a21/a22 pass) yet runtime tokens
    # stall forever at the split.
    a37_ok = True
    a37_branches_to_join: Dict[str, bool] = {}
    if prs and prj and graph.is_parallel_gateway(prs) and graph.is_parallel_gateway(prj):
        out_targets = [f.get("target", "") for f in graph.outgoing_flows(prs)]
        out_targets = [t for t in out_targets if t]
        for tgt in out_targets:
            visited = {prs}
            reaches = _reaches_avoiding(graph, tgt, prj, visited, max_depth=30)
            a37_branches_to_join[tgt] = reaches
            if not reaches:
                a37_ok = False
        split_out = len(graph.outgoing_flows(prs))
        join_in = len(graph.in_flows.get(prj, []))
        if split_out != join_in:
            a37_ok = False
            details["a37_arity_mismatch"] = {"split_out": split_out, "join_in": join_in}
    else:
        a37_ok = False
    details["a37_branches_to_join"] = a37_branches_to_join
    results["a37_parallel_split_join_symmetry"] = a37_ok

    # A39: Gateway arity sanity. Every decision gateway with >=2 outgoing flows
    # must have pairwise distinct conditions (no duplicate/tautological branches).
    # Degenerate fan-out hides semantic gaming that slips past a34 (which only
    # checks the multi-factor gateway) and d09 (tautology patterns).
    a39_issues: List[Dict[str, object]] = []
    for eid in graph.elements:
        if not graph.is_decision_gateway(eid):
            continue
        outs = graph.outgoing_flows(eid)
        if len(outs) < 2:
            continue
        conds = [(f.get("condition") or "").strip() for f in outs]
        non_empty = [c for c in conds if c]
        if len(set(non_empty)) < len(non_empty):
            a39_issues.append({"gateway": eid, "reason": "duplicate_conditions", "conds": conds})
    a39_ok = len(a39_issues) == 0
    if a39_issues:
        details["a39_issues"] = a39_issues
    results["a39_gateway_arity_sanity"] = a39_ok

    # A41: Exclusive-gateway default-flow exclusivity. When a gateway declares
    # default="flow_X", that flow must be condition-less AND at least one
    # *other* outgoing flow must carry a conditionExpression. Catches
    # degenerate gateways where the default flow has a condition, or every
    # flow is unconditional (Flowable treats the first non-default as taken).
    a41_issues: List[Dict[str, object]] = []
    for eid in graph.elements:
        if graph.get_type(eid) != "exclusiveGateway":
            continue
        elem = graph.get_element(eid)
        default_id = elem.get("default", "") if elem is not None else ""
        if not default_id:
            continue  # a33 enforces defaults only on new gateways with >=2 outs
        outs = graph.outgoing_flows(eid)
        default_flow = next((f for f in outs if f.get("flow_id") == default_id), None)
        if default_flow is None:
            a41_issues.append({"gateway": eid, "reason": "default_flow_not_found"})
            continue
        default_cond = (default_flow.get("condition") or "").strip()
        other_with_cond = any(
            (f.get("condition") or "").strip()
            for f in outs
            if f.get("flow_id") != default_id
        )
        if default_cond:
            a41_issues.append({"gateway": eid, "reason": "default_flow_has_condition"})
        elif not other_with_cond:
            a41_issues.append({"gateway": eid, "reason": "no_other_conditional_flow"})
    a41_ok = len(a41_issues) == 0
    if a41_issues:
        details["a41_issues"] = a41_issues
    results["a41_default_flow_exclusivity"] = a41_ok

    # A42: Every outgoing branch of the parallel review split must traverse at
    # least one userTask before reaching the join. Catches decorative parallel
    # blocks that only fan out through gateways (no real concurrent work).
    a42_ok = True
    a42_branch_user_tasks: Dict[str, int] = {}
    if prs and prj and graph.is_parallel_gateway(prs):
        for f in graph.outgoing_flows(prs):
            tgt = f.get("target", "")
            if not tgt:
                continue
            count = _count_user_tasks_on_path(graph, tgt, prj, {prs})
            a42_branch_user_tasks[tgt] = count
            if count < 1:
                a42_ok = False
    else:
        a42_ok = False
    details["a42_branch_user_tasks"] = a42_branch_user_tasks
    results["a42_split_branch_requires_user_task"] = a42_ok

    # A43: From senior_lead_escalation_task, every reachable path must
    # terminate at either main_end or terminal_escalation_end. Stranded
    # escalation tails (dead-end gateways / missing default flows) produce
    # tokens that never release the process instance.
    a43_ok = True
    a43_dead_ends: List[str] = []
    st = discovered.get("senior_lead_escalation_task")
    if st and st in graph.elements:
        dead = _find_non_terminating_paths(
            graph,
            start=st,
            terminal_ids={"main_end", TERMINAL_ESCALATION_END},
            max_depth=40,
        )
        if dead:
            a43_ok = False
            a43_dead_ends = dead
    else:
        a43_ok = False
    if a43_dead_ends:
        details["a43_dead_ends"] = a43_dead_ends
    results["a43_escalation_terminates"] = a43_ok

    # A44: No mixed parallel gateways. A parallelGateway with both fan-in >=2
    # AND fan-out >=2 is a "mixed" gateway; Flowable semantics wait for ALL
    # incoming before firing ALL outgoing, so if incoming sources are mutually
    # exclusive (e.g. a true-branch and the fallback of an upstream exclusive
    # gateway), the gateway can never fire and the process deadlocks.
    a44_issues: List[Dict[str, object]] = []
    for eid in graph.elements:
        if not graph.is_parallel_gateway(eid):
            continue
        inc = len(graph.in_flows.get(eid, []))
        out = len(graph.outgoing_flows(eid))
        if inc >= 2 and out >= 2:
            a44_issues.append({
                "gateway": eid,
                "in": inc,
                "out": out,
                "in_sources": [f.get("source", "") for f in graph.in_flows.get(eid, [])],
            })
    a44_ok = len(a44_issues) == 0
    if a44_issues:
        details["a44_mixed_gateways"] = a44_issues
    results["a44_no_mixed_parallel_gateways"] = a44_ok

    # A45: Parallel JOIN sources must trace back to a common parallel SPLIT.
    # A join waits for all incoming tokens; if the sources come from
    # unrelated flow regions, tokens never all arrive. Exempt: the "final
    # rendezvous" pattern (join's successor reaches main_end within 2 steps)
    # where inputs are legitimately disjoint by design.
    a45_issues: List[Dict[str, object]] = []
    for eid in graph.elements:
        if not graph.is_parallel_gateway(eid):
            continue
        inc = len(graph.in_flows.get(eid, []))
        out = len(graph.outgoing_flows(eid))
        if not (inc >= 2 and out == 1):
            continue  # not a pure join
        # Final rendezvous exemption
        if _reaches_within(graph, eid, "main_end", max_depth=2):
            continue
        # Trace each incoming source backward up to a parallel SPLIT
        split_ancestors: List[Optional[str]] = []
        for f in graph.in_flows.get(eid, []):
            src = f.get("source", "")
            ancestor = _nearest_parallel_split_ancestor(graph, src, max_depth=25)
            split_ancestors.append(ancestor)
        if any(a is None for a in split_ancestors):
            a45_issues.append({"join": eid, "reason": "source_without_split_ancestor",
                               "ancestors": split_ancestors})
            continue
        if len(set(split_ancestors)) > 1:
            a45_issues.append({"join": eid, "reason": "sources_from_different_splits",
                               "ancestors": split_ancestors})
    a45_ok = len(a45_issues) == 0
    if a45_issues:
        details["a45_issues"] = a45_issues
    results["a45_parallel_join_common_split"] = a45_ok

    # A46: Peer lane must contain userTasks covering all 5 canonical policy
    # review topics. Business rule: peer owner's standardized review covers
    # ad_ratio + ab_test + free_trial + new_merchant + review_based_promotion.
    # Partial coverage (e.g. only 3 of 5) leaves scenarios unreachable.
    peer_topics = {
        "ad_ratio": ("ad_ratio", "ad ratio", "adratio"),
        "ab_test": ("ab_test", "ab test", "abtest", "a_b_test", "a/b"),
        "free_trial": ("free_trial", "free trial", "freetrial"),
        "new_merchant": ("new_merchant", "new merchant", "newmerchant"),
        "review_based_promotion": ("review_based_promotion", "review based promotion",
                                   "review-based", "promotion_standardized"),
    }
    peer_lane_tasks = []
    for lid, ldata in graph.lanes.items():
        label = (ldata.get("name", "") + " " + lid).lower()
        if "peer" in label:
            for n in ldata.get("nodes", []):
                if graph.get_type(n) == "userTask":
                    peer_lane_tasks.append(n)
    peer_blob = ""
    for pt in peer_lane_tasks:
        peer_blob += " " + (graph.get_task_name(pt) + " " + pt + " "
                            + graph.get_documentation(pt)).lower()
    covered_topics = {}
    missing_topics = []
    for topic, fragments in peer_topics.items():
        hit = any(fr in peer_blob for fr in fragments)
        covered_topics[topic] = hit
        if not hit:
            missing_topics.append(topic)
    a46_ok = len(missing_topics) == 0
    details["a46_covered_topics"] = covered_topics
    if missing_topics:
        details["a46_missing_topics"] = missing_topics
    results["a46_peer_review_topic_coverage"] = a46_ok

    # A47: Every peer-lane userTask matching a canonical topic must trace
    # back to a parallelGateway within 2 hops, crossing only gateway nodes
    # (no intermediate userTask). Sequentializing the peer review (chaining
    # peer_* userTasks) violates the Rule 6a parallel-review intent.
    # Gateway-only intermediates are allowed (e.g. a guard exclusiveGateway).
    peer_topic_fragments = (
        "ad_ratio", "ab_test", "free_trial", "new_merchant",
        "review_based_promotion",
    )

    def _reaches_parallel_via_gateways_only(
        node: str, max_hops: int = 2
    ) -> bool:
        """Walk backward up to max_hops. Cross gateway nodes freely;
        reject if any intermediate is a userTask."""
        queue: deque = deque([(node, 0)])
        seen: Set[str] = set()
        while queue:
            cur, depth = queue.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            for pred in graph.predecessors(cur):
                if graph.is_parallel_gateway(pred):
                    return True
                if graph.get_type(pred) == "userTask":
                    continue  # blocks this branch
                if depth + 1 <= max_hops:
                    queue.append((pred, depth + 1))
        return False

    a47_offenders: List[Dict[str, object]] = []
    for pt in peer_lane_tasks:
        blob = (graph.get_task_name(pt) + " " + pt + " "
                + graph.get_documentation(pt)).lower()
        if not any(fr in blob for fr in peer_topic_fragments):
            continue
        if not _reaches_parallel_via_gateways_only(pt, max_hops=2):
            a47_offenders.append({
                "task": pt,
                "predecessors": graph.predecessors(pt),
            })
    a47_ok = len(a47_offenders) == 0
    if a47_offenders:
        details["a47_offenders"] = a47_offenders
    results["a47_peer_topic_tasks_directly_from_parallel_split"] = a47_ok

    # A48: gw_decision_outcome's default flow must land on the retry loop
    # (gw_joint_plan_retry) or terminate at escalation_final_end within 3
    # hops. Routing the "rejected" default to a new user task (e.g. a second
    # committee review) defeats Rule 12's closed-loop replan intent.
    dog = discovered.get("decision_outcome_gateway")
    a48_ok = True
    a48_detail: Dict[str, object] = {}
    if dog is None:
        a48_ok = False
        a48_detail["reason"] = "decision_outcome_gateway_missing"
    else:
        elem = graph.get_element(dog)
        default_id = elem.get("default", "") if elem is not None else ""
        default_target = None
        if default_id:
            for f in graph.outgoing_flows(dog):
                if f.get("flow_id") == default_id:
                    default_target = f.get("target", "")
                    break
        else:
            # Find the unconditional outgoing flow as implicit default
            for f in graph.outgoing_flows(dog):
                if not (f.get("condition") or "").strip():
                    default_target = f.get("target", "")
                    break
        a48_detail["default_target"] = default_target
        if not default_target:
            a48_ok = False
            a48_detail["reason"] = "no_default_flow"
        else:
            retry_ok = graph.is_reachable(default_target, "gw_joint_plan_retry", max_depth=3)
            esc_ok = graph.is_reachable(default_target, TERMINAL_ESCALATION_END, max_depth=3)
            # Also acceptable: default_target itself is the retry or escalation end.
            if default_target == "gw_joint_plan_retry" or default_target == TERMINAL_ESCALATION_END:
                retry_ok = True
            if not (retry_ok or esc_ok):
                a48_ok = False
                a48_detail["reason"] = "default_does_not_reach_retry_or_escalation_end"
    details["a48_detail"] = a48_detail
    results["a48_decision_outcome_default_loops_to_retry"] = a48_ok

    # A49: gw_peer_available's "peerOwnerAvailable == true" outgoing flow must
    # target a parallelGateway directly. Routing the available branch through
    # a userTask chain linearizes the peer review and stalls scenarios that
    # rely on concurrent peer-topic completion.
    gpa = "gw_peer_available"
    a49_ok = True
    a49_detail: Dict[str, object] = {}
    if gpa not in graph.elements:
        a49_ok = False
        a49_detail["reason"] = "gw_peer_available_missing"
    else:
        true_targets = []
        for f in graph.outgoing_flows(gpa):
            cond = (f.get("condition") or "").lower()
            if "peerowneravailable" in cond and "true" in cond:
                true_targets.append(f.get("target", ""))
        a49_detail["true_targets"] = true_targets
        if not true_targets:
            a49_ok = False
            a49_detail["reason"] = "no_peer_available_true_flow"
        else:
            ok = any(graph.is_parallel_gateway(t) for t in true_targets if t)
            if not ok:
                a49_ok = False
                a49_detail["reason"] = "true_branch_target_not_parallel_gateway"
    details["a49_detail"] = a49_detail
    results["a49_peer_available_true_branch_to_split"] = a49_ok

    # A50: No shadow tasks. Every userTask must be EITHER an anchored
    # original OR claimable by scenarios OR match a canonical domain pattern
    # (peer-topic review tasks; coordination tasks explicitly required by
    # Rule 24). Placeholder userTasks outside these categories = dead weight.
    anchored_originals = {
        "team_drafts_initial_plan", "team_executes_plan",
        "logistics_capacity_confirmation", "peer_readiness_check",
        "compliance_signal_check", "temporary_unified_decision",
        "senior_lead_expedited_decision", "emergency_delist_task",
        "brand_risk_reassessment_task", "team_joint_mature_plan_preparation",
    }
    # Canonical domain patterns implied by Rules 23 & 24.
    canonical_task_patterns = (
        "ad_ratio", "ab_test", "free_trial", "new_merchant",
        "review_based_promotion", "promotion_standardized",
        "merchant_readiness", "campaign_cadence", "merchant_admission",
    )
    # Load scenario task_outputs to know which IDs scenarios claim.
    scenario_task_ids: Set[str] = set()
    scenario_path = "starter_project/test_scenarios_L3.json"
    try:
        with open(scenario_path) as _f:
            _sc = json.load(_f)
        for s in _sc.get("scenarios", []):
            for tid in s.get("task_outputs", {}).keys():
                scenario_task_ids.add(tid)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    shadow_tasks: List[str] = []
    for eid in graph.elements:
        if graph.get_type(eid) != "userTask":
            continue
        if eid in anchored_originals or eid in scenario_task_ids:
            continue
        blob = (graph.get_task_name(eid) + " " + eid + " "
                + graph.get_documentation(eid)).lower()
        if any(p in blob for p in canonical_task_patterns):
            continue  # matches a canonical domain pattern
        shadow_tasks.append(eid)
    a50_ok = len(shadow_tasks) == 0
    if shadow_tasks:
        details["a50_shadow_tasks"] = shadow_tasks
    results["a50_no_shadow_tasks"] = a50_ok

    # A51: Data-flow closure (inverted). Every out_<var> referenced in ANY
    # gateway condition MUST be declared as a writable formProperty on some
    # task OR be a task_output in scenarios. Catches "dead condition
    # variables" (gateways referencing nonexistent outputs). We intentionally
    # do NOT require every declared out_* to be consumed, because audit
    # outputs (out_rationale, out_*_ok) are legitimately external.
    declared_outs: Set[str] = set()
    for eid in graph.elements:
        if graph.get_type(eid) != "userTask":
            continue
        props = graph.get_form_properties(eid)
        for pid, meta in props.items():
            w = (meta.get("writable") or "").strip().lower()
            if pid.startswith("out_") and w in ("true", ""):
                declared_outs.add(pid)
    # Also accept scenario task_outputs as "declared" (runtime writes them)
    try:
        with open("starter_project/test_scenarios_L3.json") as _sf:
            _sc = json.load(_sf)
        for s in _sc.get("scenarios", []):
            for _tid, outs in s.get("task_outputs", {}).items():
                if isinstance(outs, dict):
                    for k in outs.keys():
                        if k.startswith("out_"):
                            declared_outs.add(k)
                        else:
                            declared_outs.add(f"out_{k}")
                            declared_outs.add(k)  # bare too
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    # Scan all gateway / flow conditions for ${out_X} references
    referenced_out_vars: Set[str] = set()
    cond_pattern = re.compile(r"\$\{\s*(out_[A-Za-z_][A-Za-z0-9_]*)")
    for flows in graph.out_flows.values():
        for f in flows:
            cond = f.get("condition") or ""
            for m in cond_pattern.finditer(cond):
                referenced_out_vars.add(m.group(1))
    ungrounded = sorted(v for v in referenced_out_vars if v not in declared_outs)
    a51_ok = len(ungrounded) == 0
    if ungrounded:
        details["a51_ungrounded_condition_vars"] = ungrounded
    results["a51_data_flow_closure"] = a51_ok

    # A40: Anchor chain connectivity. The core decision chain
    # team_drafts_initial_plan -> joint_plan -> unified_decision -> team_executes_plan
    # must be topologically reachable step-by-step. Topology can pass a21/a22
    # and still leave the execute anchor unreachable from the planning chain.
    a40_chain = {
        "draft_to_joint_plan": False,
        "joint_plan_to_unified_decision": False,
        "unified_decision_to_execute": False,
    }
    if "team_drafts_initial_plan" in graph.elements and jp:
        a40_chain["draft_to_joint_plan"] = graph.is_reachable(
            "team_drafts_initial_plan", jp, max_depth=20
        )
    if jp and uid:
        a40_chain["joint_plan_to_unified_decision"] = graph.is_reachable(
            jp, uid, max_depth=20
        )
    if uid and "team_executes_plan" in graph.elements:
        a40_chain["unified_decision_to_execute"] = graph.is_reachable(
            uid, "team_executes_plan", max_depth=30
        )
    a40_ok = all(a40_chain.values())
    details["a40_anchor_chain"] = a40_chain
    results["a40_anchor_chain_connectivity"] = a40_ok

    return results, details


# ============================================================================
# SECTION B - TEST RESULTS
# ============================================================================


def check_test_results(results_path: str) -> Dict:
    """Section B: Load test_results.json and check pass rate.

    L3 hardened threshold: >=95% overall, AND the anti_gaming category must be
    10/10 (mandatory, doesn't count toward the 95% percentage gate - a submission
    with 98/100 but 3/10 anti_gaming still fails)."""
    try:
        with open(results_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {
            "error": str(e),
            "total_scenarios": 0,
            "passed_scenarios": 0,
            "pass_rate": 0.0,
            "meets_threshold": False,
        }

    total = data.get("total_scenarios", 0)
    passed = data.get("passed_scenarios", 0)
    pass_rate = passed / total if total > 0 else 0.0
    overall_ok = pass_rate >= 0.95
    # Anti-gaming mandatory: look at per-scenario list if provided
    anti_total = 0
    anti_passed = 0
    for s in data.get("scenarios", []):
        if str(s.get("category", "")).lower() in ("anti_gaming", "anti-gaming"):
            anti_total += 1
            if s.get("pass"):
                anti_passed += 1
    anti_ok = (anti_total == 0) or (anti_passed == anti_total)
    meets = overall_ok and anti_ok
    return {
        "total_scenarios": total,
        "passed_scenarios": passed,
        "pass_rate": pass_rate,
        "meets_threshold": meets,
        "anti_gaming_passed": anti_passed,
        "anti_gaming_total": anti_total,
        "anti_gaming_mandatory_ok": anti_ok,
    }


# ============================================================================
# SECTION C - COMPLIANCE COVERAGE + CROSS-VALIDATION
# ============================================================================


def check_compliance(
    structural_path: str,
    rules_path: str,
    graph: BPMNGraph,
    discovered: Dict[str, Optional[str]],
) -> Dict:
    """Section C: Compliance coverage check + cross-validation."""
    results: Dict[str, object] = {}

    # Structural coverage
    try:
        with open(structural_path) as f:
            struct = json.load(f)
        mods = struct.get("modifications", {})
        covered = [m for m in REQUIRED_MODIFICATIONS if m in mods]
        results["structural_coverage"] = len(covered) / len(REQUIRED_MODIFICATIONS)
        results["structural_missing"] = [m for m in REQUIRED_MODIFICATIONS if m not in mods]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        results["structural_coverage"] = 0.0
        results["structural_error"] = str(e)
        results["structural_missing"] = list(REQUIRED_MODIFICATIONS)

    # Rules coverage
    try:
        with open(rules_path) as f:
            rules_data = json.load(f)
        rule_entries = rules_data.get("rules", {})
        covered = [r for r in REQUIRED_RULES if r in rule_entries]
        results["rules_coverage"] = len(covered) / len(REQUIRED_RULES)
        results["rules_missing"] = [r for r in REQUIRED_RULES if r not in rule_entries]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        results["rules_coverage"] = 0.0
        results["rules_error"] = str(e)
        results["rules_missing"] = list(REQUIRED_RULES)

    # Cross-validation: rule_8a (logistics not modified from main flow)
    results["cross_rule_8a_logistics_notified_only"] = _check_logistics_notified_only(graph)

    # Cross-validation: rule_17 (rationale on every manual override)
    results["cross_rule_17_rationale_on_overrides"] = _check_rationale_on_overrides(graph, discovered)

    # Cross-validation: rule_13 (timer PT48H present)
    tb = discovered.get("timer_boundary_event")
    duration = ""
    if tb:
        elem = graph.get_element(tb)
        if elem is not None:
            for child in elem.iter():
                if graph._local(child.tag) == "timeDuration":
                    duration = (child.text or "").strip()
                    break
    results["cross_rule_13_timer_pt48h"] = duration.upper() == "PT48H"

    # Cross-validation: rule_6 (multi-factor >=4)
    mfg = discovered.get("multi_factor_gateway")
    hits = 0
    if mfg:
        text = gateway_conditions_text(graph, mfg).lower()
        hits = sum(1 for f in FACTOR_KEYWORDS if f in text)
    results["cross_rule_6_multi_factor_ge4"] = hits >= 4

    # C05: Every declared rule must cite at least one BPMN element ID that
    # actually exists in the graph. Binds compliance claims to concrete
    # structure and defeats pure keyword gaming. Supports two rule schemas:
    #   (a) structured: {"enforcing_elements":[{"element":<id>, ...}, ...]}
    #   (b) free-text:  {"evidence": "<sentence mentioning element IDs>"}
    c05_issues: List[Dict[str, str]] = []
    try:
        with open(rules_path) as f:
            rules_data = json.load(f)
        rule_entries = rules_data.get("rules", {})
        known_ids = {eid for eid in graph.elements.keys() if len(eid) >= 4}
        # Also count sequenceFlow IDs as valid BPMN references (reference
        # solution cites flow_* for routing rules like "non-critical rejection
        # loops back to planning").
        for flows in graph.out_flows.values():
            for f in flows:
                fid = f.get("flow_id")
                if fid and len(fid) >= 4:
                    known_ids.add(fid)
        # Accept the process key itself, plus the literal token "process",
        # so rules about the process-id can bind via <bpmn:process id=...>.
        pk = graph.process_key()
        if pk:
            known_ids.add(pk)
        known_ids.add("process")
        for rname in REQUIRED_RULES:
            entry = rule_entries.get(rname, {})
            bound = False
            # Schema (a): enforcing_elements
            enforcing = entry.get("enforcing_elements")
            if isinstance(enforcing, list):
                for item in enforcing:
                    if isinstance(item, dict):
                        eid = str(item.get("element", "")).strip()
                    else:
                        eid = str(item).strip()
                    if eid and eid in known_ids:
                        bound = True
                        break
            # Schema (b): evidence free text
            if not bound:
                evidence = str(entry.get("evidence", ""))
                if evidence.strip():
                    for eid in known_ids:
                        if eid in evidence:
                            bound = True
                            break
            if not bound:
                # Distinguish empty-claim from claim-without-valid-id
                if not enforcing and not entry.get("evidence"):
                    c05_issues.append({"rule": rname, "reason": "empty_evidence"})
                else:
                    c05_issues.append({"rule": rname, "reason": "no_existing_bpmn_id_referenced"})
        results["c05_rule_to_bpmn_binding"] = len(c05_issues) == 0
        if c05_issues:
            results["c05_issues"] = c05_issues
    except (FileNotFoundError, json.JSONDecodeError):
        results["c05_rule_to_bpmn_binding"] = False

    # C06: Structural strengthening of rule_8a. Beyond absence of main-flow
    # decision inbound (covered by cross_rule_8a_logistics_notified_only), the
    # logistics_capacity_confirmation's *outgoing* flows must only converge
    # (to a parallel gateway) or terminate / stay in the logistics lane -
    # never route into a main-flow decision or main-flow task.
    results["cross_rule_8a_logistics_structural"] = _check_logistics_outgoing_safe(graph)

    # C07: Each modification must bind to BPMN structure. Accepts either
    #   schema A: {"elements_added":[...], "elements_removed":[...]}
    #   schema B: {"elements":[...]}    (reference / codex style)
    # For schema A: >=1 added must exist in current BPMN AND (if elements_removed
    # non-empty) >=1 removed must NOT exist (confirming deletion).
    # For schema B: >=1 element must exist in the BPMN.
    c07_issues: List[Dict[str, str]] = []
    try:
        with open(structural_path) as f:
            struct = json.load(f)
        mods_data = struct.get("modifications", {})
        known_ids = set(graph.elements.keys())
        # Also accept flow IDs and process key as structural references
        for flows in graph.out_flows.values():
            for fl in flows:
                fid = fl.get("flow_id")
                if fid:
                    known_ids.add(fid)
        pk = graph.process_key()
        if pk:
            known_ids.add(pk)
        # Keys whose values are free-text prose - do not count ID mentions in
        # those (description sentences often name elements without the modification
        # actually enforcing them; require a structured field).
        free_text_keys = {"name", "title", "description", "text", "evidence", "rationale"}

        def _flatten_strs(obj):
            if isinstance(obj, str):
                yield obj
            elif isinstance(obj, list):
                for it in obj:
                    yield from _flatten_strs(it)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if str(k) in free_text_keys:
                        continue
                    yield from _flatten_strs(v)

        for mname in REQUIRED_MODIFICATIONS:
            entry = mods_data.get(mname, {})
            if not isinstance(entry, dict):
                c07_issues.append({"modification": mname, "reason": "entry_not_dict"})
                continue
            added = entry.get("elements_added")
            removed = entry.get("elements_removed")
            bound = False
            # Strict schema: elements_added / elements_removed both present
            if isinstance(added, list) and added:
                if any(str(e) in known_ids for e in added):
                    bound = True
                if isinstance(removed, list) and removed:
                    if not any(str(e) not in known_ids for e in removed):
                        c07_issues.append({
                            "modification": mname,
                            "reason": "elements_removed_still_present",
                        })
                        continue
            # Fallback: flatten all non-free-text string leaves
            if not bound:
                if any(s in known_ids for s in _flatten_strs(entry)):
                    bound = True
            if not bound:
                c07_issues.append({
                    "modification": mname,
                    "reason": "no_existing_bpmn_id_referenced",
                })
        results["c07_modification_to_bpmn_binding"] = len(c07_issues) == 0
        if c07_issues:
            results["c07_issues"] = c07_issues
    except (FileNotFoundError, json.JSONDecodeError):
        results["c07_modification_to_bpmn_binding"] = False

    # C08: Rationale form must be WRITABLE on override tasks (not just a
    # readable placeholder). Strengthens cross_rule_17 which only checks
    # presence.
    results["c08_rationale_form_writable"] = _check_rationale_writable(
        graph, discovered
    )

    # C09: Every rule entry must declare a non-empty `priority` classification
    # (Cardinal / Structural / Routing / Operational / ...). Forces rule
    # authors to intentionally categorize.
    c09_missing: List[str] = []
    try:
        with open(rules_path) as f:
            rules_data_c09 = json.load(f)
        rule_entries_c09 = rules_data_c09.get("rules", {})
        for rname in REQUIRED_RULES:
            prio = str(rule_entries_c09.get(rname, {}).get("priority", "")).strip()
            if not prio:
                c09_missing.append(rname)
    except (FileNotFoundError, json.JSONDecodeError):
        c09_missing = list(REQUIRED_RULES)
    results["c09_rule_priority_declared"] = len(c09_missing) == 0
    if c09_missing:
        results["c09_missing_priority"] = c09_missing

    # CR27: Brand-risk precedence on compliance-crisis branch. Must reach
    # brand_risk_reassessment_task on the crisis branch AND that step must
    # precede the unified decision. Brr may live inside compliance_subprocess;
    # we accept "csc reaches brr directly OR csc reaches compliance_subprocess
    # and brr is nested within it". Then brr or its containing subprocess
    # must reach the unified decision downstream.
    csc = discovered.get("compliance_signal_check_task") or "compliance_signal_check"
    brr = "brand_risk_reassessment_task"
    uid_tid = discovered.get("unified_decision_task")
    csp_tid = discovered.get("compliance_subprocess")
    cr27_ok = False
    if csc in graph.elements and brr in graph.elements and uid_tid:
        direct = graph.is_reachable(csc, brr, max_depth=15)
        via_subprocess = False
        if csp_tid and csp_tid in graph.elements:
            if graph.is_reachable(csc, csp_tid, max_depth=10):
                # brr is inside compliance_subprocess?
                brr_inside = brr in graph.subprocess_contents.get(csp_tid, set())
                if brr_inside:
                    via_subprocess = True
        a_ok = direct or via_subprocess
        # b: brr (or containing subprocess) reaches unified_decision
        b_direct = graph.is_reachable(brr, uid_tid, max_depth=15)
        b_via_sub = False
        if csp_tid and csp_tid in graph.elements:
            if graph.is_reachable(csp_tid, uid_tid, max_depth=15):
                b_via_sub = True
        b_ok = b_direct or b_via_sub
        cr27_ok = a_ok and b_ok
    results["cross_rule_27_brand_risk_before_decision"] = cr27_ok

    # CR28: Kill-switch gating. Three in_* formProperties must exist on
    # team_drafts_initial_plan: abTestEnabled, freeTrialEnabled,
    # newMerchantAdmissionEnabled (defaults "true"). Guard gateways
    # downstream must reference these variables.
    kill_switches = ("abTestEnabled", "freeTrialEnabled", "newMerchantAdmissionEnabled")
    draft_props = graph.get_form_properties("team_drafts_initial_plan")
    declared_ks: Set[str] = set()
    for pid, meta in draft_props.items():
        # in_<var> naming OR bare variable match
        if pid.startswith("in_"):
            bare = pid[3:]
        else:
            bare = pid
        if bare in kill_switches:
            declared_ks.add(bare)
    # Check gateway references - scan all conditions for each kill-switch var
    all_conds = ""
    for flows in graph.out_flows.values():
        for f in flows:
            all_conds += " " + (f.get("condition") or "")
    referenced_ks = {v for v in kill_switches if v in all_conds}
    cr28_ok = (declared_ks == set(kill_switches)) and (referenced_ks == set(kill_switches))
    if not cr28_ok:
        results["cross_rule_28_detail"] = {
            "declared": sorted(declared_ks),
            "referenced_in_gateway_cond": sorted(referenced_ks),
        }
    results["cross_rule_28_kill_switch_gating"] = cr28_ok

    # CR29: Senior-lead role-separation. Assignee on temporary_unified_decision
    # must be DIFFERENT from assignee on senior_lead_expedited_decision.
    uid_tid = discovered.get("unified_decision_task")
    sen_tid = discovered.get("senior_lead_escalation_task")
    cr29_ok = True
    if uid_tid and sen_tid:
        uid_role = graph.get_assignee(uid_tid)
        sen_role = graph.get_assignee(sen_tid)
        if uid_role and sen_role and uid_role == sen_role:
            cr29_ok = False
            results["cross_rule_29_detail"] = {
                "unified_decision_assignee": uid_role,
                "senior_escalation_assignee": sen_role,
            }
    else:
        cr29_ok = False
    results["cross_rule_29_senior_role_separation"] = cr29_ok

    # CR30: Compliance-subprocess latch declared explicitly. Strengthens
    # a36: the latch variable must exist as a writable formProperty on
    # some task inside compliance_subprocess, not merely referenced in
    # a gateway expression.
    cr30_ok = False
    csp = discovered.get("compliance_subprocess")
    latch_keywords = ("handled", "processed", "latch", "guard", "applied",
                       "complete", "done", "resolved")
    if csp and csp in graph.elements:
        csp_elem = graph.get_element(csp)
        if csp_elem is not None:
            # Find child userTasks of subprocess and scan their formProperties
            for child in csp_elem.iter():
                if graph._local(child.tag) == "formProperty":
                    pid = (child.get("id") or "").lower()
                    writable = (child.get("writable") or "").strip().lower()
                    if any(k in pid for k in latch_keywords) and writable in ("true", ""):
                        cr30_ok = True
                        break
    results["cross_rule_30_compliance_latch_explicit"] = cr30_ok

    # C10: priority_chain addresses all known conflict pairs + is consistent
    # with the BPMN. Known conflict pairs (from stakeholder_artifacts):
    #   (rule_11, rule_29), (rule_27, rule_14), (rule_7, rule_28)
    conflict_pairs = [
        ("rule_11", "rule_29"),
        ("rule_27", "rule_14"),
        ("rule_7",  "rule_28"),
    ]
    try:
        with open(rules_path) as f:
            rules_data_c10 = json.load(f)
        chain = rules_data_c10.get("priority_chain", [])
        chain_list = [str(x) for x in chain] if isinstance(chain, list) else []
        chain_set = set(chain_list)
        c10_missing_pairs: List[List[str]] = []
        for a_r, b_r in conflict_pairs:
            if a_r not in chain_set or b_r not in chain_set:
                c10_missing_pairs.append([a_r, b_r])
        c10_ok = (len(chain_list) > 0) and (len(c10_missing_pairs) == 0)
        results["c10_priority_chain_addresses_conflicts"] = c10_ok
        if not c10_ok:
            results["c10_missing_pairs"] = c10_missing_pairs
    except (FileNotFoundError, json.JSONDecodeError):
        results["c10_priority_chain_addresses_conflicts"] = False

    # C11: design_decisions.md audit. File must exist in the same directory
    # as the rules_path (submission directory). For each modification_1..18,
    # a section must be present with:
    #   - Chosen approach (non-empty)
    #   - Rejected alternatives (>=2 distinct lines, different from chosen)
    #   - Rationale citing >=1 existing rule_ID
    #   - Trade-off (non-empty)
    import os
    import re as _re
    dd_path = os.path.join(os.path.dirname(os.path.abspath(rules_path)),
                           "design_decisions.md")
    c11_issues: List[Dict[str, object]] = []
    try:
        with open(dd_path) as f:
            dd_text = f.read()
        # Split into sections keyed by "## Modification N"
        sections: Dict[str, str] = {}
        for m in _re.finditer(
            r"^##\s*Modification\s*(\d+).*?(?=^##\s*Modification\s*\d+|\Z)",
            dd_text, flags=_re.MULTILINE | _re.DOTALL
        ):
            sections[f"modification_{m.group(1)}"] = m.group(0)
        for i in range(1, 19):
            mname = f"modification_{i}"
            if mname not in sections:
                c11_issues.append({"modification": mname, "reason": "section_missing"})
                continue
            body = sections[mname].lower()
            chosen_m = _re.search(r"chosen\s*approach\s*:?\s*(.+)", body)
            chosen = (chosen_m.group(1).split("\n")[0].strip() if chosen_m else "")
            if not chosen:
                c11_issues.append({"modification": mname, "reason": "no_chosen_approach"})
                continue
            # Rejected alternatives: count non-empty bullet lines beneath
            # "rejected alternatives". End-anchor to "rationale cites" or
            # "trade-off" line (not to any "rationale" substring which may
            # legitimately appear inside bullet text like "rationale log").
            rej_block = _re.search(
                r"rejected\s*alternatives\s*:?(.*?)(?:rationale\s*(?:cites|references)|trade-?off|$)",
                body, flags=_re.DOTALL
            )
            if not rej_block:
                c11_issues.append({"modification": mname, "reason": "no_rejected_alternatives_section"})
                continue
            bullets = [
                ln.strip().lstrip("-*").strip()
                for ln in rej_block.group(1).split("\n")
                if ln.strip().lstrip("-*").strip()
            ]
            # distinct from chosen
            distinct = [b for b in bullets if b and b != chosen]
            if len(distinct) < 2:
                c11_issues.append({"modification": mname, "reason": "fewer_than_2_rejected_alternatives"})
                continue
            # Rationale must cite >=1 valid rule_ID. Anchor to the
            # "rationale cites" / "cites rule" phrase; "out_rationale"
            # mentioned in chosen_approach should not be confused for the
            # rationale bullet.
            rationale_m = _re.search(
                r"rationale\s+(?:cites|references)\s+rules?\s*:?\*{0,2}\s*([^\n]+)",
                body,
            )
            rationale = rationale_m.group(1) if rationale_m else ""
            cited_rules = [rid for rid in REQUIRED_RULES if rid in rationale]
            if not cited_rules:
                c11_issues.append({"modification": mname, "reason": "no_valid_rule_cited"})
                continue
            # Trade-off non-empty (match the bullet, not substrings of other words)
            tradeoff_m = _re.search(r"trade-?off[^\n]*:?\s*([^\n]+)", body)
            tradeoff = (tradeoff_m.group(1).strip() if tradeoff_m else "")
            if not tradeoff:
                c11_issues.append({"modification": mname, "reason": "trade_off_empty"})
        results["c11_design_decisions_audit"] = len(c11_issues) == 0
        if c11_issues:
            results["c11_issues"] = c11_issues[:10]  # truncate for report
    except FileNotFoundError:
        results["c11_design_decisions_audit"] = False
        results["c11_issues"] = [{"reason": f"file_not_found: {dd_path}"}]

    # C12: Role-authorization cross-checks against org_hierarchy.json policy.
    # Three hard-coded policy rules (from audit_post_mortem.md):
    #   (a) complianceOfficer may not assign any task inside compliance_subprocess
    #       that writes a decision-related variable;
    #   (b) peerCategoryOwner_Beauty may not be assignee of temporary_unified_decision;
    #   (c) mbTeamLead may not simultaneously own all three execution-chain
    #       anchors (team_drafts_initial_plan, team_joint_mature_plan_preparation,
    #       team_executes_plan).
    c12_violations: List[Dict[str, str]] = []
    # (b) peer owner not on unified decision
    if uid_tid:
        if graph.get_assignee(uid_tid) == "peerCategoryOwner_Beauty":
            c12_violations.append({
                "rule": "peer_owner_not_on_unified_decision",
                "task": uid_tid,
            })
    # (c) team lead not on all three
    exec_chain = ["team_drafts_initial_plan",
                  "team_joint_mature_plan_preparation",
                  "team_executes_plan"]
    team_lead_count = 0
    for eid in exec_chain:
        if eid in graph.elements and graph.get_assignee(eid) == "mbTeamLead":
            team_lead_count += 1
    if team_lead_count >= 3:
        c12_violations.append({
            "rule": "team_lead_not_on_all_three_exec_anchors",
            "team_lead_count": str(team_lead_count),
        })
    # (a) complianceOfficer inside compliance_subprocess writing decision vars
    if csp and csp in graph.elements:
        csp_elem = graph.get_element(csp)
        if csp_elem is not None:
            for child in csp_elem.iter():
                if graph._local(child.tag) == "userTask":
                    cid = child.get("id", "")
                    if not cid:
                        continue
                    if graph.get_assignee(cid) != "complianceOfficer":
                        continue
                    # Does this task write a decision-related formProperty?
                    decision_kw = ("decision", "outcome", "approv", "reject", "tier")
                    wrote_decision = False
                    for sub in child.iter():
                        if graph._local(sub.tag) == "formProperty":
                            pid = (sub.get("id") or "").lower()
                            writable = (sub.get("writable") or "").strip().lower()
                            if writable in ("true", "") and any(k in pid for k in decision_kw):
                                wrote_decision = True
                                break
                    if wrote_decision:
                        c12_violations.append({
                            "rule": "compliance_officer_writes_decision",
                            "task": cid,
                        })
    c12_ok = len(c12_violations) == 0
    results["c12_role_conflict_constraints"] = c12_ok
    if c12_violations:
        results["c12_violations"] = c12_violations

    return results


def _check_logistics_notified_only(graph: BPMNGraph) -> bool:
    """Rule 8a: no inbound sequence flow from main-flow decision targets logistics."""
    # Inbound flows to logistics_capacity_confirmation must come from a parallel
    # gateway (spawn) only, not from any main-flow decision.
    inbound = graph.in_flows.get("logistics_capacity_confirmation", [])
    for entry in inbound:
        src = entry.get("source", "")
        if src and graph.get_type(src) == "parallelGateway":
            continue  # acceptable
        if src == "startEvent" or src == "team_drafts_initial_plan":
            continue  # acceptable direct original spawn
        # Any decision gateway or main-flow userTask as source violates rule 8a.
        if graph.get_type(src) in ("exclusiveGateway", "inclusiveGateway"):
            return False
    return True


def _check_logistics_outgoing_safe(graph: BPMNGraph) -> bool:
    """Rule 8a structural: logistics_capacity_confirmation's outgoing flows
    must only converge (parallel join) or terminate; they must never route
    into a main-flow decision gateway or non-logistics userTask."""
    lcc = "logistics_capacity_confirmation"
    if lcc not in graph.elements:
        # Element removed - Layer D d05 will catch preservation; c06 coarse-passes.
        return True
    for f in graph.outgoing_flows(lcc):
        tgt = f.get("target", "")
        if not tgt:
            continue
        t_type = graph.get_type(tgt)
        if t_type == "parallelGateway":
            continue
        if t_type in ("endEvent", "terminateEventDefinition"):
            continue
        # Same-lane targets (logistics lane) are acceptable.
        lane = graph.get_lane_for_node(tgt)
        lane_label = ""
        if lane:
            lane_label = (lane.get("lane_name", "") + " " + lane.get("lane_id", "")).lower()
        if "logistics" in lane_label:
            continue
        # Any decision gateway or main-flow element as downstream violates the
        # notified-only contract.
        if graph.is_decision_gateway(tgt):
            return False
        if t_type == "userTask":
            return False
    return True


def _check_rationale_on_overrides(
    graph: BPMNGraph, discovered: Dict[str, Optional[str]]
) -> bool:
    """Rule 17: every manual override task has an out_rationale-like formProperty."""
    override_tasks = [
        discovered.get("unified_decision_task"),
        discovered.get("senior_lead_escalation_task"),
        discovered.get("emergency_delist_task"),
        discovered.get("brand_risk_reassessment_task"),
    ]
    for tid in override_tasks:
        if tid is None:
            continue
        props = graph.get_form_properties(tid)
        has_rationale = any(RATIONALE_RE.search(pid) for pid in props)
        if not has_rationale:
            return False
    return True


def _check_rationale_writable(
    graph: BPMNGraph, discovered: Dict[str, Optional[str]]
) -> bool:
    """C08: every rationale-matching formProperty on an override task must be
    writable='true'. Read-only rationales can't capture the override rationale
    the rule is supposed to enforce."""
    override_tasks = [
        discovered.get("unified_decision_task"),
        discovered.get("senior_lead_escalation_task"),
        discovered.get("emergency_delist_task"),
        discovered.get("brand_risk_reassessment_task"),
    ]
    for tid in override_tasks:
        if tid is None:
            continue
        props = graph.get_form_properties(tid)
        rationale_props = {pid: meta for pid, meta in props.items() if RATIONALE_RE.search(pid)}
        if not rationale_props:
            return False
        writable_present = any(
            (meta.get("writable") or "").strip().lower() in ("true", "")
            for meta in rationale_props.values()
        )
        if not writable_present:
            return False
    return True


# ============================================================================
# SECTION D - ANTI-GAMING CHECKS (8 forbidden patterns)
# ============================================================================


def check_anti_gaming(graph: BPMNGraph, discovered: Dict[str, Optional[str]]) -> Dict:
    results: Dict[str, object] = {}

    # D1: Single role absorbing >=2 of {execution, coordination, decision}
    exec_task = "team_executes_plan"
    coord_tasks = []
    for lid, ldata in graph.lanes.items():
        if "coordination" in ldata["name"].lower() or "coordination" in lid.lower():
            for n in ldata["nodes"]:
                if graph.get_type(n) == "userTask":
                    coord_tasks.append(n)
    if not coord_tasks:
        mct = discovered.get("merchant_coordination_task")
        cct = discovered.get("campaign_coordination_task")
        coord_tasks = [t for t in [mct, cct] if t]
    decision_task = discovered.get("unified_decision_task")

    exec_role = graph.get_assignee(exec_task) if exec_task in graph.elements else ""
    coord_roles = {graph.get_assignee(t) for t in coord_tasks if t}
    coord_roles.discard("")
    decision_role = graph.get_assignee(decision_task) if decision_task else ""

    # Count in how many sets the same role appears
    all_roles = set()
    role_in_sets: Dict[str, Set[str]] = defaultdict(set)
    if exec_role:
        role_in_sets[exec_role].add("execution")
    for cr in coord_roles:
        role_in_sets[cr].add("coordination")
    if decision_role:
        role_in_sets[decision_role].add("decision")

    max_overlap = max((len(sets) for sets in role_in_sets.values()), default=0)
    results["d01_single_role_absorption"] = max_overlap < 2
    if max_overlap >= 2:
        results["d01_issue"] = {
            role: sorted(s) for role, s in role_in_sets.items() if len(s) >= 2
        }

    # D2: No original approval/decision gateway removed (anchors preserved)
    results["d02_anchors_preserved"] = (
        "team_drafts_initial_plan" in graph.elements
        and "team_executes_plan" in graph.elements
    )

    # D3: <=30% of decisions routed to senior lead alone
    total_decision_flows = 0
    senior_terminal_flows = 0
    senior_task = discovered.get("senior_lead_escalation_task")
    for eid in graph.elements:
        if not graph.is_decision_gateway(eid):
            continue
        for f in graph.outgoing_flows(eid):
            total_decision_flows += 1
            tgt = f.get("target", "")
            if tgt == senior_task:
                senior_terminal_flows += 1
            elif senior_task and graph.is_reachable(tgt, senior_task, max_depth=3):
                # near-direct routing
                senior_terminal_flows += 1
    ratio = senior_terminal_flows / total_decision_flows if total_decision_flows > 0 else 0
    results["d03_senior_escalation_ratio_le_30"] = ratio <= 0.30
    results["d03_senior_ratio"] = round(ratio, 3)
    results["d03_senior_flows"] = senior_terminal_flows
    results["d03_total_decision_flows"] = total_decision_flows

    # D4: A/B / free-trial / new-merchant not deleted or permanently-false-guarded
    ab_ok = True
    trial_ok = True
    merchant_ok = True
    has_ab = False
    has_trial = False
    has_merchant = False
    for eid in graph.elements:
        if graph.get_type(eid) != "userTask":
            continue
        name_blob = (graph.get_task_name(eid) + " " + eid + " " + graph.get_documentation(eid)).lower()
        if "ab_test" in name_blob or "a/b" in name_blob:
            has_ab = True
            # Check guard conditions - is there any ${false} or hardcoded-off upstream?
            ab_ok = not _has_permanent_false_guard(graph, eid)
        if "free_trial" in name_blob or "free trial" in name_blob:
            has_trial = True
            trial_ok = not _has_permanent_false_guard(graph, eid)
        if "new_merchant" in name_blob or "new merchant" in name_blob:
            has_merchant = True
            merchant_ok = not _has_permanent_false_guard(graph, eid)
    results["d04_ab_test_not_shutdown"] = has_ab and ab_ok
    results["d04_free_trial_not_shutdown"] = has_trial and trial_ok
    results["d04_new_merchant_not_shutdown"] = has_merchant and merchant_ok

    # D5: Logistics lane not modified (IDs preserved, no main-flow inbound)
    results["d05_logistics_not_modified"] = (
        "logistics_lane" in graph.lanes
        and "logistics_capacity_confirmation" in graph.elements
        and _check_logistics_notified_only(graph)
    )

    # D6: Multi-person split must have convergence node (parallel join or unified decision)
    has_convergence = False
    for eid in graph.elements:
        if graph.is_parallel_gateway(eid) and len(graph.in_flows.get(eid, [])) >= 2:
            has_convergence = True
            break
    if not has_convergence and discovered.get("unified_decision_task"):
        # Unified decision node also counts as convergence
        uid = discovered["unified_decision_task"]
        if len(graph.in_flows.get(uid, [])) >= 1:
            has_convergence = True
    results["d06_convergence_node_present"] = has_convergence

    # D7: Multi-factor gateway (not single-metric) for key decisions
    mfg = discovered.get("multi_factor_gateway")
    factor_hits = 0
    if mfg:
        text = gateway_conditions_text(graph, mfg).lower()
        factor_hits = sum(1 for f in FACTOR_KEYWORDS if f in text)
    results["d07_multi_factor_gateway_ge4"] = factor_hits >= 4

    # D8: Temporary replacement not doing execution + coordination + full decision
    # If any assignee appears on execution AND coordination AND decision tasks, fail.
    results["d08_no_temp_replacement_doing_all"] = max_overlap < 3

    # D11 (bonus): No tautology / hardcoded-true-only conditions
    tautology_issues: List[str] = []
    taut_patterns = {"true", "false", "1==1", "1 == 1", "!false"}
    for eid in graph.elements:
        if not graph.is_decision_gateway(eid):
            continue
        flows = graph.outgoing_flows(eid)
        if not flows:
            continue
        all_taut = True
        has_cond = False
        for f in flows:
            cond = f.get("condition", "").strip()
            if not cond:
                all_taut = False
                continue
            has_cond = True
            inner = cond
            if inner.startswith("${") and inner.endswith("}"):
                inner = inner[2:-1]
            inner = inner.strip().lower()
            if inner not in taut_patterns:
                all_taut = False
        if has_cond and all_taut:
            tautology_issues.append(eid)
    results["d09_no_tautology_gateways"] = len(tautology_issues) == 0
    if tautology_issues:
        results["d09_tautology_issues"] = tautology_issues

    # D10: Variable-based conditions on every new gateway
    non_var_issues: List[str] = []
    for eid in graph.elements:
        if not graph.is_decision_gateway(eid):
            continue
        for f in graph.outgoing_flows(eid):
            cond = f.get("condition", "").strip()
            if not cond:
                continue
            vars_ = extract_vars_from_condition(cond)
            if not vars_:
                non_var_issues.append(f"{eid}:{f.get('flow_id', '')}")
    results["d10_all_conditions_use_variables"] = len(non_var_issues) == 0
    if non_var_issues:
        results["d10_non_variable_conditions"] = non_var_issues

    return results


def _has_permanent_false_guard(graph: BPMNGraph, task_id: str) -> bool:
    """Check if any upstream flow into task_id is guarded by ${false} or equivalent."""
    inbound = graph.in_flows.get(task_id, [])
    for entry in inbound:
        cond = entry.get("condition", "").strip()
        if not cond:
            continue
        inner = cond
        if inner.startswith("${") and inner.endswith("}"):
            inner = inner[2:-1]
        inner = inner.strip().lower()
        if inner in ("false", "!true", "1==0", "1 == 0", "0"):
            return True
    return False


# ============================================================================
# SECTION E - SOLUTION FINGERPRINT MATCH
# ============================================================================


def check_fingerprint(graph: BPMNGraph, discovered: Dict[str, Optional[str]]) -> Dict:
    """Section E: Match against Solution A/B/C fingerprints or accept novel-qualifying.

    Fingerprints:
      A - Peer-Owner-Led Committee: peer-owner lane present + unified-decision
          committee role + senior escalation bounded to critical tier
      B - Senior-Lead-Light Chair: senior lead chairs async via timer + no
          dedicated peer-owner-led committee + cross-category ops lane
      C - Team-Internal-Committee-Led: internal team committee (team lead + BA
          + merchant ops lead) drafts plan + peer owner only standardized +
          separate terminal escalation

    A submission passes Section E if it matches at least one fingerprint OR
    if all hard constraints in Sections A/C/D already pass (novel-qualifying).
    """
    results: Dict[str, object] = {}

    # Fingerprint A: Peer-Owner-Led Committee
    has_peer_lane = any(
        "peer" in ldata["name"].lower() or "peer" in lid.lower()
        for lid, ldata in graph.lanes.items()
    )
    uid = discovered.get("unified_decision_task")
    uid_assignee = graph.get_assignee(uid).lower() if uid else ""
    committee_assignment = (
        "interim" in uid_assignee
        or "committee" in uid_assignee
        or "chair" in uid_assignee
    )
    etg = discovered.get("escalation_tier_gateway")
    has_tier_gate = False
    if etg:
        text = gateway_conditions_text(graph, etg).lower()
        has_tier_gate = "critical" in text or "tier" in text
    fingerprint_a = has_peer_lane and committee_assignment and has_tier_gate

    # Fingerprint B: Senior-Lead-Light Chair (timer-driven async chair)
    tb = discovered.get("timer_boundary_event")
    has_timer = tb is not None
    senior_is_chair = uid and "senior" in uid_assignee
    fingerprint_b = has_timer and senior_is_chair

    # Fingerprint C: Team-Internal-Committee-Led
    internal_committee = uid and (
        "team" in uid_assignee
        or "business_analyst" in uid_assignee
        or "merchant_ops" in uid_assignee
    )
    separate_term_end = TERMINAL_ESCALATION_END in graph.end_events
    fingerprint_c = internal_committee and separate_term_end and has_peer_lane

    results["fingerprint_A_peer_owner_led_committee"] = fingerprint_a
    results["fingerprint_B_senior_lead_light_chair"] = fingerprint_b
    results["fingerprint_C_team_internal_committee"] = fingerprint_c
    results["any_fingerprint_matched"] = fingerprint_a or fingerprint_b or fingerprint_c

    # E06: If fingerprint A matches, the unified-decision committee task must
    # capture a decision output via a writable formProperty (out_unified_decision,
    # out_escalation_tier, etc.). Raw role-string keyword match without a
    # capturable decision output is a cosmetic fingerprint.
    e06_ok = True
    e06_detail: Dict[str, object] = {}
    if fingerprint_a:
        if not uid:
            e06_ok = False
            e06_detail["reason"] = "no_unified_decision_task"
        else:
            props = graph.get_form_properties(uid)
            decision_props = [
                pid for pid, meta in props.items()
                if (
                    pid.lower().startswith("out_")
                    and (meta.get("writable") or "").strip().lower() in ("true", "")
                )
                and ("decision" in pid.lower() or "tier" in pid.lower() or "outcome" in pid.lower())
            ]
            e06_detail["decision_props"] = decision_props
            if not decision_props:
                e06_ok = False
                e06_detail["reason"] = "no_writable_decision_output"
    results["e06_fingerprint_has_committee_form"] = e06_ok
    if e06_detail:
        results["e06_detail"] = e06_detail

    # E07: Fingerprint A concrete governance loop topology. The path
    # peer_lane_task -> gw_parallel_review_join -> temporary_unified_decision
    # -> gw_escalation_tier -> (team_joint_mature_plan_preparation OR
    # escalation_final_end) must exist as directed edges in the BPMN graph.
    # Keyword-only fingerprint match no longer suffices.
    e07_ok = True
    e07_detail: Dict[str, object] = {}
    if fingerprint_a:
        prj = discovered.get("parallel_review_join")
        uid_tid = discovered.get("unified_decision_task")
        etg = discovered.get("escalation_tier_gateway")
        jp_tid = discovered.get("joint_plan_task")
        # Require peer lane task exists upstream of prj
        peer_to_join = False
        if prj:
            for f in graph.in_flows.get(prj, []):
                src = f.get("source", "")
                if src:
                    lane = graph.get_lane_for_node(src)
                    if lane and "peer" in (lane.get("lane_name", "") + lane.get("lane_id", "")).lower():
                        peer_to_join = True
                        break
        join_to_uid = prj and uid_tid and graph.is_reachable(prj, uid_tid, max_depth=3)
        uid_to_etg = uid_tid and etg and graph.is_reachable(uid_tid, etg, max_depth=4)
        etg_to_loop = False
        if etg:
            targets_to_check = [jp_tid, TERMINAL_ESCALATION_END]
            for tgt in targets_to_check:
                if tgt and graph.is_reachable(etg, tgt, max_depth=5):
                    etg_to_loop = True
                    break
        e07_detail = {
            "peer_to_join": peer_to_join,
            "join_to_uid": bool(join_to_uid),
            "uid_to_etg": bool(uid_to_etg),
            "etg_to_loop": etg_to_loop,
        }
        e07_ok = all(e07_detail.values())
    results["e07_fingerprint_loop_topology"] = e07_ok
    if e07_detail:
        results["e07_detail"] = e07_detail

    return results


# ============================================================================
# MAIN EVALUATION
# ============================================================================


def evaluate(args) -> Dict:
    report: Dict[str, object] = {
        "task": "Z Global M&B Category Temporary Governance Restructuring (L3)",
        "level": "L3",
        "process_key": MODIFIED_PROCESS_KEY,
        "approach": "Self-contained anchor + topology + fingerprint validation",
        "sections": {},
    }

    try:
        tree = LET.parse(args.bpmn)
        root = tree.getroot()
    except LET.XMLSyntaxError as e:  # type: ignore[attr-defined]
        print(f"FATAL: Could not parse BPMN: {e}")
        report["sections"]["A_structural"] = {"error": str(e), "all_pass": False}
        report["overall_pass"] = False
        return report
    except Exception as e:
        # xml.etree.ElementTree fallback ParseError
        if hasattr(LET, "ParseError") and isinstance(e, LET.ParseError):  # type: ignore
            print(f"FATAL: Could not parse BPMN: {e}")
            report["sections"]["A_structural"] = {"error": str(e), "all_pass": False}
            report["overall_pass"] = False
            return report
        raise

    graph = BPMNGraph(root)
    discovered = discover_nodes(graph)

    # --- Section A: Structural Topology ---
    print("=" * 60)
    print("A. STRUCTURAL TOPOLOGY (50 checks, L3 + a37-a51)")
    print("=" * 60)
    try:
        structural, struct_details = check_structural(graph)
        for name in sorted(structural.keys()):
            val = structural[name]
            if isinstance(val, bool):
                status = "PASS" if val else "FAIL"
                print(f"  [{status}] {name}")
        a_bool_checks = [v for v in structural.values() if isinstance(v, bool)]
        a_pass = all(a_bool_checks)
        a_passed_count = sum(1 for v in a_bool_checks if v)
        report["sections"]["A_structural"] = {
            "checks": structural,
            "details": struct_details,
            "passed": a_passed_count,
            "total": len(a_bool_checks),
            "all_pass": a_pass,
        }
        print(f"\n  Result: {a_passed_count}/{len(a_bool_checks)} passed")
        print(f"  Section A: {'PASS' if a_pass else 'FAIL'}")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        report["sections"]["A_structural"] = {"error": str(e), "all_pass": False}
        a_pass = False

    # --- Section B: Runtime Scenarios ---
    print("\n" + "=" * 60)
    print("B. RUNTIME SCENARIOS (>=95% required AND anti_gaming mandatory, 60 scenarios for L3 hardened)")
    print("=" * 60)
    test_check = check_test_results(args.results)
    report["sections"]["B_scenarios"] = test_check
    print(f"  Passed: {test_check.get('passed_scenarios', 0)}/{test_check.get('total_scenarios', 0)}")
    print(f"  Pass rate: {test_check.get('pass_rate', 0):.1%}")
    b_pass = test_check.get("meets_threshold", False)
    print(f"  Section B: {'PASS' if b_pass else 'FAIL'}")

    # --- Section C: Compliance Coverage + Cross-Validation ---
    print("\n" + "=" * 60)
    print("C. COMPLIANCE COVERAGE (18 mods + 30 rules + c05-c12 + cross_rule 27-30)")
    print("=" * 60)
    compliance = check_compliance(args.structural, args.rules, graph, discovered)
    report["sections"]["C_compliance"] = compliance
    sc = compliance.get("structural_coverage", 0)
    rc = compliance.get("rules_coverage", 0)
    mods_covered = len(REQUIRED_MODIFICATIONS) - len(compliance.get("structural_missing", []))
    rules_covered = len(REQUIRED_RULES) - len(compliance.get("rules_missing", []))
    print(f"  Modifications: {sc:.0%} ({mods_covered}/{len(REQUIRED_MODIFICATIONS)})")
    if compliance.get("structural_missing"):
        print(f"    Missing: {compliance['structural_missing']}")
    print(f"  Rules: {rc:.0%} ({rules_covered}/{len(REQUIRED_RULES)})")
    if compliance.get("rules_missing"):
        print(f"    Missing: {compliance['rules_missing']}")
    c_extra_checks = [
        ("cross_rule_8a_logistics_notified_only", compliance.get("cross_rule_8a_logistics_notified_only")),
        ("cross_rule_8a_logistics_structural", compliance.get("cross_rule_8a_logistics_structural")),
        ("cross_rule_17_rationale_on_overrides", compliance.get("cross_rule_17_rationale_on_overrides")),
        ("cross_rule_13_timer_pt48h", compliance.get("cross_rule_13_timer_pt48h")),
        ("cross_rule_6_multi_factor_ge4", compliance.get("cross_rule_6_multi_factor_ge4")),
        ("cross_rule_27_brand_risk_before_decision", compliance.get("cross_rule_27_brand_risk_before_decision")),
        ("cross_rule_28_kill_switch_gating", compliance.get("cross_rule_28_kill_switch_gating")),
        ("cross_rule_29_senior_role_separation", compliance.get("cross_rule_29_senior_role_separation")),
        ("cross_rule_30_compliance_latch_explicit", compliance.get("cross_rule_30_compliance_latch_explicit")),
        ("c05_rule_to_bpmn_binding", compliance.get("c05_rule_to_bpmn_binding")),
        ("c07_modification_to_bpmn_binding", compliance.get("c07_modification_to_bpmn_binding")),
        ("c08_rationale_form_writable", compliance.get("c08_rationale_form_writable")),
        ("c09_rule_priority_declared", compliance.get("c09_rule_priority_declared")),
        ("c10_priority_chain_addresses_conflicts", compliance.get("c10_priority_chain_addresses_conflicts")),
        ("c11_design_decisions_audit", compliance.get("c11_design_decisions_audit")),
        ("c12_role_conflict_constraints", compliance.get("c12_role_conflict_constraints")),
    ]
    for name, val in c_extra_checks:
        status = "PASS" if val else "FAIL"
        print(f"  [{status}] {name}")
    c_pass = (
        sc == 1.0
        and rc == 1.0
        and all(v for _, v in c_extra_checks)
    )
    print(f"  Section C: {'PASS' if c_pass else 'FAIL'}")

    # --- Section D: Anti-Gaming ---
    print("\n" + "=" * 60)
    print("D. ANTI-GAMING (10 forbidden-pattern check groups, 12 boolean results)")
    print("=" * 60)
    anti = check_anti_gaming(graph, discovered)
    report["sections"]["D_anti_gaming"] = {"checks": anti}
    d_bool_checks = [v for k, v in anti.items() if isinstance(v, bool)]
    for name in sorted(anti.keys()):
        val = anti[name]
        if isinstance(val, bool):
            status = "PASS" if val else "FAIL"
            print(f"  [{status}] {name}")
    d_pass = all(d_bool_checks)
    report["sections"]["D_anti_gaming"]["all_pass"] = d_pass
    print(f"  Section D: {'PASS' if d_pass else 'FAIL'}")

    # --- Section E: Solution Fingerprint Match ---
    print("\n" + "=" * 60)
    print("E. SOLUTION FINGERPRINT (A/B/C) OR NOVEL-QUALIFYING")
    print("=" * 60)
    fp = check_fingerprint(graph, discovered)
    report["sections"]["E_fingerprint"] = fp
    for name in sorted(fp.keys()):
        val = fp[name]
        if isinstance(val, bool):
            status = "PASS" if val else "FAIL"
            print(f"  [{status}] {name}")
    # Novel-qualifying: if A/C/D all pass, fingerprint considered satisfied
    # E05: Tighten novel-qualifying escape hatch. Previously A+C+D was enough;
    # now requires B (runtime) too so a submission that passes structural/compliance
    # but fails runtime cannot slip through Section E via novel-qualifying.
    novel_qualifying = a_pass and b_pass and c_pass and d_pass
    # e06 and e07 both must hold: if fingerprint A was claimed, committee
    # form must capture a writable decision output (e06), AND the concrete
    # governance loop topology must exist (e07). Either failing vetoes E.
    e06_ok = fp.get("e06_fingerprint_has_committee_form", True)
    e07_ok = fp.get("e07_fingerprint_loop_topology", True)
    e_pass = (
        (fp.get("any_fingerprint_matched", False) or novel_qualifying)
        and e06_ok and e07_ok
    )
    report["sections"]["E_fingerprint"]["novel_qualifying"] = novel_qualifying
    report["sections"]["E_fingerprint"]["all_pass"] = e_pass
    print(f"  Novel-qualifying (A+C+D all pass): {'YES' if novel_qualifying else 'NO'}")
    print(f"  Section E: {'PASS' if e_pass else 'FAIL'}")

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print("FINAL SCORE (L3)")
    print("=" * 60)
    sections_pass = {
        "A_structural": a_pass,
        "B_scenarios": b_pass,
        "C_compliance": c_pass,
        "D_anti_gaming": d_pass,
        "E_fingerprint": e_pass,
    }
    for section, passed in sections_pass.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {section}")

    all_pass = all(sections_pass.values())
    report["overall_pass"] = all_pass
    report["sections_summary"] = sections_pass
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="L3 Evaluator for Case 2: Z Global M&B Category "
                    "Temporary Governance Restructuring"
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
    report = evaluate(args)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}")

    sys.exit(0 if report.get("overall_pass") else 1)


if __name__ == "__main__":
    main()
