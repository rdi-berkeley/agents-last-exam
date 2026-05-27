"""Scorer for computing_math/incident_response_1.

Implements the weighted rubric defined in TASK_INTAKE.md. Pure-Python — no
external deps. Importable from main.py and runnable standalone for local
sanity checks.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

HARD_GATE_SECTIONS = ("indicators_of_compromise", "compromised_hosts", "attack_timeline")

PHASE_ALIASES: dict[str, set[str]] = {
    "Reconnaissance": {"reconnaissance", "recon", "scanning"},
    "Initial Access": {"initial access", "exploitation"},
    "Privilege Escalation": {"privilege escalation", "privesc"},
    "Lateral Movement": {"lateral movement"},
    "Data Exfiltration": {"data exfiltration", "exfiltration", "exfil"},
    "Command and Control": {"command and control", "c2", "c&c"},
    "Persistence": {"persistence"},
}

REFERENCE_IOCS: set[tuple[str, str]] = {
    ("ip", "185.220.101.42"),
    ("ip", "91.234.56.78"),
    ("ip", "198.51.100.23"),
    ("domain", "evil-c2.darknet.io"),
    ("cve", "CVE-2024-53677"),
    ("ja3", "abc123def456"),
    ("rogue_account", "svc_backup"),
    ("persistence", "crontab"),
}

REFERENCE_PHASE_EVENTS: dict[str, set[str]] = {
    "Reconnaissance": {"EVT-0001", "EVT-0002", "EVT-0003", "EVT-0004"},
    "Initial Access": {"EVT-0006", "EVT-0007", "EVT-0008"},
    "Privilege Escalation": {"EVT-0009", "EVT-0010", "EVT-0011"},
    "Lateral Movement": {"EVT-0012", "EVT-0013", "EVT-0014", "EVT-0015", "EVT-0016", "EVT-0017"},
    "Data Exfiltration": {"EVT-0019", "EVT-0020", "EVT-0024"},
    "Command and Control": {"EVT-0021", "EVT-0023", "EVT-0024", "EVT-0025"},
    "Persistence": {"EVT-0027"},
}

REFERENCE_HOST_IPS: set[str] = {"10.0.1.50", "10.0.2.100", "10.0.3.200"}

HOST_METHOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "10.0.1.50": ("Struts", "CVE-2024-53677"),
    "10.0.2.100": ("Pass-the-Hash", "NTLM"),
    "10.0.3.200": ("RDP", "svc_backup"),
}

REFERENCE_EXFIL_DESTS: set[str] = {"91.234.56.78", "evil-c2.darknet.io"}
REFERENCE_EXFIL_SOURCE_HOST: str = "10.0.3.200"
REFERENCE_EXFIL_TOTAL_GB: float = 5.17
EXFIL_VOLUME_TOLERANCE: float = 0.20

REFERENCE_BENIGN_EVENTS: set[str] = {"EVT-0005", "EVT-0018", "EVT-0022", "EVT-0026", "EVT-0028"}

COMPONENT_WEIGHTS: dict[str, float] = {
    "ioc_extraction": 0.25,
    "attack_timeline": 0.20,
    "compromised_hosts": 0.15,
    "data_exfiltration": 0.15,
    "report_completeness": 0.15,
    "benign_filtering": 0.10,
}

_DEST_STRIP_PORT = re.compile(r":\d+$")


def _f1(predicted: set, reference: set) -> float:
    if not predicted and not reference:
        return 1.0
    tp = len(predicted & reference)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(reference)
    return 2 * precision * recall / (precision + recall)


def _as_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _canonicalize_phase(raw: Any) -> str | None:
    text = _as_str(raw).lower()
    if not text:
        return None
    for canonical, aliases in PHASE_ALIASES.items():
        if text in aliases:
            return canonical
    return None


def _normalize_dest(value: Any) -> str:
    text = _as_str(value)
    return _DEST_STRIP_PORT.sub("", text)


def _iter_items(value: Any) -> Iterable[Any]:
    return value if isinstance(value, list) else []


def _extract_ioc_pairs(iocs: Any) -> set[tuple[str, str]]:
    if not isinstance(iocs, dict):
        return set()
    pairs: set[tuple[str, str]] = set()
    for entry in _iter_items(iocs.get("malicious_ips")):
        if isinstance(entry, dict):
            ip = _as_str(entry.get("ip"))
            if ip:
                pairs.add(("ip", ip))
    for entry in _iter_items(iocs.get("malicious_domains")):
        if isinstance(entry, dict):
            dom = _as_str(entry.get("domain"))
            if dom:
                pairs.add(("domain", dom))
    for entry in _iter_items(iocs.get("cves_exploited")):
        if isinstance(entry, dict):
            cve = _as_str(entry.get("cve"))
            if cve:
                pairs.add(("cve", cve))
    for entry in _iter_items(iocs.get("ja3_fingerprints")):
        if isinstance(entry, dict):
            ja3 = _as_str(entry.get("hash"))
            if ja3:
                pairs.add(("ja3", ja3))
    for entry in _iter_items(iocs.get("rogue_accounts")):
        if isinstance(entry, dict):
            username = _as_str(entry.get("username"))
            if username:
                pairs.add(("rogue_account", username))
    for entry in _iter_items(iocs.get("persistence_mechanisms")):
        if isinstance(entry, dict):
            mech_type = _as_str(entry.get("type"))
            if mech_type:
                pairs.add(("persistence", mech_type))
    return pairs


def _score_ioc(report: dict[str, Any]) -> float:
    predicted = _extract_ioc_pairs(report.get("indicators_of_compromise"))
    return _f1(predicted, REFERENCE_IOCS)


def _score_timeline(report: dict[str, Any]) -> dict[str, float]:
    timeline = report.get("attack_timeline")
    phases = list(_iter_items(timeline))
    predicted_by_canonical: dict[str, list[dict[str, Any]]] = {}
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        canonical = _canonicalize_phase(phase.get("phase"))
        if canonical is None:
            continue
        predicted_by_canonical.setdefault(canonical, []).append(phase)
    predicted_set = set(predicted_by_canonical.keys())
    reference_set = set(REFERENCE_PHASE_EVENTS.keys())
    f1_phase = _f1(predicted_set, reference_set)
    matched = predicted_set & reference_set
    if matched:
        correct = 0
        for canonical in matched:
            ref_events = REFERENCE_PHASE_EVENTS[canonical]
            hit = False
            for phase in predicted_by_canonical[canonical]:
                events = phase.get("evidence_events")
                if not isinstance(events, list):
                    continue
                if any(_as_str(ev) in ref_events for ev in events):
                    hit = True
                    break
            if hit:
                correct += 1
        evidence_score = correct / len(matched)
    else:
        evidence_score = 0.0
    combined = 0.5 * f1_phase + 0.5 * evidence_score
    return {"phase_f1": f1_phase, "evidence_score": evidence_score, "score": combined}


def _score_hosts(report: dict[str, Any]) -> dict[str, float]:
    hosts = report.get("compromised_hosts")
    predicted_by_ip: dict[str, list[dict[str, Any]]] = {}
    for host in _iter_items(hosts):
        if not isinstance(host, dict):
            continue
        ip = _as_str(host.get("ip"))
        if ip:
            predicted_by_ip.setdefault(ip, []).append(host)
    predicted_ips = set(predicted_by_ip.keys())
    f1_ips = _f1(predicted_ips, REFERENCE_HOST_IPS)
    matched = predicted_ips & REFERENCE_HOST_IPS
    if matched:
        correct = 0
        for ip in matched:
            keywords = HOST_METHOD_KEYWORDS.get(ip, ())
            hit = False
            for host in predicted_by_ip[ip]:
                method = _as_str(host.get("compromise_method")).lower()
                if method and any(kw.lower() in method for kw in keywords):
                    hit = True
                    break
            if hit:
                correct += 1
        method_score = correct / len(matched)
    else:
        method_score = 0.0
    combined = 0.5 * f1_ips + 0.5 * method_score
    return {"ip_f1": f1_ips, "method_score": method_score, "score": combined}


def _score_exfil(report: dict[str, Any]) -> dict[str, float]:
    exfil = report.get("data_exfiltration")
    if not isinstance(exfil, dict):
        return {"volume": 0.0, "dest_f1": 0.0, "source_host": 0.0, "score": 0.0}
    total = exfil.get("total_volume_gb")
    try:
        total_val = float(total)
    except (TypeError, ValueError):
        total_val = None
    if total_val is None:
        volume_score = 0.0
    else:
        lo = REFERENCE_EXFIL_TOTAL_GB * (1.0 - EXFIL_VOLUME_TOLERANCE)
        hi = REFERENCE_EXFIL_TOTAL_GB * (1.0 + EXFIL_VOLUME_TOLERANCE)
        volume_score = 1.0 if lo <= total_val <= hi else 0.0
    dests: set[str] = set()
    for channel in _iter_items(exfil.get("channels")):
        if isinstance(channel, dict):
            dest = _normalize_dest(channel.get("destination"))
            if dest:
                dests.add(dest)
    dest_f1 = _f1(dests, REFERENCE_EXFIL_DESTS)
    source_match = 1.0 if _as_str(exfil.get("source_host")) == REFERENCE_EXFIL_SOURCE_HOST else 0.0
    combined = (volume_score + dest_f1 + source_match) / 3.0
    return {"volume": volume_score, "dest_f1": dest_f1, "source_host": source_match, "score": combined}


def _score_completeness(report: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["executive_summary_non_empty"] = bool(_as_str(report.get("executive_summary")))
    timeline = list(_iter_items(report.get("attack_timeline")))
    checks["phases_have_mitre_technique"] = bool(timeline) and all(
        isinstance(p, dict) and _as_str(p.get("mitre_technique")) for p in timeline
    )
    hosts = list(_iter_items(report.get("compromised_hosts")))
    checks["hosts_have_access_level"] = bool(hosts) and all(
        isinstance(h, dict) and _as_str(h.get("access_level")) for h in hosts
    )
    checks["hosts_have_os"] = bool(hosts) and all(
        isinstance(h, dict) and _as_str(h.get("os")) for h in hosts
    )
    iocs = report.get("indicators_of_compromise")
    tools = iocs.get("tools_used") if isinstance(iocs, dict) else None
    checks["tools_used_non_empty"] = isinstance(tools, list) and len(tools) >= 1
    exfil = report.get("data_exfiltration")
    channels = list(_iter_items(exfil.get("channels") if isinstance(exfil, dict) else None))
    checks["exfil_channels_have_volume_gb"] = bool(channels) and all(
        isinstance(c, dict) and isinstance(c.get("volume_gb"), (int, float)) for c in channels
    )
    passed = sum(1 for v in checks.values() if v)
    return {"checks": checks, "passed": passed, "total": len(checks), "score": passed / len(checks)}


def _score_benign(report: dict[str, Any]) -> float:
    raw = report.get("benign_events_excluded")
    predicted = {_as_str(ev) for ev in _iter_items(raw) if _as_str(ev)}
    return _f1(predicted, REFERENCE_BENIGN_EVENTS)


def _hard_gate_triggered(report: Any) -> tuple[bool, list[str]]:
    if not isinstance(report, dict):
        return True, ["report is not a JSON object"]
    missing = [key for key in HARD_GATE_SECTIONS if key not in report]
    return (len(missing) > 0, missing)


def score_report(agent_json_text: str) -> dict[str, Any]:
    """Score one agent output. Returns a dict with score + breakdown."""

    result: dict[str, Any] = {
        "score": 0.0,
        "hard_gate_triggered": False,
        "hard_gate_reasons": [],
        "components": {},
        "weights": COMPONENT_WEIGHTS,
    }
    try:
        report = json.loads(agent_json_text)
    except json.JSONDecodeError as exc:
        result["hard_gate_triggered"] = True
        result["hard_gate_reasons"] = [f"invalid json: {exc}"]
        return result
    gate, missing = _hard_gate_triggered(report)
    if gate:
        result["hard_gate_triggered"] = True
        reasons = ["missing section: " + key for key in missing] or ["report is not a JSON object"]
        result["hard_gate_reasons"] = reasons
        return result

    ioc_score = _score_ioc(report)
    timeline_detail = _score_timeline(report)
    hosts_detail = _score_hosts(report)
    exfil_detail = _score_exfil(report)
    completeness_detail = _score_completeness(report)
    benign_score = _score_benign(report)
    components = {
        "ioc_extraction": {"score": ioc_score},
        "attack_timeline": timeline_detail,
        "compromised_hosts": hosts_detail,
        "data_exfiltration": exfil_detail,
        "report_completeness": completeness_detail,
        "benign_filtering": {"score": benign_score},
    }
    weighted = sum(COMPONENT_WEIGHTS[name] * comp["score"] for name, comp in components.items())
    result["components"] = components
    result["score"] = max(0.0, min(1.0, weighted))
    return result


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: score_incident_report.py <agent_output.json>\n")
        return 2
    path = Path(argv[1])
    text = path.read_text(encoding="utf-8")
    report = score_report(text)
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
