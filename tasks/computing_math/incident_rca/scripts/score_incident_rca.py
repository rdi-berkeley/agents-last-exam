"""Deterministic scorer for computing_math/incident_rca.

Anti-distractor gate (the confirm_intended idiom): credit only when the agent names the TRUE
root cause, not a planted red herring. Mirrors task-intake/incident_rca/grade.py, on byte payloads.
No LLM judge.
"""
from __future__ import annotations

import json
import re


def _norm(s) -> str:
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())


def score_submission(rca_bytes: bytes, reference_bytes: bytes):
    """Return (score in [0,1], report). GATE: rca.json well-formed AND root_cause == true cause
    (a distractor scores 0). TIERS (base 0.5): correct fix (+0.25) and decisive evidence (+0.25)."""
    gt = json.loads(reference_bytes.decode("utf-8"))
    gt_rc, gt_fix = _norm(gt["root_cause"]), _norm(gt["fix"])
    ev_sub = str(gt["evidence_substring"])

    try:
        d = json.loads(rca_bytes.decode("utf-8"))
        # A missing key must read as empty (not the truthy string "none") so the schema gate
        # below genuinely requires a non-empty root_cause AND fix.
        raw_rc, raw_fix = d.get("root_cause"), d.get("fix")
        rc = _norm(raw_rc) if raw_rc is not None else ""
        fix = _norm(raw_fix) if raw_fix is not None else ""
        evidence = str(d.get("evidence", ""))
    except Exception as exc:
        return 0.0, {"reason": f"malformed rca.json: {exc}"}

    schema_ok = bool(rc) and bool(fix)
    if not schema_ok or rc != gt_rc:
        return 0.0, {"reason": "wrong or missing root_cause (distractor or empty)", "root_cause": rc}

    score = 0.5 + (0.25 if fix == gt_fix else 0.0) + (0.25 if ev_sub in evidence else 0.0)
    return score, {
        "score": score,
        "root_cause": rc,
        "root_cause_correct": True,
        "fix_correct": fix == gt_fix,
        "evidence_ok": ev_sub in evidence,
    }
