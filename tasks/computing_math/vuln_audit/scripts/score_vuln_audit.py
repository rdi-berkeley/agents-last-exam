"""Deterministic scorer for computing_math/vuln_audit.

Precision-gated recall over CWE families (no LLM judge). Mirrors the local, fully-tested
task-intake/vuln_audit/grade.py logic, but operates on byte payloads so main.py's evaluate()
can call it after reading output/findings.json and the hidden ground_truth.json via the session.
"""
from __future__ import annotations

import json
import os
import re

MIN_PRECISION = 0.5


def _norm_cwe(value):
    m = re.search(r"(\d+)", str(value))
    return f"CWE-{m.group(1)}" if m else None


def _norm_file(value):
    return os.path.basename(str(value).replace("\\", "/")).lower()


def score_submission(findings_bytes: bytes, reference_bytes: bytes):
    """Return (score in [0,1], report dict). One vulnerability per file; a file is credited
    when reported with a CWE in its accepted family. Precision gate defeats CWE-spam."""
    gt = json.loads(reference_bytes.decode("utf-8"))
    accept = {
        _norm_file(v["file"]): {_norm_cwe(c) for c in v.get("accept_cwes", [v["cwe"]])}
        for v in gt["vulns"]
    }
    try:
        data = json.loads(findings_bytes.decode("utf-8"))
        candidate = set()
        for f in data["findings"]:
            fn, cwe = _norm_file(f["file"]), _norm_cwe(f.get("cwe"))
            if fn and cwe:
                candidate.add((fn, cwe))
    except Exception as exc:
        return 0.0, {"reason": f"malformed findings.json: {exc}"}

    tp = {(fn, cwe) for (fn, cwe) in candidate if fn in accept and cwe in accept[fn]}
    found = {fn for (fn, _) in tp}
    precision = len(tp) / len(candidate) if candidate else 0.0
    recall = len(found) / len(accept)

    if not candidate or precision < MIN_PRECISION:
        return 0.0, {"reason": "precision gate failed", "recall": round(recall, 3), "precision": round(precision, 3)}
    return recall, {
        "score": round(recall, 3),
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "found": sorted(found),
        "missed": sorted(f for f in accept if f not in found),
    }
