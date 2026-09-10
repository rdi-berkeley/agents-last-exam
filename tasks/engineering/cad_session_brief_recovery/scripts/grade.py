"""Grader for engineering/cad_session_brief_recovery.

The agent watches a screen recording of an expert CAD session (input/sessions/<sid>/clip.mp4 plus the raw input-event
log) and must recover the task brief the expert was working from: output/<sid>.json with
  {"application": str, "deliverable": str, "objective": str, "requirements": [str, ...]}.
Hidden reference: the original task_desc.json and the designer's rubrics.json (R1..Rn requirements).
Scoring per session:
  app_score      1 if the named application matches the reference application family (deterministic keyword match), else 0
  coverage       for each hidden rubric a text judge decides (majority of ALE_JUDGE_VOTES) whether the recovered brief
                 covers it: "full" (1), "partial" (0.5) or "none" (0); coverage = mean over rubrics
  session score  = 0.15 * app_score + 0.85 * coverage
Task score = mean over sessions. Missing, unparsable or empty output = 0.
"""
from __future__ import annotations

import json
import os
import re

from tasks.engineering.cad_session_brief_recovery.scripts.judge_client import chat_json, votes

APP_FAMILIES = {"autocad": ["autocad"], "solidworks": ["solidworks", "solid works"], "nx": ["siemens nx", " nx", "nx "], "catia": ["catia"], "fusion": ["fusion"], "inventor": ["inventor"], "creo": ["creo"], "freecad": ["freecad"], "sketchup": ["sketchup"], "revit": ["revit"]}
SYSTEM = (
    "You are grading how well a candidate recovered a CAD task brief from a screen recording. You get the candidate's recovered "
    "brief and a list of requirement statements written by the original designer. For each requirement decide whether the "
    "candidate's brief covers it: 'full' if the candidate states the same requirement (same feature, count, arrangement or "
    "dimension where the requirement has one), 'partial' if the candidate mentions the feature but misses a stated count, "
    "arrangement or dimension, or 'none' if the candidate's brief does not mention it or contradicts it. Be strict; generic "
    'statements do not count as coverage. Answer with a JSON object {"coverage": {"R1": "full"|"partial"|"none", ...}}.'
)
W_APP, W_COV = 0.15, 0.85


def parse_brief(text: str) -> dict | None:
    try:
        d = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    reqs = d.get("requirements") or []
    if isinstance(reqs, str):
        reqs = [r for r in re.split(r"[\n;]+", reqs) if r.strip()]
    d["requirements"] = [str(r).strip() for r in reqs if str(r).strip()][:40]
    return d


def app_family(name: str) -> str | None:
    s = " " + (name or "").lower() + " "
    for fam, keys in APP_FAMILIES.items():
        if any(k in s for k in keys):
            return fam
    return None


def judge_coverage(brief: dict, rubrics: list[dict]) -> dict:
    ids = [r["id"] for r in rubrics]
    cand = json.dumps({k: brief.get(k) for k in ("application", "deliverable", "objective", "requirements")}, ensure_ascii=False, indent=1)[:6000]
    text = "Candidate's recovered brief:\n" + cand + "\n\nDesigner's requirements:\n" + "\n".join(f"{r['id']}: {r['requirement']}" for r in rubrics) + "\n\nReturn coverage for every id: " + ", ".join(ids) + "."
    score = {"full": 1.0, "partial": 0.5, "none": 0.0}; tallies = {i: [] for i in ids}; n = votes()
    for _ in range(n):
        res = chat_json(SYSTEM, text); cov = res.get("coverage", {})
        for i in ids:
            tallies[i].append(score.get(str(cov.get(i, "none")).lower(), 0.0))
    per = {i: sorted(v)[len(v) // 2] for i, v in tallies.items()}  # median vote
    return {"per_rubric": per, "coverage": sum(per.values()) / len(ids) if ids else 0.0, "n_votes": n}


def score_sessions(pred_texts: dict[str, str | None], ref_dir: str) -> dict:
    per = {}; sids = sorted(d for d in os.listdir(ref_dir) if os.path.isdir(os.path.join(ref_dir, d)))
    for sid in sids:
        rub = json.load(open(os.path.join(ref_dir, sid, "rubrics.json")))["rubrics"]; td = json.load(open(os.path.join(ref_dir, sid, "task_desc.json")))
        brief = parse_brief(pred_texts.get(sid) or "")
        if not brief or not brief["requirements"]:
            per[sid] = {"score": 0.0, "note": "missing or empty brief"}; continue
        app = 1.0 if app_family(str(brief.get("application", ""))) == app_family(td.get("application", "")) and app_family(td.get("application", "")) else 0.0
        try:
            j = judge_coverage(brief, rub)
        except Exception as exc:
            per[sid] = {"score": 0.0, "note": f"judge failed: {exc}"}; continue
        per[sid] = {"score": round(W_APP * app + W_COV * j["coverage"], 4), "app_score": app, "coverage": round(j["coverage"], 4), "per_rubric": j["per_rubric"], "n_rubrics": len(rub), "note": "ok"}
    return {"score": round(sum(v["score"] for v in per.values()) / len(per), 4) if per else 0.0, "per_session": per}
