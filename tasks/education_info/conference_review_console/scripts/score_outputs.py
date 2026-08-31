"""Deterministic scorer for education_info/conference_review_console.

Pure functions over file contents. Grades the console's persisted state
(state.json) plus the audit file against the generator's effective-policy
truth (reference/effective_world.json).

Published contract (input/handbook.md):
  * no state, or an assignment violating ANY effective-policy rule (including
    a missing or incorrect screening record for any active paper):
        score = invalid_screen_weight * screen_frac
  * otherwise:
        score = base + qw * q + aw * audit_f1 + cw * checklist_frac
  * never finalized -> capped at unfinalized_cap.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any


def _rows(text: str) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text.lstrip("﻿")))
    return [[c.strip() for c in row] for row in reader if any(c.strip() for c in row)]


def affinity(reviewer: dict[str, Any], paper: dict[str, Any]) -> int:
    area = 4 if paper["primary_area"] in set(reviewer["areas"]) else 0
    return area + 2 * len(set(paper["topics"]) & set(reviewer["topics"]))


def audit_f1(audit_csv: str | None, truth_triples: set[tuple[str, str, str]]) -> float:
    if not audit_csv:
        return 0.0
    rows = _rows(audit_csv)
    if not rows or rows[0] != ["reviewer_id", "paper_id", "reason_code"]:
        return 0.0
    submitted = {(r[0], r[1], r[2]) for r in rows[1:] if len(r) == 3}
    if not submitted:
        return 0.0
    tp = len(submitted & truth_triples)
    if tp == 0:
        return 0.0
    precision = tp / len(submitted)
    recall = tp / len(truth_triples)
    return 2 * precision * recall / (precision + recall)


def score_console(
    state_json: str | None,
    audit_csv: str | None,
    effective_world: dict[str, Any],
    scoring_params: dict[str, Any],
) -> dict[str, Any]:
    papers = {p["paper_id"]: p for p in effective_world["papers"]}
    reviewers = {r["reviewer_id"]: r for r in effective_world["reviewers"]}
    truth = {tuple(t) for t in effective_world["truth_triples"]}
    truth_pairs = {(a, b) for a, b, _ in truth}
    chk_truth = effective_world["checklist_answers"]
    active = [pid for pid, p in papers.items() if p["status"] == "active"]
    per_paper = int(scoring_params["reviews_per_paper"])

    out: dict[str, Any] = {"violations": [], "screen_frac": 0.0,
                           "audit_f1": 0.0, "checklist_frac": 0.0,
                           "quality": 0.0, "finalized": False}

    try:
        state = json.loads(state_json) if state_json else None
    except json.JSONDecodeError:
        state = None
    if not isinstance(state, dict):
        out["violations"] = ["no console state"]
        out["score"] = 0.0
        return out

    screenings = state.get("screenings") or {}
    n_screen_ok = 0
    for pid in active:
        rec = screenings.get(pid) or {}
        ok = (rec.get("primary_area") == papers[pid]["primary_area"]
              and rec.get("author_count") == papers[pid]["author_count"])
        if ok:
            n_screen_ok += 1
        else:
            out["violations"].append(f"{pid}: screening record missing or incorrect")
    screen_frac = n_screen_ok / len(active) if active else 0.0
    out["screen_frac"] = round(screen_frac, 6)

    assignments = state.get("assignments") or {}
    load: dict[str, int] = {}
    total = 0
    for pid, rids in assignments.items():
        if pid not in papers or papers[pid]["status"] != "active":
            out["violations"].append(f"{pid}: not an active paper")
            continue
        if len(set(rids)) != len(rids):
            out["violations"].append(f"{pid}: duplicate reviewer")
        for rid in rids:
            r = reviewers.get(rid)
            if r is None:
                out["violations"].append(f"{pid}: unknown reviewer {rid}")
                continue
            if not r["eligible"]:
                out["violations"].append(
                    f"{rid}: not eligible under effective policy (on {pid})")
            if (rid, pid) in truth_pairs:
                codes = ",".join(sorted(c for a, b, c in truth if (a, b) == (rid, pid)))
                out["violations"].append(f"conflict of interest ({codes}): {rid} on {pid}")
            load[rid] = load.get(rid, 0) + 1
            total += affinity(r, papers[pid]) if r else 0
    for pid in active:
        rids = assignments.get(pid, [])
        if len(rids) != per_paper:
            out["violations"].append(f"{pid}: has {len(rids)} reviewers, needs {per_paper}")
            continue
        known = [reviewers[r] for r in rids if r in reviewers]
        if len(known) == len(rids):
            if not any(r["senior_eff"] for r in known):
                out["violations"].append(f"{pid}: no effective-senior reviewer")
            insts = [r["canon_inst"] for r in known]
            if len(set(insts)) != len(insts):
                out["violations"].append(f"{pid}: reviewers share a canonical institution")
    for rid, n in load.items():
        r = reviewers.get(rid)
        if r and n > r["load_eff"]:
            out["violations"].append(
                f"{rid}: load {n} exceeds effective max load {r['load_eff']}")

    out["total_affinity"] = total
    out["finalized"] = bool(state.get("finalized"))

    answers = state.get("checklist_answers") or {}
    n_chk = 0
    for i, want in enumerate(chk_truth):
        try:
            if int(str(answers.get(f"q{i + 1}", "")).strip()) == int(want):
                n_chk += 1
        except ValueError:
            pass
    checklist_frac = n_chk / len(chk_truth) if chk_truth else 0.0
    out["checklist_frac"] = round(checklist_frac, 6)
    out["audit_f1"] = round(audit_f1(audit_csv, truth), 6)

    if out["violations"]:
        out["score"] = round(
            float(scoring_params["invalid_screen_weight"]) * screen_frac, 6)
        return out

    opt = int(scoring_params["optimal_affinity"])
    threshold = float(scoring_params["quality_threshold"])
    quality = (total - threshold * opt) / ((1.0 - threshold) * opt)
    quality = max(0.0, min(1.0, quality))
    out["quality"] = round(quality, 6)
    out["optimal_affinity"] = opt

    raw = (float(scoring_params["base_score"])
           + float(scoring_params["quality_weight"]) * quality
           + float(scoring_params["audit_weight"]) * out["audit_f1"]
           + float(scoring_params["checklist_weight"]) * checklist_frac)
    if not out["finalized"]:
        raw = min(raw, float(scoring_params["unfinalized_cap"]))
    out["score"] = round(raw, 6)
    return out
