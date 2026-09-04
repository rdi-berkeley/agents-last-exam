"""Deterministic scorer for education_info/conference_reviewer_assignment.

Pure functions over file contents (no I/O), so the module is directly
importable from host-side unit tests and from the task's ``evaluate`` hook.

Scoring contract (published to the agent in constraints.md):
  * any hard-constraint violation in assignment.csv -> 0.0
  * else: base + qw * q + aw * audit_f1, with
      q = clamp((A - t*OPT) / ((1 - t)*OPT), 0, 1)
      audit_f1 = F1 over (reviewer_id, paper_id, reason_code) triples
    All parameters (OPT, base, qw, aw, t) come from reference/scoring.json
    (base variant: base = 0.25, qw = 0.45, aw = 0.30, t = 0.9).

Entity rules mirror input/README.md exactly: institution comparisons are on
canonical institutions (via institutions.csv), coauthor names appear as
``First Last``, ``Last, First``, or ``F. Last`` and match per the documented
rule. The audit domain is available reviewers x active papers.
"""
from __future__ import annotations

import csv
import io
from typing import Any

REASON_CODES = ("AFFILIATION", "COAUTHOR", "DECLARED", "AUTHOR")


def _rows(text: str) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text.lstrip("﻿")))
    return [[c.strip() for c in row] for row in reader if any(c.strip() for c in row)]


def parse_institutions(text: str) -> dict[str, str]:
    return {row[0]: row[1] for row in _rows(text)[1:]}


def parse_reviewers(text: str, alias_to_canon: dict[str, str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _rows(text)[1:]:
        rid, name, affiliation, areas, topics, seniority, max_load, available = row
        out[rid] = {
            "name": name,
            "canon_inst": alias_to_canon[affiliation],
            "areas": set(areas.split("|")),
            "topics": set(topics.split("|")),
            "senior": seniority == "senior",
            "max_load": int(max_load),
            "available": available == "yes",
        }
    return out


def parse_submissions(text: str, alias_to_canon: dict[str, str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _rows(text)[1:]:
        pid, _title, status, primary_area, topics, authors_field = row
        authors = []
        for chunk in authors_field.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            name, _, affil = chunk.rstrip(")").rpartition(" (")
            authors.append((name.strip(), alias_to_canon[affil.strip()]))
        out[pid] = {
            "active": status == "active",
            "primary_area": primary_area,
            "topics": set(topics.split("|")),
            "authors": authors,
        }
    return out


def canon_name(name: str) -> str:
    name = " ".join(name.split())
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
        return f"{first} {last}"
    return name


def name_matches(coauthor_name: str, author_full: str) -> bool:
    cn = canon_name(coauthor_name)
    if "." in cn.split(" ", 1)[0]:
        first, last = cn.split(" ", 1)
        af, al = author_full.split(" ", 1)
        return al == last and af[0] == first[0]
    return cn == author_full


def build_conflict_codes(
    reviewers: dict[str, dict[str, Any]],
    submissions: dict[str, dict[str, Any]],
    coauthorships_csv: str,
    declared_csv: str,
    coi_year_cutoff: int,
) -> dict[tuple[str, str], set[str]]:
    """Truth triples over the audit domain: available reviewers x active papers."""
    recent: dict[str, list[str]] = {}
    for rid, name, year in (row for row in _rows(coauthorships_csv)[1:]):
        if int(year) >= coi_year_cutoff:
            recent.setdefault(rid, []).append(name)
    declared = {(row[0], row[1]) for row in _rows(declared_csv)[1:]}

    out: dict[tuple[str, str], set[str]] = {}

    def add(rid, pid, code):
        out.setdefault((rid, pid), set()).add(code)

    for rid, r in reviewers.items():
        if not r["available"]:
            continue
        my_recent = recent.get(rid, [])
        for pid, p in submissions.items():
            if not p["active"]:
                continue
            if (rid, pid) in declared:
                add(rid, pid, "DECLARED")
            for author, canon_inst in p["authors"]:
                if author == r["name"]:
                    add(rid, pid, "AUTHOR")
                if canon_inst == r["canon_inst"]:
                    add(rid, pid, "AFFILIATION")
                if any(name_matches(n, author) for n in my_recent):
                    add(rid, pid, "COAUTHOR")
    return out


def affinity(reviewer: dict[str, Any], paper: dict[str, Any]) -> int:
    area = 4 if paper["primary_area"] in reviewer["areas"] else 0
    return area + 2 * len(paper["topics"] & reviewer["topics"])


def score_audit(
    audit_csv: str | None,
    truth: dict[tuple[str, str], set[str]],
    reviewers: dict[str, dict[str, Any]],
    submissions: dict[str, dict[str, Any]],
) -> float:
    if not audit_csv:
        return 0.0
    rows = _rows(audit_csv)
    if not rows or rows[0] != ["reviewer_id", "paper_id", "reason_code"]:
        return 0.0
    submitted: set[tuple[str, str, str]] = set()
    for row in rows[1:]:
        if len(row) != 3:
            continue
        submitted.add((row[0], row[1], row[2]))
    truth_triples = {(rid, pid, code) for (rid, pid), cs in truth.items() for code in cs}
    if not submitted:
        return 0.0
    tp = len(submitted & truth_triples)
    precision = tp / len(submitted)
    recall = tp / len(truth_triples)
    if tp == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_submission(
    assignment_csv: str,
    submissions_csv: str,
    reviewers_csv: str,
    coauthorships_csv: str,
    declared_conflicts_csv: str,
    institutions_csv: str,
    scoring_params: dict[str, Any],
    audit_csv: str | None = None,
) -> dict[str, Any]:
    alias_to_canon = parse_institutions(institutions_csv)
    reviewers = parse_reviewers(reviewers_csv, alias_to_canon)
    submissions = parse_submissions(submissions_csv, alias_to_canon)
    per_paper = int(scoring_params["reviews_per_paper"])
    truth = build_conflict_codes(
        reviewers, submissions, coauthorships_csv, declared_conflicts_csv,
        int(scoring_params["coi_year_cutoff"]),
    )

    violations: list[str] = []
    rows = _rows(assignment_csv)
    if not rows or rows[0] != ["paper_id", "reviewer_id"]:
        return {"score": 0.0, "violations": ["bad_header"], "total_affinity": 0}

    pairs: list[tuple[str, str]] = []
    seen = set()
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            violations.append(f"line {i}: expected 2 fields")
            continue
        pid, rid = row
        if pid not in submissions:
            violations.append(f"line {i}: unknown paper_id {pid!r}")
            continue
        if rid not in reviewers:
            violations.append(f"line {i}: unknown reviewer_id {rid!r}")
            continue
        if not submissions[pid]["active"]:
            violations.append(f"{pid}: withdrawn paper must not be assigned")
            continue
        if not reviewers[rid]["available"]:
            violations.append(f"{rid}: unavailable reviewer assigned (to {pid})")
            continue
        if (pid, rid) in seen:
            violations.append(f"duplicate pair ({pid}, {rid})")
            continue
        seen.add((pid, rid))
        pairs.append((pid, rid))

    by_paper: dict[str, list[str]] = {
        pid: [] for pid, p in submissions.items() if p["active"]}
    load: dict[str, int] = {rid: 0 for rid in reviewers}
    for pid, rid in pairs:
        by_paper[pid].append(rid)
        load[rid] += 1

    for pid, rids in by_paper.items():
        if len(rids) != per_paper:
            violations.append(f"{pid}: has {len(rids)} reviewers, needs {per_paper}")
            continue
        if not any(reviewers[r]["senior"] for r in rids):
            violations.append(f"{pid}: no senior reviewer")
        insts = [reviewers[r]["canon_inst"] for r in rids]
        if len(set(insts)) != len(insts):
            violations.append(f"{pid}: reviewers share a canonical institution")
    for rid, n in load.items():
        if n > reviewers[rid]["max_load"]:
            violations.append(f"{rid}: load {n} exceeds max_load {reviewers[rid]['max_load']}")
    for pid, rid in pairs:
        if (rid, pid) in truth:
            codes = ",".join(sorted(truth[(rid, pid)]))
            violations.append(f"conflict of interest ({codes}): {rid} on {pid}")

    total = sum(affinity(reviewers[rid], submissions[pid]) for pid, rid in pairs)
    if violations:
        return {"score": 0.0, "violations": violations, "total_affinity": total}

    opt = int(scoring_params["optimal_affinity"])
    base = float(scoring_params["base_score"])
    qw = float(scoring_params["quality_weight"])
    aw = float(scoring_params["audit_weight"])
    threshold = float(scoring_params["quality_threshold"])
    quality = (total - threshold * opt) / ((1.0 - threshold) * opt)
    quality = max(0.0, min(1.0, quality))
    audit = score_audit(audit_csv, truth, reviewers, submissions)
    return {
        "score": round(base + qw * quality + aw * audit, 6),
        "violations": [],
        "total_affinity": total,
        "optimal_affinity": opt,
        "quality": round(quality, 6),
        "audit_f1": round(audit, 6),
    }
