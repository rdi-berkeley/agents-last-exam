"""Task: ``education_info/conference_reviewer_assignment``.

Assign 3 reviewers to each ACTIVE conference submission (150 papers, 6 of
them withdrawn; 60 reviewers, 6 unavailable) under tight capacity, four kinds
of conflict-of-interest rules, seniority coverage, and canonical-institution
diversity, maximizing topic/area affinity. Institution strings are aliases
(institutions.csv is the gazetteer) and coauthor names appear in three
documented formats, so the conflict set must be DERIVED, not string-matched.
A second deliverable (conflict_audit.csv, graded by exact triple F1)
externalizes that derivation. Any hard violation scores 0; 1.0 requires an
optimal assignment plus a perfect audit.

Inputs (staged to input/): submissions.csv, reviewers.csv, coauthorships.csv,
declared_conflicts.csv, institutions.csv, README.md, constraints.md. The agent
writes output/assignment.csv and output/conflict_audit.csv. reference/ holds
scoring.json (hidden optimum + weights) plus reference solutions, staged only
after the agent finishes.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from score_outputs import score_submission

logger = logging.getLogger(__name__)

DOMAIN_NAME = "education_info"
TASK_NAME = "conference_reviewer_assignment"
VARIANT_NAME = "base"

EVAL_INPUTS = (
    "submissions.csv", "reviewers.csv", "coauthorships.csv",
    "declared_conflicts.csv", "institutions.csv",
)
INPUT_FILES = EVAL_INPUTS + ("constraints.md", "README.md")


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def assignment_path(self) -> str:
        return f"{self.remote_output_dir}/assignment.csv"

    @property
    def audit_path(self) -> str:
        return f"{self.remote_output_dir}/conflict_audit.csv"

    @property
    def constraints_path(self) -> str:
        return f"{self.input_dir}/constraints.md"

    @property
    def scoring_path(self) -> str:
        return f"{self.reference_dir}/scoring.json"

    @property
    def task_description(self) -> str:
        return (
            "You are assisting the program chairs of a peer-reviewed conference "
            "with reviewer assignment.\n\n"
            f"Input data lives in {self.input_dir}/: submissions.csv (150 papers "
            "including withdrawn ones), reviewers.csv (60 reviewers including "
            "unavailable ones), coauthorships.csv, declared_conflicts.csv, and "
            "institutions.csv (the alias-to-canonical institution gazetteer). "
            f"Read {self.input_dir}/README.md (the data dictionary: institution "
            "strings are aliases, author names appear in several formats, and "
            "status/availability columns matter) and then "
            f"{self.constraints_path} (the authoritative rules and the exact "
            "scoring formula).\n\n"
            "Two deliverables:\n"
            f"1. {self.assignment_path} - header `paper_id,reviewer_id`, exactly "
            "3 reviewers for every ACTIVE paper, satisfying load limits, four "
            "kinds of conflict-of-interest rules, seniority coverage, and "
            "canonical-institution diversity. Any single hard-constraint "
            "violation scores 0.\n"
            f"2. {self.audit_path} - header `reviewer_id,paper_id,reason_code`, "
            "every conflicted pair over available reviewers x active papers "
            "with all applicable reason codes; graded by exact triple F1.\n\n"
            f"`python3 {self.software_dir}/check_format.py <assignment> "
            "[<audit>]` validates schemas only.\n\n"
            "Reviewer capacity is tight and conflicts are dense; feasibility "
            "itself is a global problem, full marks require an optimal "
            "assignment plus a perfect audit, and every entity comparison is "
            "defined on canonical forms, so derive the conflict set carefully."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "assignment_path": self.assignment_path,
            "audit_path": self.audit_path,
            "constraints_path": self.constraints_path,
            "scoring_path": self.scoring_path,
        })
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [cb.Task(
        description=cfg.task_description,
        metadata=cfg.to_metadata(),
        computer={
            "provider": "computer",
            "setup_config": {"os_type": cfg.OS_TYPE},
        },
    )]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Ensure a clean output dir, no stale reference, and staged inputs."""
    meta = task_cfg.metadata
    await session.run_command(f"mkdir -p {meta['remote_output_dir']!r}", check=False)
    await session.run_command(f"rm -f {meta['assignment_path']!r}", check=False)
    await session.run_command(f"rm -rf {meta['reference_dir']!r}", check=False)

    for name in INPUT_FILES:
        path = f"{meta['input_dir']}/{name}"
        try:
            await session.read_file(path)
        except Exception as exc:
            raise RuntimeError(f"staged input missing: {path} unreadable ({exc})")
    logger.info("[reviewer_assignment] inputs staged; output dir ready")


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    try:
        scoring_params = json.loads(await session.read_file(meta["scoring_path"]))
        inputs = {}
        for name in EVAL_INPUTS:
            inputs[name] = await session.read_file(f"{meta['input_dir']}/{name}")
    except Exception as exc:
        raise RuntimeError(f"reference/input unreadable at evaluation: {exc}")

    try:
        assignment = await session.read_file(meta["assignment_path"])
    except Exception as exc:
        logger.info("[reviewer_assignment] no output at %s: %s", meta["assignment_path"], exc)
        return [0.0]

    try:
        audit = await session.read_file(meta["audit_path"])
    except Exception:
        audit = None

    report = score_submission(
        assignment_csv=assignment,
        submissions_csv=inputs["submissions.csv"],
        reviewers_csv=inputs["reviewers.csv"],
        coauthorships_csv=inputs["coauthorships.csv"],
        declared_conflicts_csv=inputs["declared_conflicts.csv"],
        institutions_csv=inputs["institutions.csv"],
        scoring_params=scoring_params,
        audit_csv=audit,
    )
    logger.info(
        "[reviewer_assignment] score=%s affinity=%s/%s audit_f1=%s violations=%d%s",
        report["score"], report.get("total_affinity"),
        report.get("optimal_affinity"), report.get("audit_f1"),
        len(report["violations"]),
        (" first=" + report["violations"][0]) if report["violations"] else "",
    )
    return [float(report["score"])]
