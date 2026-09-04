"""Task: ``education_info/conference_review_console``.

GUI/long-horizon successor of conference_reviewer_assignment. All paper,
reviewer, conflict-register, and policy-amendment information lives ONLY
inside a local web application (the PRC Console); the server loads its
dataset into memory at boot and deletes the file, so the console's paginated
pages are the sole data source. The console enforces a multi-stage workflow:
per-paper screening (with per-paper comprehension checks) unlocks that
paper's conflict declarations and its assignability, an 8-question
integrative checklist gates finalization, and finalization is one-way.
Binding policy amendments (emeritus exclusion, junior load reduction, a
promotion, a rescinded declaration) are distributed across the inbox, two of
them on page 2.

Deliverables: the finalized in-console state (screenings + assignments +
checklist) and output/conflict_audit.csv. Grading is deterministic against
the generator's effective-policy truth: any effective-rule violation
(including an incorrect screening record) collapses to a small
screening-fraction credit; otherwise score = 0.2 + 0.35*quality +
0.25*audit_F1 + 0.2*checklist, capped at 0.15 if never finalized.
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
from score_outputs import score_console

logger = logging.getLogger(__name__)

DOMAIN_NAME = "education_info"
TASK_NAME = "conference_review_console"
VARIANT_NAME = "base"

PORT = 8765


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def state_dir(self) -> str:
        return f"{self.task_dir}/console_state"

    @property
    def state_path(self) -> str:
        return f"{self.state_dir}/state.json"

    @property
    def audit_path(self) -> str:
        return f"{self.remote_output_dir}/conflict_audit.csv"

    @property
    def server_path(self) -> str:
        return f"{self.software_dir}/server.py"

    @property
    def data_path(self) -> str:
        return f"{self.software_dir}/console_data.json"

    @property
    def scoring_path(self) -> str:
        return f"{self.reference_dir}/scoring.json"

    @property
    def world_path(self) -> str:
        return f"{self.reference_dir}/effective_world.json"

    @property
    def task_description(self) -> str:
        return (
            "You are the program chair assistant running a full review cycle "
            "in the PRC Console, a local web application at "
            f"http://127.0.0.1:{PORT} (login: chair / prc-2026; "
            f"`bash {self.software_dir}/launch.sh` opens it in Chrome). The "
            "console is the ONLY source for papers, reviewers, the conflict "
            "register, and committee announcements; there are no data files "
            "for these and the console's pages are paginated.\n\n"
            f"Start with {self.input_dir}/instructions.md and "
            f"{self.input_dir}/handbook.md. The handbook is the BASELINE "
            "policy; announcements in the console inbox are binding and "
            "supersede it, so read every inbox page before acting.\n\n"
            "Complete the enforced workflow in the console: (1) screen every "
            "active paper on its detail page (each screening asks you to "
            "confirm that paper's primary area and author count; screening "
            "unlocks the paper's conflict declarations and assignability), "
            "(2) enter a fully policy-compliant reviewer assignment (3 "
            "reviewers per active paper under the effective policy: "
            "conflicts, seniority, canonical-institution diversity, "
            "effective load limits), (3) answer the 8-question pre-submission "
            "checklist correctly, and (4) FINALIZE (one-way).\n\n"
            f"Also write {self.audit_path}: header "
            "`reviewer_id,paper_id,reason_code`, every conflicted pair over "
            "the audit domain with all applicable codes from "
            "{AFFILIATION, COAUTHOR, DECLARED, AUTHOR}.\n\n"
            "Machine-readable inputs: input/coauthorships.csv and "
            "input/institutions.csv (all institution strings anywhere are "
            "aliases; comparisons are on canonical institutions). The exact "
            "scoring formula is printed in the handbook: violations collapse "
            "the score, an unfinalized submission is capped at 0.15, and "
            "full marks require an optimal assignment, a perfect audit, and "
            "a correct checklist."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "state_dir": self.state_dir,
            "state_path": self.state_path,
            "audit_path": self.audit_path,
            "server_path": self.server_path,
            "data_path": self.data_path,
            "scoring_path": self.scoring_path,
            "world_path": self.world_path,
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
    """Boot the console (which consumes and deletes its data file) and ensure
    clean output/state. Idempotent: kills any prior server first."""
    meta = task_cfg.metadata

    await session.run_command("pkill -f 'server.py --data' || true", check=False)
    await session.run_command(f"rm -rf {meta['state_dir']!r}", check=False)
    await session.run_command(f"mkdir -p {meta['remote_output_dir']!r}", check=False)
    await session.run_command(f"rm -f {meta['audit_path']!r}", check=False)
    await session.run_command(f"rm -rf {meta['reference_dir']!r}", check=False)

    for path in (meta["server_path"], meta["data_path"]):
        try:
            await session.read_file(path)
        except Exception as exc:
            raise RuntimeError(f"staged software missing: {path} unreadable ({exc})")

    await session.run_command(
        f"nohup python3 {meta['server_path']!r} --data {meta['data_path']!r} "
        f"--state-dir {meta['state_dir']!r} --port {PORT} "
        f"> {meta['task_dir']}/server.log 2>&1 &",
        check=False,
    )
    def _stdout(res) -> str:
        return str(res["stdout"] if isinstance(res, dict) else res.stdout)

    ok = False
    for _ in range(30):
        res = await session.run_command(
            f"sleep 1; curl -s -o /dev/null -w '%{{http_code}}' "
            f"http://127.0.0.1:{PORT}/healthz", check=False)
        if "200" in _stdout(res):
            ok = True
            break
    if not ok:
        raise RuntimeError(f"PRC console did not become healthy on port {PORT}")

    res = await session.run_command(
        f"test -f {meta['data_path']!r} && echo PRESENT || echo GONE", check=False)
    if "GONE" not in _stdout(res):
        raise RuntimeError("console_data.json still present after server boot")
    logger.info("[review_console] console healthy; data file consumed")


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    try:
        scoring_params = json.loads(await session.read_file(meta["scoring_path"]))
        world = json.loads(await session.read_file(meta["world_path"]))
    except Exception as exc:
        raise RuntimeError(f"reference unreadable at evaluation: {exc}")

    try:
        state_json = await session.read_file(meta["state_path"])
    except Exception as exc:
        logger.info("[review_console] no console state at %s: %s",
                    meta["state_path"], exc)
        state_json = None

    try:
        audit = await session.read_file(meta["audit_path"])
    except Exception:
        audit = None

    report = score_console(
        state_json=state_json,
        audit_csv=audit,
        effective_world=world,
        scoring_params=scoring_params,
    )
    logger.info(
        "[review_console] score=%s finalized=%s screen=%s quality=%s audit=%s "
        "checklist=%s violations=%d%s",
        report["score"], report["finalized"], report["screen_frac"],
        report.get("quality"), report["audit_f1"], report["checklist_frac"],
        len(report["violations"]),
        (" first=" + report["violations"][0]) if report["violations"] else "",
    )
    return [float(report["score"])]
