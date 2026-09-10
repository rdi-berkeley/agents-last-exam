"""engineering/cad_session_brief_recovery (Linux).

Five screen recordings (13-18 min each) of engineers doing a CAD job in AutoCAD, SOLIDWORKS or Siemens NX, with the raw
keyboard/mouse event log but WITHOUT the brief they were given. The agent must watch the session and write the brief
back: application, deliverable, objective and the list of concrete requirements the work satisfies. The hidden
reference is the original brief and the designer's requirement list; a text judge scores requirement coverage.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "cad_session_brief_recovery"
VARIANT_NAME = "base"
N_SESSIONS = 5


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def sessions_dir(self) -> str:
        return f"{self.input_dir}/sessions"

    @property
    def readme_path(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def reference_manifest(self) -> str:
        return f"{self.reference_dir}/manifest.json"

    @property
    def task_description(self) -> str:
        return (
            "You are auditing a CAD outsourcing team. For five recorded work sessions the written brief has been lost; "
            "reconstruct it from the recording.\n\n"
            "## Input\n"
            f"- Five sessions in `{self.sessions_dir}/s01/` .. `s05/`. Each folder holds `clip.mp4` (screen recording, "
            "13-18 minutes, no audio) and `events.json` (raw mouse/keyboard event log: clicks, key presses, scrolls, "
            "active application, with timestamps).\n"
            f"- `{self.readme_path}` repeats these instructions. ffmpeg is installed for frame extraction.\n\n"
            "## What you must do\n"
            "For each session write the brief the engineer was working from, as a JSON file "
            f"`{self.remote_output_dir}/<sid>.json` (for example `s01.json`) with exactly these keys:\n"
            "- `application`: the CAD application used (for example \"AutoCAD\", \"SOLIDWORKS\", \"Siemens NX\").\n"
            "- `deliverable`: what kind of file or drawing the session produces.\n"
            "- `objective`: one or two sentences saying what was to be created.\n"
            "- `requirements`: a list of specific, checkable statements about the result: features, counts, "
            "arrangements, dimensions, annotations, views, titles. Write what the finished work actually shows, "
            "not what the engineer clicked.\n\n"
            "## Scoring\n"
            "Session score = 0.15 x (application correct) + 0.85 x coverage, where coverage is the fraction of the "
            "designer's hidden requirement list that your requirements state (a requirement mentioned without its "
            "count, arrangement or dimension counts half). Vague or generic statements earn nothing. Task score = "
            "mean over the five sessions; a missing or unparsable JSON scores 0."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({"sessions_dir": self.sessions_dir, "readme_path": self.readme_path, "reference_manifest": self.reference_manifest})
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [cb.Task(description=cfg.task_description, metadata=cfg.to_metadata(), computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}})]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    await session.run_command(f"mkdir -p {meta['remote_output_dir']!r} && rm -f {meta['remote_output_dir']!r}/*.json", check=False)
    listing = await session.run_command(f"ls {meta['sessions_dir']!r} | wc -l", check=False)
    n = int((listing.get("stdout") if isinstance(listing, dict) else getattr(listing, "stdout", "0")).strip() or 0)
    if n < N_SESSIONS:
        raise RuntimeError(f"expected {N_SESSIONS} staged sessions in {meta['sessions_dir']}, found {n}")
    if await session.file_exists(meta["reference_manifest"]):
        raise RuntimeError("reference briefs are visible to the agent; staging order is wrong")
    logger.info("[%s] input staged (%d sessions); reference hidden", TASK_NAME, n)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    from tasks.engineering.cad_session_brief_recovery.scripts.grade import score_sessions

    meta = task_cfg.metadata
    try:
        manifest = json.loads(await session.read_file(meta["reference_manifest"]))
    except Exception as exc:
        raise RuntimeError(f"reference manifest unreadable: {exc}")
    sids = [m["sid"] for m in manifest]
    with tempfile.TemporaryDirectory() as tmp:
        ref_dir = os.path.join(tmp, "ref"); preds = {}
        for sid in sids:
            os.makedirs(os.path.join(ref_dir, sid))
            for name in ("rubrics.json", "task_desc.json"):
                with open(os.path.join(ref_dir, sid, name), "w") as f:
                    f.write(await session.read_file(f"{meta['reference_dir']}/{sid}/{name}"))
            try:
                preds[sid] = await session.read_file(f"{meta['remote_output_dir']}/{sid}.json")
            except Exception:
                preds[sid] = None
        try:
            result = score_sessions(preds, ref_dir)
        except Exception as exc:
            logger.info("[%s] scoring failed on agent output: %s", TASK_NAME, exc)
            return [0.0]
    for sid, v in result["per_session"].items():
        logger.info("[%s] %s score=%.4f coverage=%s app=%s (%s)", TASK_NAME, sid, v["score"], v.get("coverage"), v.get("app_score"), v["note"])
    return [result["score"]]
