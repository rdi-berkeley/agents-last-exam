"""engineering/cad_part_from_drawing_rubric (Linux).

Six part-modelling jobs taken from real SOLIDWORKS sessions: a brief plus the 2D drawing (PDF) and photo the designer
worked from. The agent delivers one STEP solid per part. The hidden reference is the list of requirements the
original designer wrote for the part (rubrics.json); evaluate() renders the delivered solid from six fixed viewpoints
and a vision judge decides, by majority vote, which requirements the model clearly satisfies.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "cad_part_from_drawing_rubric"
VARIANT_NAME = "base"
N_PARTS = 6


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def parts_dir(self) -> str:
        return f"{self.input_dir}/parts"

    @property
    def readme_path(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def reference_manifest(self) -> str:
        return f"{self.reference_dir}/manifest.json"

    @property
    def task_description(self) -> str:
        return (
            "You are a mechanical design engineer. Six parts must be modelled from their 2D drawings.\n\n"
            "## Input\n"
            f"- Six parts in `{self.parts_dir}/p01/` .. `p06/`. Each folder holds `brief.md` (the design brief) and "
            "the drawing the designer worked from: a dimensioned PDF drawing and usually a photo or screenshot of "
            "the same sheet.\n"
            f"- `{self.readme_path}` repeats these instructions.\n\n"
            "## What you must do\n"
            "1. Read each brief and drawing; extract every feature, count, arrangement and dimension.\n"
            "2. Model the part as a closed solid with any open-source CAD tool (CadQuery, build123d, FreeCAD, OpenSCAD "
            "are installed). Follow the drawing's dimensions and proportions.\n"
            f"3. Export each part to `{self.remote_output_dir}/<stem>.step` (for example `p01.step`), STEP AP214, "
            "millimetres.\n\n"
            "## Scoring\n"
            "Each delivered solid is rendered from six fixed viewpoints and checked against the requirement list the "
            "original designer wrote for that part (features, counts, arrangement, proportions). Part score = "
            "fraction of requirements clearly satisfied, multiplied by a dimension factor when the drawing states the "
            "part's overall dimensions (1 when the solid's bounding box is within 5 % of them, 0 at 30 % off); task "
            "score = mean over the six parts. A missing, unreadable or invalid STEP scores 0 for that part."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({"parts_dir": self.parts_dir, "readme_path": self.readme_path, "reference_manifest": self.reference_manifest})
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [cb.Task(description=cfg.task_description, metadata=cfg.to_metadata(), computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}})]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    # idempotent staging (docs: wipe output/ and reference/ first, in case of a reused VM); the reference is injected only at eval time
    await session.run_command(f"rm -rf {meta['reference_dir']!r} {meta['remote_output_dir']!r} && mkdir -p {meta['remote_output_dir']!r}", check=False)
    listing = await session.run_command(f"ls {meta['parts_dir']!r} | wc -l", check=False)
    n = int((listing.get("stdout") if isinstance(listing, dict) else getattr(listing, "stdout", "0")).strip() or 0)
    if n < N_PARTS:
        raise RuntimeError(f"expected {N_PARTS} staged parts in {meta['parts_dir']}, found {n}")
    if await session.file_exists(meta["reference_manifest"]):
        raise RuntimeError("reference rubrics are visible to the agent; staging order is wrong")
    logger.info("[%s] input staged (%d parts); reference hidden", TASK_NAME, n)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Copy outputs and hidden rubrics to the host, render the solids there and ask the vision judge."""
    from tasks.engineering.cad_part_from_drawing_rubric.scripts.grade import score_parts

    meta = task_cfg.metadata
    try:
        manifest = json.loads(await session.read_file(meta["reference_manifest"]))
    except Exception as exc:
        raise RuntimeError(f"reference manifest unreadable: {exc}")
    stems = [m["stem"] for m in manifest]
    with tempfile.TemporaryDirectory() as tmp:
        ref_dir = os.path.join(tmp, "ref"); work = os.path.join(tmp, "work"); pred_paths = {}
        for stem in stems:
            os.makedirs(os.path.join(ref_dir, stem))
            for name in ("rubrics.json", "task_desc.json"):
                with open(os.path.join(ref_dir, stem, name), "w") as f:
                    f.write(await session.read_file(f"{meta['reference_dir']}/{stem}/{name}"))
            for name in ("dims.json", "drawing_images.json"):  # optional: overall dimensions and rasterised drawing pages
                src = f"{meta['reference_dir']}/{stem}/{name}"
                if await session.file_exists(src):
                    with open(os.path.join(ref_dir, stem, name), "w") as f:
                        f.write(await session.read_file(src))
            try:
                text = await session.read_file(f"{meta['remote_output_dir']}/{stem}.step")
            except Exception:
                pred_paths[stem] = None
                continue
            pp = os.path.join(tmp, f"pred_{stem}.step")
            with open(pp, "w") as f:
                f.write(text)
            pred_paths[stem] = pp
        try:
            result = score_parts(pred_paths, ref_dir, work)
        except Exception as exc:
            logger.info("[%s] scoring failed on agent output: %s", TASK_NAME, exc)
            return [0.0]
    for stem, v in result["per_part"].items():
        logger.info("[%s] %s score=%.4f %s/%s (%s)", TASK_NAME, stem, v["score"], v.get("satisfied"), v.get("n_rubrics"), v["note"])
    return [result["score"]]
