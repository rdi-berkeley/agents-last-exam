"""engineering/autocad_drafting_from_reference (Linux).

Six 2D drafting jobs taken from real AutoCAD sessions: each comes with the brief the drafter received and the reference
material they worked from (a PDF or photo of the drawing to reproduce, plus a pipeline-generated print of the expected
sheet). The agent must draft each drawing as a DXF with open-source tools. The hidden reference is the drafter's DWG
converted to DXF; evaluate() compares line geometry after normalising position and scale.
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
TASK_NAME = "autocad_drafting_from_reference"
VARIANT_NAME = "base"
N_DRAWINGS = 6


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def drawings_dir(self) -> str:
        return f"{self.input_dir}/drawings"

    @property
    def readme_path(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def reference_manifest(self) -> str:
        return f"{self.reference_dir}/manifest.json"

    @property
    def task_description(self) -> str:
        return (
            "You are a CAD drafter. A client sends you six drafting jobs that were originally done in AutoCAD and needs "
            "each drawing recreated as a DXF file.\n\n"
            "## Input\n"
            f"- Six jobs in `{self.drawings_dir}/d01/` .. `d06/`. Each folder holds `brief.md` (what the drawing must "
            "show) and the reference material the drafter worked from: a PDF or photo of the drawing to reproduce and, "
            "in most folders, a `reference_*` image that is a print of the expected finished sheet.\n"
            f"- `{self.readme_path}` repeats these instructions.\n\n"
            "## What you must do\n"
            "1. Read each brief and study the reference images; measure proportions from the given dimensions.\n"
            "2. Draft the drawing in model space with any open-source tool (ezdxf, LibreCAD, FreeCAD or QCAD are "
            "installed): all outlines, hidden and centre lines, hatches and dimensions of the reference. Text is not "
            "scored. Dimensions must be real DIMENSION entities (for example ezdxf `add_linear_dim`, `add_diameter_dim`, "
            "`add_radius_dim`); they are scored by count, and plain lines drawn to look like dimensions are not counted.\n"
            f"3. Save each job as `{self.remote_output_dir}/<stem>.dxf` (for example `d01.dxf`), any DXF version, "
            "any units.\n\n"
            "## Scoring\n"
            "Each DXF is rendered to a line raster over its own extents (so absolute position and scale do not matter, "
            "but proportions do) and compared with the reference drawing rendered the same way. Score per drawing is "
            "the F1 of line-pixel precision and recall with a tolerance of about 0.6 % of the sheet width. Task score "
            "= mean over the six drawings; a missing or unreadable DXF scores 0."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({"drawings_dir": self.drawings_dir, "readme_path": self.readme_path, "reference_manifest": self.reference_manifest})
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [cb.Task(description=cfg.task_description, metadata=cfg.to_metadata(), computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}})]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    await session.run_command(f"mkdir -p {meta['remote_output_dir']!r} && rm -f {meta['remote_output_dir']!r}/*.dxf", check=False)
    listing = await session.run_command(f"ls {meta['drawings_dir']!r} | wc -l", check=False)
    n = int((listing.get("stdout") if isinstance(listing, dict) else getattr(listing, "stdout", "0")).strip() or 0)
    if n < N_DRAWINGS:
        raise RuntimeError(f"expected {N_DRAWINGS} staged drawings in {meta['drawings_dir']}, found {n}")
    if await session.file_exists(meta["reference_manifest"]):
        raise RuntimeError("reference drawings are visible to the agent; staging order is wrong")
    logger.info("[%s] input staged (%d drawings); reference hidden", TASK_NAME, n)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    from tasks.engineering.autocad_drafting_from_reference.scripts.grade import score_parts

    meta = task_cfg.metadata
    try:
        manifest = json.loads(await session.read_file(meta["reference_manifest"]))
    except Exception as exc:
        raise RuntimeError(f"reference manifest unreadable: {exc}")
    stems = [m["stem"] for m in manifest if m.get("ok", True)]
    with tempfile.TemporaryDirectory() as tmp:
        ref_paths, pred_paths = {}, {}
        for stem in stems:
            rp = os.path.join(tmp, f"ref_{stem}.dxf")
            with open(rp, "w") as f:
                f.write(await session.read_file(f"{meta['reference_dir']}/{stem}/expert.dxf"))
            ref_paths[stem] = rp
            try:
                text = await session.read_file(f"{meta['remote_output_dir']}/{stem}.dxf")
            except Exception:
                pred_paths[stem] = None
                continue
            pp = os.path.join(tmp, f"pred_{stem}.dxf")
            with open(pp, "w") as f:
                f.write(text)
            pred_paths[stem] = pp
        try:
            result = score_parts(pred_paths, ref_paths)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.info("[%s] scoring failed on agent output: %s", TASK_NAME, exc)
            return [0.0]
    for stem, v in result["per_part"].items():
        logger.info("[%s] %s score=%.4f P=%s R=%s (%s)", TASK_NAME, stem, v["score"], v.get("precision"), v.get("recall"), v["note"])
    return [result["score"]]
