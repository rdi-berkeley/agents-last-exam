"""engineering/benchcad_views_to_step (Linux).

Twelve standard mechanical parts (helical gear, round flange, ball knob, turnbuckle, T-slot rail, bellows, ...) are
given as four orthographic renders plus a composite view, the part family and the governing standard. The agent
rebuilds each part as one solid and delivers output/<stem>.step. The hidden reference is the single solid generated
from the ground-truth CadQuery program; evaluate() scores mean volume IoU after normalisation over all 24 proper
axis-aligned rotations (scripts/grade.py). The agent-facing text lives in scripts/prompt.py.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass

import cua_bench as cb

from tasks.engineering.benchcad_views_to_step.scripts.prompt import task_text
from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "benchcad_views_to_step"
VARIANT_NAME = "base"


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
        return task_text(self.parts_dir, self.readme_path, self.remote_output_dir)

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({"parts_dir": self.parts_dir, "readme_path": self.readme_path, "reference_manifest": self.reference_manifest})
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [cb.Task(
        description=cfg.task_description,
        metadata=cfg.to_metadata(),
        computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
    )]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    await session.run_command(f"mkdir -p {meta['remote_output_dir']!r} && rm -f {meta['remote_output_dir']!r}/*.step", check=False)
    listing = await session.run_command(f"ls {meta['parts_dir']!r} | wc -l", check=False)
    n = int((listing.get("stdout") if isinstance(listing, dict) else getattr(listing, "stdout", "0")).strip() or 0)
    if n < 12:
        raise RuntimeError(f"expected 12 staged parts in {meta['parts_dir']}, found {n}")
    if not await session.file_exists(meta["readme_path"]):
        raise RuntimeError(f"{meta['readme_path']} is part of the declared starting state but was not staged")
    if await session.file_exists(meta["reference_manifest"]):
        raise RuntimeError("reference solids are visible to the agent; staging order is wrong")
    logger.info("[%s] input staged (%d parts + README); reference hidden", TASK_NAME, n)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Copy the reference and output STEP files to the host and score with CadQuery there (all 24 rotations)."""
    from tasks.engineering.benchcad_views_to_step.scripts.grade import score_parts

    meta = task_cfg.metadata
    try:
        manifest = json.loads(await session.read_file(meta["reference_manifest"]))
    except Exception as exc:
        raise RuntimeError(f"reference manifest unreadable: {exc}")
    stems = [s for s, v in manifest.items() if v.get("ok")]
    with tempfile.TemporaryDirectory() as tmp:
        ref_paths = {}; pred_paths = {}
        for stem in stems:
            rp = os.path.join(tmp, f"ref_{stem}.step")
            with open(rp, "w") as f:
                f.write(await session.read_file(f"{meta['reference_dir']}/{stem}.step"))
            ref_paths[stem] = rp
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
            result = score_parts(pred_paths, ref_paths)
        except RuntimeError:
            raise  # unusable reference: infrastructure error, surface it
        except Exception as exc:
            logger.info("[%s] scoring failed on agent output: %s", TASK_NAME, exc)
            return [0.0]
    for stem, v in result["per_part"].items():
        logger.info("[%s] %s IoU=%.4f part_score=%.4f rotations=%d (%s)", TASK_NAME, stem, v["iou"], v["part_score"], v.get("n_rotations", 0), v["note"])
    return [result["score"]]
