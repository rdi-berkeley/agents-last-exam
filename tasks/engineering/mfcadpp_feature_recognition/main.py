"""engineering/mfcadpp_feature_recognition (Linux).

The agent receives 30 B-rep parts as STEP files whose faces are named f000, f001, ... in file order,
five labelled example parts, and the 25-class feature table. It must label every face of every part
with the machining-feature class it belongs to and write output/labels.json. The hidden reference
holds the true labels; evaluate() scores macro-F1 over classes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig
from tasks.engineering.mfcadpp_feature_recognition.scripts.grade import parse_prediction, score_labels

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "mfcadpp_feature_recognition"
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
    def examples_dir(self) -> str:
        return f"{self.input_dir}/examples"

    @property
    def classes_path(self) -> str:
        return f"{self.input_dir}/feature_classes.txt"

    @property
    def readme_path(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def output_labels_path(self) -> str:
        return f"{self.remote_output_dir}/labels.json"

    @property
    def reference_labels_path(self) -> str:
        return f"{self.reference_dir}/labels.json"

    @property
    def task_description(self) -> str:
        return (
            "You are a manufacturing process-planning engineer. Before a part can be machined, every face of its "
            "B-rep model has to be assigned to the machining feature it belongs to (through hole, rectangular pocket, "
            "chamfer, ...), or marked as untouched stock.\n\n"
            f"## Input\n- 30 parts as STEP files in `{self.parts_dir}/<part_id>.step`. Every face entity is named in file "
            "order: ADVANCED_FACE('f000', ...), ADVANCED_FACE('f001', ...), and so on. Use those names as face ids.\n"
            f"- The class table `{self.classes_path}`: ids 0-23 are machining features, 24 is a stock face that belongs "
            "to no feature.\n"
            f"- Five fully labelled example parts in `{self.examples_dir}/` (a `.step` plus a `.labels.json` each).\n"
            f"- `{self.readme_path}` repeats these instructions.\n\n"
            "## What you must do\n"
            "1. Load each part with any open-source B-rep tool (python-occ / OCP / CadQuery / FreeCAD are installed) and "
            "inspect face geometry and topology.\n"
            "2. Decide the feature class of every face. Use the examples to calibrate what each class looks like.\n"
            f"3. Write `{self.output_labels_path}` as JSON: {{\"<part_id>\": {{\"f000\": <class>, \"f001\": <class>, ...}}, ...}} "
            "covering every face of every part. Integer classes only.\n\n"
            "## Scoring\n"
            "Macro-F1 over the 25 classes across all 966 faces. Unlabelled faces count as wrong. Labelling every face as "
            "stock scores close to 0; a complete correct labelling scores 1.0."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "parts_dir": self.parts_dir,
            "examples_dir": self.examples_dir,
            "classes_path": self.classes_path,
            "readme_path": self.readme_path,
            "output_labels_path": self.output_labels_path,
            "reference_labels_path": self.reference_labels_path,
        })
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
    """Ensure a clean output dir, assert staged input is present, assert the reference is not visible."""
    meta = task_cfg.metadata
    await session.run_command(f"mkdir -p {meta['remote_output_dir']!r} && rm -f {meta['output_labels_path']!r}", check=False)
    for path in (meta["classes_path"], meta["readme_path"]):
        try:
            await session.read_file(path)
        except Exception as exc:
            raise RuntimeError(f"staged input missing: {path} ({exc})")
    listing = await session.run_command(f"ls {meta['parts_dir']!r} | wc -l", check=False)
    n = int((listing.get("stdout") if isinstance(listing, dict) else getattr(listing, "stdout", "0")).strip() or 0)
    if n < 30:
        raise RuntimeError(f"expected 30 staged parts in {meta['parts_dir']}, found {n}")
    if await session.file_exists(meta["reference_labels_path"]):
        raise RuntimeError("reference labels are visible to the agent; staging order is wrong")
    logger.info("[%s] input staged (%d parts); reference hidden", TASK_NAME, n)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        ref = json.loads(await session.read_file(meta["reference_labels_path"]))
    except Exception as exc:
        raise RuntimeError(f"reference unreadable at {meta['reference_labels_path']}: {exc}")
    try:
        text = await session.read_file(meta["output_labels_path"])
    except Exception as exc:
        logger.info("[%s] no output at %s: %s", TASK_NAME, meta["output_labels_path"], exc)
        return [0.0]
    result = score_labels(parse_prediction(text), ref)
    logger.info("[%s] macro-F1 %.4f accuracy %.4f labelled %d/%d", TASK_NAME, result["macro_f1"], result["accuracy"], result["labelled"], result["faces"])
    return [result["score"]]
