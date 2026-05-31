"""neuroscience_pub_figure_reproduce_1 — Adobe Illustrator schematic recreation."""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.psychology_neuro.neuroscience_pub_figure_reproduce_1.scripts.score_figure_reproduction import (
    evaluate_files,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [("base", "virtual plume behavioral paradigm schematic")]


async def _path_missing(session: cb.DesktopSession, path: str, *, label: str, tag: str) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


@dataclass
class NeurosciencePubFigureReproduceConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "psychology_neuro"
    TASK_NAME: str = "neuroscience_pub_figure_reproduce_1"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def input_reference_pdf(self) -> str:
        return rf"{self.input_dir}\reference_schematic.pdf"

    @property
    def input_caption(self) -> str:
        return rf"{self.input_dir}\caption.txt"

    @property
    def input_template_ai(self) -> str:
        return rf"{self.input_dir}\template_fly.ai"

    @property
    def input_task_brief(self) -> str:
        return rf"{self.input_dir}\task_brief.md"

    @property
    def output_pdf(self) -> str:
        return rf"{self.output_dir}\schematic_reproduce.pdf"

    @property
    def output_ai(self) -> str:
        return rf"{self.output_dir}\schematic.ai"

    @property
    def reference_pdf(self) -> str:
        return rf"{self.reference_dir}\expert_reproduction.pdf"

    @property
    def reference_ai(self) -> str:
        return rf"{self.reference_dir}\expert_reproduction.ai"

    @property
    def source_hashes(self) -> str:
        return rf"{self.reference_dir}\source_hashes.json"

    @property
    def illustrator_launcher(self) -> str:
        return rf"{self.software_dir}\launch_illustrator.bat"

    @property
    def task_description(self) -> str:
        return f"""\
You are recreating a neuroscience experimental schematic in Adobe Illustrator.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Runtime Entry Point
- Open Adobe Illustrator from: `{self.illustrator_launcher}`

## Input Files
- Visual reference PDF: `{self.input_reference_pdf}`
- Scientific caption: `{self.input_caption}`
- Reusable fly component: `{self.input_template_ai}`
- Brief: `{self.input_task_brief}`

## What You Must Do
1. Open Illustrator using `{self.illustrator_launcher}`.
2. Inspect the reference schematic and caption.
3. Use the fly component where useful, but create a genuine editable Illustrator reproduction.
4. Preserve the key content: fly on spherical treadmill, visual landmark arena, odor tube, odor zones 1 and 2, and Protocol A/B block structures.
5. Save the editable Illustrator file to `{self.output_ai}`.
6. Export the final PDF to `{self.output_pdf}`.

## Output Requirements
- Save exactly one final Illustrator file at `{self.output_ai}`.
- Save exactly one final PDF export at `{self.output_pdf}`.
- Do not submit a direct copy of an input file as the final answer.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_dir": self.input_dir,
                "input_reference_pdf": self.input_reference_pdf,
                "input_caption": self.input_caption,
                "input_template_ai": self.input_template_ai,
                "input_task_brief": self.input_task_brief,
                "output_pdf": self.output_pdf,
                "output_ai": self.output_ai,
                "reference_pdf": self.reference_pdf,
                "reference_ai": self.reference_ai,
                "source_hashes": self.source_hashes,
                "illustrator_launcher": self.illustrator_launcher,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=NeurosciencePubFigureReproduceConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=NeurosciencePubFigureReproduceConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": "windows"},
            },
        )
        for variant_name, variant_label in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]

    for key, label in [
        ("output_pdf", "output PDF"),
        ("output_ai", "output Illustrator file"),
        ("reference_pdf", "hidden reference PDF"),
        ("input_reference_pdf", "visible reference PDF"),
        ("input_template_ai", "visible template AI"),
    ]:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            logger.error("[%s] Missing %s at %s", tag, label, meta[key])
            return [0.0]

    with tempfile.TemporaryDirectory(prefix="neuroscience_pub_figure_reproduce_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_paths = {
            "output_pdf": tmp / "output_schematic_reproduce.pdf",
            "output_ai": tmp / "output_schematic.ai",
            "reference_pdf": tmp / "expert_reproduction.pdf",
            "input_reference_pdf": tmp / "input_reference_schematic.pdf",
            "input_template_ai": tmp / "input_template_fly.ai",
        }
        try:
            for key, local_path in local_paths.items():
                local_path.write_bytes(await session.read_bytes(meta[key]))
            result = evaluate_files(
                output_pdf=local_paths["output_pdf"],
                output_ai=local_paths["output_ai"],
                reference_pdf=local_paths["reference_pdf"],
                input_pdf=local_paths["input_reference_pdf"],
                input_template_ai=local_paths["input_template_ai"],
            )
        except Exception as exc:
            logger.exception("[%s] Evaluation failed: %s", tag, exc)
            return [0.0]

    logger.info(
        "[%s] score=%.3f reason=%s details=%s",
        tag,
        result["score"],
        result["reason"],
        result.get("details"),
    )
    return [float(result.get("score", 0.0))]
