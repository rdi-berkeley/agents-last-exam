"""AgentHLE task: psychology_neuro/reddit_ai_post_codebook_boolean_coding."""

import json
import logging
from dataclasses import dataclass
from pathlib import PureWindowsPath

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.psychology_neuro.reddit_ai_post_codebook_boolean_coding.scripts.score_annotation_workbook import (
    score_workbooks,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


def win_join(*parts: str) -> str:
    return str(PureWindowsPath(*parts))


def _is_runtime_output_dir(path: str) -> bool:
    return PureWindowsPath(path).name.lower() == "output"


@dataclass
class RedditAIPostCodebookConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "psychology_neuro"
    TASK_NAME: str = "reddit_ai_post_codebook_boolean_coding"
    VARIANT_NAME: str = "base"

    OUTPUT_FILENAME: str = "ai_addiction_annotations_output.xlsx"
    INPUT_WORKBOOK_FILENAME: str = "ai_addiction_dataset_n95.xlsx"
    CODEBOOK_FILENAME: str = "ai_addiction_codebook.pdf"
    REFERENCE_FILENAME: str = "ai_addiction_human_annotations_n95.xlsx"

    @property
    def input_dir(self) -> str:
        return win_join(self.task_dir, "input")

    @property
    def input_workbook(self) -> str:
        return win_join(self.input_dir, self.INPUT_WORKBOOK_FILENAME)

    @property
    def codebook_pdf(self) -> str:
        return win_join(self.input_dir, self.CODEBOOK_FILENAME)

    @property
    def output_workbook(self) -> str:
        return win_join(self.output_dir, self.OUTPUT_FILENAME)

    @property
    def reference_workbook(self) -> str:
        return win_join(self.reference_dir, self.REFERENCE_FILENAME)

    @property
    def open_dataset_entry(self) -> str:
        return win_join(self.software_dir, "open_dataset.bat")

    @property
    def open_codebook_entry(self) -> str:
        return win_join(self.software_dir, "open_codebook.bat")

    @property
    def open_inputs_entry(self) -> str:
        return win_join(self.software_dir, "open_inputs.bat")

    @property
    def task_description(self) -> str:
        return f"""\
You are completing a psychology coding workbook on a Windows VM.

## Input Files
- Dataset workbook: `{self.input_workbook}`
- Codebook PDF: `{self.codebook_pdf}`

## Software
- Open the workbook with `{self.open_dataset_entry}`
- Open the codebook with `{self.open_codebook_entry}`
- Or open both with `{self.open_inputs_entry}`

## What You Must Do
1. Read the codebook and inspect the Reddit post workbook.
2. For each nonempty post row (Excel rows 2 through 96), fill the annotation columns F through AR.
3. Choose exactly one focus label in columns F through H for each post row.
4. Save the completed workbook exactly to `{self.output_workbook}`.

## Rules
- Keep the existing post metadata in columns A through E intact.
- Use boolean annotations in columns F through AR.
- Do not modify files under `input\\`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "input_workbook": self.input_workbook,
                "codebook_pdf": self.codebook_pdf,
                "output_workbook": self.output_workbook,
                "reference_workbook": self.reference_workbook,
                "open_dataset_entry": self.open_dataset_entry,
                "open_codebook_entry": self.open_codebook_entry,
                "open_inputs_entry": self.open_inputs_entry,
            }
        )
        return metadata


config = RedditAIPostCodebookConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    try:
        if not (await session.file_exists(meta["output_workbook"]) or await session.directory_exists(meta["output_workbook"])):
            logger.warning("Missing output workbook: %s", meta["output_workbook"])
            return [0.0]
        if not (await session.file_exists(meta["reference_workbook"]) or await session.directory_exists(meta["reference_workbook"])):
            logger.warning("Missing hidden reference workbook: %s", meta["reference_workbook"])
            return [0.0]

        candidate_bytes = await session.read_bytes(meta["output_workbook"])
        reference_bytes = await session.read_bytes(meta["reference_workbook"])
        result = score_workbooks(candidate_bytes, reference_bytes)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return [0.0]

    logger.info("evaluation=%s", json.dumps(result, sort_keys=True))
    return [float(result.get("score", 0.0))]
