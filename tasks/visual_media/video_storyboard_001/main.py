"""AgentHLE task implementation for visual_media/video_storyboard_001."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import PureWindowsPath

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.visual_media.video_storyboard_001.scripts.evaluate_storyboard import \
    score_storyboard_docx

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "visual_media"
TASK_NAME = "video_storyboard_001"
VARIANT_NAME = "base"
VIDEO_NAME = "Yasuji_Murata-_A_Wolf_is_a_Wolf_(1931).ogv"
QA_NAME = "video_storyboard_qa_set.docx"
OUTPUT_NAME = "storyboard.docx"


def win_join(*parts: str) -> str:
    return str(PureWindowsPath(*parts))


@dataclass
class VideoStoryboardConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = os.environ.get("REMOTE_ROOT_DIR", r"E:\agenthle")
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def input_dir(self) -> str:
        return win_join(self.task_dir, "input")

    @property
    def video_file(self) -> str:
        return win_join(self.input_dir, VIDEO_NAME)

    @property
    def question_file(self) -> str:
        return win_join(self.input_dir, QA_NAME)

    @property
    def word_processor_launcher(self) -> str:
        return win_join(self.software_dir, "launch_word_processor.cmd")

    @property
    def output_file(self) -> str:
        return win_join(self.remote_output_dir, OUTPUT_NAME)

    @property
    def reference_storyboard(self) -> str:
        return win_join(self.reference_dir, "reference_storyboard.docx")

    @property
    def task_description(self) -> str:
        return f"""\
## Context

A small restoration studio is about to begin a scan + remaster pass on a public-domain print of a vintage Yasuji Murata animation. Before any restoration work starts, the pipeline requires an **archival shot log / continuity report** of the source print: a temporal, per-shot factual record of what is on screen, written by an archival assistant editor.

The QC team has handed you a **restoration fact-check brief** (a list of factual visual questions about the film). Your shot log must preserve enough detail that downstream QC engineers can verify every item in the brief by reading the log alone. As the archivist, you record facts — you do **not** answer the brief's questions or insert interpretation. Answering is the QC engineers' job.

## Input Files
- Source print: `{self.video_file}`
- QC fact-check brief: `{self.question_file}`

## Software
- Use any DOCX-capable workflow available on the VM.
- A task-local launcher is available at: `{self.word_processor_launcher}`

## Your Task
1. Inspect the full source print end-to-end.
2. Read the QC brief so the log preserves the facts each item in the brief depends on.
3. Produce a concise, complete shot log in temporal order.
4. Organize the log as shots or segments. Each shot or segment must include:
   - shot or segment ID
   - start time and end time
   - brief visual description
   - key actions or events
   - important objects, people, animals, and scene details
   - visible on-screen text or dialogue relevant to the brief
5. Do **not** answer the brief's questions in the log. Record only the observable facts a QC engineer would need to answer them.
6. Use only information grounded in the source print. Do not add external knowledge, plot context, or interpretation.

## Output
- Save the final readable Word document exactly here: `{self.output_file}`
- The output file must be a valid `.docx` file named `{OUTPUT_NAME}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "video_file": self.video_file,
                "question_file": self.question_file,
                "word_processor_launcher": self.word_processor_launcher,
                "output_file": self.output_file,
                "reference_storyboard": self.reference_storyboard,
                "judge_model": os.environ.get("VIDEO_STORYBOARD_JUDGE_MODEL"),
            }
        )
        return metadata


config = VideoStoryboardConfig()


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
    output_file = meta["output_file"]
    question_file = meta["question_file"]

    try:
        if not await session.exists(output_file):
            logger.warning("Missing storyboard output: %s", output_file)
            return [0.0]
        if not await session.exists(question_file):
            logger.warning("Missing question file: %s", question_file)
            return [0.0]

        storyboard_docx = await session.read_bytes(output_file)
        question_docx = await session.read_bytes(question_file)
        result = await score_storyboard_docx(
            storyboard_docx=storyboard_docx,
            question_docx=question_docx,
            model=meta.get("judge_model"),
        )
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return [0.0]

    logger.info("evaluation=%s", json.dumps(result, sort_keys=True))
    return [float(result.get("score", 0.0))]
