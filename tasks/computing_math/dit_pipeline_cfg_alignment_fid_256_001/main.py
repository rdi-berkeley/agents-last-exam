"""AgentHLE task: dit_pipeline_cfg_alignment_fid_256_001."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local import fallback only

    class _FallbackTask:
        def __init__(self, description, metadata, computer):
            self.description = description
            self.metadata = metadata
            self.computer = computer

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    cb = SimpleNamespace(
        Task=_FallbackTask,
        DesktopSession=object,
        tasks_config=_identity_decorator,
        setup_task=_identity_decorator,
        evaluate_task=_identity_decorator,
    )

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup  # noqa: E402
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import ScoreResult, score_submission_text  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "dit_pipeline_cfg_alignment_fid_256_001"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "OUTPUT_SUBDIR must be one of: " + ", ".join(sorted(ALLOWED_OUTPUT_DIRS))
        )
    return normalized


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


@dataclass
class DitPipelineCfgAlignmentConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    @property
    def starter_pipeline(self) -> str:
        return f"{self.input_dir}/pipeline_dit.py"

    @property
    def sample_script(self) -> str:
        return f"{self.input_dir}/sample.py"

    @property
    def run_script(self) -> str:
        return f"{self.input_dir}/run_sample.sh"

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def software_note(self) -> str:
        return f"{self.software_dir}/README.md"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/pipeline_dit.py"

    @property
    def output_dir_name(self) -> str:
        return _canonical_output_dir_name(self.OUTPUT_SUBDIR)

    @property
    def output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/pipeline_dit.py"

    @property
    def task_description(self) -> str:
        return f"""\
You are repairing a standalone Diffusers-based DiT pipeline on a Linux VM.

Task directory:
- `{self.task_dir}`

Visible files:
- Buggy starter pipeline: `{self.starter_pipeline}`
- Raw sampling harness context: `{self.sample_script}`
- Raw run script context: `{self.run_script}`
- Benchmark-owned prompt: `{self.task_prompt_file}`

Your task:
1. Read `{self.task_prompt_file}` first.
2. Repair the standalone starter file at `{self.starter_pipeline}`.
3. Keep the fix localized to the pipeline file.
4. Save exactly one final file to:
   `{self.output_file}`

Do not modify files under `input/`.
Do not rely on hidden evaluator files while solving.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "starter_pipeline": self.starter_pipeline,
                "sample_script": self.sample_script,
                "run_script": self.run_script,
                "task_prompt_file": self.task_prompt_file,
                "software_note": self.software_note,
                "reference_file": self.reference_file,
                "output_file": self.output_file,
                "output_dir_name": self.output_dir_name,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = DitPipelineCfgAlignmentConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not (await session.file_exists(meta["output_file"]) or await session.directory_exists(meta["output_file"])):
        logger.info("missing output file: %s", meta["output_file"])
        return [0.0]

    submission_text = _as_text(await session.read_file(meta["output_file"]))
    reference_text = _as_text(await session.read_file(meta["reference_file"]))

    result: ScoreResult = await asyncio.to_thread(
        score_submission_text, submission_text, reference_text
    )
    logger.info("evaluation result: %s", json.dumps(result.to_dict(), sort_keys=True))
    return [float(result.score)]


if __name__ == "__main__":
    for task in load():
        print(task.description)
