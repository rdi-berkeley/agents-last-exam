"""AgentHLE task for Apple FY2024 balance-sheet reconstruction."""

from __future__ import annotations

import json
import logging
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

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "financial_stmt_reconstruction_aapl_fy2024"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"
ALLOWED_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}
ADMIN_OUTPUT_PREFIX = "output_admin_"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
SCORE_SCRIPT = (SCRIPTS_DIR / "score_outputs.py").read_text(encoding="utf-8")


def _normalize_output_dir_name(raw: str) -> str:
    normalized = raw.replace("\\", "/").strip("/")
    if not normalized or "/" in normalized:
        raise ValueError(f"OUTPUT_SUBDIR must be a single directory name, got {raw!r}")
    if normalized in ALLOWED_OUTPUT_DIR_NAMES or normalized.startswith(ADMIN_OUTPUT_PREFIX):
        return normalized
    raise ValueError(
        "OUTPUT_SUBDIR must be one of "
        f"{sorted(ALLOWED_OUTPUT_DIR_NAMES)} or start with {ADMIN_OUTPUT_PREFIX!r}"
    )


@dataclass
class AAPLBalanceSheetConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    @property
    def output_dir_name(self) -> str:
        return _normalize_output_dir_name(self.OUTPUT_SUBDIR)

    @property
    def output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/TASK_PROMPT.md"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/balance_sheet.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM as a financial statement extraction analyst.

## Task Directory
`{self.task_dir}`

## Visible Inputs
- Task prompt: `{self.task_prompt_file}`
- Apple FY2024 10-K PDF: `{self.input_dir}/aapl-2024-10k.pdf`
- SEC EDGAR HTML: `{self.input_dir}/aapl-20240928.htm`
- Task metadata: `{self.input_dir}/task.json`
- Output schema: `{self.input_dir}/output_schema.json`
- Source notes: `{self.input_dir}/material_sources.md`
- Tool wrappers: `{self.software_dir}/python`, `{self.software_dir}/pdftotext`, `{self.software_dir}/grep`

## Your Task
Reconstruct Apple Inc.'s Consolidated Balance Sheet for fiscal year 2024
(period ended September 28, 2024) as structured JSON.

Write the final JSON to `{self.output_file}`. Follow the schema in
`{self.input_dir}/output_schema.json`, use USD millions exactly as reported, and
do not modify files under `{self.input_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": self.VARIANT_NAME,
                "output_dir_name": self.output_dir_name,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "reference_dir": self.reference_dir,
                "software_dir": self.software_dir,
                "output_dir": self.output_dir,
                "task_prompt_file": self.task_prompt_file,
                "output_file": self.output_file,
                "eval_tmp_dir": EVAL_TMP_DIR,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = AAPLBalanceSheetConfig()


@cb.tasks_config(split="train")
def load():
    cfg = AAPLBalanceSheetConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    reference_file = f"{meta['reference_dir']}/aapl_fy2024_balance_sheet_reference.json"
    if not (await session.file_exists(reference_file) or await session.directory_exists(reference_file)):
        logger.error("missing evaluator reference file: %s", reference_file)
        return [0.0]

    score_script = f"{meta['eval_tmp_dir']}/score_outputs.py"
    try:
        await session.interface.create_dir(meta["eval_tmp_dir"])
        await session.write_file(score_script, SCORE_SCRIPT)
        result = await session.run_command(
            f'python "{score_script}" '
            f'--output "{meta["output_dir"]}" '
            f'--reference "{meta["reference_dir"]}"'
        )
        if result.get("return_code", 1) != 0:
            logger.warning("score script failed: %s", result.get("stderr", ""))
            return [0.0]
        report = json.loads(result.get("stdout", ""))
        logger.info("score report: %s", report)
        return [float(report.get("score", 0.0))]
    except Exception as exc:
        logger.error("evaluation failed: %s", exc)
        return [0.0]
