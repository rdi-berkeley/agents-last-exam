"""AgentHLE task: business_finance/pe_screening_memo_1."""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local fallback only

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

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_screening_memo import (ScoreResult,  # noqa: E402
                                  score_screening_memo)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "pe_screening_memo_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "zscaler_fy2025"


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8-sig") if isinstance(payload, bytes) else str(payload)


def _log_score(result: ScoreResult) -> None:
    logger.info(
        "score=%.6f passed=%s hard_gate=%s reason=%s",
        result.score,
        result.passed,
        result.hard_gate,
        result.reason,
    )
    logger.info("details=%s", json.dumps(result.to_dict(), ensure_ascii=True))


@dataclass
class PEScreeningMemoConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    OUTPUT_FILENAME: str = "screening_memo.md"

    @property
    def task_brief_file(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def memo_template_file(self) -> str:
        return f"{self.input_dir}/memo_template.md"

    @property
    def source_manifest_file(self) -> str:
        return f"{self.input_dir}/source_manifest.json"

    @property
    def software_readme_file(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/{self.OUTPUT_FILENAME}"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/reference_memo.md"

    @property
    def evaluation_contract_file(self) -> str:
        return f"{self.reference_dir}/evaluation_contract.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux finance benchmark task.

Read these staged files first:
- `{self.task_brief_file}`
- `{self.memo_template_file}`
- `{self.source_manifest_file}`

Task-local software notes:
- `{self.software_readme_file}`

Your job is to write the requested private-equity screening memo for Zscaler
using only the staged packet in `{self.input_dir}`.

Write exactly one required file:
- `{self.output_file}`

Output requirements:
- filename must be exactly `{self.OUTPUT_FILENAME}`
- output must be valid UTF-8 Markdown
- follow the structure from `memo_template.md`
- make an explicit `Go`, `No-Go`, or `Hold / Needs More Diligence` recommendation

Constraints:
1. Use only the staged packet.
2. Do not use web search.
4. Write only your final memo under `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": self.VARIANT_NAME,
                "task_brief_file": self.task_brief_file,
                "memo_template_file": self.memo_template_file,
                "source_manifest_file": self.source_manifest_file,
                "software_readme_file": self.software_readme_file,
                "output_file": self.output_file,
                "output_filename": self.OUTPUT_FILENAME,
                "reference_file": self.reference_file,
                "evaluation_contract_file": self.evaluation_contract_file,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = PEScreeningMemoConfig()


@cb.tasks_config(split="train")
def load():
    cfg = PEScreeningMemoConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_file = meta["output_file"]
    reference_file = meta["reference_file"]
    contract_file = meta["evaluation_contract_file"]

    if not await session.exists(output_file):
        logger.error("Missing output file: %s", output_file)
        return [0.0]
    if not await session.exists(reference_file):
        logger.error("Missing hidden reference file: %s", reference_file)
        return [0.0]
    if not await session.exists(contract_file):
        logger.error("Missing hidden evaluation contract: %s", contract_file)
        return [0.0]

    try:
        candidate_memo = await session.read_file(output_file)
        reference_memo = await session.read_file(reference_file)
        contract_text = await session.read_file(contract_file)
    except Exception as exc:
        logger.error("Failed to read task artifacts from VM: %s", exc)
        return [0.0]

    result = score_screening_memo(
        _as_text(candidate_memo),
        _as_text(reference_memo),
        contract_text=_as_text(contract_text),
    )
    _log_score(result)
    return [float(result.score)]
