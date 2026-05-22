"""AgentHLE task: legal/legal_dr_fees_01."""

from __future__ import annotations

import json
import logging
import sys
import types
from dataclasses import dataclass

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.legal.legal_dr_fees_01.scripts.score_outputs import (
    score_submission,
)
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "legal"
TASK_NAME = "legal_dr_fees_01"
VARIANT_NAME = "base"

sys.modules.setdefault(__name__, types.ModuleType(__name__))


@dataclass
class LegalDrFeesConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def task_brief_file(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def cases_file(self) -> str:
        return f"{self.input_dir}/cases.json"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def document_manifest_file(self) -> str:
        return f"{self.input_dir}/document_manifest.json"

    @property
    def documents_dir(self) -> str:
        return f"{self.input_dir}/documents"

    @property
    def runtime_env_file(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def python_entrypoint(self) -> str:
        return f"{self.software_dir}/python3.12"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/arbitration_fee_results.json"

    @property
    def answer_key_file(self) -> str:
        return f"{self.reference_dir}/answer_key.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are calculating arbitration fees under the staged Beijing Arbitration Commission
(BAC) materials.

## Closed-Book Rule

- Use only the staged materials under `{self.input_dir}`.
- Do not use outside web search or external regulatory knowledge.

## Inputs

- Task brief: `{self.task_brief_file}`
- Case facts: `{self.cases_file}`
- Output contract: `{self.output_contract_file}`
- Document manifest: `{self.document_manifest_file}`
- BAC PDF documents: `{self.documents_dir}`
- Optional Python runtime manifest: `{self.runtime_env_file}`
- Python 3.12 entry point: `{self.python_entrypoint}`

## Your Task

Calculate, for each of the five listed cases:

- institution fee in RMB;
- arbitrator remuneration in RMB;
- total fee in RMB.

All values must be accurate to two decimal places. Case 4 has a special
condition: the parties reached a settlement agreement regarding the dispute
before applying for arbitration.

If you want Python PDF tooling, install/use the package manifest under
`input/runtime_env/`; do not modify files under `input/`.

## Deliverable

Write exactly one JSON file here:

`{self.output_file}`

Follow the schema in `{self.output_contract_file}`. The evaluated numeric fields
for each case are:

- `institution_fee_rmb`
- `arbitrator_remuneration_rmb`
- `total_fee_rmb`
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_brief_file": self.task_brief_file,
                "cases_file": self.cases_file,
                "output_contract_file": self.output_contract_file,
                "document_manifest_file": self.document_manifest_file,
                "documents_dir": self.documents_dir,
                "runtime_env_file": self.runtime_env_file,
                "python_entrypoint": self.python_entrypoint,
                "output_file": self.output_file,
                "answer_key_file": self.answer_key_file,
                "canonical_gcs_root": f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = LegalDrFeesConfig()


@cb.tasks_config(split="train")
def load():
    cfg = LegalDrFeesConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


async def _missing(session: cb.DesktopSession, path: str) -> bool:
    return not await session.exists(path)


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]

    if await _missing(session, meta["output_file"]):
        logger.error("[%s] Missing output file: %s", tag, meta["output_file"])
        return [0.0]
    if await _missing(session, meta["answer_key_file"]):
        logger.error("[%s] Missing evaluator answer key: %s", tag, meta["answer_key_file"])
        return [0.0]

    try:
        payload = json.loads((await session.read_bytes(meta["output_file"])).decode("utf-8-sig"))
        _ = json.loads((await session.read_bytes(meta["answer_key_file"])).decode("utf-8-sig"))
        result = score_submission(payload)
    except Exception as exc:
        logger.exception("[%s] Evaluation failed: %s", tag, exc)
        return [0.0]

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, ensure_ascii=False, sort_keys=True))
    return [float(result.get("score", 0.0))]


if __name__ == "__main__":
    print(__file__)
