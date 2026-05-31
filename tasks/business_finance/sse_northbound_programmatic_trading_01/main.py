"""AgentHLE task: business_finance/sse_northbound_programmatic_trading_01."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.business_finance.sse_northbound_programmatic_trading_01.scripts.score_research_answers import \
    score_submission
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()


logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "sse_northbound_programmatic_trading_01"
VARIANT_NAME = "base"


@dataclass
class SseNorthboundProgrammaticTradingConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def task_brief_file(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def question_set_file(self) -> str:
        return f"{self.input_dir}/question_set.json"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def document_manifest_file(self) -> str:
        return f"{self.input_dir}/document_manifest.json"

    @property
    def source_documents_dir(self) -> str:
        return f"{self.input_dir}/source_documents"

    @property
    def extracted_text_dir(self) -> str:
        return f"{self.input_dir}/extracted_text"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/research_answers.json"

    @property
    def answer_key_file(self) -> str:
        return f"{self.reference_dir}/answer_key.json"

    @property
    def evaluator_documents_file(self) -> str:
        return f"{self.reference_dir}/evaluator_documents.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are acting as a regulatory consultant on Shanghai Stock Exchange Northbound programmatic-trading reporting.

## Closed-Book Rule

- Use only the staged materials under `{self.input_dir}`.
- Do not use outside web search or external regulatory knowledge.

## Inputs

- Task brief: `{self.task_brief_file}`
- Question set: `{self.question_set_file}`
- Output contract: `{self.output_contract_file}`
- Document manifest: `{self.document_manifest_file}`
- Original source files: `{self.source_documents_dir}`
- Extracted UTF-8 text mirrors: `{self.extracted_text_dir}`

## Your Task

Answer the three client questions in the staged brief:

1. whether leverage-related information must be reported separately through each broker for same-LEI accounts;
2. whether the staged rules provide specific numerical amounts for `流量费` and `撤单费`;
3. whether a French firm can report using only the original foreign-language software name.

## Deliverable

Write exactly one JSON file here:

- `{self.output_file}`

Each top-level question key must contain:

- `conclusion`: exactly `Yes`, `No`, or `Unknown`
- `citation_document`: the Simplified Chinese source document name you rely on
- `evidence_snippet`: an exact Simplified Chinese excerpt copied from the staged source
- `answer_text`: a short English explanation
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_brief_file": self.task_brief_file,
                "question_set_file": self.question_set_file,
                "output_contract_file": self.output_contract_file,
                "document_manifest_file": self.document_manifest_file,
                "source_documents_dir": self.source_documents_dir,
                "extracted_text_dir": self.extracted_text_dir,
                "output_file": self.output_file,
                "answer_key_file": self.answer_key_file,
                "evaluator_documents_file": self.evaluator_documents_file,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


config = SseNorthboundProgrammaticTradingConfig()


@cb.tasks_config(split="train")
def load():
    cfg = SseNorthboundProgrammaticTradingConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


async def _missing(session: cb.DesktopSession, path: str) -> bool:
    return not (await session.file_exists(path) or await session.directory_exists(path))


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]

    if not (await session.file_exists(meta["output_file"]) or await session.directory_exists(meta["output_file"])):
        logger.error("[%s] missing output file: %s", tag, meta["output_file"])
        return [0.0]

    for label, key in [
        ("answer key", "answer_key_file"),
        ("evaluator documents", "evaluator_documents_file"),
        ("document manifest", "document_manifest_file"),
    ]:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            raise RuntimeError(
                f"[{tag}] evaluator-controlled {label} missing: {meta[key]}"
            )

    with tempfile.TemporaryDirectory(prefix="sse_northbound_programmatic_trading_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output = tmp / "research_answers.json"
        local_answer_key = tmp / "answer_key.json"
        local_docs = tmp / "evaluator_documents.json"
        local_manifest = tmp / "document_manifest.json"
        local_output.write_bytes(await session.read_bytes(meta["output_file"]))
        local_answer_key.write_bytes(await session.read_bytes(meta["answer_key_file"]))
        local_docs.write_bytes(await session.read_bytes(meta["evaluator_documents_file"]))
        local_manifest.write_bytes(await session.read_bytes(meta["document_manifest_file"]))
        result = score_submission(
            submission_payload=json.loads(
                local_output.read_text(encoding="utf-8", errors="replace")
            ),
            answer_key=json.loads(
                local_answer_key.read_text(encoding="utf-8", errors="replace")
            ),
            evaluator_documents=json.loads(
                local_docs.read_text(encoding="utf-8", errors="replace")
            ),
            document_manifest=json.loads(
                local_manifest.read_text(encoding="utf-8", errors="replace")
            ),
        )

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, ensure_ascii=False, sort_keys=True))
    return [float(result.get("score", 0.0))]


if __name__ == "__main__":
    for task in load():
        print(task.description)
