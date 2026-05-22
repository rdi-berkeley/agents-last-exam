"""AgentHLE task: business_finance/legal_ma_consistency_audit_01."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

# cua_bench may import task modules via exec_module without first inserting
# them into sys.modules; dataclass annotation handling expects the module to be
# registered already.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.business_finance.legal_ma_consistency_audit_01.scripts.score_audit_report import \
    score_report_text
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "legal_ma_consistency_audit_01"
VARIANT_NAME = "base"


@dataclass
class LegalMAConsistencyAuditConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def task_brief_file(self) -> str:
        return f"{self.input_dir}/task_brief.md"

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
        return f"{self.remote_output_dir}/audit_report.md"

    @property
    def gold_findings_file(self) -> str:
        return f"{self.reference_dir}/gold_findings.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are reviewing four Chinese regulatory filings about the same share-transfer transaction.

## Task
Read the staged materials, cross-check the filings for factual inconsistencies, and write one
English audit report.

## Visible Inputs
- Task root: `{self.task_dir}`
- Task brief: `{self.task_brief_file}`
- Document manifest: `{self.document_manifest_file}`
- Original PDFs: `{self.source_documents_dir}`
- Page-labeled extracted text: `{self.extracted_text_dir}`
- Optional Python helper: `software/python` (relative to the task root above)

## What To Deliver
Save exactly one Markdown file here:

- `{self.output_file}`

## Report Requirements
1. Write in English.
2. Report every material inconsistency you can support from the staged documents.
3. For each finding, include:
   - a short English title;
   - a brief explanation of the conflict;
   - original Chinese evidence text;
   - a document name and page-number location marker.
4. Do not invent unsupported issues.
5. If you want to script against the staged files, use `software/python` from the task root.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_brief_file": self.task_brief_file,
                "document_manifest_file": self.document_manifest_file,
                "source_documents_dir": self.source_documents_dir,
                "extracted_text_dir": self.extracted_text_dir,
                "output_file": self.output_file,
                "gold_findings_file": self.gold_findings_file,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


config = LegalMAConsistencyAuditConfig()


@cb.tasks_config(split="train")
def load():
    cfg = LegalMAConsistencyAuditConfig()
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

    required_eval_paths = {
        "output file": meta["output_file"],
        "gold findings": meta["gold_findings_file"],
    }
    missing_eval_paths = [
        f"{label} ({path})"
        for label, path in required_eval_paths.items()
        if not await session.exists(path)
    ]
    if missing_eval_paths:
        logger.error("[%s] Missing evaluation assets: %s", tag, ", ".join(missing_eval_paths))
        return [0.0]

    with tempfile.TemporaryDirectory(prefix="legal_ma_consistency_audit_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output = tmp / "audit_report.md"
        local_reference = tmp / "gold_findings.json"
        try:
            local_output.write_bytes(await session.read_bytes(meta["output_file"]))
            local_reference.write_bytes(await session.read_bytes(meta["gold_findings_file"]))
            result = score_report_text(
                report_text=local_output.read_text(encoding="utf-8", errors="replace"),
                reference_payload=json.loads(
                    local_reference.read_text(encoding="utf-8", errors="replace")
                ),
            )
        except Exception as exc:
            logger.exception("[%s] Evaluation failed: %s", tag, exc)
            return [0.0]

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, ensure_ascii=False, sort_keys=True))
    return [float(result.get("score", 0.0))]


if __name__ == "__main__":
    for task in load():
        print(task.description)
