"""AgentHLE task: AGORA governance classification instance 1."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
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

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.legal.agora_governance_classify_instance_1.scripts.score_outputs import (
    score_output_file,
)
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


async def _missing(session: cb.DesktopSession, path: str, *, label: str) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("Missing %s: %s", label, path)
    return True


@dataclass
class AgoraGovernanceClassifyConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "legal"
    TASK_NAME: str = "agora_governance_classify_instance_1"
    VARIANT_NAME: str = "base"
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")

    @property
    def task_spec_file(self) -> str:
        return f"{self.input_dir}/task_spec.md"

    @property
    def taxonomy_file(self) -> str:
        return f"{self.input_dir}/taxonomy_and_instructions.md"

    @property
    def document_index_file(self) -> str:
        return f"{self.input_dir}/document_index.json"

    @property
    def documents_dir(self) -> str:
        return f"{self.input_dir}/documents"

    @property
    def software_readme(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def software_python_file(self) -> str:
        return f"{self.software_dir}/python"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/agent_output.json"

    @property
    def reference_truth_file(self) -> str:
        return f"{self.reference_dir}/ground_truth_FINAL.json"

    @property
    def reference_docs_file(self) -> str:
        return f"{self.reference_dir}/document_texts.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM as a policy analyst.

Visible task files:
- `{self.task_spec_file}`
- `{self.taxonomy_file}`
- `{self.document_index_file}`
- `{self.documents_dir}/`
- `{self.software_readme}`
- `{self.software_python_file}` (optional Python helper entry point)

Your task:
1. Read `task_spec.md`, the sanitized taxonomy instructions, and all three cached document text files.
2. Classify AGORA IDs 768, 1293, and 2047 across legislative status, technical scope, and AI lifecycle stage.
3. For every TRUE label, provide a verbatim evidence quote from the corresponding cached document text.
4. Produce a cross-document matrix and a 150-300 word gap analysis.
5. Write valid JSON to:
   - `{self.output_file}`

Rules:
- Do not modify files under `input/`.
- Use the cached documents under `input/documents/` as the canonical evidence source.
- Keep the final deliverable at exactly `{self.output_file}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{self.DOMAIN_NAME}/{self.TASK_NAME}",
                "task_spec_file": self.task_spec_file,
                "taxonomy_file": self.taxonomy_file,
                "document_index_file": self.document_index_file,
                "documents_dir": self.documents_dir,
                "software_readme": self.software_readme,
                "software_python_file": self.software_python_file,
                "output_file": self.output_file,
                "reference_truth_file": self.reference_truth_file,
                "reference_docs_file": self.reference_docs_file,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


config = AgoraGovernanceClassifyConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    for key, label in [
        ("output_file", "candidate output"),
        ("reference_truth_file", "hidden consensus labels"),
        ("reference_docs_file", "hidden document texts"),
    ]:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            logger.error("Missing %s at %s", label, meta[key])
            return [0.0]

    with tempfile.TemporaryDirectory(prefix="agora_governance_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        candidate = tmp / "agent_output.json"
        truth = tmp / "ground_truth_FINAL.json"
        docs = tmp / "document_texts.json"
        try:
            candidate.write_bytes(await session.read_bytes(meta["output_file"]))
            truth.write_bytes(await session.read_bytes(meta["reference_truth_file"]))
            docs.write_bytes(await session.read_bytes(meta["reference_docs_file"]))
            result = score_output_file(candidate, truth, docs)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Evaluation failed: %s", exc)
            return [0.0]

    logger.info("evaluation=%s", json.dumps(result, sort_keys=True)[:4000])
    return [float(result.get("score", 0.0))]
