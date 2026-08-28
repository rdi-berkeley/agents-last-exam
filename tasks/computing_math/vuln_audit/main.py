"""ALE task: computing_math/vuln_audit.

Patterned on the canonical Linux demo tasks/demo/hello/main.py and the reference task
tasks/business_finance/american_option_pricing_ls/main.py. The framework auto-stages input/ +
reference/ (REQUIRES_TASK_DATA); start() just ensures the output dir and sanity-checks input;
evaluate() reads output/findings.json + the hidden reference and scores via scripts/. No LLM judge.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_vuln_audit import score_submission  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "vuln_audit"
VARIANT_NAME = "base"


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def repo_dir(self) -> str:
        return f"{self.input_dir}/repo"

    @property
    def readme_file(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def findings_output(self) -> str:
        return f"{self.remote_output_dir}/findings.json"

    @property
    def ground_truth_reference(self) -> str:
        return f"{self.reference_dir}/ground_truth.json"

    @property
    def task_description(self) -> str:
        return (
            "You are a security analyst working on an Ubuntu machine.\n\n"
            f"The directory `{self.repo_dir}` is a small Python service. Audit it for security "
            f"vulnerabilities and write your findings to `{self.findings_output}`.\n\n"
            'Schema (see the README for full details and the CWE scope):\n'
            '{"findings": [{"file": "<file>.py", "cwe": "CWE-NNN", "line": 7, "explanation": "..."}]}\n\n'
            "Rules:\n"
            "- Report each distinct vulnerability class once per file (a file + CWE pair).\n"
            "- Precision matters: false positives are penalized. Not every file is vulnerable.\n"
            f"- See `{self.readme_file}` for the schema and scope. Write exactly to `{self.findings_output}`."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update(
            {
                "repo_dir": self.repo_dir,
                "readme_file": self.readme_file,
                "findings_output": self.findings_output,
                "ground_truth_reference": self.ground_truth_reference,
            }
        )
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    await session.run_command(f"mkdir -p {meta['remote_output_dir']!r}", check=False)
    try:
        await session.read_file(meta["readme_file"])  # framework auto-stages input/
    except Exception as exc:
        raise RuntimeError(f"staged input missing: {meta['readme_file']} unreadable ({exc})")
    logger.info("[vuln_audit] output dir ready; input staged")


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        findings_bytes = await session.read_bytes(meta["findings_output"])
    except Exception as exc:
        logger.info("[vuln_audit] output unreadable at %s: %s", meta["findings_output"], exc)
        return [0.0]
    try:
        reference_bytes = await session.read_bytes(meta["ground_truth_reference"])
        score, report = score_submission(findings_bytes, reference_bytes)
    except Exception as exc:
        logger.exception("[vuln_audit] evaluation failed: %s", exc)
        return [0.0]
    logger.info("[vuln_audit] eval_report=%s", json.dumps(report, sort_keys=True))
    return [score]
