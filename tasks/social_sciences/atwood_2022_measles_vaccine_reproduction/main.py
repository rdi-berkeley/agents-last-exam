"""AgentHLE task for Atwood 2022 measles vaccine reproduction."""

from __future__ import annotations

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

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "social_sciences"
TASK_NAME = "atwood_2022_measles_vaccine_reproduction"
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
class AtwoodReproductionConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    @property
    def output_dir_name(self) -> str:
        return _normalize_output_dir_name(self.OUTPUT_SUBDIR)

    @property
    def hidden_gold_dir(self) -> str:
        return f"{self.reference_dir}/hidden_gold"

    @property
    def output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/TASK_PROMPT.md"

    @property
    def paper_file(self) -> str:
        return f"{self.input_dir}/paper.pdf"

    @property
    def replication_package(self) -> str:
        return f"{self.input_dir}/replication_package.zip"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM as an applied-economics replication analyst.

## Task Directory
`{self.task_dir}`

## Visible Inputs
- Task prompt: `{self.task_prompt_file}`
- Published paper: `{self.paper_file}`
- Online appendix: `{self.input_dir}/paper-online-appendix.pdf`
- Archived replication package: `{self.replication_package}`
- Task metadata: `{self.input_dir}/task.json`
- Environment manifest: `{self.input_dir}/environment_manifest.json`
- Allowed repairs: `{self.input_dir}/allowed_repairs.json`
- Output schemas/examples: `{self.input_dir}/output_schemas`

## Your Task
Use the staged public materials to reproduce the coefficient report for Alicia
Atwood (2022), Table 2, row "Vaccination effect".

Write all final files under `{self.output_dir}`:
- `execution_log.txt`
- `repair_log.json`
- `paper_coefficients.json`
- `code_coefficients.json`
- `coefficient_matches.csv`
- `verdict.json`

Follow the file contracts in `{self.task_prompt_file}` and the schemas under
`{self.input_dir}/output_schemas`. Do not modify files under `{self.input_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": self.VARIANT_NAME,
                "output_dir_name": self.output_dir_name,
                "hidden_gold_dir": self.hidden_gold_dir,
                "output_dir": self.output_dir,
                "task_prompt_file": self.task_prompt_file,
                "paper_file": self.paper_file,
                "replication_package": self.replication_package,
                "eval_tmp_dir": EVAL_TMP_DIR,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = AtwoodReproductionConfig()


@cb.tasks_config(split="train")
def load():
    cfg = AtwoodReproductionConfig()
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
    required_reference_paths = [
        meta["reference_dir"],
        meta["hidden_gold_dir"],
        f"{meta['hidden_gold_dir']}/gold_paper_coefficients.json",
        f"{meta['hidden_gold_dir']}/gold_verdict.json",
    ]
    missing_reference = [
        path for path in required_reference_paths if not (await session.file_exists(path) or await session.directory_exists(path))
    ]
    if missing_reference:
        logger.error("missing evaluator reference paths: %s", "; ".join(missing_reference))
        return [0.0]

    score_script = f"{meta['eval_tmp_dir']}/score_outputs.py"
    try:
        await session.interface.create_dir(meta["eval_tmp_dir"])
        await session.write_file(score_script, SCORE_SCRIPT)
        result = await session.run_command(
            f'python "{score_script}" '
            f'--output "{meta["output_dir"]}" '
            f'--reference "{meta["hidden_gold_dir"]}"'
        )
        stdout = result.get("stdout", "")
        if result.get("return_code", 1) != 0:
            logger.warning("score script failed: %s", result.get("stderr", ""))
            return [0.0]
        report = json.loads(stdout)
        logger.info("score report: %s", report)
        return [float(report.get("score", 0.0))]
    except Exception as exc:
        logger.error("evaluation failed: %s", exc)
        return [0.0]
