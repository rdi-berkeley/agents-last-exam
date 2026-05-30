"""AgentHLE task: reproduce the Fama-French five-factor model from public data."""

import json
import logging
import os
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

from score_outputs import (REQUIRED_COLUMNS, ScoreResult,  # noqa: E402
                           score_factor_csv)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "ff5_public_reconstruction"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
OBSERVABLE_TRACE_ENV = "FF5_OBSERVABLE_TRACE_PATHS"


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8-sig") if isinstance(payload, bytes) else str(payload)


def _split_trace_paths(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [path for path in raw_value.split(os.pathsep) if path]


def _log_score(result: ScoreResult) -> None:
    logger.info(
        "score=%.6f passed=%s aligned_rows=%s hard_gate=%s reason=%s",
        result.score,
        result.passed,
        result.aligned_rows,
        result.hard_gate,
        result.reason,
    )
    logger.info("details=%s", json.dumps(result.to_dict(), ensure_ascii=True))


@dataclass
class FF5PublicReconstructionConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    OUTPUT_FILENAME: str = "ff5_factors_monthly_2015_onward.csv"

    @property
    def agent_prompt_file(self) -> str:
        return f"{self.input_dir}/agent_prompt.md"

    @property
    def visible_readme_file(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def visible_manifest_file(self) -> str:
        return f"{self.input_dir}/manifest.json"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def runtime_pyproject(self) -> str:
        return f"{self.runtime_env_dir}/pyproject.toml"

    @property
    def runtime_lock(self) -> str:
        return f"{self.runtime_env_dir}/uv.lock"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python"

    @property
    def uv_wrapper(self) -> str:
        return f"{self.software_dir}/uv"

    @property
    def browser_wrapper(self) -> str:
        return f"{self.software_dir}/browser"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/{self.OUTPUT_FILENAME}"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/normalized_gold_2015_onward.csv"

    @property
    def evaluation_contract_file(self) -> str:
        return f"{self.reference_dir}/evaluation_contract.json"

    @property
    def task_description(self) -> str:
        columns = ",".join(REQUIRED_COLUMNS)
        return f"""\
You are working on a Linux finance benchmark task.

Read these staged files first:
- `{self.agent_prompt_file}`
- `{self.visible_readme_file}`
- `{self.visible_manifest_file}`

Task-local entry points:
- Browser launcher: `{self.browser_wrapper}`
- Python wrapper: `{self.python_wrapper}`
- UV wrapper: `{self.uv_wrapper}`

If you want the pinned Python runtime manifest, it is staged at:
- `{self.runtime_pyproject}`
- `{self.runtime_lock}`

Your job is to reconstruct the monthly U.S. Fama-French five factors from public data starting at `2015-01`, following the full instructions in `agent_prompt.md`.

Write exactly one required file:
- `{self.output_file}`

Output requirements:
- filename must be exactly `{self.OUTPUT_FILENAME}`
- columns must be exactly `{columns}`
- `date` format must be `YYYY-MM`
- values must be monthly factor values in percent units
- no duplicate dates

Constraints:
1. Follow the allowlist and workflow restrictions in `agent_prompt.md` and `manifest.json`.
2. If you install Python dependencies, use the staged runtime manifest rather than modifying benchmark-owned evaluator state.
3. Write only your task output under `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": self.VARIANT_NAME,
                "agent_prompt_file": self.agent_prompt_file,
                "visible_readme_file": self.visible_readme_file,
                "visible_manifest_file": self.visible_manifest_file,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lock": self.runtime_lock,
                "python_wrapper": self.python_wrapper,
                "uv_wrapper": self.uv_wrapper,
                "browser_wrapper": self.browser_wrapper,
                "output_file": self.output_file,
                "output_filename": self.OUTPUT_FILENAME,
                "required_columns": list(REQUIRED_COLUMNS),
                "reference_file": self.reference_file,
                "evaluation_contract_file": self.evaluation_contract_file,
                "observable_trace_env": OBSERVABLE_TRACE_ENV,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = FF5PublicReconstructionConfig()


@cb.tasks_config(split="train")
def load():
    cfg = FF5PublicReconstructionConfig()
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
        output_csv = await session.read_file(output_file)
        reference_csv = await session.read_file(reference_file)
        contract_text = await session.read_file(contract_file)
    except Exception as exc:
        logger.error("Failed to read task artifacts from VM: %s", exc)
        return [0.0]

    trace_paths = _split_trace_paths(os.environ.get(meta["observable_trace_env"]))
    trace_payloads: list[str] = []
    for trace_path in trace_paths:
        try:
            if await session.exists(trace_path):
                trace_payloads.append(_as_text(await session.read_file(trace_path)))
        except Exception as exc:
            logger.warning("Could not read observable trace path %s: %s", trace_path, exc)

    result = score_factor_csv(
        output_csv,
        reference_csv,
        contract_text=contract_text,
        trace_texts=trace_payloads,
        trace_paths=trace_paths,
    )
    _log_score(result)
    return [float(result.score)]
