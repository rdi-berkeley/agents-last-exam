"""AgentHLE task: data_pipeline_etl_instance_1."""

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
from tasks.computing_math.data_pipeline_etl_instance_1.scripts.score_outputs import (
    score_output_bundle,
)
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


async def _missing(session: cb.DesktopSession, path: str, *, label: str) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("Missing %s: %s", label, path)
    return True


async def _executable(session: cb.DesktopSession, path: str, *, label: str) -> bool:
    result = await session.run_command(f'test -x "{path}" && printf "__ok__"', check=False)
    if result.get("stdout", "").strip() == "__ok__":
        return True
    logger.error("Non-executable %s: %s", label, path)
    return False


@dataclass
class DataPipelineEtlConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "computing_math"
    TASK_NAME: str = "data_pipeline_etl_instance_1"
    VARIANT_NAME: str = "base"
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def schema_spec_file(self) -> str:
        return f"{self.input_dir}/target_schema_spec.json"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def transactions_dir(self) -> str:
        return f"{self.input_dir}/raw_transactions"

    @property
    def customers_file(self) -> str:
        return f"{self.input_dir}/raw_customers/customers.json"

    @property
    def products_file(self) -> str:
        return f"{self.input_dir}/raw_products/product_catalog.tsv"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def software_readme(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_etl_env.sh"

    @property
    def output_db(self) -> str:
        return f"{self.remote_output_dir}/warehouse.db"

    @property
    def output_report(self) -> str:
        return f"{self.remote_output_dir}/data_quality_report.json"

    @property
    def output_summary(self) -> str:
        return f"{self.remote_output_dir}/warehouse_summary.json"

    @property
    def runtime_scratch_dir(self) -> str:
        return f"{self.remote_output_dir}/_runtime"

    @property
    def reference_db(self) -> str:
        return f"{self.reference_dir}/warehouse.db"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to build a cleaned SQLite warehouse from messy retail source files.

Visible task files:
- `{self.task_prompt_file}`
- `{self.schema_spec_file}`
- `{self.output_contract_file}`
- `{self.transactions_dir}/`
- `{self.customers_file}`
- `{self.products_file}`
- `{self.runtime_env_dir}/`
- `{self.software_readme}`
- `{self.python_wrapper}`

What you must do:
1. Read the staged task prompt, schema spec, and output contract.
2. Build the final SQLite warehouse and truthful JSON sidecars from the staged raw inputs.
3. Write these three deliverables at the output-directory root:
   - `{self.output_db}`
   - `{self.output_report}`
   - `{self.output_summary}`
4. If you use the staged helper wrapper, its runtime scratch may live under `{self.runtime_scratch_dir}`.

Rules:
- Do not modify files under `input/`.
- Keep the final deliverables at the output-directory root.
- Write valid SQLite / JSON outputs only.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{self.DOMAIN_NAME}/{self.TASK_NAME}",
                "task_prompt_file": self.task_prompt_file,
                "schema_spec_file": self.schema_spec_file,
                "output_contract_file": self.output_contract_file,
                "transactions_dir": self.transactions_dir,
                "customers_file": self.customers_file,
                "products_file": self.products_file,
                "runtime_env_dir": self.runtime_env_dir,
                "software_readme": self.software_readme,
                "python_wrapper": self.python_wrapper,
                "output_db": self.output_db,
                "output_report": self.output_report,
                "output_summary": self.output_summary,
                "runtime_scratch_dir": self.runtime_scratch_dir,
                "reference_db": self.reference_db,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


config = DataPipelineEtlConfig()


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
        ("output_db", "candidate warehouse"),
        ("output_report", "candidate quality report"),
        ("output_summary", "candidate summary"),
        ("reference_db", "hidden reference warehouse"),
    ]:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            logger.error("Missing %s at %s", label, meta[key])
            return [0.0]

    with tempfile.TemporaryDirectory(prefix="data_pipeline_etl_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        candidate_dir = tmp / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        reference_db = tmp / "reference_warehouse.db"

        try:
            (candidate_dir / "warehouse.db").write_bytes(await session.read_bytes(meta["output_db"]))
            (candidate_dir / "data_quality_report.json").write_bytes(
                await session.read_bytes(meta["output_report"])
            )
            (candidate_dir / "warehouse_summary.json").write_bytes(
                await session.read_bytes(meta["output_summary"])
            )
            reference_db.write_bytes(await session.read_bytes(meta["reference_db"]))
            result = score_output_bundle(
                output_dir=candidate_dir,
                reference_db_path=reference_db,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Evaluation failed: %s", exc)
            return [0.0]

    logger.info("evaluation=%s", json.dumps(result, sort_keys=True)[:4000])
    return [float(result.get("score", 0.0))]
