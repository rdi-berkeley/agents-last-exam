"""AgentHLE task: public_health_mask_mandate_ratio."""

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

# cua_bench may exec_module without pre-registering the module.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.health_medicine.public_health_mask_mandate_ratio.scripts.score_outputs import (
    score_output_bundle,
)


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


async def _missing(session: cb.DesktopSession, path: str, *, label: str) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("Missing %s: %s", label, path)
    return True


@dataclass
class PublicHealthMaskMandateRatioConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "health_medicine"
    TASK_NAME: str = "public_health_mask_mandate_ratio"
    VARIANT_NAME: str = "base"
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def output_schema_file(self) -> str:
        return f"{self.input_dir}/output_schema.json"

    @property
    def county_panel_file(self) -> str:
        return f"{self.input_dir}/county_panel.csv"

    @property
    def matched_pairs_file(self) -> str:
        return f"{self.input_dir}/matched_pairs.csv"

    @property
    def software_readme(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def output_results(self) -> str:
        return f"{self.remote_output_dir}/results.json"

    @property
    def reference_output(self) -> str:
        return f"{self.reference_dir}/reference_output.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to estimate the moment-in-time effect of a county-level mask mandate.

Visible task files:
- `{self.task_prompt_file}`
- `{self.output_schema_file}`
- `{self.county_panel_file}`
- `{self.matched_pairs_file}`
- `{self.software_readme}`

What you must do:
1. Read the staged task prompt and output schema.
2. Build the stacked matched-pair county-day analysis frame and engineer the required lagged features.
3. Fit the specified quasi-Poisson mixed model with the spline-by-treatment interaction.
4. Compute the policy-effect ratios at relative days 14, 28, and 42, plus the arithmetic mean over days 1..42.
5. Write exactly one file to `{self.remote_output_dir}`:
   - `{self.output_results}`

Output rules:
- Write valid JSON only.
- Use exactly the six keys listed in `output_schema.json`.
- Output numeric values, not strings.
- Do not modify the staged input files and do not write outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{self.DOMAIN_NAME}/{self.TASK_NAME}",
                "task_prompt_file": self.task_prompt_file,
                "output_schema_file": self.output_schema_file,
                "county_panel_file": self.county_panel_file,
                "matched_pairs_file": self.matched_pairs_file,
                "software_readme": self.software_readme,
                "output_results": self.output_results,
                "reference_output": self.reference_output,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


config = PublicHealthMaskMandateRatioConfig()


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
        ("output_results", "results JSON"),
        ("reference_output", "hidden reference JSON"),
    ]:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            logger.error("Missing %s at %s", label, meta[key])
            return [0.0]

    with tempfile.TemporaryDirectory(prefix="mask_mandate_ratio_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_results = tmp / "results.json"
        local_reference = tmp / "reference_output.json"

        try:
            local_results.write_bytes(await session.read_bytes(meta["output_results"]))
            local_reference.write_bytes(await session.read_bytes(meta["reference_output"]))
            result = score_output_bundle(
                results_path=local_results,
                reference_path=local_reference,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Evaluation failed: %s", exc)
            return [0.0]

    logger.info("evaluation=%s", json.dumps(result, sort_keys=True)[:2000])
    return [float(result.get("score", 0.0))]
