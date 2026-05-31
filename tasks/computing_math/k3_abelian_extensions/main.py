"""AgentHLE task: computing_math/k3_abelian_extensions."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

VERIFY_SCRIPT_PATH = SCRIPTS_DIR / "verify_outputs.py"

VARIANTS = (
    ("h_4_4_4_m_1_8", "(Z/4)^3, m=1..8", [4, 4, 4], [1, 8]),
    ("h_2_6_6_m_1_6", "(Z/6)^2 x Z/2, m=1..6", [2, 6, 6], [1, 6]),
    ("h_3_3_6_m_1_6", "Z/6 x (Z/3)^2, m=1..6", [3, 3, 6], [1, 6]),
    ("h_2_4_8_m_1_8", "Z/8 x Z/4 x Z/2, m=1..8", [2, 4, 8], [1, 8]),
    ("h_2_2_2_2_2_m_1_4", "(Z/2)^5, m=1..4", [2, 2, 2, 2, 2], [1, 4]),
    ("h_2_2_2_4_m_1_4", "Z/4 x (Z/2)^3, m=1..4", [2, 2, 2, 4], [1, 4]),
)


def _product(values: list[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def _load_verify_module():
    spec = importlib.util.spec_from_file_location(
        "k3_abelian_extensions_verify_outputs", VERIFY_SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load verifier module from {VERIFY_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load_verify_module()
from oracle import build_expected_output  # noqa: E402


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: float | None = None,
    check: bool = False,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


@dataclass
class AbelianExtensionsConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "computing_math"
    TASK_NAME: str = "k3_abelian_extensions"
    VARIANT_NAME: str = ""
    VARIANT_TITLE: str = ""
    H_INVARIANT_FACTORS: list[int] | None = None
    M_SEARCH_RANGE: list[int] | None = None

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.task_dir}/output_test_neg"

    @property
    def config_file(self) -> str:
        return f"{self.input_dir}/config.json"

    @property
    def task_spec_file(self) -> str:
        return f"{self.input_dir}/task_specification.md"

    @property
    def gap_wrapper(self) -> str:
        return f"{self.software_dir}/run_gap.sh"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/results.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Your Task
Solve the staged finite abelian group extension classification problem for variant `{self.VARIANT_NAME}` ({self.VARIANT_TITLE}).

## Visible Files
- Variant config: `{self.config_file}`
- Mathematical task specification: `{self.task_spec_file}`
- GAP launcher: `{self.gap_wrapper}`

You may also use `/usr/bin/gap`, Python, shell scripts, or text editors available on the VM.

## What You Must Do
1. Read `{self.config_file}` and `{self.task_spec_file}` carefully.
2. Enumerate every finite abelian group `G` in the required short exact sequences across the full `m` search range.
3. Determine the `product_type` classification for every valid extension group.
4. Save exactly one JSON file at `{self.output_file}`.

## Required Output Schema
Write one JSON object with these top-level keys:
- `total_extensions`
- `extensions`
- `non_product_type_count`
- `non_product_type`

Each object in `extensions` and `non_product_type` must contain:
- `m`
- `G_invariant_factors`
- `G_order`
- `product_type`

Use invariant factors in ascending order with the divisibility condition satisfied. Order `extensions` by `(m, G_invariant_factors)`.

## Output Rules
- Save your final deliverable only under `{self.output_dir}`.
- Do not modify files under `{self.input_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "output_dir": self.output_dir,
                "config_file": self.config_file,
                "task_spec_file": self.task_spec_file,
                "gap_wrapper": self.gap_wrapper,
                "output_file": self.output_file,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
                "variant_title": self.VARIANT_TITLE,
                "h_invariant_factors": list(self.H_INVARIANT_FACTORS or []),
                "h_order": int(_product(list(self.H_INVARIANT_FACTORS or []))),
                "m_search_range": list(self.M_SEARCH_RANGE or []),
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    tasks = []
    for variant_name, variant_title, h_invariant_factors, m_search_range in VARIANTS:
        cfg = AbelianExtensionsConfig(
            VARIANT_NAME=variant_name,
            VARIANT_TITLE=variant_title,
            H_INVARIANT_FACTORS=list(h_invariant_factors),
            M_SEARCH_RANGE=list(m_search_range),
        )
        tasks.append(
            cb.Task(
                description=cfg.task_description,
                metadata=cfg.to_metadata(),
                computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
            )
        )
    return tasks


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_file = meta["output_file"]

    if not (await session.file_exists(output_file) or await session.directory_exists(output_file)):
        logger.error("[%s] Missing agent output at %s", meta["variant_name"], output_file)
        return [0.0]

    try:
        agent_output = (await session.read_bytes(output_file)).decode("utf-8-sig")
    except Exception as exc:
        logger.error("[%s] Unable to read agent output: %s", meta["variant_name"], exc)
        return [0.0]

    try:
        reference_output = json.dumps(
            build_expected_output(
                h_invariant_factors=list(meta["h_invariant_factors"]),
                m_search_range=list(meta["m_search_range"]),
            ),
            indent=2,
            sort_keys=True,
        )
        report = VERIFY_MODULE.verify_submission_texts(agent_output, reference_output)
    except Exception as exc:
        logger.error("[%s] Verification failed: %s", meta["variant_name"], exc)
        return [0.0]

    score = float(report.get("score", 0.0))
    logger.info(
        "[%s] score=%.3f passed=%s reason=%s",
        meta["variant_name"],
        score,
        report.get("passed"),
        report.get("reason"),
    )
    return [score]
