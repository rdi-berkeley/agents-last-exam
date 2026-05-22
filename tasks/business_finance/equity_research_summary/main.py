"""equity_research_summary — Tesla one-page equity research workbook task."""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_workbook import WorkbookScoreResult, score_workbook_bytes

logger = logging.getLogger(__name__)

VARIANTS = [
    (
        "tsla_fy2023",
        "Single Tesla equity-research workbook using live Yahoo pages plus fixed FY2023 SEC values.",
    )
]


def win_join(*parts: str) -> str:
    return str(PureWindowsPath(*parts))


@dataclass
class EquityResearchSummaryConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "business_finance"

    TASK_NAME: str = "equity_research_summary"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_dir(self) -> str:
        return win_join(self.task_dir, "input")

    @property
    def instruction_file(self) -> str:
        return win_join(self.input_dir, "instruction.md")

    @property
    def source_urls_file(self) -> str:
        return win_join(self.input_dir, "source_urls.txt")

    @property
    def financials_file(self) -> str:
        return win_join(self.input_dir, "tsla_fy2023_financials.md")

    @property
    def output_workbook(self) -> str:
        return win_join(self.remote_output_dir, "TSLA_Financial_Summary.xlsx")

    @property
    def reference_manifest(self) -> str:
        return win_join(self.reference_dir, "reference_manifest.json")

    @property
    def calc_launcher(self) -> str:
        return win_join(self.software_dir, "launch_calc.bat")

    @property
    def task_description(self) -> str:
        return f"""\
You are an equity research analyst preparing a one-page Tesla workbook in LibreOffice Calc.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Runtime Entry Points
- Read the staged instructions: `{self.instruction_file}`
- Read the FY2023 financial data: `{self.financials_file}`
- Read the allowed public source URLs: `{self.source_urls_file}`
- Launch LibreOffice Calc from: `{self.calc_launcher}`

## Your Task
Create a single worksheet for a Tesla financial summary workbook.

Use Yahoo Finance for the live market-data / valuation fields.
Use the exact FY2023 values from `{self.financials_file}` for the fixed
annual financial-statement sections — do NOT look them up separately.

Read `{self.instruction_file}` for the full list of required sections,
exact label text, and which fields must be formulas.

## Output Requirements
- Save exactly one workbook to: `{self.output_workbook}`
- The filename must be exactly `TSLA_Financial_Summary.xlsx`
- Use spreadsheet formulas (starting with `=`) for all calculated fields
- Section headers should be bold with a colored background fill
- Keep the live Yahoo Finance cells populated with actual numeric values
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_dir": self.input_dir,
                "instruction_file": self.instruction_file,
                "financials_file": self.financials_file,
                "source_urls_file": self.source_urls_file,
                "output_workbook": self.output_workbook,
                "reference_manifest": self.reference_manifest,
                "calc_launcher": self.calc_launcher,
            }
        )
        return metadata


def _cfg_for_variant(spec: tuple[str, str]) -> EquityResearchSummaryConfig:
    return EquityResearchSummaryConfig(
        VARIANT_NAME=spec[0],
        VARIANT_LABEL=spec[1],
    )


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_cfg_for_variant(spec).task_description,
            metadata=_cfg_for_variant(spec).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
        for spec in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _log_result(tag: str, result: WorkbookScoreResult) -> None:
    logger.info(
        "[%s] score=%.1f passed=%s reasons=%s",
        tag,
        result.score,
        result.passed,
        result.reasons,
    )


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_workbook = meta["output_workbook"]
    reference_manifest = meta["reference_manifest"]

    if not await session.exists(reference_manifest):
        logger.error("Missing reference manifest: %s", reference_manifest)
        return [0.0]
    if not await session.exists(output_workbook):
        logger.error("Missing output workbook: %s", output_workbook)
        return [0.0]

    try:
        agent_bytes = await session.read_bytes(output_workbook)
        manifest_text = await session.read_file(reference_manifest)
    except Exception as exc:
        logger.error("Failed to read staged workbook/manifest: %s", exc)
        return [0.0]

    try:
        manifest = json.loads(manifest_text)
        result = score_workbook_bytes(agent_bytes=agent_bytes, manifest=manifest)
    except Exception as exc:
        logger.error("Workbook scoring failed: %s", exc)
        return [0.0]

    _log_result(meta["variant_name"], result)
    return [result.score]
