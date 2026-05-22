"""Stage 2 implementation for business_finance/saas_onepager_brand_refresh_instance_1."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_onepager_output import score_output

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "saas_onepager_brand_refresh_instance_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANTS = (("base", "NorthstarOS one-pager brand refresh"),)


@dataclass
class OnePagerBrandRefreshConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "base"
    VARIANT_LABEL: str = "NorthstarOS one-pager brand refresh"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def original_png(self) -> str:
        return rf"{self.input_dir}\original_onepager.png"

    @property
    def edit_request(self) -> str:
        return rf"{self.input_dir}\edit_request.txt"

    @property
    def brand_guide(self) -> str:
        return rf"{self.input_dir}\brand_guide.pdf"

    @property
    def metrics_csv(self) -> str:
        return rf"{self.input_dir}\metrics.csv"

    @property
    def chart_csv(self) -> str:
        return rf"{self.input_dir}\chart_data.csv"

    @property
    def task_brief(self) -> str:
        return rf"{self.input_dir}\task_brief.md"

    @property
    def output_contract(self) -> str:
        return rf"{self.input_dir}\output_contract.json"

    @property
    def assets_dir(self) -> str:
        return rf"{self.input_dir}\assets"

    @property
    def powerpoint_launcher(self) -> str:
        return rf"{self.software_dir}\OpenPowerPoint.bat"

    @property
    def output_png(self) -> str:
        return rf"{self.remote_output_dir}\edited_onepager.png"

    @property
    def output_pptx(self) -> str:
        return rf"{self.remote_output_dir}\edited_onepager.pptx"

    @property
    def reference_png(self) -> str:
        return rf"{self.reference_dir}\edited_onepager.png"

    @property
    def reference_pptx(self) -> str:
        return rf"{self.reference_dir}\edited_onepager.pptx"

    @property
    def reference_regions(self) -> str:
        return rf"{self.reference_dir}\edited_regions.json"

    @property
    def reference_text(self) -> str:
        return rf"{self.reference_dir}\expected_text_fields.json"

    @property
    def reference_numeric(self) -> str:
        return rf"{self.reference_dir}\expected_numeric_fields.json"

    @property
    def reference_chart(self) -> str:
        return rf"{self.reference_dir}\expected_chart_data.json"

    @property
    def reference_constraints(self) -> str:
        return rf"{self.reference_dir}\structural_constraints.json"

    @property
    def reference_thresholds(self) -> str:
        return rf"{self.reference_dir}\evaluation_thresholds.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are editing a SaaS one-pager on a Windows VM.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Runtime Entry Point
- Open PowerPoint from: `{self.powerpoint_launcher}`

## Visible Inputs
- Source one-pager PNG: `{self.original_png}`
- Edit brief: `{self.edit_request}`
- Brand guide PDF: `{self.brand_guide}`
- KPI data CSV: `{self.metrics_csv}`
- Chart data CSV: `{self.chart_csv}`
- Replacement assets directory: `{self.assets_dir}`
- Task brief: `{self.task_brief}`
- Output contract: `{self.output_contract}`

## What You Must Do
1. Read the staged brief and input files.
2. Rebuild the one-pager as a meaningfully editable single-slide PowerPoint.
3. Apply the requested NorthstarOS brand refresh while preserving the overall composition.
4. Save the editable PowerPoint exactly to `{self.output_pptx}`.
5. Export the final slide exactly to `{self.output_png}`.

## Output Requirements
- The `.pptx` must remain meaningfully editable.
- Save only the final required deliverables inside `{self.remote_output_dir}`.
- Do not modify files under `input/`.
"""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "domain_name": self.DOMAIN_NAME,
            "task_name": self.TASK_NAME,
            "variant_name": self.VARIANT_NAME,
            "requires_task_data": self.REQUIRES_TASK_DATA,
            "task_dir": self.task_dir,
            "software_dir": self.software_dir,
            "reference_dir": self.reference_dir,
            "reference_gcs_prefix": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/reference",
            "remote_output_dir": self.remote_output_dir,
            "task_id": TASK_ID,
            "variant_label": self.VARIANT_LABEL,
            "input_dir": self.input_dir,
            "original_png": self.original_png,
            "edit_request": self.edit_request,
            "brand_guide": self.brand_guide,
            "metrics_csv": self.metrics_csv,
            "chart_csv": self.chart_csv,
            "task_brief": self.task_brief,
            "output_contract": self.output_contract,
            "assets_dir": self.assets_dir,
            "powerpoint_launcher": self.powerpoint_launcher,
            "output_png": self.output_png,
            "output_pptx": self.output_pptx,
            "reference_png": self.reference_png,
            "reference_pptx": self.reference_pptx,
            "reference_regions": self.reference_regions,
            "reference_text": self.reference_text,
            "reference_numeric": self.reference_numeric,
            "reference_chart": self.reference_chart,
            "reference_constraints": self.reference_constraints,
            "reference_thresholds": self.reference_thresholds,
        }


def _config_for_variant(variant_name: str, variant_label: str) -> OnePagerBrandRefreshConfig:
    return OnePagerBrandRefreshConfig(VARIANT_NAME=variant_name, VARIANT_LABEL=variant_label)


async def _reset_remote_output_dir(session: cb.DesktopSession, remote_output_dir: str) -> None:
    escaped = remote_output_dir.replace("'", "''")
    command = (
        "powershell -NoProfile -Command "
        f"\"$p = '{escaped}'; "
        "if (Test-Path -LiteralPath $p) { "
        "Get-ChildItem -LiteralPath $p -Force | Remove-Item -Recurse -Force -ErrorAction Stop "
        "} else { "
        "New-Item -ItemType Directory -Path $p | Out-Null "
        '}"'
    )
    result = await session.run_command(command, check=False)
    if result.get("return_code", 1) != 0:
        raise RuntimeError(
            "Failed to reset remote output dir "
            f"{remote_output_dir}: {result.get('stderr') or result.get('stdout')}"
        )
    await session.makedirs(remote_output_dir)


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_config_for_variant(name, label).task_description,
            metadata=_config_for_variant(name, label).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
        for name, label in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _read_required_bytes(
    session: cb.DesktopSession, paths: dict[str, str]
) -> dict[str, bytes]:
    payload: dict[str, bytes] = {}
    for key, remote_path in paths.items():
        if not await session.exists(remote_path):
            raise FileNotFoundError(remote_path)
        payload[key] = await session.read_bytes(remote_path)
    return payload


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_paths = {
        "edited_onepager.png": meta["output_png"],
        "edited_onepager.pptx": meta["output_pptx"],
    }

    try:
        output_bytes = await _read_required_bytes(session, output_paths)
    except FileNotFoundError as exc:
        logger.error("[%s] Missing evaluation artifact: %s", meta["variant_name"], exc)
        return [0.0]
    except Exception as exc:
        logger.error(
            "[%s] Failed to read remote evaluation artifacts: %s", meta["variant_name"], exc
        )
        return [0.0]

    reference_paths = {
        "edited_onepager.png": meta["reference_png"],
        "edited_onepager.pptx": meta["reference_pptx"],
        "edited_regions.json": meta["reference_regions"],
        "expected_text_fields.json": meta["reference_text"],
        "expected_numeric_fields.json": meta["reference_numeric"],
        "expected_chart_data.json": meta["reference_chart"],
        "structural_constraints.json": meta["reference_constraints"],
        "evaluation_thresholds.json": meta["reference_thresholds"],
    }
    try:
        reference_bytes = await _read_required_bytes(session, reference_paths)
    except FileNotFoundError as exc:
        logger.error("[%s] Missing staged evaluator reference: %s", meta["variant_name"], exc)
        return [0.0]
    except Exception as exc:
        logger.error(
            "[%s] Failed to read staged evaluator reference: %s", meta["variant_name"], exc
        )
        return [0.0]

    with tempfile.TemporaryDirectory(prefix=f"{TASK_NAME}_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        output_dir = tmp_root / "output"
        reference_dir = tmp_root / "reference"
        output_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)

        for name, payload in output_bytes.items():
            (output_dir / name).write_bytes(payload)
        for name, payload in reference_bytes.items():
            (reference_dir / name).write_bytes(payload)

        result = score_output(output_dir, reference_dir)
        logger.info(
            "[%s] evaluation result: %s",
            meta["variant_name"],
            json.dumps(result, ensure_ascii=False),
        )
        return [1.0 if result.get("pass") else 0.0]
