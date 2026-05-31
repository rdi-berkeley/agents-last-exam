"""taxform_4_1 — browser-based tax form completion task."""

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

from score_taxform_outputs import score_variant_outputs  # noqa: E402

logger = logging.getLogger(__name__)


def win_join(*parts: str) -> str:
    return str(PureWindowsPath(*parts))


SOURCE_FORM_INFO = {
    "formw2": {
        "title": "Form W-2 — Wage and Tax Statement",
        "url": "http://localhost:8080/forms/formw2.html?data=/",
    },
    "form1099int": {
        "title": "Form 1099-INT — Interest Income",
        "url": "http://localhost:8080/forms/form1099int.html?data=/",
    },
    "form1099r": {
        "title": "Form 1099-R — Distributions From Pensions/Annuities",
        "url": "http://localhost:8080/forms/form1099r.html?data=/",
    },
    "form1099msc": {
        "title": "Form 1099-MISC — Miscellaneous Information",
        "url": "http://localhost:8080/forms/form1099msc.html?data=/",
    },
}

OUTPUT_FORM_INFO = {
    "form1040": {
        "title": "Form 1040 — U.S. Individual Income Tax Return",
        "url": "http://localhost:8080/forms/form1040.html",
        "output_file": "form1040_output.json",
    },
    "form1040s1": {
        "title": "Schedule 1 — Additional Income and Adjustments",
        "url": "http://localhost:8080/forms/form1040s1.html",
        "output_file": "form1040s1_output.json",
    },
}


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    variant_label: str
    source_forms: tuple[str, ...]
    output_forms: tuple[str, ...]


VARIANTS = [
    VariantSpec(
        variant_name="variant_1",
        variant_label="W-2 income with a single Form 1040 output.",
        source_forms=("formw2",),
        output_forms=("form1040",),
    ),
    VariantSpec(
        variant_name="variant_2",
        variant_label="1099-INT and 1099-R source forms with a single Form 1040 output.",
        source_forms=("form1099int", "form1099r"),
        output_forms=("form1040",),
    ),
    VariantSpec(
        variant_name="variant_3",
        variant_label="W-2 plus 1099-R source forms, including dependent handling on Form 1040.",
        source_forms=("formw2", "form1099r"),
        output_forms=("form1040",),
    ),
    VariantSpec(
        variant_name="variant_4",
        variant_label="W-2, 1099-INT, and 1099-R source forms with child-tax-credit-related fields on Form 1040.",
        source_forms=("formw2", "form1099int", "form1099r"),
        output_forms=("form1040",),
    ),
    VariantSpec(
        variant_name="variant_5",
        variant_label="W-2, 1099-INT, 1099-R, and 1099-MISC source forms with both Schedule 1 and Form 1040 outputs.",
        source_forms=("formw2", "form1099int", "form1099r", "form1099msc"),
        output_forms=("form1040s1", "form1040"),
    ),
]


@dataclass
class TaxFormTaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "business_finance"

    TASK_NAME: str = "taxform_4_1"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""
    SOURCE_FORMS: tuple[str, ...] = ()
    OUTPUT_FORMS: tuple[str, ...] = ()

    @property
    def input_dir(self) -> str:
        return win_join(self.task_dir, "input")

    @property
    def instruction_file(self) -> str:
        return win_join(self.input_dir, "instruction.txt")

    @property
    def http_server_bat(self) -> str:
        return win_join(self.software_dir, "start_http_server.bat")

    @property
    def required_output_files(self) -> list[str]:
        return [OUTPUT_FORM_INFO[form_name]["output_file"] for form_name in self.OUTPUT_FORMS]

    @property
    def task_description(self) -> str:
        source_lines = "\n".join(
            f"- {SOURCE_FORM_INFO[form_name]['title']}: `{SOURCE_FORM_INFO[form_name]['url']}`"
            for form_name in self.SOURCE_FORMS
        )
        output_lines = "\n".join(
            f"- {OUTPUT_FORM_INFO[form_name]['title']}: `{OUTPUT_FORM_INFO[form_name]['url']}`"
            for form_name in self.OUTPUT_FORMS
        )
        output_file_lines = "\n".join(f"- `{name}`" for name in self.required_output_files)
        return f"""\
You are preparing a U.S. tax return inside a browser-based local task environment.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Runtime Entry Points
- Start the local HTTP server by running: `{self.http_server_bat}`
- Read the taxpayer instructions at: `{self.instruction_file}`

## Source Forms To Read
These pages are prefilled reference documents. Read them, but do not edit them.
{source_lines}

## Output Forms To Complete
Fill these blank return pages in the browser, then click `Save Results` on each one.
{output_lines}

## What You Must Do
1. Run `{self.http_server_bat}` to serve the staged `input/` directory on `http://localhost:8080`.
2. Read `{self.instruction_file}` carefully.
3. Open every source form listed above and extract the needed values.
4. Open every output form listed above and complete the tax return.
5. Click `Save Results` on each required output form.
6. Move the downloaded JSON file(s) into `{self.output_dir}`.

## Required Final Output Files
Place these exact files in `{self.output_dir}`:
{output_file_lines}
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_dir": self.input_dir,
                "instruction_file": self.instruction_file,
                "http_server_bat": self.http_server_bat,
                "source_forms": list(self.SOURCE_FORMS),
                "output_forms": list(self.OUTPUT_FORMS),
                "required_output_files": self.required_output_files,
            }
        )
        return metadata


def _cfg_for_variant(spec: VariantSpec) -> TaxFormTaskConfig:
    return TaxFormTaskConfig(
        VARIANT_NAME=spec.variant_name,
        VARIANT_LABEL=spec.variant_label,
        SOURCE_FORMS=spec.source_forms,
        OUTPUT_FORMS=spec.output_forms,
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


async def _read_json_remote(session: cb.DesktopSession, path: str) -> dict[str, Any]:
    return json.loads(await session.read_file(path))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_dir = meta["output_dir"]
    reference_dir = meta["reference_dir"]
    output_forms: list[str] = meta["output_forms"]

    reference_payloads: dict[str, dict[str, Any]] = {}
    output_payloads: dict[str, dict[str, Any]] = {}
    alt_reference_payloads: dict[str, dict[str, Any]] = {}

    for form_name in output_forms:
        reference_path = win_join(reference_dir, f"{form_name}_gt.json")
        alt_reference_path = win_join(reference_dir, f"{form_name}_gt_alt.json")
        output_path = win_join(output_dir, f"{form_name}_output.json")

        logger.info("[%s] Checking reference file %s", meta["variant_name"], reference_path)
        if not (await session.file_exists(reference_path) or await session.directory_exists(reference_path)):
            logger.error("[%s] Missing reference file: %s", meta["variant_name"], reference_path)
            return [0.0]
        logger.info("[%s] Checking output file %s", meta["variant_name"], output_path)
        if not (await session.file_exists(output_path) or await session.directory_exists(output_path)):
            logger.error("[%s] Missing agent output: %s", meta["variant_name"], output_path)
            return [0.0]

        try:
            logger.info("[%s] Reading reference JSON %s", meta["variant_name"], reference_path)
            reference_payloads[form_name] = await _read_json_remote(session, reference_path)
            logger.info("[%s] Reading output JSON %s", meta["variant_name"], output_path)
            output_payloads[form_name] = await _read_json_remote(session, output_path)
            if (await session.file_exists(alt_reference_path) or await session.directory_exists(alt_reference_path)):
                logger.info(
                    "[%s] Reading alt reference JSON %s", meta["variant_name"], alt_reference_path
                )
                alt_reference_payloads[form_name] = await _read_json_remote(
                    session, alt_reference_path
                )
        except Exception as exc:
            logger.error("[%s] Failed to read JSON payloads: %s", meta["variant_name"], exc)
            return [0.0]

    result = score_variant_outputs(
        reference_payloads, output_payloads, alt_reference_payloads or None
    )
    logger.info(
        "[%s] Evaluation result: %s", meta["variant_name"], json.dumps(result, ensure_ascii=True)
    )
    return [float(result["score"])]
