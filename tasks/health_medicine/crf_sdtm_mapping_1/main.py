"""AgentHLE task: CRF to SDTM mapping specification."""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_crf_sdtm_mapping import ScoreResult, score_mapping_csv  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "crf_sdtm_mapping_1"

VARIANTS = [
    {
        "variant": "base",
        "domain_label": "Concomitant Medications",
        "primary_dataset": "CM",
        "supp_dataset": "SUPPCM",
        "output_file": "cm_mapping.csv",
        "example_row": (
            "CONCOMITANT MEDICATIONS - BASELINE (CONMED BSL),"
            "What is the medication identifier?,"
            "Sponsor-Defined Identifier,"
            "CM,CMSPID,Identifier,CRF,"
            "Map the sponsor-defined medication identifier to CMSPID.,,"
            "NO,"
            "aCRF page 15 annotates Sponsor-Defined Identifier to CMSPID "
            "for baseline concomitant medications."
        ),
    },
    {
        "variant": "vs",
        "domain_label": "Vital Signs",
        "primary_dataset": "VS",
        "supp_dataset": "SUPPVS",
        "output_file": "vs_mapping.csv",
        "example_row": (
            "VITAL SIGNS - BASELINE (VITALS BSL),"
            "Date,Date:,"
            "VS,VSDTC,Timing,CRF,"
            "Convert the collected date to ISO 8601 and populate VSDTC "
            "for each VS test record created from the form.,"
            "ISO 8601 date/datetime,"
            "NO,"
            "aCRF baseline vital signs page maps Date to VSDTC."
        ),
    },
    {
        "variant": "ds",
        "domain_label": "Disposition",
        "primary_dataset": "DS",
        "supp_dataset": "SUPPDS",
        "output_file": "ds_mapping.csv",
        "example_row": (
            "MAIN INFORMED CONSENT (CONSENT),"
            "Consent Was,Consent Was:,"
            "DS,DSTERM,Topic,CRF,"
            "\"When Consent Was = OBTAINED, populate DSTERM with "
            "'INFORMED CONSENT OBTAINED'.\","
            "OBTAINED -> INFORMED CONSENT OBTAINED,"
            "NO,"
            "aCRF page 17 explicitly annotates DSSTDTC when "
            "DSTERM/DSDECOD = INFORMED CONSENT OBTAINED."
        ),
    },
]

SOURCE_DOCUMENTS = [
    "sample_crf.pdf",
    "annotated_crf.pdf",
    "sdtm_define.xml",
    "adam_supp_define.xml",
]


def _as_text(payload: Any) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8-sig")
    return str(payload)


class CrfSdtmMappingConfig(LinuxTaskConfig):
    def __init__(
        self,
        *,
        DOMAIN_NAME: str,
        TASK_NAME: str,
        VARIANT_NAME: str,
        domain_label: str,
        primary_dataset: str,
        supp_dataset: str,
        output_filename: str,
        example_row: str,
    ) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
        )
        self.domain_label = domain_label
        self.primary_dataset = primary_dataset
        self.supp_dataset = supp_dataset
        self.output_filename = output_filename
        self.example_row = example_row

    @property
    def task_brief_file(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def source_documents_dir(self) -> str:
        return f"{self.input_dir}/source_documents"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/{self.output_filename}"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/{self.output_filename}"

    @property
    def task_description(self) -> str:
        return f"""You are preparing a field-level CRF-to-SDTM mapping specification on Linux.

## Your Task
Create the {self.domain_label} mapping for study C4591001.

## Visible Inputs
- Task brief: `{self.task_brief_file}`
- Output contract: `{self.output_contract_file}`
- Source documents directory: `{self.source_documents_dir}`
- Optional Python runtime manifest: `{self.runtime_env_dir}`

The source documents include the sample CRF PDF, annotated CRF PDF, SDTM define.xml,
and supplemental define.xml material. Use those files to identify form fields and
their target SDTM variables for `{self.primary_dataset}` and, where applicable,
`{self.supp_dataset}` supplemental qualifiers.

## Required Output
Save exactly one CSV file:

```text
{self.output_file}
```

The CSV must contain exactly these columns in this order:

```text
crf_form, crf_field_label, crf_item_or_placeholder, sdtm_dataset, sdtm_variable, role, origin, mapping_rule, controlled_terms_or_expected_values, goes_to_suppqual, notes
```

## Example Row

The following shows one correctly formatted CSV row to illustrate the expected
schema (header + data). Include only rows for CRF-sourced fields; omit derived
or protocol-level variables (e.g. STUDYID, DOMAIN, USUBJID) that have no
corresponding CRF field.

```csv
crf_form,crf_field_label,crf_item_or_placeholder,sdtm_dataset,sdtm_variable,role,origin,mapping_rule,controlled_terms_or_expected_values,goes_to_suppqual,notes
{self.example_row}
```

## Constraints
- Produce a mapping specification only, not subject-level records or XPT datasets.
- Keep all generated files inside `{self.output_dir}`.
- Do not modify files under `{self.input_dir}`.
- Use the visible task files only; do not use external web sources.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "task_brief_file": self.task_brief_file,
                "output_contract_file": self.output_contract_file,
                "runtime_env_dir": self.runtime_env_dir,
                "source_documents_dir": self.source_documents_dir,
                "source_documents": [
                    f"{self.source_documents_dir}/{filename}" for filename in SOURCE_DOCUMENTS
                ],
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "output_filename": self.output_filename,
                "primary_dataset": self.primary_dataset,
                "supp_dataset": self.supp_dataset,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


def _config_from_variant(spec: dict[str, str]) -> CrfSdtmMappingConfig:
    return CrfSdtmMappingConfig(
        DOMAIN_NAME=DOMAIN_NAME,
        TASK_NAME=TASK_NAME,
        VARIANT_NAME=spec["variant"],
        domain_label=spec["domain_label"],
        primary_dataset=spec["primary_dataset"],
        supp_dataset=spec["supp_dataset"],
        output_filename=spec["output_file"],
        example_row=spec["example_row"],
    )


@cb.tasks_config(split="train")
def load():
    tasks = []
    for spec in VARIANTS:
        config = _config_from_variant(spec)
        tasks.append(
            cb.Task(
                description=config.task_description,
                metadata=config.to_metadata(),
                computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
            )
        )
    return tasks


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _log_score(result: ScoreResult, variant: str) -> None:
    logger.info("CRF SDTM mapping evaluation %s: %s", variant, json.dumps(result.to_dict()))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    variant = meta["variant_name"]

    for path in (meta["output_file"], meta["reference_file"]):
        if not (await session.file_exists(path) or await session.directory_exists(path)):
            logger.error("Missing evaluation artifact: %s", path)
            return [0.0]

    try:
        agent_csv = _as_text(await session.read_file(meta["output_file"]))
        reference_csv = _as_text(await session.read_file(meta["reference_file"]))
    except Exception as exc:
        logger.exception("Failed to read CRF SDTM evaluation artifacts: %s", exc)
        return [0.0]

    result = score_mapping_csv(agent_csv, reference_csv, variant=variant)
    _log_score(result, variant)
    return [float(result.score)]
