"""AgentHLE task: CRF to SDTM mapping specification."""

import json
import logging
import os
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
TASK_NAME = "crf_sdtm_mapping_4"

VARIANTS = [
    {
        "variant": "base",
        "domain_label": "Adverse Events",
        "primary_dataset": "AE",
        "supp_dataset": "SUPPAE",
        "output_file": "ae_mapping.csv",
        "flag_column": "goes_to_suppqual",
    },
    {
        "variant": "dm",
        "domain_label": "Demography",
        "primary_dataset": "DM",
        "supp_dataset": "SUPPDM",
        "output_file": "dm_mapping.csv",
        "flag_column": "derived_or_assigned",
    },
]

SOURCE_DOCUMENTS = [
    "sample_crf.pdf",
    "annotated_crf.pdf",
    "sdtm_define.xml",
    "supp_define.xml",
]


def _as_text(payload: Any) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8-sig")
    return str(payload)


class CrfSdtmMappingConfig(LinuxTaskConfig):
    def __init__(
        self,
        *,
        REMOTE_OUTPUT_DIR: str,
        DOMAIN_NAME: str,
        TASK_NAME: str,
        VARIANT_NAME: str,
        domain_label: str,
        primary_dataset: str,
        supp_dataset: str,
        output_filename: str,
        flag_column: str,
    ) -> None:
        super().__init__(
            REMOTE_OUTPUT_DIR=REMOTE_OUTPUT_DIR,
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
        )
        self.domain_label = domain_label
        self.primary_dataset = primary_dataset
        self.supp_dataset = supp_dataset
        self.output_filename = output_filename
        self.flag_column = flag_column

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
        return f"{self.remote_output_dir}/{self.output_filename}"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/{self.output_filename}"

    @property
    def task_description(self) -> str:
        flag_description = (
            "`goes_to_suppqual` must mark AE rows as NO and SUPPAE rows as YES."
            if self.flag_column == "goes_to_suppqual"
            else "`derived_or_assigned` must identify whether each DM/SUPPDM mapping is CRF, DERIVED, or ASSIGNED."
        )
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

The CSV must follow the column order and constraints in `{self.output_contract_file}`.
{flag_description}

## Constraints
- Produce a mapping specification only, not subject-level records or XPT datasets.
- Keep all generated files inside `{self.remote_output_dir}`.
- Do not modify files under `{self.input_dir}`.
- Use the visible task files only; do not use external web sources.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
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
                "flag_column": self.flag_column,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


def _config_from_variant(spec: dict[str, str]) -> CrfSdtmMappingConfig:
    return CrfSdtmMappingConfig(
        REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        DOMAIN_NAME=DOMAIN_NAME,
        TASK_NAME=TASK_NAME,
        VARIANT_NAME=spec["variant"],
        domain_label=spec["domain_label"],
        primary_dataset=spec["primary_dataset"],
        supp_dataset=spec["supp_dataset"],
        output_filename=spec["output_file"],
        flag_column=spec["flag_column"],
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
        if not await session.exists(path):
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
