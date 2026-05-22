"""AgentHLE task: health_medicine/sa_aki_phenotyping."""

import asyncio
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

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.health_medicine.sa_aki_phenotyping.scripts.score_outputs import ScoreResult, score_csv_texts


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "sa_aki_phenotyping"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
OUTPUT_FILENAME = "sa_aki_patients.csv"


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


def _log_score(result: ScoreResult) -> None:
    logger.info(
        "score=%.6f passed=%s reason=%s hard_gate=%s exact_match=%s",
        result.score,
        result.passed,
        result.reason,
        result.hard_gate,
        result.exact_match,
    )
    logger.info("details=%s", json.dumps(result.to_dict(), ensure_ascii=True))


@dataclass
class SAAKIPhenotypingConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def mimic_root(self) -> str:
        return rf"{self.input_dir}\mimic_synthetic"

    @property
    def hosp_dir(self) -> str:
        return rf"{self.mimic_root}\hosp"

    @property
    def icu_dir(self) -> str:
        return rf"{self.mimic_root}\icu"

    @property
    def note_dir(self) -> str:
        return rf"{self.mimic_root}\note"

    @property
    def admissions_file(self) -> str:
        return rf"{self.hosp_dir}\admissions.csv.gz"

    @property
    def labevents_file(self) -> str:
        return rf"{self.hosp_dir}\labevents.csv.gz"

    @property
    def microbiology_file(self) -> str:
        return rf"{self.hosp_dir}\microbiologyevents.csv.gz"

    @property
    def patients_file(self) -> str:
        return rf"{self.hosp_dir}\patients.csv.gz"

    @property
    def icustays_file(self) -> str:
        return rf"{self.icu_dir}\icustays.csv.gz"

    @property
    def chartevents_file(self) -> str:
        return rf"{self.icu_dir}\chartevents.csv.gz"

    @property
    def outputevents_file(self) -> str:
        return rf"{self.icu_dir}\outputevents.csv.gz"

    @property
    def datetimeevents_file(self) -> str:
        return rf"{self.icu_dir}\datetimeevents.csv.gz"

    @property
    def software_python(self) -> str:
        return rf"{self.software_dir}\python.bat"

    @property
    def output_file(self) -> str:
        return rf"{self.remote_output_dir}\{OUTPUT_FILENAME}"

    @property
    def reference_file(self) -> str:
        return rf"{self.reference_dir}\gold_labels.csv"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Windows clinical phenotyping task over staged synthetic MIMIC-IV style tables.

Use the staged data under `{self.input_dir}` and the stable Python entry point `{self.software_python}`.

Your job is to identify patients who developed sepsis-associated acute kidney injury (SA-AKI) during an ICU stay by operationalizing:
- Sepsis-3 onset
- KDIGO AKI staging
- the requirement that AKI begins within 7 days after sepsis onset during the same ICU stay

Visible raw data locations:
- Hospital tables: `{self.hosp_dir}`
- ICU tables: `{self.icu_dir}`
- Note tables: `{self.note_dir}`

Important raw files you will likely need:
- `{self.admissions_file}`
- `{self.labevents_file}`
- `{self.microbiology_file}`
- `{self.patients_file}`
- `{self.icustays_file}`
- `{self.chartevents_file}`
- `{self.outputevents_file}`
- `{self.datetimeevents_file}`

What you must do:
1. Read the staged raw CSV tables directly from `{self.mimic_root}`.
2. Determine which ICU patients satisfy the SA-AKI definition from the staged cohort.
3. Write exactly one CSV file to `{self.output_file}`.

Output rules:
- The file must be named exactly `{OUTPUT_FILENAME}`.
- The header row must be exactly `subject_id`.
- Each subsequent row must contain one integer `subject_id`.
- Do not add any other columns, commentary, or sidecar outputs.
- Do not modify the staged input files and do not write outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "input_dir": self.input_dir,
                "task_dir": self.task_dir,
                "mimic_root": self.mimic_root,
                "hosp_dir": self.hosp_dir,
                "icu_dir": self.icu_dir,
                "note_dir": self.note_dir,
                "admissions_file": self.admissions_file,
                "labevents_file": self.labevents_file,
                "microbiology_file": self.microbiology_file,
                "patients_file": self.patients_file,
                "icustays_file": self.icustays_file,
                "chartevents_file": self.chartevents_file,
                "outputevents_file": self.outputevents_file,
                "datetimeevents_file": self.datetimeevents_file,
                "software_python": self.software_python,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = SAAKIPhenotypingConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    for path in [meta["output_file"], meta["reference_file"]]:
        if not await session.exists(path):
            logger.error("Missing required evaluation path: %s", path)
            return [0.0]

    try:
        candidate_csv = _as_text(await session.read_file(meta["output_file"]))
        reference_csv = _as_text(await session.read_file(meta["reference_file"]))
    except Exception as exc:
        logger.error("Failed to read candidate/reference CSV: %s", exc)
        return [0.0]

    result = score_csv_texts(candidate_csv_text=candidate_csv, reference_csv_text=reference_csv)
    _log_score(result)
    return [float(result.score)]
