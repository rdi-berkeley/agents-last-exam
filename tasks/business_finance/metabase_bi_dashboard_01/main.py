"""Metabase BI Dashboard task — business_finance/metabase_bi_dashboard_01.

The agent must build a business analytics dashboard in Metabase using the
pre-loaded Business Analytics SQLite database, then export key metrics
as a structured JSON file.
"""

import logging
import os
from dataclasses import dataclass

import cua_bench as cb

from tasks.business_finance.metabase_bi_dashboard_01.scripts.evaluate_metrics import \
    score_metrics
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


@dataclass
class MetabaseDashboardConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "business_finance"
    TASK_NAME: str = "metabase_bi_dashboard_01"
    VARIANT_NAME: str = "base"
    OS_TYPE: str = "windows"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def output_file(self) -> str:
        return rf"{self.remote_output_dir}\dashboard_metrics.json"

    @property
    def reference_file(self) -> str:
        return rf"{self.reference_dir}\reference_metrics.json"

    @property
    def software_launcher(self) -> str:
        return rf"{self.software_dir}\launch_metabase.bat"

    @property
    def task_description(self) -> str:
        return f"""\
You are a business analyst building a BI dashboard in Metabase.

## Environment
- Metabase is available on this Windows machine. To start it, run the launcher script:
  `{self.software_launcher}`
  Then wait for the server to become ready and open http://localhost:3000 in the browser.
- Login credentials are in: `{self.input_dir}\\metabase_credentials.txt`
- The database "Business Analytics" is already connected in Metabase.

## Your Task
1. Read the task requirements at `{self.input_dir}\\requirements.md`
2. Read the output schema at `{self.input_dir}\\output_schema.json`
3. Launch Metabase using the launcher script and log in
4. Using the Metabase GUI, build the required dashboard with all 6 charts and a heading text card
5. Query the Business Analytics database to extract the required metrics
6. Save the final metrics as a JSON file conforming to the output schema

## Output
- Save your JSON report to: `{self.output_file}`
- The JSON must conform to the schema in `{self.input_dir}\\output_schema.json`
- All numerical values must be rounded to 2 decimal places
- Revenue calculations must only include orders with status = 'completed'
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "software_launcher": self.software_launcher,
            }
        )
        return metadata


config = MetabaseDashboardConfig(
    REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
)


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta.get("variant_name", "base")
    output_file = meta["output_file"]
    reference_file = meta["reference_file"]

    logger.info("[%s] Starting evaluation", tag)

    # Read agent output
    if not (await session.file_exists(output_file) or await session.directory_exists(output_file)):
        logger.error("[%s] Output file not found: %s", tag, output_file)
        return [0.0]

    try:
        agent_json = await session.read_file(output_file)
    except Exception as exc:
        logger.error("[%s] Failed to read output: %s", tag, exc)
        return [0.0]

    # Read reference
    try:
        ref_json = await session.read_file(reference_file)
    except Exception as exc:
        logger.error("[%s] Failed to read reference: %s", tag, exc)
        return [0.0]

    result = score_metrics(agent_json, ref_json)
    logger.info("[%s] score=%.4f details=%s", tag, result["score"], result["details"])
    return [result["score"]]


if __name__ == "__main__":
    for task in load():
        print(task.description)
