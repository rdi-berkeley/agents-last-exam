"""Digital marketing audience segmentation benchmark task."""

import logging
import os
from dataclasses import dataclass

import cua_bench as cb

from tasks.business_finance.digital_marketing_audience_segmentation_1.scripts.evaluate_segmentation import \
    score_segmentation
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


@dataclass
class SegmentationConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "business_finance"
    TASK_NAME: str = "digital_marketing_audience_segmentation_1"
    VARIANT_NAME: str = "base"

    @property
    def segment_def_output(self) -> str:
        return f"{self.remote_output_dir}/segment_definition.json"

    @property
    def roster_output(self) -> str:
        return f"{self.remote_output_dir}/audience_roster.csv"

    @property
    def overlap_output(self) -> str:
        return f"{self.remote_output_dir}/overlap_report.tsv"

    @property
    def segment_def_ref(self) -> str:
        return f"{self.reference_dir}/segment_definition.json"

    @property
    def roster_ref(self) -> str:
        return f"{self.reference_dir}/audience_roster.csv"

    @property
    def overlap_ref(self) -> str:
        return f"{self.reference_dir}/overlap_report.tsv"

    @property
    def task_description(self) -> str:
        inp = self.input_dir
        out = self.remote_output_dir
        return (
            "You are a digital marketing analyst performing audience segmentation "
            "for a re-engagement campaign.\n\n"
            f"## Task Directory\n`{self.task_dir}`\n\n"
            "## Your Task\n"
            f"1. Read the segmentation brief at `{inp}/segmentation_brief.md` and the "
            f"governance policies at `{inp}/governance_policies.yaml`. Use "
            f"`{inp}/data_dictionary.tsv` for field reference.\n"
            f"2. Load customer profiles from `{inp}/unified_profiles.parquet`.\n"
            "3. Identify customers matching the target criteria in the brief "
            "(high-transaction but email-disengaged).\n"
            "4. Apply governance rules: suppress customers with active support tickets, "
            "respect channel opt-in/opt-out, compute per-customer SMS and push eligibility.\n"
            "5. Strip geo/demographic identifier columns (age, gender, city, state) from "
            "the output roster for PII compliance.\n"
            f"6. Analyze overlap between the qualifying audience and each existing audience "
            f"in `{inp}/existing_audiences.csv`.\n"
            "7. Produce three output files.\n\n"
            f"## Output Files\nSave all files to `{out}/`:\n\n"
            "**`segment_definition.json`** \u2014 JSON object with keys:\n"
            "- `segment_name`: descriptive name\n"
            "- `created_date`: date string\n"
            "- `version`: version string\n"
            "- `filter_predicates`: list of {field, operator, value}\n"
            "- `suppression_rules`: list of {field, operator, value, reason}\n"
            "- `activation_channels`: mapping of channel name \u2192 {eligibility_field, required_value}\n"
            "- `governance_applied`: {pii_fields_removed: [...], opt_out_compliance: bool, "
            "support_suppression: bool}\n"
            "- `audience_stats`: {total_qualifying, sms_eligible, push_eligible, "
            "any_channel_eligible, pct_of_total_profiles}\n\n"
            "**`audience_roster.csv`** \u2014 CSV, one row per qualifying customer after "
            "suppression. Include customer_id and profile/engagement columns, but exclude "
            "age, gender, city, state. Add columns: `sms_eligible`, `push_eligible`, "
            "`any_channel_eligible` (1 or 0).\n\n"
            "**`overlap_report.tsv`** \u2014 TSV with columns: existing_audience_id, "
            "existing_audience_name, overlap_count, overlap_pct, flag_high_overlap. "
            "One row per existing audience. "
            "overlap_pct = (overlap_count / total qualifying audience size) * 100.\n\n"
            "## Environment\n"
            "Python 3.12 and `uv` are available on this machine. "
            f"A dependency manifest is at `{inp}/runtime_env/pyproject.toml`. "
            f"Install with: `cd {inp}/runtime_env && uv sync`\n"
            f"Then run scripts with: "
            f"`uv run --project {inp}/runtime_env python your_script.py`\n"
        )

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "segment_def_output": self.segment_def_output,
                "roster_output": self.roster_output,
                "overlap_output": self.overlap_output,
                "segment_def_ref": self.segment_def_ref,
                "roster_ref": self.roster_ref,
                "overlap_ref": self.overlap_ref,
            }
        )
        return metadata


config = SegmentationConfig(
    REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
)


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

    # Read agent output files
    try:
        seg_out = (await session.read_bytes(meta["segment_def_output"])).decode("utf-8")
    except Exception as exc:
        logger.error("Cannot read segment_definition.json output: %s", exc)
        return [0.0]
    try:
        roster_out = (await session.read_bytes(meta["roster_output"])).decode("utf-8")
    except Exception as exc:
        logger.error("Cannot read audience_roster.csv output: %s", exc)
        return [0.0]
    try:
        overlap_out = (await session.read_bytes(meta["overlap_output"])).decode("utf-8")
    except Exception as exc:
        logger.error("Cannot read overlap_report.tsv output: %s", exc)
        return [0.0]

    # Read reference files
    try:
        seg_ref = (await session.read_bytes(meta["segment_def_ref"])).decode("utf-8")
    except Exception as exc:
        logger.error("Cannot read reference segment_definition.json: %s", exc)
        return [0.0]
    try:
        roster_ref = (await session.read_bytes(meta["roster_ref"])).decode("utf-8")
    except Exception as exc:
        logger.error("Cannot read reference audience_roster.csv: %s", exc)
        return [0.0]
    try:
        overlap_ref = (await session.read_bytes(meta["overlap_ref"])).decode("utf-8")
    except Exception as exc:
        logger.error("Cannot read reference overlap_report.tsv: %s", exc)
        return [0.0]

    result = score_segmentation(
        seg_out=seg_out,
        roster_out=roster_out,
        overlap_out=overlap_out,
        seg_ref=seg_ref,
        roster_ref=roster_ref,
        overlap_ref=overlap_ref,
    )
    logger.info("score=%.4f details=%s", result["score"], result["details"])
    return [result["score"]]


if __name__ == "__main__":
    for task in load():
        print(task.description)
