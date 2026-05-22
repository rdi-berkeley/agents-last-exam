"""AgentHLE task: business_finance/llm_ecosystem_privacy_audit_realdata_1.

Given pre-classified data collection profiles for 2,253 GPT Actions plus the
verbatim OpenAI GPT Actions Usage Policy (with its Policy-to-Taxonomy Mapping
Guide), the agent must produce two output artifacts:

- ``output/policy_violations.json`` — one entry per violating Action field.
- ``output/cross_domain_exposure_report.csv`` — amplification report for every
  backend domain served by at least three distinct GPTs.

Evaluation is fully local: ``evaluate()`` fetches the agent's outputs plus the
hidden reference files from the remote VM and scores them with
``scripts/score_outputs.py``.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import score_submission, zero_report

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "llm_ecosystem_privacy_audit_realdata_1"
VARIANT_NAME = "base"

POLICY_OUTPUT_FILENAME = "policy_violations.json"
EXPOSURE_OUTPUT_FILENAME = "cross_domain_exposure_report.csv"
POLICY_REFERENCE_FILENAME = "policy_violations_reference.json"
EXPOSURE_REFERENCE_FILENAME = "cross_domain_exposure_reference.csv"


async def _missing(session: cb.DesktopSession, path: str, *, label: str, tag: str) -> bool:
    if await session.exists(path):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


def _decode(data) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return str(data)


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def pp_action_data_entities(self) -> str:
        return f"{self.input_dir}/pp_action_data_entities.json"

    @property
    def taxonomy_csv(self) -> str:
        return f"{self.input_dir}/taxonomy.csv"

    @property
    def policy_md(self) -> str:
        return f"{self.input_dir}/openai_gpt_actions_usage_policy.md"

    @property
    def gpt_action_metadata(self) -> str:
        return f"{self.input_dir}/gpt_action_metadata.json"

    @property
    def python_entry(self) -> str:
        return f"{self.software_dir}/python"

    @property
    def policy_output_file(self) -> str:
        return f"{self.remote_output_dir}/{POLICY_OUTPUT_FILENAME}"

    @property
    def exposure_output_file(self) -> str:
        return f"{self.remote_output_dir}/{EXPOSURE_OUTPUT_FILENAME}"

    @property
    def policy_reference_file(self) -> str:
        return f"{self.reference_dir}/{POLICY_REFERENCE_FILENAME}"

    @property
    def exposure_reference_file(self) -> str:
        return f"{self.reference_dir}/{EXPOSURE_REFERENCE_FILENAME}"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to perform a compliance audit of GPT Actions
against OpenAI's published GPT Actions Usage Policy. All inputs are staged
locally on this VM. No network access is required or permitted.

## Input Files (read-only)
- Pre-classified Action data: `{self.pp_action_data_entities}`
- Data taxonomy (CSV, UTF-8 BOM): `{self.taxonomy_csv}`
- OpenAI GPT Actions Usage Policy + Policy-to-Taxonomy Mapping Guide: `{self.policy_md}`
- GPT → Action metadata: `{self.gpt_action_metadata}`

## Your Task
1. Parse all four input files using only Python 3 standard library modules
   (`json`, `csv`, `re`, `collections`).
   - If you want to script the audit, use the canonical task-local Python
     entry point `{self.python_entry}`. It `exec`s the preinstalled system
     interpreter at `/usr/bin/python`.
2. Apply the Policy-to-Taxonomy Mapping Guide inside the policy markdown
   verbatim — it is the authoritative mapping from taxonomy `data_type`
   values to CRITICAL and HIGH violations.
3. Produce `{self.policy_output_file}` as a JSON document with:
   - `total_violations`: integer
   - `severity_breakdown`: object with `CRITICAL` and `HIGH` counts
   - `violations`: list, one entry per offending Action field. Each entry
     must include `severity` (CRITICAL or HIGH), `policy_clause` (the
     specific clause from the policy markdown), `action_domain` (the
     backend domain, i.e. the left-hand side of the `"<domain>, <api_name>"`
     key in `pp_action_data_entities.json`), `api_name`, `data_field_name`,
     and `data_type`. Entries missing `policy_clause` or `action_domain`
     will be discarded before scoring.
4. Produce `{self.exposure_output_file}` with this exact header row:
   `domain,gpt_count,action_count,avg_datatypes_per_action,union_datatypes_count,amplification_factor,union_datatypes`
   Include exactly one row for every backend `domain` that is served by at
   least 3 distinct GPTs (measured from `gpt_action_metadata.json`).
   - `action_count` is the number of Actions on that backend domain that
     appear in `pp_action_data_entities.json`.
   - `avg_datatypes_per_action` is the mean number of distinct `data_type`
     values collected per Action on that domain.
   - `union_datatypes_count` is the number of distinct `data_type` values
     aggregated across all Actions on that domain.
   - `amplification_factor` is the string
     `f"{{round(union_datatypes_count / avg_datatypes_per_action, 3)}}x"`
     (note the trailing `x` suffix; three-decimal numeric prefix).
   - `union_datatypes` is a semicolon-delimited, alphabetically sorted list
     of the distinct `data_type` values.

## Hints
- `taxonomy.csv` begins with a UTF-8 BOM — open it with `encoding="utf-8-sig"`.
- Keys in `pp_action_data_entities.json` are literal `"<domain>, <api_name>"`
  strings separated by `", "` (comma + single space). Split on the first
  `", "` only; some `api_name` values contain commas.
- Do not modify any file under `{self.input_dir}`.
- Write only to `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "pp_action_data_entities": self.pp_action_data_entities,
                "taxonomy_csv": self.taxonomy_csv,
                "policy_md": self.policy_md,
                "gpt_action_metadata": self.gpt_action_metadata,
                "python_entry": self.python_entry,
                "policy_output_file": self.policy_output_file,
                "exposure_output_file": self.exposure_output_file,
                "policy_reference_file": self.policy_reference_file,
                "exposure_reference_file": self.exposure_reference_file,
                "canonical_gcs_root": (
                    "gs://ale-data-all/business_finance/" "llm_ecosystem_privacy_audit_realdata_1/base"
                ),
            }
        )
        return metadata


config = TaskConfig()


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

    try:
        policy_agent_text = _decode(await session.read_file(meta["policy_output_file"]))
    except Exception as exc:
        logger.error("Failed to read agent policy output %s: %s", meta["policy_output_file"], exc)
        return [0.0]

    try:
        exposure_agent_text = _decode(await session.read_file(meta["exposure_output_file"]))
    except Exception as exc:
        logger.error(
            "Failed to read agent exposure output %s: %s", meta["exposure_output_file"], exc
        )
        return [0.0]

    try:
        policy_ref_text = _decode(await session.read_file(meta["policy_reference_file"]))
        exposure_ref_text = _decode(await session.read_file(meta["exposure_reference_file"]))
    except Exception as exc:
        logger.error("Failed to read hidden reference: %s", exc)
        return [0.0]

    try:
        policy_agent_json = json.loads(policy_agent_text)
    except Exception as exc:
        logger.error("agent policy_violations.json is not valid JSON: %s", exc)
        report = zero_report("agent policy_violations.json is not valid JSON")
        logger.info("evaluation_report=%s", json.dumps(report, sort_keys=True))
        return [0.0]

    try:
        policy_ref_json = json.loads(policy_ref_text)
    except Exception as exc:
        logger.exception("reference policy_violations JSON is not valid: %s", exc)
        return [0.0]

    try:
        report = score_submission(
            policy_agent_json,
            policy_ref_json,
            exposure_agent_text,
            exposure_ref_text,
        )
    except Exception as exc:
        logger.exception("scoring failed: %s", exc)
        return [0.0]

    logger.info("evaluation_report=%s", json.dumps(report, sort_keys=True))
    return [float(report["overall_score"])]
