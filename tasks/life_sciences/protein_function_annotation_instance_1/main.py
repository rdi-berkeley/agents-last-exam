"""AgentHLE task: protein_function_annotation_instance_1."""

import json
import logging
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local fallback only

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

from dataclasses import dataclass

from tasks.common_setup import BaseTaskSetup
from tasks.life_sciences.protein_function_annotation_instance_1.scripts.score_outputs import (
    ScoreReport,
    score_output_payloads,
)
from tasks.linux_runtime import DATA_ROOT, LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "protein_function_annotation_instance_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
INTERPRO_VERSION = "5.77-108.0"


def _remote_join(*parts: str) -> str:
    return str(PurePosixPath(*parts))


async def _read_text(session: cb.DesktopSession, path: str) -> str:
    try:
        return await session.read_file(path)
    except Exception:
        data = await session.read_bytes(path)
        return data.decode("utf-8")


async def _list_output_files(session: cb.DesktopSession, path: str) -> list[str]:
    result = await session.run_command(
        "bash -lc " + json.dumps(f'find "{path}" -mindepth 1 -maxdepth 1 -printf "%f\\n" | sort'),
        check=False,
    )
    if result.get("return_code") != 0:
        raise RuntimeError(f"unable to list output directory: {path}")
    return [line.strip() for line in (result.get("stdout") or "").splitlines() if line.strip()]


@dataclass
class ProteinFunctionAnnotationConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    @property
    def visible_output_dir(self) -> str:
        return f"{self.task_dir}/output"

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.data_task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.data_task_dir}/output_test_neg"

    @property
    def input_fasta(self) -> str:
        return _remote_join(self.input_dir, "protein_sequence.fasta")

    @property
    def organism_file(self) -> str:
        return _remote_join(self.input_dir, "organism_name.txt")

    @property
    def interproscan_wrapper(self) -> str:
        return _remote_join(self.software_dir, "interproscan.sh")

    @property
    def interpro2go_file(self) -> str:
        return _remote_join(self.software_dir, "interpro2go")

    @property
    def go_namespace_lookup_file(self) -> str:
        return _remote_join(self.software_dir, "go_namespace_lookup.tsv")

    @property
    def interpro_domains_output(self) -> str:
        return _remote_join(self.remote_output_dir, "interpro_domains.tsv")

    @property
    def go_terms_output(self) -> str:
        return _remote_join(self.remote_output_dir, "go_terms.tsv")

    @property
    def summary_output(self) -> str:
        return _remote_join(self.remote_output_dir, "functional_summary.txt")

    @property
    def reference_interpro_domains(self) -> str:
        return _remote_join(self.reference_dir, "expected_interpro_domains.tsv")

    @property
    def reference_go_terms(self) -> str:
        return _remote_join(self.reference_dir, "expected_go_terms.tsv")

    @property
    def reference_summary(self) -> str:
        return _remote_join(self.reference_dir, "expected_functional_summary.txt")

    @property
    def install_script(self) -> str:
        return _remote_join(self.software_dir, "install_software.sh")

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to annotate one staged protein with InterProScan.

## Visible Task Directory
- `{self.task_dir}`

## Visible Inputs
- Protein FASTA: `{self.input_fasta}`
- Organism name: `{self.organism_file}`
- InterProScan install script: `{self.install_script}`
- InterProScan wrapper: `{self.interproscan_wrapper}`
- InterPro2GO mapping file: `{self.interpro2go_file}`
- GO namespace lookup: `{self.go_namespace_lookup_file}`

## Your Task
1. Install InterProScan by running `{self.install_script}`. This downloads and sets up InterProScan {INTERPRO_VERSION} (~15 GB download, requires Java 11 which is already available). The script is idempotent and will skip if already installed.
2. Run the staged InterProScan {INTERPRO_VERSION} wrapper on `{self.input_fasta}`.
3. Create `{self.visible_output_dir}/interpro_domains.tsv` with exactly these columns:
   `sequence_id,interpro_accession,interpro_name,start,end,e_value`
4. Retain exactly one row per non-empty InterPro accession. If multiple InterProScan rows map to the same accession, merge them into one retained row by using the minimum `start`, the maximum `end`, the shared `interpro_name` for that accession, and the smallest numeric score as `e_value`. If an accession has no numeric score in its retained rows, write `0.00`.
5. Sort the retained domain rows by `start` ascending, then by `interpro_accession` ascending.
6. Round every retained `e_value` to 2 decimal places.
7. Map the retained InterPro accessions to GO terms using `{self.interpro2go_file}`, use `{self.go_namespace_lookup_file}` to fill `go_namespace`, and write `{self.visible_output_dir}/go_terms.tsv` with exactly these columns:
   `go_id,go_name,go_namespace,source_interpro_accession`
8. Remove exact duplicate GO rows.
9. Write `{self.visible_output_dir}/functional_summary.txt` as a concise 1-2 sentence summary consistent with the retained domains and GO terms.

## Important Requirements
- The summary must include gamma-tubulin family identity plus microtubule nucleation and minus-end initiation or equivalent wording.
- Write only these three files under `{self.visible_output_dir}`.
- Do not modify files under `{self.input_dir}` or `{self.software_dir}`.
- Do not read or modify hidden evaluator-owned directories.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": self.VARIANT_NAME,
                "workspace_dir": self.task_dir,
                "task_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "visible_output_dir": self.visible_output_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "remote_output_dir": self.remote_output_dir,
                "remote_output_name": self.REMOTE_OUTPUT_DIR,
                "input_fasta": self.input_fasta,
                "organism_file": self.organism_file,
                "interproscan_wrapper": self.interproscan_wrapper,
                "install_script": self.install_script,
                "interpro2go_file": self.interpro2go_file,
                "go_namespace_lookup_file": self.go_namespace_lookup_file,
                "interpro_domains_output": self.interpro_domains_output,
                "go_terms_output": self.go_terms_output,
                "summary_output": self.summary_output,
                "reference_interpro_domains": self.reference_interpro_domains,
                "reference_go_terms": self.reference_go_terms,
                "reference_summary": self.reference_summary,
                "canonical_gcs_root": "gs://ale-data-all/life_sciences/protein_function_annotation_instance_1/base/",
            }
        )
        return metadata


config = ProteinFunctionAnnotationConfig()


@cb.tasks_config(split="train")
def load():
    cfg = ProteinFunctionAnnotationConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _log_report(report: ScoreReport) -> None:
    logger.info(
        "[%s] score=%.1f passed=%s hard_fail=%s",
        TASK_ID,
        report.score,
        report.passed,
        report.hard_fail_reason,
    )
    logger.info("[%s] details=%s", TASK_ID, json.dumps(report.to_dict(), ensure_ascii=True))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    required_outputs = [
        meta["interpro_domains_output"],
        meta["go_terms_output"],
        meta["summary_output"],
    ]
    expected_output_files = {
        "interpro_domains.tsv",
        "go_terms.tsv",
        "functional_summary.txt",
    }
    output_exists = {path: await session.exists(path) for path in required_outputs}
    if not all(output_exists.values()):
        missing = [path for path, exists in output_exists.items() if not exists]
        logger.error("missing agent outputs: %s", missing)
        return [0.0]
    try:
        output_files = set(await _list_output_files(session, meta["remote_output_dir"]))
    except Exception as exc:
        logger.exception("failed to inspect output directory: %s", exc)
        return [0.0]
    if output_files != expected_output_files:
        logger.error(
            "output directory must contain only %s but found %s",
            expected_output_files,
            output_files,
        )
        return [0.0]

    try:
        agent_domains = await _read_text(session, meta["interpro_domains_output"])
        agent_go = await _read_text(session, meta["go_terms_output"])
        agent_summary = await _read_text(session, meta["summary_output"])
        reference_domains = await _read_text(session, meta["reference_interpro_domains"])
        reference_go = await _read_text(session, meta["reference_go_terms"])
        reference_summary = await _read_text(session, meta["reference_summary"])
    except Exception as exc:
        logger.exception("failed to read staged task files: %s", exc)
        return [0.0]

    report = score_output_payloads(
        agent_interpro_tsv=agent_domains,
        agent_go_tsv=agent_go,
        agent_summary_text=agent_summary,
        reference_interpro_tsv=reference_domains,
        reference_go_tsv=reference_go,
        reference_summary_text=reference_summary,
    )
    _log_report(report)
    return [float(report.score)]
