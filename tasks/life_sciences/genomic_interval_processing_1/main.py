"""Ubuntu-native ENCODE CTCF genomic interval union benchmark."""

from __future__ import annotations

import json
import logging
import os
import posixpath
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import INPUT_BED_FILES, REQUIRED_FILES, score_submission  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "genomic_interval_processing_1"
VARIANT_NAME = "base"
ALLOWED_OUTPUT_DIRS = {
    "output",
    "output_test_pos",
    "output_test_neg",
    "output_admin_pos",
    "output_admin_neg",
}


def _normalize_output_dir_name(raw: str) -> str:
    normalized = posixpath.normpath(raw.replace("\\", "/").strip())
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: "
            "output, output_test_pos, output_test_neg, output_admin_pos, output_admin_neg"
        )
    return normalized


class TaskConfig(LinuxTaskConfig):
    def __init__(self, remote_output_dir: str | None = None) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=remote_output_dir or os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def output_dir_name(self) -> str:
        return _normalize_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def output_files(self) -> dict[str, str]:
        return {name: f"{self.remote_output_dir}/{name}" for name in REQUIRED_FILES}

    @property
    def task_instructions_file(self) -> str:
        return f"{self.input_dir}/task_instructions.md"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def operation_specification_file(self) -> str:
        return f"{self.input_dir}/operation_specification.txt"

    @property
    def reference_bed_file(self) -> str:
        return f"{self.reference_dir}/union_ref.bed"

    @property
    def input_hashes_file(self) -> str:
        return f"{self.reference_dir}/input_hashes.json"

    @property
    def task_description(self) -> str:
        input_lines = "\n".join(f"- `{self.input_dir}/{name}`" for name in INPUT_BED_FILES)
        return f"""\
You are a bioinformatics analyst working on a Linux genomic interval processing task.

Task directory:
- `{self.task_dir}`

Visible input files:
{input_lines}
- Operation specification: `{self.operation_specification_file}`
- Detailed task instructions: `{self.task_instructions_file}`
- Output contract: `{self.output_contract_file}`
- BEDTools v2.31.1 entry point: `{self.software_dir}/bedtools`
- Software note: `{self.software_dir}/README.txt`

What you must do:
1. Read the operation specification and apply it to the three ENCODE CTCF narrowPeak BED files.
2. Produce the final non-overlapping CTCF union peak set as a 3-column BED file with columns `chrom`, `start`, and `end`.
3. Sort the final BED lexicographically by chromosome, then numerically by start and end.
4. Record the commands or script steps you used.
5. Write a JSON summary whose input and output counts match the files you used and produced.
6. Write exactly these required files under `{self.remote_output_dir}`:
   - `union_peaks.bed`
   - `commands.sh`
   - `summary.json`

Use `{self.software_dir}/bedtools` when you need BEDTools; do not rely on a PATH-resolved `bedtools`.
Do not modify files under `{self.input_dir}`.
Do not write final outputs outside `{self.remote_output_dir}`.
Use local staged files only; do not download replacement data from the internet.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "operation_specification_file": self.operation_specification_file,
                "task_instructions_file": self.task_instructions_file,
                "output_contract_file": self.output_contract_file,
                "output_dir_name": self.output_dir_name,
                "output_files": self.output_files,
                "reference_bed_file": self.reference_bed_file,
                "input_hashes_file": self.input_hashes_file,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    task_config = TaskConfig()
    return [
        cb.Task(
            description=task_config.task_description,
            metadata=task_config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": task_config.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _read_required_output_files(session: cb.DesktopSession, output_files: dict[str, str]):
    payloads: dict[str, bytes] = {}
    missing: list[str] = []
    for name, path in output_files.items():
        if not await session.exists(path):
            missing.append(name)
            continue
        payloads[name] = await session.read_bytes(path)
    return payloads, missing


async def _read_text(session: cb.DesktopSession, path: str) -> str:
    raw = await session.read_file(path)
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return raw


async def _input_hashes_match(
    session: cb.DesktopSession, input_dir: str, expected: dict[str, str]
) -> bool:
    for rel_path, expected_hash in expected.items():
        remote_path = str(PurePosixPath(input_dir, rel_path))
        result = await session.run_command(f'sha256sum "{remote_path}"', check=False)
        if result.get("return_code", 1) != 0:
            logger.error(
                "failed to hash %s: %s", remote_path, result.get("stderr") or result.get("stdout")
            )
            return False
        observed = result.get("stdout", "").split()[0]
        if observed != expected_hash:
            logger.error("input hash mismatch for %s", remote_path)
            return False
    return True


async def _count_input_rows(session: cb.DesktopSession, input_dir: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for filename in INPUT_BED_FILES:
        path = f"{input_dir}/{filename}"
        result = await session.run_command(f'grep -cve "^$" "{path}"', check=False)
        if result.get("return_code", 1) != 0:
            raise RuntimeError(f"failed to count rows in {path}: {result.get('stderr')}")
        counts[filename] = int(result.get("stdout", "0").strip())
    return counts


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    outputs, missing = await _read_required_output_files(session, meta["output_files"])
    if missing:
        logger.info("missing output files: %s", missing)
        return [0.0]

    for ref_key in ("input_hashes_file", "reference_bed_file"):
        if not await session.exists(meta[ref_key]):
            raise RuntimeError(
                f"evaluator-controlled {ref_key} missing: {meta[ref_key]}"
            )

    input_hashes = json.loads(await _read_text(session, meta["input_hashes_file"]))
    if not await _input_hashes_match(session, meta["input_dir"], input_hashes):
        return [0.0]

    reference_bed = await session.read_bytes(meta["reference_bed_file"])
    input_counts = await _count_input_rows(session, meta["input_dir"])
    report = score_submission(outputs, reference_bed=reference_bed, input_counts=input_counts)
    logger.info("Evaluation report: %s", json.dumps(report.to_dict(), sort_keys=True))
    return [report.score]


if __name__ == "__main__":
    for task in load():
        print(task.description)
