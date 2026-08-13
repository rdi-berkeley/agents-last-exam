"""Ubuntu-native germline variant-calling benchmark."""

import logging
import os
import posixpath
import re
import shlex
import tempfile
from pathlib import Path
from typing import Optional

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.life_sciences.WGS_Variant_Calling.scoring import (
    MAX_COMPRESSED_VCF_BYTES,
    score_vcf_bytes,
    validate_summary_csv,
    validate_tabix_index,
)

logger = logging.getLogger(__name__)

CANONICAL_OUTPUT_DIR_NAMES = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIR_NAMES:
        raise ValueError(
            f"REMOTE_OUTPUT_DIR must normalize to one of: {CANONICAL_OUTPUT_DIR_NAMES}"
        )
    return normalized


class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "life_sciences"
    VARIANT_NAME: str = "base"
    MIN_MAPPING_RATE: float = 95.0
    MAX_DUPLICATION_RATE: float = 20.0
    MIN_SNP_F1: float = 0.99
    MIN_SNP_PRECISION: float = 0.99
    MIN_SNP_RECALL: float = 0.99
    MIN_INDEL_F1: float = 0.94
    MIN_INDEL_PRECISION: float = 0.97
    MIN_INDEL_RECALL: float = 0.91

    FASTQC_R1_FILE: str = "region_R1_fastqc.html"
    FASTQC_R2_FILE: str = "region_R2_fastqc.html"
    MULTIQC_FILE: str = "multiqc_report.html"
    FLAGSTAT_FILE: str = "flagstat.txt"
    DUPLICATION_FILE: str = "duplication_metrics.txt"
    VCF_FILE: str = "variants.filtered.vcf.gz"
    VCF_INDEX_FILE: str = "variants.filtered.vcf.gz.tbi"
    RTG_SUMMARY_FILE: str = "rtg_summary.csv"

    @property
    def output_dir_name(self) -> str:
        return _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return f"{self.data_task_dir}/{self.output_dir_name}"

    @property
    def task_description(self) -> str:
        return f"""\
You are given paired-end HG002 sequencing reads covering a 1 Mb region of GRCh38 chromosome 17, \
plus the GRCh38 chromosome 17 reference.

Your task is to run an Ubuntu-native germline variant-calling workflow using `bwa`, `samtools`, \
and `bcftools`, then produce a filtered VCF containing SNP and INDEL calls.

## Task Directory
`{self.task_dir}`

## Input Directory
`{self.input_dir}`

## Available Environment
- `bwa`, `samtools`, `bcftools`, FastQC, MultiQC, and RTG Tools are preinstalled on `PATH`.

## Required Outputs
Save all outputs under `{self.remote_output_dir}`:
- `{self.FASTQC_R1_FILE}` - FastQC report for mate 1
- `{self.FASTQC_R2_FILE}` - FastQC report for mate 2
- `{self.MULTIQC_FILE}` - MultiQC aggregate report
- `{self.FLAGSTAT_FILE}` - samtools flagstat output
- `{self.DUPLICATION_FILE}` - Picard-style duplication metrics
- `{self.VCF_FILE}` - filtered variant calls (bgzipped)
- `{self.VCF_INDEX_FILE}` - tabix index for the VCF
- `{self.RTG_SUMMARY_FILE}` - informational CSV with the exact header \
`Type,Precision,Sensitivity,F_measure` and rows for `SNP` and `INDEL`; metric values may be `NA`.

Do not ask for confirmation. Execute directly.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "output_dir_name": self.output_dir_name,
                "min_mapping_rate": self.MIN_MAPPING_RATE,
                "max_duplication_rate": self.MAX_DUPLICATION_RATE,
                "min_snp_f1": self.MIN_SNP_F1,
                "min_snp_precision": self.MIN_SNP_PRECISION,
                "min_snp_recall": self.MIN_SNP_RECALL,
                "min_indel_f1": self.MIN_INDEL_F1,
                "min_indel_precision": self.MIN_INDEL_PRECISION,
                "min_indel_recall": self.MIN_INDEL_RECALL,
            }
        )
        return metadata


config = TaskConfig(TASK_NAME="WGS_Variant_Calling", DOMAIN_NAME="life_sciences")


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)
    output_dir = shlex.quote(task_cfg.metadata["remote_output_dir"])
    await session.run_command(
        f"if [ -d {output_dir} ]; then "
        f"find {output_dir} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +; "
        f"else mkdir -p -- {output_dir}; fi"
    )


def parse_mapping_rate(flagstat_text: str) -> Optional[float]:
    for line in flagstat_text.splitlines():
        if "mapped (" in line and "primary" not in line:
            match = re.search(r"\(([\d.]+)%", line)
            if match:
                return float(match.group(1))
    return None


def parse_duplication_rate(metrics_text: str) -> Optional[float]:
    lines = [line for line in metrics_text.strip().splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.startswith("LIBRARY") and idx + 1 < len(lines):
            headers = line.split("\t")
            values = lines[idx + 1].split("\t")
            try:
                dup_col = headers.index("PERCENT_DUPLICATION")
                return float(values[dup_col]) * 100
            except (ValueError, IndexError):
                continue
        if line.lower().startswith("percent_duplication"):
            try:
                return float(line.split()[-1]) * 100
            except ValueError:
                continue
    read_total = None
    dup_total = None
    for line in lines:
        if line.startswith("READ:"):
            try:
                read_total = int(line.split(":")[-1])
            except ValueError:
                pass
        elif line.startswith("DUPLICATE TOTAL:"):
            try:
                dup_total = int(line.split(":")[-1])
            except ValueError:
                pass
    if read_total and dup_total is not None:
        return dup_total / read_total * 100
    return None


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    output_dir = task_cfg.metadata["remote_output_dir"]
    score = 0.0
    vcf_bytes: bytes | None = None

    try:
        qc_files = [
            f"{output_dir}/{config.FASTQC_R1_FILE}",
            f"{output_dir}/{config.FASTQC_R2_FILE}",
            f"{output_dir}/{config.MULTIQC_FILE}",
        ]
        all_exist = True
        for path in qc_files:
            data = await session.read_bytes(path)
            if not data or len(data) < 100:
                all_exist = False
                break
        if all_exist:
            score += 0.10
            logger.info("Checkpoint 0 PASSED")
        else:
            logger.info("Checkpoint 0 FAILED")
    except Exception as exc:
        logger.info("Checkpoint 0 FAILED: %s", exc)

    try:
        flagstat_bytes = await session.read_bytes(f"{output_dir}/{config.FLAGSTAT_FILE}")
        mapping_rate = parse_mapping_rate(flagstat_bytes.decode()) if flagstat_bytes else None
        if mapping_rate is not None and mapping_rate >= task_cfg.metadata["min_mapping_rate"]:
            score += 0.10
            logger.info("Checkpoint 1 PASSED")
        else:
            logger.info("Checkpoint 1 FAILED")
    except Exception as exc:
        logger.info("Checkpoint 1 FAILED: %s", exc)

    try:
        dup_bytes = await session.read_bytes(f"{output_dir}/{config.DUPLICATION_FILE}")
        dup_rate = parse_duplication_rate(dup_bytes.decode()) if dup_bytes else None
        if dup_rate is not None and dup_rate <= task_cfg.metadata["max_duplication_rate"]:
            score += 0.10
            logger.info("Checkpoint 2 PASSED")
        else:
            logger.info("Checkpoint 2 FAILED")
    except Exception as exc:
        logger.info("Checkpoint 2 FAILED: %s", exc)

    try:
        vcf_bytes = await session.read_bytes(f"{output_dir}/{config.VCF_FILE}")
        tbi_bytes = await session.read_bytes(f"{output_dir}/{config.VCF_INDEX_FILE}")
        summary_bytes = await session.read_bytes(f"{output_dir}/{config.RTG_SUMMARY_FILE}")
        if (
            vcf_bytes
            and len(vcf_bytes) > 50
            and len(vcf_bytes) <= MAX_COMPRESSED_VCF_BYTES
            and validate_tabix_index(tbi_bytes or b"")
            and validate_summary_csv(summary_bytes or b"")
        ):
            score += 0.10
            logger.info("Checkpoint 3 PASSED")
        else:
            logger.info("Checkpoint 3 FAILED")
    except Exception as exc:
        logger.info("Checkpoint 3 FAILED: %s", exc)

    if not vcf_bytes:
        logger.info("Checkpoint 4 FAILED: submitted VCF is missing")
        return [score]

    evaluator_data_dir = Path(
        os.environ.get(
            "WGS_VARIANT_CALLING_EVAL_DATA_DIR",
            Path(__file__).resolve().parents[3]
            / "secret"
            / "eval_data"
            / "WGS_Variant_Calling",
        )
    )
    required_evaluator_files = (
        "chr17.fa",
        "truth_chr17.vcf.gz",
        "truth_chr17.vcf.gz.tbi",
        "eval_region_confident.bed",
    )

    if all((evaluator_data_dir / name).is_file() for name in required_evaluator_files):
        report = score_vcf_bytes(vcf_bytes, evaluator_data_dir)
    else:
        reference_dir = task_cfg.metadata["reference_dir"]
        try:
            with tempfile.TemporaryDirectory(prefix="wgs-evaluator-") as temp_dir:
                staged_evaluator_dir = Path(temp_dir)
                for name in required_evaluator_files:
                    payload = await session.read_bytes(f"{reference_dir}/{name}")
                    if not payload:
                        raise FileNotFoundError(
                            f"staged evaluator reference is missing or empty: {name}"
                        )
                    (staged_evaluator_dir / name).write_bytes(payload)
                report = score_vcf_bytes(vcf_bytes, staged_evaluator_dir)
        except Exception as exc:
            raise RuntimeError(
                "WGS evaluator data is unavailable both locally and in the staged reference"
            ) from exc

    for value, threshold in [
        (report.snp.f1, task_cfg.metadata["min_snp_f1"]),
        (report.snp.precision, task_cfg.metadata["min_snp_precision"]),
        (report.snp.recall, task_cfg.metadata["min_snp_recall"]),
        (report.indel.f1, task_cfg.metadata["min_indel_f1"]),
        (report.indel.precision, task_cfg.metadata["min_indel_precision"]),
        (report.indel.recall, task_cfg.metadata["min_indel_recall"]),
    ]:
        if value >= threshold:
            score += 0.10
    logger.info("Checkpoint 4 independently recomputed metrics=%s", report)

    return [score]
