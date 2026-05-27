"""Ubuntu-native germline variant-calling benchmark."""

import csv
import io
import logging
import posixpath
import re
from typing import Optional

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

MINIFORGE_ROOT = "/opt/toolchains/miniforge3"
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
    CONDA_ENV: str = "wf1-env"

    MIN_MAPPING_RATE: float = 80.0
    MAX_DUPLICATION_RATE: float = 30.0
    MIN_SNP_F1: float = 0.95
    MIN_SNP_PRECISION: float = 0.95
    MIN_SNP_RECALL: float = 0.95
    MIN_INDEL_F1: float = 0.90
    MIN_INDEL_PRECISION: float = 0.90
    MIN_INDEL_RECALL: float = 0.90

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
You are given a tiny paired-end genomic sequencing dataset plus a small reference and truth set.

Your task is to run an Ubuntu-native germline variant-calling workflow using `bwa`, `samtools`, \
and `bcftools`, then summarize benchmark metrics against the provided truth calls.

## Task Directory
`{self.task_dir}`

## Input Directory
`{self.input_dir}`

## Available Environment
- Open a Linux terminal yourself, `cd "{self.task_dir}"`, then activate the preinstalled \
conda environment: `source "{MINIFORGE_ROOT}/etc/profile.d/conda.sh" && conda activate "{self.CONDA_ENV}"`
- conda environment `{self.CONDA_ENV}` is preinstalled with bwa, samtools, bcftools, FastQC, MultiQC

## Required Outputs
Save all outputs under `{self.remote_output_dir}`:
- `{self.FASTQC_R1_FILE}` — FastQC report for mate 1
- `{self.FASTQC_R2_FILE}` — FastQC report for mate 2
- `{self.MULTIQC_FILE}` — MultiQC aggregate report
- `{self.FLAGSTAT_FILE}` — samtools flagstat output
- `{self.DUPLICATION_FILE}` — Picard-style duplication metrics
- `{self.VCF_FILE}` — filtered variant calls (bgzipped)
- `{self.VCF_INDEX_FILE}` — tabix index for the VCF
- `{self.RTG_SUMMARY_FILE}` — CSV starting with the exact header line \
`Type,Precision,Sensitivity,F_measure`, then your benchmark rows appended below

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


config = TaskConfig(TASK_NAME="WGS_Variant_Calling")


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


def parse_rtg_summary(summary_text: str) -> Optional[dict]:
    reader = csv.DictReader(io.StringIO(summary_text))
    results: dict[str, float] = {}
    for row in reader:
        row = {key.strip(): value.strip() for key, value in row.items()}
        variant_type = row.get("Type", "").upper()
        try:
            precision = float(row.get("Precision", 0))
            sensitivity = float(row.get("Sensitivity", 0))
            f_measure = float(row.get("F_measure", 0))
        except (TypeError, ValueError):
            continue
        if variant_type == "SNP":
            results["snp_precision"] = precision
            results["snp_recall"] = sensitivity
            results["snp_f1"] = f_measure
        elif variant_type == "INDEL":
            results["indel_precision"] = precision
            results["indel_recall"] = sensitivity
            results["indel_f1"] = f_measure
    return results if results else None


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    output_dir = task_cfg.metadata["remote_output_dir"]
    score = 0.0

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
        if vcf_bytes and len(vcf_bytes) > 50 and tbi_bytes and len(tbi_bytes) > 0:
            score += 0.10
            logger.info("Checkpoint 3 PASSED")
        else:
            logger.info("Checkpoint 3 FAILED")
    except Exception as exc:
        logger.info("Checkpoint 3 FAILED: %s", exc)

    try:
        summary_bytes = await session.read_bytes(f"{output_dir}/{config.RTG_SUMMARY_FILE}")
        metrics = parse_rtg_summary(summary_bytes.decode()) if summary_bytes else None
        if metrics:
            for key, threshold in [
                ("snp_f1", task_cfg.metadata["min_snp_f1"]),
                ("snp_precision", task_cfg.metadata["min_snp_precision"]),
                ("snp_recall", task_cfg.metadata["min_snp_recall"]),
                ("indel_f1", task_cfg.metadata["min_indel_f1"]),
                ("indel_precision", task_cfg.metadata["min_indel_precision"]),
                ("indel_recall", task_cfg.metadata["min_indel_recall"]),
            ]:
                if metrics.get(key, 0.0) >= threshold:
                    score += 0.10
            logger.info("Checkpoint 4 metrics=%s", metrics)
        else:
            logger.info("Checkpoint 4 FAILED")
    except Exception as exc:
        logger.info("Checkpoint 4 FAILED: %s", exc)

    return [score]
