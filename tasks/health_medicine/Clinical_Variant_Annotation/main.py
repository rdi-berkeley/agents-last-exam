"""Ubuntu-native clinical variant annotation benchmark."""

import logging
import re

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)


class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "health_medicine"
    VARIANT_NAME: str = "base"

    EXPECTED_VARIANT_COUNT: int = 200
    PATHOGENIC_CHROM: str = "chr17"
    PATHOGENIC_POS: str = "43106487"
    PATHOGENIC_REF: str = "T"
    PATHOGENIC_ALT: str = "G"
    EXPECTED_GENE: str = "BRCA1"

    VARIANT_COUNT_FILE: str = "variant_count.txt"
    VEP_RESULTS_FILE: str = "vep_results.csv"
    GNOMAD_RESULTS_FILE: str = "gnomad_results.csv"
    CLINVAR_RESULTS_FILE: str = "clinvar_results.csv"
    FINAL_CANDIDATES_FILE: str = "final_candidates.csv"

    @property
    def vcf_path(self) -> str:
        return f"{self.input_dir}/patient_variants.vcf"

    @property
    def task_description(self) -> str:
        return f"""You are given a small clinical VCF containing 200 candidate variants from a breast-cancer case.

Your task is to identify the single pathogenic BRCA1-like candidate and document the evidence in the required output files.

Task directory:
- `{self.task_dir}`

Input:
- `{self.vcf_path}`

Available tools:
- Open a Linux terminal yourself and `cd "{self.task_dir}"`
- `python3`
- `curl`
- `jq`

External resources:
1. gnomAD GraphQL API
2. Ensembl VEP REST API
3. NCBI ClinVar / E-utilities

Required outputs under `{self.remote_output_dir}`:
- `{self.VARIANT_COUNT_FILE}` with a single integer
- `{self.VEP_RESULTS_FILE}` with exact header `CHROM,POS,REF,ALT,GENE,CONSEQUENCE,IMPACT,SIFT,POLYPHEN`
- `{self.GNOMAD_RESULTS_FILE}` with exact header `CHROM,POS,REF,ALT,ALLELE_FREQ`
- `{self.CLINVAR_RESULTS_FILE}` with exact header `CHROM,POS,REF,ALT,CLINVAR_RESULT`
- `{self.FINAL_CANDIDATES_FILE}` with exact header `CHROM,POS,REF,ALT,JUSTIFICATION`

Rules:
- Create each CSV file with the exact header line above, then append your rows.
- It is acceptable to use `NA` for unavailable fields.
- The final candidate file must include the pathogenic evidence in the justification column.
- Do not ask for confirmation. Execute directly.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "vcf_path": self.vcf_path,
                "expected_variant_count": self.EXPECTED_VARIANT_COUNT,
                "pathogenic_chrom": self.PATHOGENIC_CHROM,
                "pathogenic_pos": self.PATHOGENIC_POS,
                "pathogenic_ref": self.PATHOGENIC_REF,
                "pathogenic_alt": self.PATHOGENIC_ALT,
                "expected_gene": self.EXPECTED_GENE,
            }
        )
        return metadata


config = TaskConfig(TASK_NAME="Clinical_Variant_Annotation", DOMAIN_NAME="health_medicine")


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


def _data_lines(csv_text: str, prefix: str) -> list[str]:
    return [
        line.strip()
        for line in csv_text.strip().splitlines()
        if line.strip() and not line.upper().startswith(prefix)
    ]


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    output_dir = task_cfg.metadata["remote_output_dir"]
    score = 0.0

    try:
        variant_count_bytes = await session.read_bytes(f"{output_dir}/{config.VARIANT_COUNT_FILE}")
        if variant_count_bytes:
            text = variant_count_bytes.decode().strip()
            numbers = re.findall(r"\d+", text)
            if numbers and int(numbers[0]) == task_cfg.metadata["expected_variant_count"]:
                score += 0.2
                logger.info("Checkpoint 0 PASSED: correct variant count")
            else:
                logger.info("Checkpoint 0 FAILED: got %s", text)
        else:
            logger.info("Checkpoint 0 FAILED: missing variant_count.txt")
    except Exception as exc:
        logger.info("Checkpoint 0 FAILED: %s", exc)

    try:
        gnomad_bytes = await session.read_bytes(f"{output_dir}/{config.GNOMAD_RESULTS_FILE}")
        if gnomad_bytes:
            lines = _data_lines(gnomad_bytes.decode(), "CHROM")
            target = next((line for line in lines if task_cfg.metadata["pathogenic_pos"] in line), None)
            if target:
                parts = target.split(",", 4)
                freq_field = parts[4].strip().strip('"') if len(parts) >= 5 else ""
                has_freq_data = bool(re.search(r"\d+\.?\d*(?:[eE][-+]?\d+)?", freq_field))
            else:
                has_freq_data = False
            if target and has_freq_data:
                score += 0.2
                logger.info("Checkpoint 1 PASSED: gnomAD results populated")
            else:
                logger.info("Checkpoint 1 FAILED")
        else:
            logger.info("Checkpoint 1 FAILED: missing gnomad_results.csv")
    except Exception as exc:
        logger.info("Checkpoint 1 FAILED: %s", exc)

    try:
        vep_bytes = await session.read_bytes(f"{output_dir}/{config.VEP_RESULTS_FILE}")
        if vep_bytes:
            lines = _data_lines(vep_bytes.decode(), "CHROM")
            target = next((line for line in lines if task_cfg.metadata["pathogenic_pos"] in line), None)
            if target and task_cfg.metadata["expected_gene"] in target and "missense" in target.lower():
                score += 0.2
                logger.info("Checkpoint 2 PASSED: VEP annotation looks correct")
            else:
                logger.info("Checkpoint 2 FAILED")
        else:
            logger.info("Checkpoint 2 FAILED: missing vep_results.csv")
    except Exception as exc:
        logger.info("Checkpoint 2 FAILED: %s", exc)

    try:
        clinvar_bytes = await session.read_bytes(f"{output_dir}/{config.CLINVAR_RESULTS_FILE}")
        if clinvar_bytes:
            lines = _data_lines(clinvar_bytes.decode(), "CHROM")
            target = next((line for line in lines if task_cfg.metadata["pathogenic_pos"] in line), None)
            if target and "pathogenic" in target.lower():
                score += 0.2
                logger.info("Checkpoint 3 PASSED: ClinVar annotation looks correct")
            else:
                logger.info("Checkpoint 3 FAILED")
        else:
            logger.info("Checkpoint 3 FAILED: missing clinvar_results.csv")
    except Exception as exc:
        logger.info("Checkpoint 3 FAILED: %s", exc)

    try:
        candidates_bytes = await session.read_bytes(f"{output_dir}/{config.FINAL_CANDIDATES_FILE}")
        if candidates_bytes:
            lines = _data_lines(candidates_bytes.decode(), "CHROM")
            target = next((line for line in lines if task_cfg.metadata["pathogenic_pos"] in line), None)
            if target:
                parts = target.split(",", 4)
                has_ref_alt = (
                    len(parts) >= 4
                    and parts[2].strip() == task_cfg.metadata["pathogenic_ref"]
                    and parts[3].strip() == task_cfg.metadata["pathogenic_alt"]
                )
                has_justification = len(parts) >= 5 and "pathogenic" in parts[4].lower()
                if has_ref_alt and has_justification:
                    score += 0.2
                    logger.info("Checkpoint 4 PASSED: final candidate looks correct")
                else:
                    logger.info("Checkpoint 4 FAILED")
            else:
                logger.info("Checkpoint 4 FAILED")
        else:
            logger.info("Checkpoint 4 FAILED: missing final_candidates.csv")
    except Exception as exc:
        logger.info("Checkpoint 4 FAILED: %s", exc)

    return [score]
