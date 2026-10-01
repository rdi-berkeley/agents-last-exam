"""Ubuntu-native clinical variant annotation benchmark."""

import csv
import io
import logging
import re
from decimal import Decimal, InvalidOperation

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
    PATHOGENIC_REF: str = "A"
    PATHOGENIC_ALT: str = "C"
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
        return f"""You are given a small synthetic clinical VCF containing 200 candidate variants for a breast-cancer annotation exercise.

Your task is to identify the single pathogenic BRCA1-like candidate and document the evidence in the required output files.

Task directory:
- `{self.task_dir}`

Input:
- `{self.vcf_path}`
- Coordinates and REF/ALT use the GRCh38 forward genomic strand, not transcript-oriented alleles.

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
- `{self.VEP_RESULTS_FILE}` with columns `CHROM,POS,REF,ALT,GENE,CONSEQUENCE,IMPACT,SIFT,POLYPHEN`
- `{self.GNOMAD_RESULTS_FILE}` with columns `CHROM,POS,REF,ALT,ALLELE_FREQ`
- `{self.CLINVAR_RESULTS_FILE}` with columns `CHROM,POS,REF,ALT,CLINVAR_RESULT`
- `{self.FINAL_CANDIDATES_FILE}` with columns `CHROM,POS,REF,ALT,JUSTIFICATION`

Rules:
- Create each CSV file using the column names above, then append your rows.
- It is acceptable to use `NA` for unavailable annotations, except the final candidate's `ALLELE_FREQ`: provide a numeric population allele frequency in [0, 1] from gnomAD v4 exomes or genomes. A public source quoting that dataset is acceptable if the direct API is unavailable. Document the dataset and evidence source in the final justification; do not invent missing frequencies.
- Identify variants by the complete CHROM, POS, REF, ALT combination, not position alone. Quoted CSV fields, column reordering, and `17`/`chr17` chromosome spelling are equivalent; retain the specified column names.
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


CSV_COLUMNS = {
    "gnomad_results.csv": ("CHROM", "POS", "REF", "ALT", "ALLELE_FREQ"),
    "vep_results.csv": (
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "GENE",
        "CONSEQUENCE",
        "IMPACT",
        "SIFT",
        "POLYPHEN",
    ),
    "clinvar_results.csv": ("CHROM", "POS", "REF", "ALT", "CLINVAR_RESULT"),
    "final_candidates.csv": ("CHROM", "POS", "REF", "ALT", "JUSTIFICATION"),
}


def _target_row(content: str, columns: tuple[str, ...], metadata: dict) -> dict | None:
    reader = csv.reader(io.StringIO(content.lstrip("\ufeff")), strict=True)
    rows = [row for row in reader if any(value.strip() for value in row)]
    if not rows:
        return None
    header = [value.strip() for value in rows[0]]
    if len(header) != len(columns) or set(header) != set(columns):
        return None
    candidates = []
    expected = (
        str(metadata["pathogenic_chrom"]).lower().removeprefix("chr"),
        int(metadata["pathogenic_pos"]),
        metadata["pathogenic_ref"].upper(),
        metadata["pathogenic_alt"].upper(),
    )
    for values in rows[1:]:
        if len(values) != len(header):
            return None
        row = {name: value.strip() for name, value in zip(header, values)}
        if not re.fullmatch(r"[0-9]+", row["POS"]):
            return None
        identity = (
            row["CHROM"].lower().removeprefix("chr"),
            int(row["POS"]),
            row["REF"].upper(),
            row["ALT"].upper(),
        )
        if identity == expected:
            candidates.append(row)
    return candidates[0] if len(candidates) == 1 else None


def _pathogenic_evidence(text: str) -> bool:
    normalized = re.sub(r"[_-]+", " ", text.lower())
    return bool(re.search(r"\bpathogenic\b", normalized)) and not re.search(
        r"\b(?:not|non)\s+(?:likely\s+)?pathogenic\b|\bno\s+(?:evidence\s+of\s+)?pathogenicity\b",
        normalized,
    )


def score_output_bundle(files: dict[str, str], metadata: dict) -> dict:
    count = files.get(config.VARIANT_COUNT_FILE, "").strip()
    checks = {
        "variant_count": bool(re.fullmatch(r"[0-9]+", count))
        and count.lstrip("0") == str(metadata["expected_variant_count"])
    }
    for filename, columns in CSV_COLUMNS.items():
        try:
            row = _target_row(files.get(filename, ""), columns, metadata)
            passed = False
            if row is not None:
                if filename == config.GNOMAD_RESULTS_FILE:
                    frequency = Decimal(row["ALLELE_FREQ"])
                    passed = frequency.is_finite() and 0 <= frequency <= 1
                elif filename == config.VEP_RESULTS_FILE:
                    genes = re.split(r"[\s,;|]+", row["GENE"].upper())
                    consequences = re.sub(r"[_-]", " ", row["CONSEQUENCE"].lower())
                    passed = metadata["expected_gene"].upper() in genes and bool(
                        re.search(r"\bmissense(?:\s+variant)?\b", consequences)
                    )
                else:
                    field = (
                        "CLINVAR_RESULT"
                        if filename == config.CLINVAR_RESULTS_FILE
                        else "JUSTIFICATION"
                    )
                    passed = _pathogenic_evidence(row[field])
            checks[filename] = bool(passed)
        except (csv.Error, InvalidOperation, ValueError, OverflowError):
            checks[filename] = False
    return {"score": sum(checks.values()) / 5.0, "checks": checks}


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    output_dir = task_cfg.metadata["remote_output_dir"]
    files = {}
    for filename in (config.VARIANT_COUNT_FILE, *CSV_COLUMNS):
        path = f"{output_dir}/{filename}"
        if not await session.file_exists(path):
            continue
        try:
            content = await session.read_bytes(path)
            if content is not None:
                files[filename] = content.decode("utf-8-sig")
        except (OSError, UnicodeError) as exc:
            logger.info("Cannot read %s: %s", filename, exc)
    result = score_output_bundle(files, task_cfg.metadata)
    logger.info("Clinical variant annotation: %s", result)
    return [result["score"]]
