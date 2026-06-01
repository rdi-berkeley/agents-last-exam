"""AgentHLE task: hg002_chr22_germline_variant_pipeline."""

from __future__ import annotations

import logging
import shutil
import sys
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

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import evaluate as score_submission  # noqa: E402

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "hg002_chr22_germline_variant_pipeline"
VARIANT_NAME = "base"
CANONICAL_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}
EVAL_TMP_ROOT = Path("/tmp/agenthle_eval") / TASK_NAME


def _remote_join(*parts: str) -> str:
    return str(PurePosixPath(*parts))


def _canonical_output_dir_name(name: str) -> str:
    normalized = str(PurePosixPath(name))
    if normalized not in CANONICAL_OUTPUT_DIRS:
        raise ValueError(
            f"REMOTE_OUTPUT_DIR must normalize to one of {sorted(CANONICAL_OUTPUT_DIRS)}"
        )
    return normalized


class Hg002Chr22Config(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def output_dir_name(self) -> str:
        return _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return _remote_join(self.task_dir, self.output_dir_name)

    @property
    def submission_root(self) -> str:
        return _remote_join(self.remote_output_dir, "submission")

    @property
    def hidden_truth_dir(self) -> str:
        return _remote_join(self.reference_dir, "hidden_truth")

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux germline-variant-calling task.

Visible solve-time input root:
- `{self.input_dir}/starter_project`

Your job is to repair the starter workflow and write a completed `submission/`
tree under:
- `{self.submission_root}`

Inside the visible starter project you have:
- paired-end reads under `fastq/`
- the chr22 reference FASTA and known-sites inputs under `reference/`
- the chr22 confident BED under `truth/`
- VEP resources under `vep_cache/`
- intentionally broken `samplesheet.csv`, `nextflow.config`, and `run_vep.sh`

Write these required artifacts under `{self.submission_root}`:
- `pipeline/samplesheet.csv`
- `pipeline/nextflow.config`
- `pipeline/known_sites_chr22/dbsnp_138.chr22.vcf.gz` and `.tbi`
- `pipeline/known_sites_chr22/Mills_and_1000G_gold_standard.indels.chr22.vcf.gz` and `.tbi`
- `results/variants/HG002.filtered.vcf.gz` and `.tbi`
- `results/annotation/HG002.filtered.annotated.vcf.gz` and `.tbi`
- `results/reports/multiqc_report.html`
- `results/reports/multiqc_data/multiqc_general_stats.txt`
- `results/reports/multiqc_data/multiqc_software_versions.txt`
- `results/qc/qc_summary.json`
- `DECISIONS.md`

Requirements:
1. Correct the samplesheet so it is valid for the HG002 chr22 run.
2. Rename the Mills contigs to `chr22` and re-index the corrected file.
3. Rebuild the missing `bwa-mem2` index.
4. Tune the workflow config for the available host.
5. Produce a hard-filtered chr22 VCF and a VEP-annotated VCF.
6. Aggregate QC into MultiQC and write `results/qc/qc_summary.json` as a
   flat JSON object with at least these top-level keys:
   `{{"alignment_rate": NUMBER, "dup_rate": NUMBER, "mean_coverage_chr22": NUMBER}}`.
7. Explain any filtering choices in `DECISIONS.md` with concrete numeric values.

Rules:
- Treat `{self.input_dir}` plus any intended visible `software/` entry points as
  the solve-time surface.
- Write solver-created files only under `{self.remote_output_dir}`.
- Hidden benchmarking against evaluator-side truth happens after you finish.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "task_id": f"{DOMAIN_NAME}/{TASK_NAME}",
                "output_dir_name": self.output_dir_name,
                "submission_root": self.submission_root,
                "hidden_truth_dir": self.hidden_truth_dir,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = Hg002Chr22Config(DOMAIN_NAME=DOMAIN_NAME, TASK_NAME=TASK_NAME)


@cb.tasks_config(split="train")
def load():
    cfg = Hg002Chr22Config(DOMAIN_NAME=DOMAIN_NAME, TASK_NAME=TASK_NAME)
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


async def _download_if_present(
    session: cb.DesktopSession, remote_path: str, local_path: Path
) -> bool:
    if not (await session.file_exists(remote_path) or await session.directory_exists(remote_path)):
        return False
    payload = await session.read_bytes(remote_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(payload if isinstance(payload, bytes) else str(payload).encode("utf-8"))
    return True


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    eval_root = EVAL_TMP_ROOT / meta["output_dir_name"]
    if eval_root.exists():
        shutil.rmtree(eval_root)
    submission_dir = eval_root / "submission"
    hidden_truth_dir = eval_root / "hidden_truth"
    submission_dir.mkdir(parents=True, exist_ok=True)
    hidden_truth_dir.mkdir(parents=True, exist_ok=True)

    submission_relpaths = [
        "DECISIONS.md",
        "pipeline/samplesheet.csv",
        "pipeline/nextflow.config",
        "pipeline/known_sites_chr22/dbsnp_138.chr22.vcf.gz",
        "pipeline/known_sites_chr22/dbsnp_138.chr22.vcf.gz.tbi",
        "pipeline/known_sites_chr22/Mills_and_1000G_gold_standard.indels.chr22.vcf.gz",
        "pipeline/known_sites_chr22/Mills_and_1000G_gold_standard.indels.chr22.vcf.gz.tbi",
        "results/variants/HG002.filtered.vcf.gz",
        "results/variants/HG002.filtered.vcf.gz.tbi",
        "results/annotation/HG002.filtered.annotated.vcf.gz",
        "results/annotation/HG002.filtered.annotated.vcf.gz.tbi",
        "results/reports/multiqc_report.html",
        "results/reports/multiqc_data/multiqc_general_stats.txt",
        "results/reports/multiqc_data/multiqc_software_versions.txt",
        "results/qc/qc_summary.json",
    ]
    for rel in submission_relpaths:
        await _download_if_present(
            session,
            _remote_join(meta["submission_root"], rel),
            submission_dir / rel,
        )

    hidden_relpaths = [
        "HG002_GRCh38_v4.2.1_chr22.vcf.gz",
        "HG002_GRCh38_v4.2.1_chr22.vcf.gz.tbi",
        "HG002_GRCh38_v4.2.1_chr22_confident.bed",
        "clinvar.chr22.vcf.gz",
        "clinvar.chr22.vcf.gz.tbi",
    ]
    for rel in hidden_relpaths:
        ok = await _download_if_present(
            session,
            _remote_join(meta["hidden_truth_dir"], rel),
            hidden_truth_dir / rel,
        )
        if not ok:
            raise RuntimeError(f"missing hidden truth path during evaluation: {rel}")

    report = score_submission(submission_dir, hidden_truth_dir)
    if report.max_points <= 0:
        return [0.0]
    return [report.total_points / report.max_points]
