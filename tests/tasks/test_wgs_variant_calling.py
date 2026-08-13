from __future__ import annotations

import gzip
from types import SimpleNamespace

import pytest

from tasks.life_sciences.WGS_Variant_Calling.main import config, evaluate, start


class FakeSession:
    def __init__(self, vcf_payload: bytes):
        self.vcf_payload = vcf_payload

    async def read_bytes(self, path: str) -> bytes:
        if path.endswith("variants.filtered.vcf.gz"):
            return self.vcf_payload
        if path.endswith("variants.filtered.vcf.gz.tbi"):
            return gzip.compress(b"TBI\x01test-index")
        if path.endswith("rtg_summary.csv"):
            return b"Type,Precision,Sensitivity,F_measure\nSNP,1.0,1.0,1.0\nINDEL,1.0,1.0,1.0\n"
        if path.endswith(("region_R1_fastqc.html", "region_R2_fastqc.html", "multiqc_report.html")):
            return b"<html>" + b"x" * 100 + b"</html>"
        if path.endswith("flagstat.txt"):
            return b"100 + 0 mapped (99.90% : N/A)\n"
        if path.endswith("duplication_metrics.txt"):
            return b"percent_duplication 0.01\n"
        raise AssertionError(path)


class StagedReferenceSession(FakeSession):
    def __init__(self, vcf_payload: bytes, evaluator_dir):
        super().__init__(vcf_payload)
        self.evaluator_dir = evaluator_dir

    async def read_bytes(self, path: str) -> bytes:
        if "/reference/" in path:
            return (self.evaluator_dir / path.rsplit("/", 1)[-1]).read_bytes()
        return await super().read_bytes(path)


class StartSession:
    def __init__(self):
        self.commands: list[str] = []

    async def run_command(self, command: str) -> None:
        self.commands.append(command)


def _vcf_payload(reference: str, *, include_variants: bool) -> bytes:
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=chr17,length={len(reference)}>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]
    if include_variants:
        snp_ref = reference[19]
        snp_alt = "A" if snp_ref != "A" else "C"
        indel_ref = reference[49]
        lines.extend(
            [
                f"chr17\t20\t.\t{snp_ref}\t{snp_alt}\t100\tPASS\t.",
                f"chr17\t50\t.\t{indel_ref}\t{indel_ref}T\t100\tPASS\t.",
            ]
        )
    return gzip.compress(("\n".join(lines) + "\n").encode())


def _write_evaluator_data(tmp_path, reference: str) -> None:
    (tmp_path / "chr17.fa").write_text(">chr17\n" + reference + "\n")
    (tmp_path / "eval_region_confident.bed").write_text(f"chr17\t0\t{len(reference)}\n")
    (tmp_path / "truth_chr17.vcf.gz").write_bytes(_vcf_payload(reference, include_variants=True))
    (tmp_path / "truth_chr17.vcf.gz.tbi").write_bytes(gzip.compress(b"TBI\x01test-index"))


async def test_evaluator_recomputes_metrics_from_submitted_vcf(tmp_path, monkeypatch):
    reference = ("ACGTTGCAAG" * 20)[:200]
    _write_evaluator_data(tmp_path, reference)
    monkeypatch.setenv("WGS_VARIANT_CALLING_EVAL_DATA_DIR", str(tmp_path))
    task_cfg = SimpleNamespace(metadata=config.to_metadata())

    score = await evaluate(
        task_cfg,
        FakeSession(_vcf_payload(reference, include_variants=True)),
    )

    assert score == [0.9999999999999999]


async def test_perfect_self_report_cannot_replace_variant_calls(tmp_path, monkeypatch):
    reference = ("ACGTTGCAAG" * 20)[:200]
    _write_evaluator_data(tmp_path, reference)
    monkeypatch.setenv("WGS_VARIANT_CALLING_EVAL_DATA_DIR", str(tmp_path))
    task_cfg = SimpleNamespace(metadata=config.to_metadata())

    score = await evaluate(
        task_cfg,
        FakeSession(_vcf_payload(reference, include_variants=False)),
    )

    assert score == [0.4]


async def test_missing_evaluator_bundle_is_not_scored_as_partial_success(tmp_path, monkeypatch):
    reference = ("ACGTTGCAAG" * 20)[:200]
    missing_dir = tmp_path / "missing"
    monkeypatch.setenv("WGS_VARIANT_CALLING_EVAL_DATA_DIR", str(missing_dir))
    task_cfg = SimpleNamespace(metadata=config.to_metadata())

    with pytest.raises(RuntimeError, match="evaluator data is unavailable"):
        await evaluate(
            task_cfg,
            FakeSession(_vcf_payload(reference, include_variants=True)),
        )


async def test_staged_gated_reference_is_used_when_local_bundle_is_absent(
    tmp_path, monkeypatch
):
    reference = ("ACGTTGCAAG" * 20)[:200]
    evaluator_dir = tmp_path / "reference"
    evaluator_dir.mkdir()
    _write_evaluator_data(evaluator_dir, reference)
    monkeypatch.setenv("WGS_VARIANT_CALLING_EVAL_DATA_DIR", str(tmp_path / "missing"))
    task_cfg = SimpleNamespace(metadata=config.to_metadata())

    score = await evaluate(
        task_cfg,
        StagedReferenceSession(
            _vcf_payload(reference, include_variants=True),
            evaluator_dir,
        ),
    )

    assert score == [0.9999999999999999]


async def test_start_clears_stale_output():
    session = StartSession()
    task_cfg = SimpleNamespace(metadata={"remote_output_dir": "/task path/output"})

    await start(task_cfg, session)

    assert session.commands == [
        "if [ -d '/task path/output' ]; then "
        "find '/task path/output' -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; "
        "else mkdir -p -- '/task path/output'; fi"
    ]
