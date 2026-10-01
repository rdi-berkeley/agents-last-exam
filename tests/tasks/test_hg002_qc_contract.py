"""Synthetic contract regressions; historical artifact scores are recorded separately."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tasks.life_sciences.hg002_chr22_germline_variant_pipeline.scripts import (
    score_outputs as scorer,
)


MODULES = ["fastqc", "picard", "samtools", "mosdepth"]
METRICS = ["percent_gc", "PERCENT_DUPLICATION", "mapped_passed", "mean_coverage"]
VERSIONS = [
    ("FastQC", "0.11.9"),
    ("GATK4", "4.5.0.0"),
    ("samtools", "1.13"),
    ("mosdepth", "0.3.14"),
]


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)


def write_multiqc(sub, modules=MODULES, versions=None):
    reports = sub / "results/reports"
    data = reports / "multiqc_data"
    write_tsv(
        data / "multiqc_general_stats.txt",
        [
            ["Sample"] + [f"{module}-{metric}" for module, metric in zip(modules, METRICS)],
            ["HG002", "47", "0.00036", "99.99", "29.99"],
        ],
    )
    write_tsv(
        data / "multiqc_software_versions.txt", versions or [["Software", "Version"], *VERSIONS]
    )
    (reports / "multiqc_report.html").write_text("<html><body>QC report</body></html>")
    return data


def write_samplesheet(sub, **changes):
    row = {
        "sample": "HG002",
        "sex": "XY",
        "lane": "1",
        "fastq_1": "fastq/HG002_R1.fastq.gz",
        "fastq_2": "fastq/HG002_R2.fastq.gz",
    }
    row.update(changes)
    write_tsv_path = sub / "pipeline/samplesheet.csv"
    write_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with write_tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)


@pytest.mark.parametrize(
    "family,alias",
    [
        (0, "fastqc"),
        (0, "FASTQC_RAW"),
        (0, "FastQC-status-check"),
        (1, "Picard"),
        (1, "gatk4_markduplicates"),
        (1, "GATK MarkDuplicates"),
        (2, "Samtools"),
        (2, "samtools_stats"),
        (2, "SAMTOOLS_FLAGSTAT"),
        (3, "mosdepth"),
    ],
)
@pytest.mark.parametrize("native", [False, True])
def test_finite_module_aliases_and_native_namespaces(tmp_path, family, alias, native):
    modules = MODULES.copy()
    modules[family] = alias
    if native:
        modules = [f"{module}_mqc-generalstats-{module}" for module in modules]
    write_multiqc(tmp_path, modules)
    result = scorer.gate_multiqc_sections(tmp_path)
    assert result.passed and result.points == result.max_points == 5, result.detail


@pytest.mark.parametrize("shape", ["software", "tool", "grouped", "headerless", "matrix"])
@pytest.mark.parametrize(
    "gatk,version", [("GATK4", "4.5.0.0"), ("gatk", "v4.6.2.0"), ("GaTk", "4.4.0.0+build.1")]
)
def test_version_identity_and_table_equivalences(tmp_path, shape, gatk, version):
    entries = [(gatk, version), *[entry for entry in VERSIONS if entry[0] != "GATK4"]]
    if shape == "grouped":
        rows = [["Group", "Software", "Version"], *[["run", *entry] for entry in entries]]
    elif shape == "matrix":
        rows = [
            ["Sample", *[entry[0] for entry in entries]],
            ["run", *[entry[1] for entry in entries]],
        ]
    elif shape == "headerless":
        rows = entries
    else:
        rows = [[shape, "Version"], *entries]
    write_multiqc(tmp_path, versions=rows)
    assert scorer.gate_multiqc_sections(tmp_path).points == 5


def test_sparse_native_data_separate_sample_rows_and_zero_measurements(tmp_path):
    data = write_multiqc(tmp_path)
    write_tsv(
        data / "multiqc_general_stats.txt",
        [
            [
                "Sample",
                "fastqc-percent_gc",
                "picard-PERCENT_DUPLICATION",
                "samtools_stats-reads_mapped",
                "mosdepth-mean_coverage",
            ],
            ["# comment", "", "", "", ""],
            ["HG002_R1", "47", "", "", ""],
            ["HG002", "", "0", "1e7", "29.99"],
        ],
    )
    assert scorer.gate_multiqc_sections(tmp_path).points == 5


@pytest.mark.parametrize("family", range(4))
@pytest.mark.parametrize("mutation", ["missing", "substring", "empty", "nan", "prose"])
def test_missing_or_nonmetric_modules_cannot_earn_qc_points(tmp_path, family, mutation):
    data = write_multiqc(tmp_path)
    header = ["Sample"] + [f"{module}-{metric}" for module, metric in zip(MODULES, METRICS)]
    row = ["HG002", "47", "0.00036", "99.99", "29.99"]
    column = family + 1
    if mutation == "missing":
        header.pop(column)
        row.pop(column)
    elif mutation == "substring":
        header[column] = "not" + header[column]
    else:
        row[column] = {"empty": "", "nan": "NaN", "prose": "tool ran successfully"}[mutation]
    write_tsv(data / "multiqc_general_stats.txt", [header, row])
    assert scorer.gate_multiqc_sections(tmp_path).points == 0


@pytest.mark.parametrize("tool", [entry[0] for entry in VERSIONS])
@pytest.mark.parametrize(
    "mutation", ["missing", "wrong-tool", "empty", "placeholder", "keyword-version"]
)
def test_tool_names_must_have_actual_version_records(tmp_path, tool, mutation):
    entries = []
    for name, version in VERSIONS:
        if name == tool:
            if mutation == "missing":
                continue
            if mutation == "wrong-tool":
                name = "not" + name
            else:
                version = {"empty": "", "placeholder": "unknown", "keyword-version": tool}[mutation]
        entries.append((name, version))
    write_multiqc(tmp_path, versions=[["Software", "Version"], *entries])
    assert scorer.gate_multiqc_sections(tmp_path).points == 0


@pytest.mark.parametrize(
    "tool,version",
    [("GATK", "3.8"), ("GATK4", "3.8"), ("Picard", "4.5.0.0"), ("GATK4 MarkDuplicates", "4.5.0.0")],
)
def test_picard_or_wrong_major_version_is_not_gatk4(tmp_path, tool, version):
    entries = [(tool, version), *[entry for entry in VERSIONS if entry[0] != "GATK4"]]
    write_multiqc(tmp_path, versions=[["Software", "Version"], *entries])
    assert scorer.gate_multiqc_sections(tmp_path).points == 0


@pytest.mark.parametrize("filename", ["multiqc_general_stats.txt", "multiqc_software_versions.txt"])
@pytest.mark.parametrize(
    "payload",
    [
        "",
        "fastqc picard samtools mosdepth gatk4\nHG002\n",
        "Sample\tfastqc\nHG002\t1\textra\n",
        '"unclosed\t1\n',
        "# fastqc picard samtools mosdepth gatk4\n",
    ],
)
def test_malformed_tables_fail_safely(tmp_path, filename, payload):
    data = write_multiqc(tmp_path)
    (data / filename).write_text(payload)
    assert scorer.gate_multiqc_sections(tmp_path).points == 0


def test_versions_in_group_labels_do_not_count(tmp_path):
    write_multiqc(
        tmp_path,
        versions=[
            ["Group", "Software", "Version"],
            *[[tool, "unrelated", version] for tool, version in VERSIONS],
        ],
    )
    assert scorer.gate_multiqc_sections(tmp_path).points == 0


@pytest.mark.parametrize(
    "basename_pair",
    [("HG002_R1.fastq.gz", "HG002_R2.fastq.gz"), ("reads_1.fq.gz", "reads_2.fq.gz")],
)
def test_mate_validation_ignores_parent_directory_names(tmp_path, basename_pair):
    first, second = basename_pair
    write_samplesheet(tmp_path, fastq_1=f"run_R2/fastq/{first}", fastq_2=f"run_R1/fastq/{second}")
    assert scorer.gate_samplesheet_valid(tmp_path).points == 5


@pytest.mark.parametrize(
    "changes",
    [
        {"fastq_1": "run_R1/HG002_R2.fastq.gz", "fastq_2": "run_R2/HG002_R1.fastq.gz"},
        {"sex": "XX"},
        {"sex": ""},
        {"lane": ""},
        {"fastq_1": ""},
        {"fastq_1": "/data/HG002_R1.fastq.gz"},
        {"fastq_2": "C:\\data\\HG002_R2.fastq.gz"},
    ],
)
def test_samplesheet_scientific_and_portability_checks_remain(tmp_path, changes):
    write_samplesheet(tmp_path, **changes)
    assert scorer.gate_samplesheet_valid(tmp_path).points == 0


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        1,
        "text",
        {},
        {"alignment_rate": 1, "dup_rate": 0},
        *[
            {"alignment_rate": 1, "dup_rate": value, "mean_coverage_chr22": 30}
            for value in [None, {}, [], True, "0.01", -0.1, 12, float("nan"), float("inf")]
        ],
    ],
)
def test_malformed_qc_json_cannot_pass_hard_gate(tmp_path, payload):
    path = tmp_path / "results/qc/qc_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    assert not scorer.gate_qc_json_valid(tmp_path).passed


@pytest.mark.parametrize("alignment", [0, 1, 0.99998, 99.998, 100])
def test_descriptive_alignment_accepts_fraction_or_percentage(tmp_path, alignment):
    path = tmp_path / "results/qc/qc_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"alignment_rate": alignment, "dup_rate": 0.01, "mean_coverage_chr22": 30})
    )
    assert scorer.gate_qc_json_valid(tmp_path).passed


@pytest.mark.parametrize(
    "field,value",
    [
        ("alignment_rate", 101),
        ("alignment_rate", -1),
        ("alignment_rate", True),
        ("mean_coverage_chr22", -1),
        ("mean_coverage_chr22", float("inf")),
        ("mean_coverage_chr22", 10**400),
        ("mean_coverage_chr22", "30"),
    ],
)
def test_qc_number_contract_rejects_invalid_values(tmp_path, field, value):
    path = tmp_path / "results/qc/qc_summary.json"
    path.parent.mkdir(parents=True)
    qc = {"alignment_rate": 1, "dup_rate": 0.01, "mean_coverage_chr22": 30}
    qc[field] = value
    path.write_text(json.dumps(qc))
    assert not scorer.gate_qc_json_valid(tmp_path).passed


@pytest.mark.parametrize("missing", ["html", "stats", "versions"])
def test_required_multiqc_artifacts_cannot_be_omitted(tmp_path, missing):
    data = write_multiqc(tmp_path)
    paths = {
        "html": data.parent / "multiqc_report.html",
        "stats": data / "multiqc_general_stats.txt",
        "versions": data / "multiqc_software_versions.txt",
    }
    paths[missing].unlink()
    assert scorer.gate_multiqc_sections(tmp_path).points == 0


@pytest.mark.parametrize("mutation", ["unnamed", "header-only", "no-metric-columns", "empty-html"])
def test_structurally_empty_qc_cannot_earn_points(tmp_path, mutation):
    data = write_multiqc(tmp_path)
    path = data / "multiqc_general_stats.txt"
    rows = list(csv.reader(path.open(), delimiter="\t"))
    if mutation == "unnamed":
        rows[1][0] = ""
    elif mutation == "header-only":
        rows = rows[:1]
    elif mutation == "no-metric-columns":
        rows[0] = ["Sample", *MODULES]
    else:
        (data.parent / "multiqc_report.html").write_text("")
    write_tsv(path, rows)
    assert scorer.gate_multiqc_sections(tmp_path).points == 0


def write_vcf(path, records, csq=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("##fileformat=VCFv4.2\n##contig=<ID=chr22>\n")
        if csq:
            handle.write(
                '##INFO=<ID=CSQ,Number=.,Type=String,Description="Format: Allele|IMPACT|ClinVar_CLNSIG|gnomAD_AF">\n'
            )
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for position, alt in records:
            info = f"CSQ={alt}|MODERATE|benign|0.1" if csq else "."
            handle.write(f"chr22\t{position}\t.\tA\t{alt}\t100\tPASS\t{info}\n")


@pytest.fixture
def complete_submission(tmp_path):
    sub = tmp_path / "submission"
    hidden = tmp_path / "truth"
    variants = [(position, "G" if position <= 7200 else "C") for position in range(1, 10201)]
    variants += [(position, "AT") for position in range(10201, 10301)]
    write_vcf(sub / "results/variants/HG002.filtered.vcf.gz", variants)
    write_vcf(hidden / "HG002_GRCh38_v4.2.1_chr22.vcf.gz", variants)
    (hidden / "HG002_GRCh38_v4.2.1_chr22_confident.bed").write_text("chr22\t0\t20000\n")
    write_vcf(sub / "results/annotation/HG002.filtered.annotated.vcf.gz", variants[:10], csq=True)
    write_vcf(hidden / "clinvar.chr22.vcf.gz", variants[:10])
    write_samplesheet(sub)
    (sub / "pipeline/nextflow.config").write_text("process.cpus = 4\n")
    for name in ("dbsnp_138.chr22", "Mills_and_1000G_gold_standard.indels.chr22"):
        path = sub / f"pipeline/known_sites_chr22/{name}.vcf.gz"
        write_vcf(path, variants[:10])
        path.with_suffix(".gz.tbi").write_bytes(b"test index presence only")
    write_multiqc(sub)
    qc = sub / "results/qc/qc_summary.json"
    qc.parent.mkdir(parents=True)
    qc.write_text(
        json.dumps({"alignment_rate": 0.999, "dup_rate": 0.01, "mean_coverage_chr22": 30})
    )
    (sub / "DECISIONS.md").write_text("Hard filters: QD < 2.0 and FS > 60.0.\n")
    return sub, hidden


def test_complete_synthetic_submission_preserves_scoring_weights(complete_submission):
    report = scorer.evaluate(*complete_submission)
    assert report.hard_pass and report.pass_
    assert report.total_points == report.max_points == 100
    assert {gate.name: gate.max_points for gate in report.scored_gates} == {
        "snp_f1": 25,
        "indel_f1": 20,
        "snp_ti_tv": 10,
        "mean_coverage": 10,
        "dup_rate": 5,
        "samplesheet_valid": 5,
        "known_sites_chr22_only": 5,
        "vep_annotation_coverage": 10,
        "multiqc_sections": 5,
        "decisions_md_cites_metric": 5,
    }


@pytest.mark.parametrize(
    "mutation,gate",
    [
        ("wrong_variants", "snp_f1"),
        ("wrong_variants", "indel_f1"),
        ("annotation_provenance", "vep_annotation_coverage"),
        ("clinvar_provenance", "vep_annotation_coverage"),
        ("qc_missing", "multiqc_sections"),
        ("coverage", "mean_coverage"),
        ("duplication", "dup_rate"),
    ],
)
def test_valid_multiqc_never_bypasses_scientific_gates(complete_submission, mutation, gate):
    sub, hidden = complete_submission
    if mutation == "wrong_variants":
        write_vcf(
            sub / "results/variants/HG002.filtered.vcf.gz",
            [(position, "T") for position in range(1, 10301)],
        )
    elif mutation == "annotation_provenance":
        write_vcf(
            sub / "results/annotation/HG002.filtered.annotated.vcf.gz", [(15000, "G")], csq=True
        )
    elif mutation == "qc_missing":
        (sub / "results/reports/multiqc_data/multiqc_general_stats.txt").unlink()
    elif mutation == "clinvar_provenance":
        write_vcf(hidden / "clinvar.chr22.vcf.gz", [(15000, "G")])
    else:
        path = sub / "results/qc/qc_summary.json"
        qc = json.loads(path.read_text())
        qc["mean_coverage_chr22" if mutation == "coverage" else "dup_rate"] = (
            10 if mutation == "coverage" else 0.13
        )
        path.write_text(json.dumps(qc))
    report = scorer.evaluate(sub, hidden)
    assert report.hard_pass and report.total_points < 100
    assert next(result for result in report.scored_gates if result.name == gate).points == 0
    if mutation == "qc_missing":
        assert report.total_points == 95
    else:
        assert (
            next(
                result for result in report.scored_gates if result.name == "multiqc_sections"
            ).points
            == 5
        )


def test_prompt_card_expose_identical_qc_contract():
    from tasks.life_sciences.hg002_chr22_germline_variant_pipeline import main

    card = json.loads(Path(main.__file__).with_name("task_card.json").read_text())
    prompt = main.config.task_description
    contract = prompt.split("6. Aggregate QC")[1].split("7. Explain")[0]
    assert "relative FASTQ paths" in prompt and "relative FASTQ paths" in card["taskPrompt"]
    assert contract == card["taskPrompt"].split("6. Aggregate QC")[1].split("7. Explain")[0]
    for requirement in [
        "five-point",
        "FastQC",
        "MarkDuplicates",
        "samtools",
        "mosdepth",
        "GATK4",
        "GATK 4",
        "Software",
        "Version",
        "fraction",
        "flat JSON",
    ]:
        assert requirement in contract


@pytest.mark.parametrize("has_unrelated_evaluator", [False, True])
def test_task_import_ignores_another_tasks_scorer(has_unrelated_evaluator):
    script = f"""
import sys
import types

unrelated = types.ModuleType('score_outputs')
if {has_unrelated_evaluator!r}:
    unrelated.evaluate = object()
sys.modules['score_outputs'] = unrelated
from tasks.life_sciences.hg002_chr22_germline_variant_pipeline import main
from tasks.life_sciences.hg002_chr22_germline_variant_pipeline.scripts import score_outputs
assert main.score_submission is score_outputs.evaluate
assert sys.modules['score_outputs'] is unrelated
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
