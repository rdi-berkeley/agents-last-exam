import asyncio
import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.health_medicine.Clinical_Variant_Annotation import main as task


ROOT = Path(__file__).resolve().parents[2]
METADATA = task.config.to_metadata()


def csv_text(columns, values, **kwargs):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, **kwargs)
    writer.writerow(columns)
    writer.writerow(values)
    return stream.getvalue()


@pytest.fixture
def positive():
    values = {
        "CHROM": "chr17",
        "POS": "43106487",
        "REF": "A",
        "ALT": "C",
        "ALLELE_FREQ": "1.721047167707097e-5",
        "GENE": "BRCA1",
        "CONSEQUENCE": "missense_variant",
        "IMPACT": "MODERATE",
        "SIFT": "NA",
        "POLYPHEN": "NA",
        "CLINVAR_RESULT": "Pathogenic; ClinVar VCV000017661",
        "JUSTIFICATION": "BRCA1 p.Cys61Gly, ClinVar pathogenic; gnomAD v4 exome AF 25/1452604.",
    }
    return {
        "variant_count.txt": "200\n",
        **{
            filename: csv_text(columns, [values[name] for name in columns])
            for filename, columns in task.CSV_COLUMNS.items()
        },
    }


def test_positive(positive):
    result = task.score_output_bundle(positive, METADATA)
    assert result["score"] == 1.0
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    "style",
    [
        "quoted",
        "reordered",
        "bom",
        "blank_lines",
        "crlf",
        "bare_chromosome",
        "lower_alleles",
        "numeric_equivalent",
    ],
)
def test_equivalent_csv_remains_full_credit(positive, style):
    for filename, columns in task.CSV_COLUMNS.items():
        header, values = list(csv.reader(io.StringIO(positive[filename])))
        if style == "reordered":
            header.reverse()
            values.reverse()
        elif style == "bare_chromosome":
            values[0] = "17"
        elif style == "lower_alleles":
            values[2:4] = [value.lower() for value in values[2:4]]
        elif style == "numeric_equivalent" and filename == "gnomad_results.csv":
            values[-1] = "0.00001721047167707097"
        positive[filename] = csv_text(
            header,
            values,
            quoting=csv.QUOTE_ALL if style == "quoted" else csv.QUOTE_MINIMAL,
            lineterminator="\r\n" if style == "crlf" else "\n",
        )
        if style == "bom":
            positive[filename] = "\ufeff" + positive[filename]
        elif style == "blank_lines":
            positive[filename] = "\n" + positive[filename] + "\n"
    assert task.score_output_bundle(positive, METADATA)["score"] == 1.0


@pytest.mark.parametrize("filename", list(task.CSV_COLUMNS))
@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_chrom",
        "wrong_pos",
        "wrong_ref",
        "wrong_alt",
        "duplicate",
        "ragged",
        "wrong_header",
        "unclosed_quote",
    ],
)
def test_incorrect_identity_and_malformed_tables_lose_only_their_component(
    positive, filename, mutation
):
    header, values = list(csv.reader(io.StringIO(positive[filename])))
    if mutation.startswith("wrong_") and mutation != "wrong_header":
        index = {"wrong_chrom": 0, "wrong_pos": 1, "wrong_ref": 2, "wrong_alt": 3}[mutation]
        values[index] = {0: "chr18", 1: "431064870", 2: "T", 3: "G"}[index]
    elif mutation == "wrong_header":
        header[-1] = "UNKNOWN"
    elif mutation == "ragged":
        values.append("extra")
    content = csv_text(header, values)
    if mutation == "duplicate":
        content += csv_text(header, values).split("\n", 1)[1]
    elif mutation == "unclosed_quote":
        content += '"unterminated'
    positive[filename] = content
    result = task.score_output_bundle(positive, METADATA)
    assert result["score"] == 0.8
    assert result["checks"][filename] is False


@pytest.mark.parametrize(
    "frequency",
    [
        "NA",
        "",
        "NaN",
        "Infinity",
        "-0.01",
        "1.1",
        "no data 2026",
        "0.01 percent",
        "0; data unavailable",
    ],
)
def test_frequency_must_be_a_finite_fraction(positive, frequency):
    positive["gnomad_results.csv"] = csv_text(
        task.CSV_COLUMNS["gnomad_results.csv"], ["chr17", "43106487", "A", "C", frequency]
    )
    assert task.score_output_bundle(positive, METADATA)["score"] == 0.8


@pytest.mark.parametrize("count", ["199", "201", "200 variants", "-200", "200.0", "", "9" * 5000])
def test_variant_count_must_be_a_single_correct_integer(positive, count):
    positive["variant_count.txt"] = count
    assert task.score_output_bundle(positive, METADATA)["score"] == 0.8


@pytest.mark.parametrize(
    "classification",
    [
        "benign",
        "uncertain significance",
        "not pathogenic",
        "non-pathogenic",
        "no evidence of pathogenicity",
    ],
)
def test_wrong_pathogenic_evidence(positive, classification):
    for filename in ("clinvar_results.csv", "final_candidates.csv"):
        positive[filename] = csv_text(
            task.CSV_COLUMNS[filename], ["chr17", "43106487", "A", "C", classification]
        )
    assert task.score_output_bundle(positive, METADATA)["score"] == 0.6


@pytest.mark.parametrize(
    "gene,consequence",
    [("BRCA10", "missense_variant"), ("BRCA1", "intron_variant"), ("NA", "missense_variant")],
)
def test_incorrect_vep_annotation(positive, gene, consequence):
    positive["vep_results.csv"] = csv_text(
        task.CSV_COLUMNS["vep_results.csv"],
        ["chr17", "43106487", "A", "C", gene, consequence, "MODERATE", "NA", "NA"],
    )
    assert task.score_output_bundle(positive, METADATA)["score"] == 0.8


def test_async_evaluator_reads_original_guest_artifacts(positive):
    class Session:
        async def file_exists(self, path):
            return Path(path).name in positive

        async def read_bytes(self, path):
            return positive[Path(path).name].encode()

    config = type("Config", (), {"metadata": METADATA})()
    assert asyncio.run(task.evaluate(config, Session())) == [1.0]


@pytest.mark.parametrize("present_count", range(6))
def test_async_evaluator_preserves_partial_credit_with_missing_files(positive, present_count):
    available = dict(list(positive.items())[:present_count])

    class Session:
        async def file_exists(self, path):
            return Path(path).name in available

        async def read_bytes(self, path):
            name = Path(path).name
            if name not in available:
                raise RuntimeError(f"[Errno 2] No such file or directory: {path}")
            return available[name].encode()

    result = asyncio.run(task.evaluate(SimpleNamespace(metadata=METADATA), Session()))
    assert result == [present_count / 5]


def test_presence_transport_failure_is_not_a_missing_submission():
    class Session:
        async def file_exists(self, path):
            raise RuntimeError("CUA API unavailable")

    with pytest.raises(RuntimeError, match="CUA API unavailable"):
        asyncio.run(task.evaluate(SimpleNamespace(metadata=METADATA), Session()))


@pytest.mark.skipif(
    not os.environ.get("CLINICAL_AUDIT_CUA_URL"), reason="No retained clinical audit guest"
)
def test_real_retained_partial_artifact_evaluation():
    from cua_bench.computers.remote import RemoteDesktopSession

    async def run():
        session = RemoteDesktopSession(
            api_url=os.environ["CLINICAL_AUDIT_CUA_URL"], os_type="linux"
        )
        try:
            assert await session.file_exists(f"{METADATA['remote_output_dir']}/variant_count.txt")
            assert not await session.file_exists(
                f"{METADATA['remote_output_dir']}/gnomad_results.csv"
            )
            score = await task.evaluate(SimpleNamespace(metadata=METADATA), session)
        finally:
            await session.close()
        assert score == [0.2]

    asyncio.run(run())


def test_absent_artifacts_score_zero():
    assert task.score_output_bundle({}, METADATA)["score"] == 0.0


def test_task_card_matches_runtime():
    card = json.loads(
        (ROOT / "tasks/health_medicine/Clinical_Variant_Annotation/task_card.json").read_text()
    )
    configured = task.TaskConfig(
        TASK_NAME="Clinical_Variant_Annotation", DOMAIN_NAME="health_medicine"
    )
    prompt = configured.task_description.replace(configured.task_dir, "base")
    assert card["taskPrompt"] == prompt
    assert "GRCh38 forward genomic strand" in prompt
    assert "except the final candidate's `ALLELE_FREQ`" in prompt
    assert "pre-created" not in prompt


def test_installed_fixture_has_genomic_target():
    root = ROOT / "task-data-hf/extracted/health_medicine/Clinical_Variant_Annotation/base"
    if not root.exists():
        pytest.skip("Clinical annotation release data is not installed")
    reference = json.loads((root / "reference/expected_candidate.json").read_text())
    assert reference["ref"] == "A" and reference["alt"] == "C"
    lines = (root / "input/patient_variants.vcf").read_text().splitlines()
    records = [line.split("\t") for line in lines if not line.startswith("#")]
    assert len(records) == 200
    assert "##reference=GRCh38" in lines
    matches = [row for row in records if row[1] == "43106487"]
    assert len(matches) == 1 and matches[0][3:5] == ["A", "C"]
    assert all(row[3] != row[4] for row in records)
