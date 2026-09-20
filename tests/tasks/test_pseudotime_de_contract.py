import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tasks.life_sciences.pseudotime_de import main as task
from tasks.life_sciences.pseudotime_de.scripts import score_outputs as scorer

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "tasks/life_sciences/pseudotime_de"


def gene_csv(genes, header="gene", **writer_options):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, **writer_options)
    writer.writerow([header])
    writer.writerows([gene] for gene in genes)
    return stream.getvalue().encode("utf-8")


def score(candidate, reference=b"x\nG1\nG2\nG3\nG4\nG5\n"):
    payloads = {} if candidate is None else {"de_genes.csv": candidate}
    return scorer.score_submission(payloads, reference)


@pytest.mark.parametrize(
    ("candidate", "precision", "recall", "expected"),
    [
        (["G1", "G2", "G3", "G4", "G5"], 1, 1, 1),
        (["G1", "G2", "G3", "G4"], 1, 0.8, 1),
        (["G1", "G2", "G3", "G4", "D1"], 0.8, 0.8, 1),
        (["G1", "G2", "G3", "G4", "D1", "D2"], 4 / 6, 0.8, 0),
        (["G1", "G2", "G3"], 1, 0.6, 0),
        (["G1", "G2", "G3", "G4", "G5", "D1", "D2"], 5 / 7, 1, 0),
        (["D1", "D2"], 0, 0, 0),
    ],
)
def test_precision_and_recall_must_both_pass(candidate, precision, recall, expected):
    report = score(gene_csv(candidate))
    assert report.score == expected
    assert report.precision == precision
    assert report.recall == recall
    assert not report.error
    assert report.agent_gene_count == len(candidate)
    assert report.reference_gene_count == 5
    assert report.false_positive_count == len(candidate) - report.intersection_count
    assert report.false_negative_count == 5 - report.intersection_count
    assert json.loads(json.dumps(report.to_dict())) == report.to_dict()


def test_f1_cannot_trade_away_original_recall_requirement():
    report = score(gene_csv(["G1", "G2", "G3"]), b"x\nG1\nG2\nG3\nG4\n")
    assert 2 * report.precision * report.recall / (report.precision + report.recall) > 0.8
    assert report.recall == 0.75
    assert report.score == 0


@pytest.mark.parametrize("dimension", ["precision", "recall"])
def test_metrics_are_not_rounded_before_threshold_comparison(monkeypatch, dimension):
    class CountedSet:
        def __init__(self, count):
            self.count = count

        def __len__(self):
            return self.count

        def __and__(self, other):
            return CountedSet(4_000_000)

    sizes = {"gene": 4_000_000, "x": 4_000_000}
    sizes["gene" if dimension == "precision" else "x"] = 5_000_001
    monkeypatch.setattr(scorer, "_parse_gene_set", lambda raw, header: CountedSet(sizes[header]))
    report = score(b"unused")
    assert round(getattr(report, dimension), 6) == 0.8
    assert getattr(report, dimension) < 0.8
    assert report.score == 0


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("quoting", [csv.QUOTE_MINIMAL, csv.QUOTE_ALL])
def test_equivalent_csv_serializations(bom, line_ending, quoting):
    candidate = gene_csv(
        ["G5", "G2", "G4", "G1", "G3", "G1", "G5"],
        header="  GeNe  ",
        lineterminator=line_ending,
        quoting=quoting,
    )
    report = score(bom + candidate)
    assert report.score == 1
    assert report.agent_gene_count == 5


def test_blank_lines_outer_whitespace_and_no_final_newline_are_equivalent():
    report = score(b'\n\ngene\n\n" G1 "\n\nG2\nG3\nG4\nG5')
    assert report.score == 1


@pytest.mark.parametrize("gene", ["NA", "NAN", "C1orf54", "HLA-DRA", "RP11-1.1"])
def test_symbols_are_strings_not_missing_values_or_uppercase_coerced(gene):
    assert score(gene_csv([gene]), gene_csv([gene], header="x")).score == 1
    assert score(gene_csv([gene.lower()]), gene_csv([gene], header="x")).score == 0


def test_duplicates_cannot_improve_precision_or_recall():
    original = ["G1", "G2", "G3", "G4", "D1", "D2"]
    assert (
        score(gene_csv(original)).to_dict()
        == score(gene_csv(original + ["G1"] * 10000 + ["D1"] * 10000)).to_dict()
    )


INVALID_CANDIDATES = [
    b"",
    b"\xef\xbb\xbf",
    b"\n\n",
    b"gene\n",
    b"GeneSymbol\nG1\nG2\nG3\nG4\nG5\n",
    b"x\nG1\nG2\nG3\nG4\nG5\n",
    b",gene\n1,G1\n",
    b"gene,score\nG1,1\n",
    b"score,gene\n1,G1\n",
    b"gene,gene\nG1,G1\n",
    b"gene,\nG1,\n",
    b"gene\nG1,0\n",
    b"gene\nG1,\n",
    b'gene\n"G1\n',
    b'gene\n"G1"trailing\n',
    b'gene\nG"1\n',
    b'gene\n"G1,G2"\n',
    b'gene\n"G1\nG2"\n',
    b'gene\n"G1""G2"\n',
    b'gene\n""\n',
    b"gene\n   \n",
    b"gene\nG1 G2\n",
    b"gene\nG1\tG2\n",
    b"gene\nG1\x00\n",
    b"gene\nG1\x7f\n",
    b"gene\nG1\xff\n",
    "gene\nG1\u200b\n".encode(),
]


@pytest.mark.parametrize("candidate", INVALID_CANDIDATES)
def test_malformed_candidates_are_zero_not_partial_success(candidate):
    report = score(candidate)
    assert report.score == 0
    assert report.error
    assert report.precision == report.recall == 0
    assert report.agent_gene_count == 0
    assert report.reference_gene_count == 5


@pytest.mark.parametrize("suffix", [b"G6,extra\n", b'"G6\n', b"G6\xff\n", b'""\n'])
def test_bad_record_after_all_gold_invalidates_whole_candidate(suffix):
    report = score(gene_csv(["G1", "G2", "G3", "G4", "G5"]) + suffix)
    assert report.score == 0
    assert report.error
    assert report.intersection_count == 0


def test_missing_candidate_is_zero_with_valid_reference():
    report = score(None)
    assert report.score == 0
    assert "missing" in report.error
    assert report.reference_gene_count == 5


@pytest.mark.parametrize(
    "reference",
    [
        b"x\nG1\nG2\nG1\n",
        b'"","x"\n"1","G1"\n"2","G2"\n"3","G1"\n',
        b'\xef\xbb\xbf""," X "\r\n"2","G2"\r\n"1"," G1 "\r\n',
    ],
)
def test_reference_single_column_and_actual_r_row_index_format(reference):
    report = score(b"gene\nG2\nG1\n", reference)
    assert report.score == 1
    assert report.reference_gene_count == 2


INVALID_REFERENCES = [
    b"",
    b"x\n",
    b"gene\nG1\n",
    b"wrong\nG1\n",
    b"x,extra\nG1,ignored\n",
    b"x,x\nG1,G1\n",
    b",x\n",
    b",x\n1\n",
    b",x\n1,G1,extra\n",
    b",x\n1,\n",
    b",x\nrow,G1\n",
    b",x\n0,G1\n",
    b",x\n-1,G1\n",
    b",x\n1,G1\n1,G2\n",
    b'x\n"G1\n',
    b"x\nG1\xff\n",
    b"x\nG1\x00\n",
]


@pytest.mark.parametrize("reference", INVALID_REFERENCES)
@pytest.mark.parametrize("candidate", [None, b"bad", b"gene\nG1\n"])
def test_reference_errors_never_become_candidate_zero(reference, candidate):
    with pytest.raises(scorer.ReferenceValidationError, match="evaluator-controlled reference"):
        score(candidate, reference)


def test_unexpected_scorer_bug_is_not_converted_to_zero(monkeypatch):
    def broken_parser(raw, expected_column):
        if expected_column == "x":
            return {"G1"}
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(scorer, "_parse_gene_set", broken_parser)
    with pytest.raises(RuntimeError, match="unexpected failure"):
        score(b"gene\nG1\n")


def test_random_sets_against_independent_integer_oracle():
    generator = random.Random(27)
    for _ in range(250):
        reference = {f"G{index}" for index in range(generator.randint(1, 80))}
        selected = set(generator.sample(sorted(reference), generator.randint(0, len(reference))))
        selected.update(f"D{index}" for index in range(generator.randint(0, 30)))
        report = score(gene_csv(selected), gene_csv(reference, header="x"))
        matches = sum(gene in reference for gene in selected)
        expected = (
            bool(selected)
            and 5 * matches >= 4 * len(selected)
            and 5 * matches >= 4 * len(reference)
        )
        assert report.score == float(expected)
        assert report.intersection_count == matches
        assert report.precision == (matches / len(selected) if selected else 0)
        assert report.recall == matches / len(reference)


def make_session(candidate=b"gene\nG1\n", reference=b"x\nG1\n"):
    files = {task.config.reference_file: reference, task.config.output_file: candidate}
    return SimpleNamespace(
        file_exists=AsyncMock(side_effect=lambda path: files.get(path) is not None),
        read_bytes=AsyncMock(side_effect=lambda path: files[path]),
        reference_sha256=hashlib.sha256(reference or b"").hexdigest(),
    )


def evaluate(session):
    with patch.object(task, "REFERENCE_SHA256", session.reference_sha256):
        return asyncio.run(
            task.evaluate(SimpleNamespace(metadata=task.config.to_metadata()), session)
        )


@pytest.mark.parametrize("candidate", [None, b"bad", b"gene\nG1\xff\n", b"gene\nG1\n"])
def test_entrypoint_wrong_reference_version_is_infrastructure_before_candidate(candidate):
    session = make_session(candidate=candidate)
    with pytest.raises(scorer.ReferenceValidationError, match="reference hash mismatch"):
        asyncio.run(task.evaluate(SimpleNamespace(metadata=task.config.to_metadata()), session))
    session.file_exists.assert_awaited_once_with(task.config.reference_file)
    session.read_bytes.assert_awaited_once_with(task.config.reference_file)


@pytest.mark.parametrize(
    "candidate", [b"gene\nG1\n", b"gene\nD1\n", b"malformed", b"gene\nG1\xff\n", None]
)
def test_real_entrypoint_and_report_logging(candidate, caplog):
    session = make_session(candidate=candidate)
    with caplog.at_level(logging.INFO, logger=task.__name__):
        assert evaluate(session) == [1.0 if candidate == b"gene\nG1\n" else 0.0]
    assert '"precision":' in caplog.text
    assert '"recall":' in caplog.text


@pytest.mark.parametrize("reference", [None, b"", b"x\n", b"bad", b"x\nG1\xff\n"])
@pytest.mark.parametrize("candidate", [None, b"bad", b"gene\nG1\n"])
def test_entrypoint_reference_errors_propagate(reference, candidate):
    with pytest.raises(RuntimeError, match="reference"):
        evaluate(make_session(candidate, reference))


@pytest.mark.parametrize("path_kind", ["reference", "candidate"])
@pytest.mark.parametrize("operation", ["file_exists", "read_bytes"])
@pytest.mark.parametrize(
    "failure", [OSError("transport failure"), TimeoutError("transport timeout")]
)
def test_transport_errors_are_not_candidate_zero(path_kind, operation, failure):
    session = make_session()
    path = task.config.reference_file if path_kind == "reference" else task.config.output_file
    original = getattr(session, operation).side_effect

    def fail(target):
        if target == path:
            raise failure
        return original(target)

    getattr(session, operation).side_effect = fail
    with pytest.raises(type(failure), match="transport"):
        evaluate(session)


def test_scorer_import_is_task_scoped_in_loader_and_shared_process():
    script = """
import sys
import types
from ale_run.tasks.loader import TaskLoader
unrelated = types.ModuleType('score_outputs')
sys.modules['score_outputs'] = unrelated
before = list(sys.path)
from tasks.life_sciences.pseudotime_de import main
from tasks.life_sciences.pseudotime_de.scripts import score_outputs
assert main.score_submission is score_outputs.score_submission
assert main.REQUIRED_FILES is score_outputs.REQUIRED_FILES
assert sys.modules['score_outputs'] is unrelated
assert sys.path == before
loader = TaskLoader('tasks/life_sciences/pseudotime_de')
loaded = loader.load()
assert loader._load_module().score_submission is score_outputs.score_submission
assert loaded['description'] == main.config.task_description
assert 'score_outputs' not in sys.modules
assert sys.path == before
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_prompt_card_and_raw_brief_agree():
    card = json.loads((TASK_DIR / "task_card.json").read_text())
    prompt = task.config.task_description.replace(task.config.task_dir, "base")
    raw = (TASK_DIR / "task_description.txt").read_text()
    assert card["taskPrompt"] == prompt
    for prefix in ("Format: ", "Scoring: "):
        paragraph = next(line[2:] for line in prompt.splitlines() if line.startswith("- " + prefix))
        assert paragraph in raw
    for text in (prompt, raw):
        normalized = " ".join(text.split())
        assert "do not apply a separate absolute logFC or fcMedian filter" in normalized
        assert "all tested genes" in normalized
        assert "raw-count expression matrix (all genes, not only the 1500 HVGs)" in normalized
        assert "Run5_164698952452459" in text
        assert "Run5_131097901611291" in text
        assert "Run5_134936662236454" in text
        assert "Run4_200562869397916" in text
        assert "num_waypoints=500" in text
        assert "nknots=6" in text
        assert "l2fc=log2(1.5)" in text
    assert "BOTH precision >= 0.80 AND recall >= 0.80" in card["evaluation"]
    assert card["vm"] == {
        "machineType": "c4-standard-4",
        "snapshot": "cpu-free-ubuntu",
        "timeout": 7200,
    }


@pytest.fixture
def actual_reference():
    path = ROOT / "task-data-hf/extracted/life_sciences/pseudotime_de/base/reference/degs.csv"
    if not path.is_file():
        pytest.skip("Local benchmark reference not installed; synthetic contracts still run")
    raw = path.read_bytes()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True))
    assert rows[0] == ["", "x"]
    assert all(len(row) == 2 for row in rows[1:])
    return raw, {row[1] for row in rows[1:]}


def test_actual_reference_positive_and_decoy_attack(actual_reference):
    raw, genes = actual_reference
    assert score(gene_csv(genes), raw).score == 1
    decoys = {f"DECOY{index}" for index in range(10000)}
    assert genes.isdisjoint(decoys)
    report = score(gene_csv(genes | decoys), raw)
    assert report.score == 0
    assert report.recall == 1
    assert report.precision == len(genes) / (len(genes) + 10000)
    assert report.false_positive_count == 10000


def test_preserved_unified_log_output_and_data_override(actual_reference):
    evidence_path = os.environ.get("PSEUDOTIME_REPAIR_EVIDENCE")
    if not evidence_path:
        pytest.skip("Set PSEUDOTIME_REPAIR_EVIDENCE to replay preserved local output")
    evidence = Path(evidence_path)
    raw, genes = actual_reference
    candidate = (evidence / "preserved/de_genes.csv").read_bytes()
    selected = {
        row["gene"]
        for row in csv.DictReader(io.StringIO(candidate.decode("utf-8-sig"), newline=""))
    }
    matches = sum(gene in genes for gene in selected)
    expected = float(5 * matches >= 4 * len(genes) and 5 * matches >= 4 * len(selected))
    report = score(candidate, raw)
    assert report.precision == matches / len(selected)
    assert report.recall == matches / len(genes)
    assert report.score == expected
    assert evaluate(make_session(candidate, raw)) == [expected]
    assert json.loads((evidence / "preserved/eval_result.json").read_text())["score"] == 0.0
    override = (
        evidence.parent
        / "data-overrides/life_sciences/pseudotime_de/base/input/task_description.txt"
    )
    assert override.read_bytes() == (TASK_DIR / "task_description.txt").read_bytes()
