"""Submission rejection is separate from reference and infrastructure failures."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tasks.life_sciences.gene_expression_differential_analysis_functional_enrichment_analysis_1.scripts import (
    score_outputs as scorer,
)
from tests.tasks.test_deg_zero_probability import deg_rows, tsv


@pytest.fixture
def payloads():
    enrichment = tsv(
        scorer.ENRICH_REQUIRED_COLUMNS,
        [
            {
                "Gene_set": "KEGG_2021_Human",
                "Term": "Synthetic term",
                "Overlap": "1/2",
                "P-value": 0.01,
                "Adjusted P-value": 0.02,
                "Old P-value": 0,
                "Old Adjusted P-value": 0,
                "Odds Ratio": 2,
                "Combined Score": 4,
                "Genes": "UP",
            }
        ],
    )
    return dict(
        zip(
            scorer.REQUIRED_FILES,
            [tsv(scorer.DEG_REQUIRED_COLUMNS, deg_rows()), enrichment, enrichment],
        )
    )


def score(candidate, reference):
    return scorer.score_submission(
        output_payloads=candidate,
        reference_deg_payload=reference[scorer.REQUIRED_FILES[0]],
        reference_up_payload=reference[scorer.REQUIRED_FILES[1]],
        reference_down_payload=reference[scorer.REQUIRED_FILES[2]],
    )


@pytest.mark.parametrize("name", scorer.REQUIRED_FILES)
@pytest.mark.parametrize(
    "defect", ["utf8", "ragged", "duplicate-header", "unclosed-quote", "empty", "columns"]
)
def test_malformed_submission_scores_zero_but_reference_raises(payloads, name, defect):
    malformed = dict(payloads)
    text = malformed[name].decode()
    if defect == "utf8":
        malformed[name] = b"\xff"
    elif defect == "ragged":
        malformed[name] += b"a\tb\n"
    elif defect == "duplicate-header":
        lines = text.splitlines()
        lines[0] += "\t" + lines[0].split("\t")[0]
        malformed[name] = "\n".join(lines).encode()
    elif defect == "unclosed-quote":
        malformed[name] = b'id\tgene\n"unterminated'
    elif defect == "empty":
        malformed[name] = b""
    else:
        malformed[name] = b"wrong\nvalue\n"
    report = score(malformed, payloads)
    assert report.score == 0 and report.failures
    with pytest.raises(scorer.ReferenceDataError):
        score(payloads, malformed)
    with pytest.raises(scorer.ReferenceDataError):
        score({}, malformed)


def test_reference_without_significant_directions_raises(payloads):
    rows = deg_rows()
    for row in rows:
        row.update(padj=0.5, significant="no significant")
    malformed = {**payloads, scorer.REQUIRED_FILES[0]: tsv(scorer.DEG_REQUIRED_COLUMNS, rows)}
    with pytest.raises(scorer.ReferenceDataError, match="both significant directions"):
        score(payloads, malformed)


@pytest.fixture(params=["package", "task_loader"])
def evaluator(monkeypatch, request):
    package = scorer.__name__.rsplit(".scripts.", 1)[0]
    module_name = (
        package + ".boundary_test"
        if request.param == "package"
        else "_task_" + package.removeprefix("tasks.").replace(".", "_")
    )
    framework = ModuleType("cua_bench")
    framework.tasks_config = framework.setup_task = framework.evaluate_task = lambda **kwargs: (
        lambda function: function
    )
    common = ModuleType("tasks.common_setup")
    common.BaseTaskSetup = object
    runtime = ModuleType("tasks.linux_runtime")
    runtime.LinuxTaskConfig = SimpleNamespace
    collision = ModuleType("score_outputs")
    collision.REQUIRED_FILES = ["wrong-task.tsv"]
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "cua_bench", framework)
        patch.setitem(sys.modules, "tasks.common_setup", common)
        patch.setitem(sys.modules, "tasks.linux_runtime", runtime)
        patch.setitem(sys.modules, "score_outputs", collision)
        spec = importlib.util.spec_from_file_location(
            module_name, Path(scorer.__file__).parents[1] / "main.py"
        )
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, module_name, module)
        if request.param == "task_loader":
            assert module.__package__ == ""
        before = list(sys.path)
        spec.loader.exec_module(module)
        assert sys.path == before
        assert module.score_submission is scorer.score_submission
        yield module


class Session:
    def __init__(self, files, directories=(), error=None):
        self.files = files
        self.directories = directories
        self.error = error

    async def file_exists(self, path):
        return path in self.files

    async def directory_exists(self, path):
        return path in self.directories

    async def read_bytes(self, path):
        if self.error and self.error[0] == path:
            raise self.error[1]
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


@pytest.fixture
def session_config(payloads):
    files = {
        prefix + name: value
        for prefix in ("output/", "reference/")
        for name, value in payloads.items()
    }
    metadata = dict(
        zip(
            ["reference_deg_file", "reference_up_file", "reference_down_file"],
            ["reference/" + name for name in scorer.REQUIRED_FILES],
        )
    )
    metadata["output_files"] = {name: "output/" + name for name in scorer.REQUIRED_FILES}
    return files, SimpleNamespace(metadata=metadata)


def test_evaluator_independent_synthetic_correct_one(evaluator, session_config):
    files, config = session_config
    assert asyncio.run(evaluator.evaluate(config, Session(files))) == [1.0]


@pytest.mark.parametrize("defect", ["missing", "directory", "race", "malformed"])
def test_missing_or_nonfile_submission_zero(evaluator, session_config, defect):
    files, config = session_config
    path = "output/" + scorer.REQUIRED_FILES[0]
    directories, error = [], None
    if defect == "missing":
        del files[path]
    elif defect == "directory":
        directories = [path]
    elif defect == "race":
        error = (path, FileNotFoundError(path))
    else:
        files[path] = b"\xff"
    assert asyncio.run(evaluator.evaluate(config, Session(files, directories, error))) == [0.0]


@pytest.mark.parametrize("prefix", ["output/", "reference/"])
@pytest.mark.parametrize(
    "exception", [TimeoutError, ConnectionError, PermissionError, RuntimeError]
)
def test_transport_and_unexpected_errors_propagate(evaluator, session_config, prefix, exception):
    files, config = session_config
    session = Session(files, error=(prefix + scorer.REQUIRED_FILES[0], exception("transport")))
    with pytest.raises(exception, match="transport"):
        asyncio.run(evaluator.evaluate(config, session))


@pytest.mark.parametrize("defect", ["missing", "malformed"])
def test_reference_failures_propagate_with_missing_candidate(evaluator, session_config, defect):
    files, config = session_config
    del files["output/" + scorer.REQUIRED_FILES[0]]
    path = "reference/" + scorer.REQUIRED_FILES[0]
    if defect == "missing":
        del files[path]
    else:
        files[path] = b"\xff"
    with pytest.raises(FileNotFoundError if defect == "missing" else scorer.ReferenceDataError):
        asyncio.run(evaluator.evaluate(config, Session(files)))
