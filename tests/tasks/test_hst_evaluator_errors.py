import asyncio
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tasks.physical_sciences.hst_acs_wfc_visit_reduction import main as task


@pytest.fixture
def session():
    return SimpleNamespace(
        directory_exists=AsyncMock(return_value=True),
        file_exists=AsyncMock(return_value=True),
        read_bytes=AsyncMock(return_value=b"reference"),
        run_command=AsyncMock(return_value={"return_code": 0, "stdout": ""}),
    )


def evaluate(session):
    return asyncio.run(task.evaluate(SimpleNamespace(metadata=task.config.to_metadata()), session))


@pytest.fixture
def scorer_result(monkeypatch):
    monkeypatch.setattr(task, "_pull_tree", AsyncMock(return_value={"visit/file": b"data"}))
    monkeypatch.setattr(task, "_run_candidate", AsyncMock(return_value="/scratch/generated"))
    result = subprocess.CompletedProcess([], 0, stdout='{"score": 0.5}', stderr="")
    monkeypatch.setattr(task.subprocess, "run", lambda *args, **kwargs: result)
    return result


@pytest.mark.parametrize("score", [0, 0.5, 1])
def test_valid_scores_are_unchanged(session, scorer_result, score):
    scorer_result.stdout = json.dumps({"score": score})
    assert evaluate(session) == [score]


@pytest.mark.parametrize("failure", [RuntimeError("Request failed"), TimeoutError("timed out")])
def test_reference_transport_error_is_not_agent_zero(session, monkeypatch, scorer_result, failure):
    monkeypatch.setattr(task, "_pull_tree", AsyncMock(side_effect=failure))
    with pytest.raises(type(failure), match=str(failure)):
        evaluate(session)


def test_output_transport_error_is_not_agent_zero(session, monkeypatch, scorer_result):
    monkeypatch.setattr(
        task, "_pull_tree", AsyncMock(side_effect=[{"visit/file": b"data"}, OSError("read failed")])
    )
    with pytest.raises(OSError, match="read failed"):
        evaluate(session)


def test_candidate_transport_error_is_not_a_process_exit(session, monkeypatch, scorer_result):
    monkeypatch.setattr(task, "_run_candidate", AsyncMock(side_effect=TimeoutError("transport")))
    with pytest.raises(TimeoutError, match="transport"):
        evaluate(session)


def test_empty_reference_is_not_a_score(session, monkeypatch, scorer_result):
    monkeypatch.setattr(task, "_pull_tree", AsyncMock(return_value={}))
    with pytest.raises(RuntimeError, match="reference"):
        evaluate(session)
    task._run_candidate.assert_not_awaited()


@pytest.mark.parametrize("returncode", [1, 2, -9, 137])
def test_failed_scorer_is_not_a_score_even_with_json(session, scorer_result, returncode):
    scorer_result.returncode = returncode
    with pytest.raises(RuntimeError, match="scorer"):
        evaluate(session)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "traceback",
        "[]",
        "null",
        "{}",
        '{"score": null}',
        '{"score": true}',
        '{"score": "0.5"}',
        '{"score": NaN}',
        '{"score": Infinity}',
        '{"score": -0.1}',
        '{"score": 1.1}',
    ],
)
def test_invalid_scorer_results_are_errors(session, scorer_result, stdout):
    scorer_result.stdout = stdout
    with pytest.raises(RuntimeError, match="scorer"):
        evaluate(session)


def test_existing_json_line_fallback_is_retained(session, scorer_result):
    scorer_result.stdout = 'diagnostic\n{"score": 0.5}\n'
    assert evaluate(session) == [0.5]


@pytest.mark.parametrize("return_code", [1, 2, None])
def test_incomplete_tree_listing_is_not_accepted(session, return_code):
    session.run_command.return_value = {
        "return_code": return_code,
        "stdout": "visit/partial.csv\n",
        "stderr": "listing failed",
    }
    with pytest.raises(RuntimeError, match="list"):
        asyncio.run(task._pull_tree(session, "/reference"))
    session.read_bytes.assert_not_awaited()


def test_missing_reference_directory_is_an_error(session):
    session.directory_exists.return_value = False
    with pytest.raises(RuntimeError, match="missing"):
        asyncio.run(task._pull_tree(session, "/reference"))


def test_missing_candidate_output_directory_remains_empty(session):
    session.directory_exists.return_value = False
    assert asyncio.run(task._pull_tree(session, "/output", allow_missing=True)) == {}


def test_missing_candidate_is_still_zero(session):
    session.file_exists.side_effect = lambda path: path != task.config.candidate_script
    session.run_command.return_value = {"return_code": 0, "stdout": "visit/reference.csv\n"}
    assert evaluate(session) == [0.0]


def test_nonzero_candidate_exit_is_still_zero(session):
    session.run_command.side_effect = [
        {"return_code": 0, "stdout": "visit/reference.csv\n"},
        {"return_code": 0, "stdout": ""},
        {"return_code": 1, "stdout": "", "stderr": "solver fit did not converge"},
    ]
    assert evaluate(session) == [0.0]


def test_candidate_setup_failure_is_not_an_agent_score(session):
    session.run_command.side_effect = [
        {"return_code": 0, "stdout": "visit/reference.csv\n"},
        {"return_code": 1, "stdout": "", "stderr": "No space left on device"},
    ]
    with pytest.raises(RuntimeError, match="prepare"):
        evaluate(session)


def test_missing_candidate_exit_marker_is_not_an_agent_score(session):
    session.run_command.side_effect = [
        {"return_code": 0, "stdout": "visit/reference.csv\n"},
        {"return_code": 0, "stdout": ""},
        {"stdout": "unknown execution state"},
    ]
    with pytest.raises(RuntimeError, match="exit"):
        evaluate(session)
