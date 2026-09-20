import asyncio
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tasks.life_sciences.idp_ensemble_scoring import main as task
from tasks.life_sciences.idp_ensemble_scoring.scripts.verify_output import REQUIRED_COLUMNS


TASK_DIR = Path(task.__file__).parent


def valid_csv():
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(REQUIRED_COLUMNS)
    writer.writerows([[f"Model{index}", 0.5, 0.5, 0.5, 0.5] for index in range(1, 6)])
    return stream.getvalue()


def verifier(tmp_path, reference, output):
    reference_path = tmp_path / "reference.csv"
    output_path = tmp_path / "output.csv"
    if reference is not None:
        reference_path.write_bytes(reference)
    if output is not None:
        output_path.write_bytes(output)
    result = subprocess.run(
        [
            sys.executable,
            str(TASK_DIR / "scripts/verify_output.py"),
            "--output-file",
            str(output_path),
            "--reference-file",
            str(reference_path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode, json.loads(result.stdout)


@pytest.mark.parametrize(
    "damage", ["missing", "empty", "nan", "extra", "duplicate", "ragged", "encoding", "quote"]
)
@pytest.mark.parametrize("candidate_present", [False, True])
def test_reference_failures_never_become_candidate_zero(tmp_path, damage, candidate_present):
    content = valid_csv().encode()
    if damage == "missing":
        content = None
    elif damage == "empty":
        content = b""
    elif damage == "nan":
        content = content.replace(b"0.5", b"NaN", 1)
    elif damage == "extra":
        content = content.replace(b"Method,", b"Unknown,", 1)
    elif damage == "duplicate":
        content = content.replace(b"Model2", b"Model1")
    elif damage == "ragged":
        content = content.replace(b"Model1,0.5,", b"Model1,")
    elif damage == "encoding":
        content += b"\xff"
    else:
        content += b'"unclosed'
    status, result = verifier(
        tmp_path, content, valid_csv().encode() if candidate_present else None
    )
    assert status == 2
    assert result["error"] == "invalid_reference"
    assert "score" not in result


@pytest.mark.parametrize("candidate", [None, b"", b"\xff", b'"unclosed'])
def test_bad_or_missing_candidate_is_valid_zero(tmp_path, candidate):
    status, result = verifier(tmp_path, valid_csv().encode(), candidate)
    assert status == 0
    assert result["score"] == 0
    assert not result["passed"]


def test_ranking_is_diagnostic_not_an_extra_gate(tmp_path):
    candidate = valid_csv().replace("Model5,0.5", "Model5,0.50001")
    status, result = verifier(tmp_path, valid_csv().encode(), candidate.encode())
    assert status == 0
    assert result["score"] == 1
    assert result["passed"]
    assert "Ranking mismatch" in result["reasons"][0]
    status, result = verifier(tmp_path, valid_csv().encode(), valid_csv().encode())
    assert result["reasons"] == []


@pytest.mark.parametrize(
    "response",
    [
        dict(return_code=2, stdout='{"error":"invalid_reference"}'),
        dict(return_code=1, stdout=""),
        dict(return_code=0, stdout="not json"),
        dict(return_code=0, stdout="[]"),
        dict(return_code=0, stdout="{}"),
        dict(return_code=0, stdout='{"score":NaN}'),
        dict(return_code=0, stdout='{"score":1.1}'),
        dict(return_code=0, stdout='{"score":0,"error":"invalid_reference"}'),
    ],
)
def test_evaluator_infrastructure_errors_raise(response):
    session = SimpleNamespace(
        file_exists=AsyncMock(return_value=True),
        directory_exists=AsyncMock(return_value=False),
        interface=SimpleNamespace(create_dir=AsyncMock()),
        write_file=AsyncMock(),
        run_command=AsyncMock(return_value=response),
    )
    config = SimpleNamespace(
        metadata=task.IDPEnsembleScoringConfig(VARIANT_NAME="default").to_metadata()
    )
    with pytest.raises(RuntimeError):
        asyncio.run(task.evaluate(config, session))


@pytest.mark.parametrize("score", [0, 0.35, 0.95, 1])
def test_evaluator_preserves_valid_cell_credit(score):
    session = SimpleNamespace(
        file_exists=AsyncMock(return_value=True),
        directory_exists=AsyncMock(return_value=False),
        interface=SimpleNamespace(create_dir=AsyncMock()),
        write_file=AsyncMock(),
        run_command=AsyncMock(
            return_value=dict(return_code=0, stdout=json.dumps(dict(score=score)))
        ),
    )
    config = SimpleNamespace(
        metadata=task.IDPEnsembleScoringConfig(VARIANT_NAME="default").to_metadata()
    )
    assert asyncio.run(task.evaluate(config, session)) == [score]


def test_public_contract_and_card_match():
    config = task.IDPEnsembleScoringConfig(VARIANT_NAME="default")
    card = json.loads((TASK_DIR / "task_card.json").read_text())
    prompt = config.task_description.replace(config.input_dir, "/input").replace(
        config.remote_output_dir, "/output"
    )
    assert card["taskPrompt"] == prompt
    for required in [
        "--pH 5",
        "TP=False, ML=True",
        "heavy_atom_substitute=False",
        "pool_size=200",
        "five proteins",
        "0.5",
        "round(float(value), 2)",
    ]:
        assert required in prompt
    evidence = os.environ.get("IDP_SCIENTIFIC_EVIDENCE")
    if not evidence:
        pytest.skip("Set IDP_SCIENTIFIC_EVIDENCE to validate staged public data overrides")
    override = Path(evidence).parent / "data-overrides/life_sciences/idp_ensemble_scoring/default"
    assert (override / "task_description.txt").read_text() == config.task_description
    assert (override / "input/cached_cspred.py").read_bytes() == (
        TASK_DIR / "scripts/cached_cspred.py"
    ).read_bytes()
    assert (override / "input/runtime-requirements.txt").read_bytes() == (
        TASK_DIR / "runtime-requirements.txt"
    ).read_bytes()
