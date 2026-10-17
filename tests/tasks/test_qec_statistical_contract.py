from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.computing_math.clustered_cyclic_code_circuit_level_simulation import main as task
from tasks.computing_math.clustered_cyclic_code_circuit_level_simulation.scripts.circuit_contract import (
    circuit_signature,
)
from tasks.computing_math.clustered_cyclic_code_circuit_level_simulation.scripts.score_logical_error_rates import (
    EXPECTED_COLUMNS,
    _close_numeric,
    _consistent_rate,
    score_logical_error_rates_bytes,
)


def encode_rows(rows):
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=EXPECTED_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def table(counts):
    rows = []
    for index, failures in enumerate(counts):
        probability = int(failures) / 5000
        rows.append(
            {
                "code": "[24,8,3]",
                "n": 24,
                "k": 8,
                "d": 3,
                "p_physical": str((index + 1) / 10000),
                "num_rounds": 42,
                "num_shots": 5000,
                "num_failures": int(failures),
                "p_logical": probability,
                "lfr_per_round": 1 - (1 - probability) ** (1 / 42),
                "lfr_per_round_per_qubit": 1 - (1 - probability) ** (1 / 336),
            }
        )
    return rows


def test_independent_samples_and_batching_need_not_match_counts():
    generator = np.random.default_rng(90210)
    probabilities = np.linspace(0.01, 0.99, 60)
    reference = encode_rows(table(generator.binomial(5000, probabilities)))
    failures = 0
    for _ in range(100):
        sampled = generator.binomial(5000, probabilities)
        result = score_logical_error_rates_bytes(
            agent_bytes=encode_rows(table(sampled)),
            reference_bytes=reference,
        )
        failures += not result.passed
    assert failures <= 1


@pytest.mark.parametrize("kind", ["zero", "flat", "half_noise", "no_decode", "reversed"])
def test_wrong_experiments_fail_even_with_consistent_rates(kind):
    reference_counts = np.arange(60) * 75 + 100
    if kind == "zero":
        counts = np.zeros(60)
    elif kind == "flat":
        counts = np.full(60, 500)
    elif kind == "half_noise":
        counts = reference_counts // 2
    elif kind == "no_decode":
        counts = np.full(60, 4900)
    else:
        counts = reference_counts[::-1]
    result = score_logical_error_rates_bytes(
        agent_bytes=encode_rows(table(counts)),
        reference_bytes=encode_rows(table(reference_counts)),
    )
    assert not result.passed
    assert any("distribution_mismatch" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_shots", "4999"),
        ("num_rounds", "0"),
        ("k", "0"),
        ("num_failures", "NaN"),
        ("num_failures", "Infinity"),
        ("num_failures", "-1"),
        ("num_failures", "5001"),
        ("p_logical", "NaN"),
        ("p_logical", "0.9"),
        ("lfr_per_round", "0.5"),
        ("lfr_per_round_per_qubit", "0.5"),
    ],
)
def test_invalid_parameters_and_inconsistent_rates_fail(field, value):
    reference = encode_rows(table([250, 750]))
    rows = table([250, 750])
    rows[0][field] = value
    assert not score_logical_error_rates_bytes(
        agent_bytes=encode_rows(rows),
        reference_bytes=reference,
    ).passed


def test_missing_extra_and_duplicate_rows_fail():
    rows = table([250, 750])
    for malformed in (rows[:1], rows + [rows[0]], table([250, 750, 900])):
        assert not score_logical_error_rates_bytes(
            agent_bytes=encode_rows(malformed),
            reference_bytes=encode_rows(rows),
        ).passed


def test_reference_error_is_not_a_solver_zero():
    with pytest.raises(ValueError, match="reference"):
        score_logical_error_rates_bytes(agent_bytes=encode_rows(table([1])), reference_bytes=b"bad")


@pytest.mark.parametrize("shots", [1, 100, 249])
def test_undersized_candidate_is_not_a_reference_error(shots):
    rows = table([0])
    rows[0]["num_shots"] = shots
    result = score_logical_error_rates_bytes(
        agent_bytes=encode_rows(rows), reference_bytes=encode_rows(table([250]))
    )
    assert not result.passed
    assert any("num_shots_mismatch" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_failures", "NaN"),
        ("num_failures", "Infinity"),
        ("num_failures", "bad"),
        ("num_failures", "1.5"),
        ("num_failures", "-1"),
        ("num_failures", "5001"),
        ("num_shots", "0"),
        ("num_shots", "bad"),
        ("num_rounds", "0"),
        ("k", "NaN"),
    ],
)
def test_malformed_reference_counts_raise_independently_of_candidate(field, value):
    reference = table([250])
    reference[0][field] = value
    with pytest.raises(ValueError, match="Invalid QEC reference"):
        score_logical_error_rates_bytes(
            agent_bytes=b"invalid candidate", reference_bytes=encode_rows(reference)
        )


@pytest.mark.parametrize("rounds", [20, 42])
@pytest.mark.parametrize("numeric_format", [".12g", ".12e", ".17g"])
def test_documented_probability_precision_passes(rounds, numeric_format):
    rows = table([0, 1, 2, 2499, 2500, 4999, 5000])
    for row in rows:
        probability = row["p_logical"]
        row["num_rounds"] = rounds
        for field, divisor in (("lfr_per_round", rounds), ("lfr_per_round_per_qubit", rounds * 8)):
            row[field] = -math.expm1(math.log1p(-probability) / divisor) if probability < 1 else 1.0
    reference = encode_rows(rows)
    for row in rows:
        for field in ("p_logical", "lfr_per_round", "lfr_per_round_per_qubit"):
            row[field] = format(row[field], numeric_format)
    assert score_logical_error_rates_bytes(
        agent_bytes=encode_rows(rows), reference_bytes=reference
    ).passed
    requirements = (task.ASSETS_DIR / "output_requirements.txt").read_text()
    assert "at least 12\nsignificant digits" in requirements
    assert "Scientific\nnotation is accepted" in requirements


def test_twelve_significant_digits_cover_every_allowed_failure_count():
    for failures in range(5001):
        probability = failures / 5000
        for rounds in (20, 42):
            for logical_qubits in (1, 8):
                rate = (
                    -math.expm1(math.log1p(-probability) / (rounds * logical_qubits))
                    if probability < 1
                    else 1.0
                )
                assert _close_numeric(
                    float(format(rate, ".12g")),
                    _consistent_rate(probability, rounds, logical_qubits),
                )


SMALL_CIRCUIT = "R 0\nX_ERROR(0.1) 0\nTICK\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"


def test_circuit_structure_accepts_equivalent_serialization():
    equivalent = "# same experiment\nREPEAT 1 {\n" + SMALL_CIRCUIT + "}\nTICK\n"
    assert circuit_signature(equivalent) == circuit_signature(SMALL_CIRCUIT)


def test_circuit_structure_rejects_wrong_noise_and_logical_observable():
    expected = circuit_signature(SMALL_CIRCUIT)
    assert circuit_signature(SMALL_CIRCUIT.replace("0.1", "0.05")) != expected
    assert circuit_signature(SMALL_CIRCUIT.replace("INCLUDE(0)", "INCLUDE(1)")) != expected
    with pytest.raises(ValueError):
        circuit_signature("REPEAT 999999999 {\n" + SMALL_CIRCUIT + "}\n")


class LocalSession:
    async def file_exists(self, path):
        return Path(path).is_file()

    async def directory_exists(self, path):
        return Path(path).is_dir()

    async def write_file(self, path, content):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

    async def read_bytes(self, path):
        return Path(path).read_bytes()

    async def run_command(self, command):
        subprocess.run(command, shell=True, check=True)


@pytest.mark.asyncio
async def test_setup_removes_solution_and_installs_complete_contract(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "run_simulation.py").write_text("answer-producing helper")
    (input_dir / "fd_wrapper.py").write_text("low-level decoder")
    cfg = SimpleNamespace(metadata={"input_dir": str(input_dir)}, description="new task")
    await task.start(cfg, LocalSession())
    assert not (input_dir / "run_simulation.py").exists()
    assert (input_dir / "fd_wrapper.py").read_text() == "low-level decoder"
    assert (input_dir / "task_description.txt").read_text() == "new task"
    assert "run_simulation.py" not in "\n".join(
        path.read_text() for path in task.ASSETS_DIR.iterdir() if path.is_file()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "signature_case", ["valid", "empty", "partial", "malformed", "nonfinite", "bad_hash"]
)
async def test_evaluate_checks_circuits_not_just_csv(tmp_path, monkeypatch, signature_case):
    signatures = tmp_path / "private"
    signatures.mkdir()
    entries = [circuit_signature(SMALL_CIRCUIT) for _ in range(60)]
    if signature_case == "empty":
        entries = []
    elif signature_case == "partial":
        entries = entries[:59]
    elif signature_case == "malformed":
        entries[0] = {}
    elif signature_case == "nonfinite":
        entries[0]["qubits"] = float("nan")
    elif signature_case == "bad_hash":
        entries[0]["detector_error_distribution_sha256"] = "z" * 64
    (signatures / "circuit_signatures.json").write_text(json.dumps(entries))
    monkeypatch.setattr(task, "SCRIPTS_DIR", signatures)
    output = tmp_path / "output"
    (output / "circuits").mkdir(parents=True)
    (output / "source").mkdir()
    (output / "source/simulate.py").write_text("print('simulation implementation')")
    for row_index in range(60):
        (output / f"circuits/row_{row_index:03}.stim").write_text(SMALL_CIRCUIT)
    (output / "simulation_manifest.json").write_text(
        json.dumps({"sampling_seeds": list(range(60))})
    )
    csv_path = output / "logical_error_rates_3codes.csv"
    csv_path.write_bytes(encode_rows(table([250] * 60)))
    reference_path = tmp_path / "reference.csv"
    reference_path.write_bytes(csv_path.read_bytes())
    (tmp_path / "stage1_manifest.json").write_text(
        json.dumps(
            {
                "decoder_representation": "stim-flattened-error-columns-v1",
                "reference_csv_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            }
        )
    )
    cfg = SimpleNamespace(
        metadata={
            "output_csv": str(csv_path),
            "reference_csv": str(reference_path),
            "remote_output_dir": str(output),
            "variant_name": "base",
        }
    )
    if signature_case != "valid":
        with pytest.raises(RuntimeError, match="Invalid QEC reference circuit signature"):
            await task.evaluate(cfg, LocalSession())
        return
    assert await task.evaluate(cfg, LocalSession()) == [1.0]
    (output / "circuits/row_000.stim").write_text(SMALL_CIRCUIT.replace("0.1", "0.01"))
    assert await task.evaluate(cfg, LocalSession()) == [0.0]
    reference_path.unlink()
    with pytest.raises(RuntimeError, match="reference"):
        await task.evaluate(cfg, LocalSession())


@pytest.mark.asyncio
@pytest.mark.parametrize("manifest_case", ["missing", "invalid_json", "wrong_type", "legacy", "stale_csv"])
async def test_evaluate_rejects_stale_decoder_reference_as_setup_error(tmp_path, manifest_case):
    reference = encode_rows(table([250]))
    reference_path = tmp_path / "reference.csv"
    reference_path.write_bytes(reference)
    output_path = tmp_path / "output.csv"
    output_path.write_bytes(reference)
    manifest = {
        "decoder_representation": "stim-flattened-error-columns-v1",
        "reference_csv_sha256": hashlib.sha256(reference).hexdigest(),
    }
    if manifest_case == "legacy":
        manifest["decoder_representation"] = "quits-detector-set-merge"
    elif manifest_case == "stale_csv":
        manifest["reference_csv_sha256"] = "0" * 64
    manifest_path = tmp_path / "stage1_manifest.json"
    if manifest_case == "invalid_json":
        manifest_path.write_text("invalid")
    elif manifest_case == "wrong_type":
        manifest_path.write_text("[]")
    elif manifest_case != "missing":
        manifest_path.write_text(json.dumps(manifest))
    cfg = SimpleNamespace(metadata={"output_csv": str(output_path), "reference_csv": str(reference_path)})
    with pytest.raises(RuntimeError, match="QEC.*reference"):
        await task.evaluate(cfg, LocalSession())
