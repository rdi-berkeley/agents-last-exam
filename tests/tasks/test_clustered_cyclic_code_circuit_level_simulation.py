from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest

from tasks.computing_math.clustered_cyclic_code_circuit_level_simulation.scripts.score_logical_error_rates import score_logical_error_rates_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = (
    REPO_ROOT
    / "task-data-hf"
    / "extracted"
    / "computing_math"
    / "clustered_cyclic_code_circuit_level_simulation"
    / "base"
    / "reference"
    / "logical_error_rates_3codes.csv"
)

pytestmark = pytest.mark.skipif(
    not REFERENCE_PATH.is_file(), reason="Requires the gated published reference dataset"
)


def test_published_reference_scores_one_against_itself() -> None:
    reference = REFERENCE_PATH.read_bytes()
    result = score_logical_error_rates_bytes(
        agent_bytes=reference,
        reference_bytes=reference,
    )

    assert result.score == 1.0
    assert result.passed

    rows = list(csv.DictReader(StringIO(reference.decode("utf-8"))))
    assert {row["code"] for row in rows} == {
        "[24,8,3]",
        "[40,8,5]",
        "[56,8,7]",
    }


def test_consistent_but_different_counts_are_rejected() -> None:
    reference = REFERENCE_PATH.read_bytes()
    rows = list(csv.DictReader(StringIO(reference.decode("utf-8"))))
    for index, row in enumerate(rows):
        shots = 5_000
        failures = int(row["num_failures"])
        if index == 0:
            failures += 1
        probability = failures / shots
        rounds = int(row["num_rounds"])
        logical_qubits = int(row["k"])
        row["num_shots"] = str(shots)
        row["num_failures"] = str(failures)
        row["p_logical"] = str(probability)
        row["lfr_per_round"] = str(1 - (1 - probability) ** (1 / rounds))
        row["lfr_per_round_per_qubit"] = str(
            1 - (1 - probability) ** (1 / (logical_qubits * rounds))
        )
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    result = score_logical_error_rates_bytes(
        agent_bytes=stream.getvalue().encode(),
        reference_bytes=reference,
    )

    assert not result.passed
    assert any("num_failures_mismatch" in reason for reason in result.reasons)


def test_wrong_shot_count_is_rejected() -> None:
    reference = REFERENCE_PATH.read_bytes()
    rows = list(csv.DictReader(StringIO(reference.decode("utf-8"))))
    rows[0]["num_shots"] = "4999"
    rows[0]["num_failures"] = "0"
    rows[0]["p_logical"] = "0"
    rows[0]["lfr_per_round"] = "0"
    rows[0]["lfr_per_round_per_qubit"] = "0"
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    result = score_logical_error_rates_bytes(
        agent_bytes=stream.getvalue().encode(),
        reference_bytes=reference,
    )

    assert not result.passed
    assert any("num_shots_mismatch" in reason for reason in result.reasons)


def test_consistent_but_fabricated_flat_curve_fails() -> None:
    reference = REFERENCE_PATH.read_bytes()
    rows = list(csv.DictReader(StringIO(reference.decode("utf-8"))))
    for row in rows:
        shots = 5_000
        failures = 500
        probability = failures / shots
        rounds = int(row["num_rounds"])
        logical_qubits = int(row["k"])
        row["num_shots"] = str(shots)
        row["num_failures"] = str(failures)
        row["p_logical"] = str(probability)
        row["lfr_per_round"] = str(1 - (1 - probability) ** (1 / rounds))
        row["lfr_per_round_per_qubit"] = str(
            1 - (1 - probability) ** (1 / (logical_qubits * rounds))
        )
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    result = score_logical_error_rates_bytes(
        agent_bytes=stream.getvalue().encode(),
        reference_bytes=reference,
    )

    assert not result.passed
    assert any("num_failures_mismatch" in reason for reason in result.reasons)


def test_consistent_but_non_monotonic_curve_fails() -> None:
    reference = REFERENCE_PATH.read_bytes()
    rows = list(csv.DictReader(StringIO(reference.decode("utf-8"))))
    for index, row in enumerate(
        item for item in rows if item["code"] == "[24,8,3]"
    ):
        shots = 5_000
        probability = 0.8 if index == 0 else 0.05
        failures = round(probability * shots)
        rounds = int(row["num_rounds"])
        logical_qubits = int(row["k"])
        row["num_shots"] = str(shots)
        row["num_failures"] = str(failures)
        row["p_logical"] = str(probability)
        row["lfr_per_round"] = str(1 - (1 - probability) ** (1 / rounds))
        row["lfr_per_round_per_qubit"] = str(
            1 - (1 - probability) ** (1 / (logical_qubits * rounds))
        )
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    result = score_logical_error_rates_bytes(
        agent_bytes=stream.getvalue().encode(),
        reference_bytes=reference,
    )

    assert not result.passed
    assert any("num_failures_mismatch" in reason for reason in result.reasons)
