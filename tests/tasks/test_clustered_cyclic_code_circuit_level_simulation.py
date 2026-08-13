from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from tasks.computing_math.clustered_cyclic_code_circuit_level_simulation.scripts.score_logical_error_rates import (
    SUPPRESSION_LADDER,
    score_logical_error_rates_bytes,
)


REFERENCE_PATH = Path(
    "/mnt/data/agenthle/computing_math/clustered_cyclic_code_circuit_level_simulation/"
    "base/reference/logical_error_rates_3codes.csv"
)


def test_published_reference_scores_one_against_itself() -> None:
    reference = REFERENCE_PATH.read_bytes()
    result = score_logical_error_rates_bytes(
        agent_bytes=reference,
        reference_bytes=reference,
    )

    assert SUPPRESSION_LADDER == ["[24,8,3]", "[40,8,5]", "[56,8,7]"]
    assert result.score == 1.0
    assert result.passed

    rows = list(csv.DictReader(StringIO(reference.decode("utf-8"))))
    assert {row["code"] for row in rows} == set(SUPPRESSION_LADDER)
