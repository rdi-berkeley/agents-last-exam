from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.health_medicine.nsclc_radiomics_cox_signature_v1 import main


class _Session:
    def __init__(self, output_dir: Path, reference_dir: Path) -> None:
        self.output_dir = output_dir
        self.reference_dir = reference_dir
        self.commands: list[tuple[str, bool]] = []
        self.written: dict[str, str] = {}
        self.verifier_result: subprocess.CompletedProcess[str] | None = None

    async def write_file(self, path: str, content: str) -> None:
        self.written[path] = content

    async def run_command(self, command: str, *, check: bool = True) -> dict[str, object]:
        self.commands.append((command, check))
        if "verify_outputs.py" not in command:
            return {"return_code": 0, "stdout": "OK\n", "stderr": ""}
        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                str(main.SCRIPTS_DIR / "verify_outputs.py"),
                "--output-dir",
                str(self.output_dir),
                "--reference-dir",
                str(self.reference_dir),
            ],
            check=check,
            capture_output=True,
            text=True,
        )
        self.verifier_result = result
        return {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


@pytest.mark.parametrize(
    ("risks", "expected_score", "expected_code"),
    [
        ((3, 2, 1), 1.0, 0),
        ((3, 1, 2), 0.666667, 0),
        ((1, 2, 3), 0.0, 1),
    ],
    ids=["full-credit", "partial-credit", "below-threshold"],
)
async def test_evaluate_parses_normal_verifier_outcomes(
    tmp_path, risks, expected_score, expected_code
):
    output_dir = tmp_path / "output"
    reference_dir = tmp_path / "reference"
    output_dir.mkdir()
    reference_dir.mkdir()
    (reference_dir / "ground_truth.csv").write_text(
        "PatientID,survival_time_days,deadstatus.event\np1,10,1\np2,20,1\np3,30,1\n"
    )
    (output_dir / "risk_scores.csv").write_text(
        "PatientID,risk_score\n" + "".join(f"p{i},{risk}\n" for i, risk in enumerate(risks, 1))
    )
    cfg = SimpleNamespace(
        metadata={
            "risk_scores_file": str(output_dir / "risk_scores.csv"),
            "remote_output_dir": str(output_dir),
            "reference_dir": str(reference_dir),
        }
    )
    session = _Session(output_dir, reference_dir)

    assert await main.evaluate(cfg, session) == [expected_score]

    assert session.verifier_result is not None
    assert session.verifier_result.returncode == expected_code
    payload = json.loads(session.verifier_result.stdout)
    assert payload["passed"] is (expected_code == 0)
    assert payload["score"] == expected_score
    assert [check for _, check in session.commands] == [True, True, False]
    assert session.written[f"{main.EVAL_TMP_DIR}/verify_outputs.py"] == (
        main.SCRIPTS_DIR / "verify_outputs.py"
    ).read_text(encoding="utf-8")
