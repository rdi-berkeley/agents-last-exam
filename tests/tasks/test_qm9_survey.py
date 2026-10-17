import copy
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("rdkit")

TASK = Path(__file__).resolve().parents[2] / "tasks/physical_sciences/qm9_mmff94_forcefield_survey_1"
SPEC = importlib.util.spec_from_file_location("qm9_verify", TASK / "scripts/verify.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def test_official_population_exclusions():
    excluded = json.loads((TASK / "assets/uncharacterized_ids.json").read_text())
    assert len(excluded) == len(set(excluded)) == 3054
    assert min(excluded) >= 1 and max(excluded) <= 133885
    assert 133885 - len(excluded) == 130831


def test_absolute_discrepancy_includes_shorter_distances():
    rows = [{"Molecule_ID": str(index), "QM9_Dist_A": 4.0,
             "RDKit_Dist_A": 2.0, "Discrepancy_A": 2.0} for index in range(10)]
    checks, _ = verify._score_phase1(rows, rows)
    assert all(checks)
    negative_signed = copy.deepcopy(rows)
    negative_signed[0]["Discrepancy_A"] = -2.0
    checks, _ = verify._score_phase1(negative_signed, rows)
    assert not checks[1]


def test_missing_spot_checked_rows_cannot_pass():
    rows = [{"Molecule_ID": str(index), "QM9_Dist_A": 4.0,
             "RDKit_Dist_A": 2.0, "Discrepancy_A": 2.0} for index in range(10)]
    checks, _ = verify._score_phase1(rows[:1], rows)
    assert checks[2:] == [False, False]


def test_missing_reference_is_evaluation_error(tmp_path):
    with pytest.raises(RuntimeError, match="Invalid evaluator reference"):
        verify.score(tmp_path, tmp_path / "missing")


def test_evaluator_dependency_failure_is_not_solver_zero(monkeypatch, capsys):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(
        command, 1, "", "dependency installation failed"
    ))
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(TASK / "scripts/verify.py"), run_name="qm9_dependency_probe")
    assert error.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert "error" in payload and "score" not in payload


def test_inconsistent_reference_energy_is_evaluation_error(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "hard_gate", lambda directory: None)
    content = {rank: {"data": {"quantum_forced_energy": 17.545,
                               "mmff94_global_min_energy": 17.353,
                               "delta_e": 0.158}} for rank in verify.RANK_KEYS}
    (tmp_path / "phase4_pes_results.json").write_text(json.dumps(content))
    with pytest.raises(RuntimeError, match="inconsistent delta_e"):
        verify.score(tmp_path, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_reference", "process", "payload"])
async def test_main_propagates_evaluation_errors(failure):
    from types import SimpleNamespace
    from tasks.physical_sciences.qm9_mmff94_forcefield_survey_1 import main

    session = SimpleNamespace(
        file_exists=AsyncMock(return_value=failure != "missing_reference"),
        directory_exists=AsyncMock(return_value=False),
        interface=SimpleNamespace(create_dir=AsyncMock()),
        write_file=AsyncMock(),
        run_command=AsyncMock(return_value={"return_code": 2 if failure == "process" else 0,
                                           "stdout": "not JSON", "stderr": "failure"}),
    )
    config = SimpleNamespace(metadata={"variant_name": "base", "remote_output_dir": "output",
                                       "reference_dir": "reference", "eval_tmp_dir": "eval"})
    with pytest.raises(RuntimeError):
        await main.evaluate(config, session)
