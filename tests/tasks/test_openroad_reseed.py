import asyncio
import hashlib
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tasks.engineering.openroad_sky130_ibex_pnr_signoff.scripts import verify_submission as verifier
from tasks.engineering.openroad_sky130_ibex_pnr_signoff import main as task


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM openroad/orfs@sha256:" + "a" * 64 + "\n")
    (tmp_path / "Makefile").write_text("run:\n\t$(MAKE) -s -f flow.mk\n")
    (tmp_path / "flow.mk").write_text("export OR_SEED = 42\nall:\n\t@echo $(OR_SEED)\n")
    return tmp_path


def test_effective_seed_overrides_configuration_in_recursive_make(workspace, monkeypatch):
    native_run = subprocess.run
    observations = []

    def run_make(command, **kwargs):
        assert command[-3:-1] == ["bash", "-lc"]
        result = native_run(command[-3:], cwd=workspace, capture_output=True, text=True)
        observations.append(result.stdout.strip())
        return result

    before = hashlib.sha256((workspace / "flow.mk").read_bytes()).hexdigest()
    monkeypatch.setattr(verifier.subprocess, "run", run_make)
    assert verifier._run_docker_flow_once(workspace) == 0
    assert verifier._run_docker_flow_once(workspace, routing_seed=20260417) == 0
    assert observations == ["make -s -f flow.mk\n42", "make -s -f flow.mk\n20260417"]
    assert hashlib.sha256((workspace / "flow.mk").read_bytes()).hexdigest() == before


@pytest.mark.parametrize("seed", [-1, 2**31, True, "1; exit 0"])
def test_invalid_seed_never_launches_docker(workspace, monkeypatch, seed):
    launch = Mock()
    monkeypatch.setattr(verifier.subprocess, "run", launch)
    with pytest.raises(ValueError, match="routing_seed"):
        verifier._run_docker_flow_once(workspace, routing_seed=seed)
    launch.assert_not_called()


def test_native_failure_retry_preserves_selected_seed(workspace, monkeypatch):
    run = Mock(side_effect=[1, 0])
    monkeypatch.setattr(verifier, "_run_docker_flow_once", run)
    monkeypatch.setattr(verifier, "_cts_illegal_instruction_detected", lambda directory: True)
    assert verifier.run_docker_flow(workspace, routing_seed=123) == 0
    assert run.call_args_list[0].kwargs == {"env": None, "routing_seed": 123}
    assert run.call_args_list[1].kwargs == {
        "env": {"SKIP_CTS_REPAIR_TIMING": "1"}, "routing_seed": 123
    }


@pytest.mark.parametrize("original,expected", [("42", 20260417), ("20260417", 20260418)])
@pytest.mark.parametrize("passes", [True, False])
def test_reseed_uses_different_seed_and_retains_physical_gates(tmp_path, monkeypatch, original, expected, passes):
    starter = tmp_path / "starter"
    (starter / "flow/designs/sky130hd/ibex").mkdir(parents=True)
    configuration = tmp_path / "config.mk"
    configuration.write_text(f"export OR_SEED = {original}\n")
    journal = tmp_path / "JOURNAL.md"
    journal.write_text("completed flow\n")
    run = Mock(return_value=0)
    monkeypatch.setattr(verifier, "run_docker_flow", run)
    report = verifier.Report()
    for gate_id in ("G2", "G3", "G4", "G5", "G6", "G7", "G8"):
        weight = verifier.RUBRIC[gate_id]["weight"]
        report.add(verifier.GateResult(gate_id, gate_id, weight, weight if passes else 0, passes))
    monkeypatch.setattr(verifier, "score_submission", lambda *arguments: report)
    result = verifier.gate_g9_reseed(starter, configuration, journal, tmp_path / "scratch", {})
    assert run.call_args.kwargs == {"routing_seed": expected}
    assert result.score == (10 if passes else 0)
    assert result.passed == passes
    assert configuration.read_text() == f"export OR_SEED = {original}\n"


@pytest.mark.parametrize("recognized", [True, False])
def test_setup_corrects_only_unsupported_seed_names(tmp_path, monkeypatch, recognized):
    source = (
        "Physical targets and required artifacts stay unchanged.\n"
        "All physical targets above must remain satisfied with a different OpenROAD "
        "global-placement random seed (`GPL_RANDOM_SEED`, `YOSYS_SEED`, `OPENROAD_SEED`).\n"
    ) if recognized else "Unknown version of task instructions.\n"
    specification = tmp_path / "SPEC.md"
    specification.write_text(source)
    starter_config = tmp_path / "flow/designs/sky130hd/ibex/config.mk"
    starter_config.parent.mkdir(parents=True)
    starter_text = (
        "# This file is the one artifact you are expected to edit. Seven variables below\n"
        "# are set to values that will either prevent signoff or waste area / power.\n"
        "#\n\nexport DESIGN_NICKNAME = ibex\n"
        "# NOTE: Starter values below cause downstream failures. Tune them.\n"
        "export CORE_UTILIZATION = 70 # STARTER (too high — causes congestion)\n"
        "# STARTER (forces fixed techmap; reference leaves empty — tune it out).\n"
        "export ADDER_MAP_FILE = $(PLATFORM_DIR)/cells_adders_hd.v\n"
        "# Source and platform paths remain frozen.\n"
    )
    starter_config.write_text(starter_text)
    setup = AsyncMock()
    monkeypatch.setattr(task, "_setup", setup)

    async def command(session, command, *, check):
        assert check
        process = subprocess.run(shlex.split(command), capture_output=True, text=True, check=True)
        return {"return_code": process.returncode, "stdout": process.stdout}

    monkeypatch.setattr(task, "_run_command", command)
    configuration = SimpleNamespace(metadata={"starter_project_dir": str(tmp_path)})
    if not recognized:
        with pytest.raises(subprocess.CalledProcessError):
            asyncio.run(task.start(configuration, object()))
        assert specification.read_text() == source
        return
    asyncio.run(task.start(configuration, object()))
    repaired = specification.read_text()
    assert repaired.startswith("Physical targets and required artifacts stay unchanged.\n")
    assert "`OR_SEED`" in repaired
    assert "GPL_RANDOM_SEED" not in repaired
    assert "20260417" not in repaired
    assert verifier.parse_config_mk_exports(starter_config.read_text()) == verifier.parse_config_mk_exports(starter_text)
    assert "STARTER" not in starter_config.read_text()
    assert "Seven variables" not in starter_config.read_text()
    assert "Source and platform paths remain frozen" in starter_config.read_text()
    assert "reference leaves empty" not in starter_config.read_text()
    cleaned_config = starter_config.read_text()
    asyncio.run(task.start(configuration, object()))
    assert specification.read_text() == repaired
    assert starter_config.read_text() == cleaned_config
