import json
import math
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.engineering.cailian_road_highway_alignment_2 import main


@pytest.fixture
def extracted_alignment():
    return {
        "alignment_info": {
            "start_x": main.START_X,
            "start_y": main.START_Y,
            "end_x": main.END_X,
            "end_y": main.END_Y,
            "length": 2300.0,
            "n_curves": 2,
            "curves": [{"radius": 500.0}, {"radius": 500.0}],
            "spirals": [],
        },
        "profile_info": {"count": 1},
        "tsv_exists": True,
        "tsv_headers": ["Station", "X", "Y", "Z"],
    }


def test_runtime_prompt_and_card_disclose_unchanged_ratio():
    card = json.loads(Path(main.__file__).with_name("task_card.json").read_text())
    for prompt in (main.config.task_description, card["taskPrompt"]):
        assert "straight-line distance between the specified control points" in prompt
        assert f"at least {main.MIN_PATH_OVER_CHORD}" in prompt
        assert f"{main.CHORD_LENGTH:.6f} m" in prompt
        assert f"{main.MIN_PATH_OVER_CHORD * main.CHORD_LENGTH:.6f} m" in prompt
    assert main.MIN_PATH_OVER_CHORD == 1.05
    assert (main.MIN_TOTAL_LENGTH, main.MAX_TOTAL_LENGTH) == (1800.0, 2400.0)


@pytest.mark.parametrize("difference", [-0.01, 0.0, 0.01])
def test_existing_ratio_gate_boundary_is_unchanged(extracted_alignment, difference):
    extracted_alignment["alignment_info"]["length"] = (
        main.MIN_PATH_OVER_CHORD * main.CHORD_LENGTH + difference
    )
    result = main._score_from_verifier(extracted_alignment)
    assert result["admin_gates_passed"] is (difference >= 0)
    assert any(
        "admin_gate_8_path_over_chord" in failure for failure in result["hard_gate_failures"]
    ) is (difference < 0)
    assert result["final_score"] == (0.0 if difference < 0 else 50.0)


@pytest.mark.parametrize("length", [1900.0, 2100.0, 2200.0])
def test_original_broad_window_does_not_override_ratio(extracted_alignment, length):
    extracted_alignment["alignment_info"]["length"] = length
    result = main._score_from_verifier(extracted_alignment)
    assert not result["admin_gates_passed"]
    assert result["final_score"] == 0.0


@pytest.mark.parametrize("length", [1799.9, 2400.1])
def test_total_length_gate_still_applies(extracted_alignment, length):
    extracted_alignment["alignment_info"]["length"] = length
    result = main._score_from_verifier(extracted_alignment)
    assert result["hard_gate_failures"][0].startswith("gate_5_length_out_of_range")
    assert result["final_score"] == 0.0


def test_geometric_constraints_have_a_two_arc_solution():
    half_angle = 0.65
    radius = main.CHORD_LENGTH / (4.0 * math.sin(half_angle))
    length = 4.0 * radius * half_angle
    first_arc_displacement = 2.0 * radius * math.sin(half_angle)
    heading = math.atan2(main.END_Y - main.START_Y, main.END_X - main.START_X)
    end_x = main.START_X + 2.0 * first_arc_displacement * math.cos(heading)
    end_y = main.START_Y + 2.0 * first_arc_displacement * math.sin(heading)
    assert (end_x, end_y) == pytest.approx((main.END_X, main.END_Y), abs=1e-8)
    assert radius >= main.MIN_CURVE_RADIUS
    assert main.MIN_TOTAL_LENGTH <= length <= main.MAX_TOTAL_LENGTH
    assert length / main.CHORD_LENGTH >= main.MIN_PATH_OVER_CHORD
    assert 1900.0 < main.CHORD_LENGTH - 2.0 * main.CONTROL_POINT_TOLERANCE


@pytest.fixture
def native_result(extracted_alignment):
    extracted_alignment.update({
        "native_road_verified": True,
        "surface_elevations": [2.91, 4.0, 6.80],
        "tsv_rows": [
            {"Station": "0", "Z": "2.91"},
            {"Station": "20", "Z": "4.0"},
            {"Station": "30", "Z": "6.80"},
        ],
    })
    return extracted_alignment


def test_native_score_requires_independent_extraction(native_result):
    assert main._score_native_road(native_result)["final_score"] == 100
    native_result.pop("native_road_verified")
    assert main._score_native_road(native_result)["final_score"] == 0


def test_native_wrong_elevation_fails_original_pass_threshold(native_result):
    native_result["tsv_rows"][1]["Z"] = "5.0"
    result = main._score_native_road(native_result)
    assert result["vertical_subscore"] == 0
    assert result["final_score"] == 60 < main.PASS_THRESHOLD


def test_native_wrong_endpoint_fails(native_result):
    native_result["alignment_info"]["start_x"] += 1
    result = main._score_native_road(native_result)
    assert result["final_score"] == 0
    assert result["hard_gate_failures"][0].startswith("gate_3_start_point_too_far")


@pytest.mark.parametrize("problem", ["error", "missing_surface", "nan"])
def test_native_invalid_evidence_never_uses_tsv_fallback(native_result, problem):
    if problem == "error":
        native_result["error"] = "native reopen failed"
    elif problem == "missing_surface":
        native_result["surface_elevations"] = []
    else:
        native_result["tsv_rows"][1]["Z"] = "nan"
    assert main._score_native_road(native_result)["final_score"] == 0


def test_native_does_not_add_endpoint_height_gate(native_result):
    assert native_result["surface_elevations"][0] != main.START_Z
    assert native_result["surface_elevations"][-1] != main.END_Z
    assert main._score_native_road(native_result)["final_score"] == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [{"error": "native reopen failed"}, [], {}])
async def test_native_entry_runs_limited_verifier_and_fails_closed(receipt):
    class Session:
        def __init__(self):
            self.commands = []

        async def run_command(self, command, check=False):
            command = shlex.split(command)[-1]
            self.commands.append(command)
            return {"return_code": 0, "stdout": json.dumps({"exit_code": 0, "stdout": json.dumps(receipt) if command.startswith("cat ") else "", "stderr": ""})}

    metadata = {
        "cad_backend": "freecad-road",
        "native_road_work_dir": "/private/work",
        "native_road_adapter_dir": "/trusted/adapter",
        "native_road_runtime": "/installed/runtime",
        "topo_surface_file": "/input/terrain.obj",
        "alignment_fcstd": "/output/alignment.FCStd",
        "alignment_tsv": "/output/alignment_metrics.tsv",
    }
    session = Session()
    if isinstance(receipt, dict) and receipt.get("error"):
        assert await main.evaluate(SimpleNamespace(metadata=metadata), session) == [0.0]
    else:
        with pytest.raises(main.NativeRoadEnvironmentError):
            await main.evaluate(SimpleNamespace(metadata=metadata), session)
    assert len(session.commands) == 3
    execution = session.commands[1]
    for option in ("CPUQuota=100%", "MemoryMax=3G", "MemorySwapMax=0", "xvfb-run -a", "--user-cfg"):
        assert option in execution
    assert "--tsv-only" not in execution


def test_formal_linux_configuration_is_self_contained():
    metadata = main.config.to_metadata()
    assert main.config.OS_TYPE == "linux"
    assert metadata["cad_backend"] == "freecad-road"
    assert metadata["requires_task_data"] is False
    assert metadata["alignment_fcstd"].endswith("/output/alignment.FCStd")
    assert metadata["native_road_adapter_dir"] == metadata["software_dir"]
    assert "license-oss" not in json.dumps(metadata)


def test_public_surface_has_no_private_alignment():
    import hashlib
    import zipfile
    import xml.etree.ElementTree as element_tree

    assets = Path(main.__file__).parent / "assets"
    assert hashlib.sha256((assets / "declared-survey-terrain.obj").read_bytes()).hexdigest() == "f5b136d233c8bb62bfbb6f1cfba44a48470a6cccf4077a7e883c9286b9f356df"
    with zipfile.ZipFile(assets / "declared-survey-terrain.FCStd") as archive:
        document = element_tree.fromstring(archive.read("Document.xml"))
    names = {obj.attrib.get("name") for obj in document.findall(".//Objects/Object")}
    assert "Terrain" in names
    assert "Alignment" not in names and "Profile" not in names
    assert not list(assets.glob("*route*"))
    assert "def build(" not in (Path(main.__file__).parent / "scripts/native_adapter.py").read_text()


@pytest.mark.asyncio
async def test_formal_setup_stages_runtime_and_source_assets():
    class Session:
        def __init__(self):
            self.commands = []

        async def run_command(self, command, check=False):
            if command.startswith("python3 -c "):
                command = shlex.split(command)[-1]
            self.commands.append(command)
            receipt = {"runtime": "/installed/runtime", "launcher": "/software/open_road.sh", "source_files_modified": False}
            return {"return_code": 0, "stdout": json.dumps({"exit_code": 0, "stdout": json.dumps(receipt), "stderr": ""})}

    session = Session()
    await main.start(SimpleNamespace(metadata=main.config.to_metadata()), session)
    assert any(command.startswith("tar -xzf ") for command in session.commands)
    assert "prepare_runtime.py" in session.commands[-1]
    assert all("witness" not in command for command in session.commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [17, 127])
async def test_installed_session_exit_loss_cannot_create_a_task_zero(exit_code):
    import asyncio

    from computer.interface.models import CommandResult
    from cua_bench.computers.remote import RemoteDesktopSession
    from tasks.engineering.cailian_road_highway_alignment_2.native_commands import (
        NativeRoadEnvironmentError,
        run_native_command,
    )

    class Interface:
        async def run_command(self, command):
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return CommandResult(stdout.decode(), stderr.decode(), process.returncode)

    session = RemoteDesktopSession(api_url="http://127.0.0.1:1", os_type="linux")
    session._computer = SimpleNamespace(interface=Interface())
    session._initialized = True
    with pytest.raises(NativeRoadEnvironmentError, match=str(exit_code)):
        await run_native_command(session, f"exit {exit_code}")
