import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


SCRIPT = Path(__file__).parents[2] / "tasks/visual_media/project_migration/scripts/project_check.py"
SPEC = importlib.util.spec_from_file_location("monitor_project_check", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def snapshots(tmp_path, before, after, *, original_flags="MasterOut,OrderSet",
              candidate_flags="MasterOut,OrderSet", direction="Output"):
    paths = []
    for name, connections, flags in [("original", before, original_flags),
                                      ("candidate", after, candidate_flags)]:
        tree = ET.Element("Session")
        route = ET.SubElement(ET.SubElement(tree, "Routes"), "Route", name="Stereo Out")
        ET.SubElement(route, "PresentationInfo", flags=flags)
        output = ET.SubElement(route, "IO", direction=direction)
        for index, targets in enumerate(connections):
            port = ET.SubElement(output, "Port", type="audio", name=f"Stereo Out/audio_out {index + 1}")
            for target in targets:
                ET.SubElement(port, "Connection", other=target)
        path = tmp_path / (name + ".ardour")
        ET.ElementTree(tree).write(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("before,after", [
    ([[], []], [["system:playback_1"], ["system:playback_2"]]),
    ([["system:playback_1"], ["system:playback_2"]], [[], []]),
    ([["system:playback_1"], ["system:playback_2"]], [["system:playback_3"], ["system:playback_4"]]),
])
def test_master_monitor_device_change_preserves_internal_routing(tmp_path, before, after):
    assert CHECKER.compare_project(*snapshots(tmp_path, before, after))["passed"]


@pytest.mark.parametrize("changes", [
    {"original_flags": "AudioBus", "candidate_flags": "AudioBus"},
    {"original_flags": "AudioBus"},
    {"candidate_flags": "AudioBus"},
    {"direction": "Input"},
])
def test_non_master_and_input_connections_are_preserved(tmp_path, changes):
    result = CHECKER.compare_project(*snapshots(
        tmp_path, [[], []], [["system:playback_1"], ["system:playback_2"]], **changes
    ))
    assert not result["passed"]


@pytest.mark.parametrize("target", [
    "STRINGS/audio_in 1", "system:capture_1", "other_app:input_1",
    "system:playback_0", "system:playback_1_extra",
])
def test_other_connections_are_not_monitor_devices(tmp_path, target):
    result = CHECKER.compare_project(*snapshots(tmp_path, [[], []], [[target], []]))
    assert not result["passed"]


def test_existing_internal_master_destination_cannot_be_removed(tmp_path):
    result = CHECKER.compare_project(*snapshots(
        tmp_path, [["Recorder/audio_in 1"], []], [["system:playback_1"], []]
    ))
    assert not result["passed"]
