import hashlib
import importlib
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import wave

import pytest

from tasks.visual_media.project_migration.scripts.native_batch import (
    UnsupportedBatch,
    prepare_batch,
    render_batch,
    verify_solo_graph,
)


@pytest.fixture
def project(tmp_path):
    root = ET.Element("Session", {"sample-rate": "48000", "id-counter": "1000"})
    ET.SubElement(root, "VCAManager")
    ET.SubElement(root, "RouteGroups")
    locations = ET.SubElement(root, "Locations")
    ET.SubElement(locations, "Location", flags="IsSessionRange", start="0", end="4800000")
    sources = ET.SubElement(root, "Sources")
    ET.SubElement(sources, "Source", id="300", name="original.mid")
    routes = ET.SubElement(root, "Routes")
    for index, (name, target) in enumerate(
        (("Direct", "Master"), ("Grouped", "Group"), ("Group", "Master"), ("Master", None)), 1
    ):
        route = ET.SubElement(routes, "Route", id=str(index), name=name)
        ET.SubElement(
            route, "PresentationInfo", flags="MasterOut" if target is None else "AudioBus"
        )
        for direction in ("Input", "Output"):
            ports = ET.SubElement(route, "IO", name=name, direction=direction)
            for channel in (1, 2):
                port = ET.SubElement(
                    ports,
                    "Port",
                    type="audio",
                    name=f"{name}/audio_{'in' if direction == 'Input' else 'out'} {channel}",
                )
                if direction == "Output" and target:
                    ET.SubElement(port, "Connection", other=f"{target}/audio_in {channel}")
        processor = ET.SubElement(route, "Processor", id=str(100 + index), type="lv2", active="1")
        automation = ET.SubElement(processor, "Automation")
        curve = ET.SubElement(automation, "AutomationList", id=str(200 + index), state="Play")
        ET.SubElement(curve, "events").text = "0 0.5\n48000 0.75\n"
    snapshot = tmp_path / "original.ardour"
    ET.ElementTree(root).write(snapshot)
    jobs = [
        ("first", "Direct", {"passages": [{"start": 5, "duration": 12}]}),
        ("second", "Grouped", {"passages": [{"start": 8, "duration": 12}]}),
    ]
    return snapshot, root, jobs


def test_instrument_taps_preserve_all_routes_effects_and_original(project, tmp_path):
    snapshot, original, jobs = project
    before = snapshot.read_bytes()
    group_plugin = original.find("Routes/Route[@name='Group']/Processor")
    ET.SubElement(group_plugin, "lv2", {"state-dir": "retained-reverb"})
    ET.ElementTree(original).write(snapshot)
    before = snapshot.read_bytes()
    prepared = prepare_batch(snapshot, jobs, tmp_path / "batch")
    result = ET.parse(prepared).getroot()
    assert snapshot.read_bytes() == before
    assert ET.tostring(result.find("Routes")) == ET.tostring(original.find("Routes"))
    assert ET.tostring(result.find("Sources")) == ET.tostring(original.find("Sources"))
    assert result.find("Locations/Location").get("end") == "960000"
    for label, source, _ in jobs:
        config = ET.parse(tmp_path / "batch" / (label + ".xml"))
        assert [port.get("name") for port in config.findall("Channel/Port")] == [
            f"{source}/audio_out 1",
            f"{source}/audio_out 2",
        ]


@pytest.mark.parametrize("problem", ["sidechain", "input", "duplicate", "vca", "send"])
def test_coupled_batch_falls_back_before_mutation(project, tmp_path, problem):
    snapshot, root, jobs = project
    group = root.find("Routes/Route[@name='Group']")
    if problem == "sidechain":
        processor = ET.SubElement(group, "Processor", type="sidechain")
        ET.SubElement(processor, "Connection", other="Direct/audio_out 1")
    elif problem == "input":
        ET.SubElement(
            root.find("Routes/Route[@name='Direct']/IO[@direction='Input']/Port"),
            "Connection",
            other="Grouped/audio_out 1",
        )
    elif problem == "duplicate":
        jobs[1] = jobs[0]
    elif problem == "send":
        ET.SubElement(group, "Processor", type="intsend")
    else:
        ET.SubElement(root.find("VCAManager"), "VCA")
    ET.ElementTree(root).write(snapshot)
    before = snapshot.read_bytes()
    outputs, receipt = render_batch(snapshot, jobs, tmp_path / "batch", {}, Path("helper.so"))
    assert outputs == {} and receipt["route"] == "serial_fallback"
    assert snapshot.read_bytes() == before
    assert not (tmp_path / "batch").exists()


def test_native_failure_retains_diagnostic(project, tmp_path, monkeypatch):
    snapshot, _, jobs = project

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(139, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    outputs, receipt = render_batch(snapshot, jobs, tmp_path / "batch", {}, Path("helper.so"))
    assert outputs == {} and receipt["route"] == "serial_fallback"
    assert (tmp_path / "batch/failure.json").is_file()


def test_single_and_full_export_keep_same_instrument_tap(project, tmp_path):
    snapshot, root, jobs = project
    ET.SubElement(root.find("VCAManager"), "VCA")
    ET.ElementTree(root).write(snapshot)
    prepared = prepare_batch(snapshot, jobs[:1], tmp_path / "single", full=True)
    assert ET.parse(prepared).find("Locations/Location").get("end") == "4800000"
    config = ET.parse(tmp_path / "single/first.xml")
    assert config.find("Channel/Port").get("name") == "Direct/audio_out 1"


def test_master_is_not_a_scored_instrument_tap(project, tmp_path):
    snapshot, _, jobs = project
    with pytest.raises(UnsupportedBatch, match="master"):
        prepare_batch(snapshot, [("master", "Master", jobs[0][2])], tmp_path / "invalid")


def test_solo_verification_checks_source_ports(project, tmp_path):
    snapshot, _, jobs = project
    evidence = tmp_path / "batch"
    prepared = prepare_batch(snapshot, jobs, evidence)
    tree = ET.parse(prepared)
    for _, source, _ in jobs:
        ET.SubElement(
            tree.find(f"Routes/Route[@name='{source}']"),
            "Controllable",
            {"name": "solo", "self-solo": "1"},
        )
    terminal = tree.find("Routes/Route[@name='Grouped']/Controllable")
    terminal.set("self-solo", "0")
    tree.write(prepared)
    with pytest.raises(RuntimeError, match="Grouped"):
        verify_solo_graph(prepared, evidence)
    terminal.set("self-solo", "1")
    tree.write(prepared)
    verify_solo_graph(prepared, evidence)


@pytest.fixture
def native_module(monkeypatch):
    scripts = Path(__file__).parents[2] / "tasks/visual_media/project_migration/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("native_audio")


def test_missing_helper_cannot_silently_export_master(
    project, tmp_path, monkeypatch, native_module
):
    monkeypatch.delenv("ARDOUR_BATCH_HELPER", raising=False)
    with pytest.raises(RuntimeError, match="requires the provisioned"):
        native_module.render_native_stems(project[0], {"stems": []}, tmp_path / "evidence")


@pytest.mark.parametrize("fallback", [False, True])
def test_compact_retention_and_serial_tap(project, tmp_path, monkeypatch, native_module, fallback):
    snapshot, _, _ = project
    helper = tmp_path / "helper.so"
    helper.touch()
    monkeypatch.setenv("ARDOUR_BATCH_HELPER", str(helper))
    monkeypatch.setenv("ARDOUR_BATCH_SIZE", "2")
    records = [
        {
            "valid": True,
            "reference": name + ".wav",
            "reference_audio": {"audible": True},
            "passages": [{"start": 0.25, "duration": 0.25}],
        }
        for name in ("Direct", "Grouped", "Group")
    ]
    previous = []
    expected = {}
    calls = []

    def batch(snapshot, jobs, evidence, environment, helper, **kwargs):
        calls.append(len(jobs))
        assert not any(path.exists() for path in previous)
        if fallback and len(jobs) > 1:
            return {}, {"route": "serial_fallback", "reason": "coupled"}
        evidence.mkdir()
        outputs = {}
        for label, _, _ in jobs:
            path = evidence / (label + ".wav")
            with wave.open(str(path), "wb") as stream:
                stream.setparams((2, 2, 48000, 48000, "NONE", "not compressed"))
                stream.writeframes(b"\x34\x12\x34\x12" * 48000)
            outputs[label] = path
            expected[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            previous.append(path)
        return outputs, {"route": "native_batch", "snapshot": str(snapshot)}

    monkeypatch.setattr(native_module, "render_batch", batch)
    receipts = native_module.render_native_stems(
        snapshot, {"stems": records}, tmp_path / "evidence"
    )
    assert calls == ([2, 1, 1, 1] if fallback else [2, 1])
    assert len(receipts) == 3 and not any(path.exists() for path in previous)
    for receipt in receipts:
        assert receipt["audio"]["sha256"] == expected[receipt["render"]]
        assert receipt["audio"]["frames"] == 48000
        assert receipt["tap_point"] == "instrument_output_inserts_strip_before_group_master"
    assert all(Path(record["passages"][0]["native_path"]).is_file() for record in records)
