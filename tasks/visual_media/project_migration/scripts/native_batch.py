"""Export instrument route outputs through the verified Ardour channel API."""

import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


TAP_POINT = "instrument_output_inserts_strip_before_group_master"


class UnsupportedBatch(ValueError):
    pass


def verify_solo_graph(snapshot, evidence):
    routes = {route.get("name"): route for route in ET.parse(snapshot).findall("Routes/Route")}
    for name in (evidence / "routes.tsv").read_text().splitlines():
        solo = routes[name].find("Controllable[@name='solo']")
        if solo is None or float(solo.get("self-solo", "0")) <= 0:
            raise RuntimeError("Native source solo is missing: " + name)


def prepare_batch(snapshot, jobs, evidence, *, full=False):
    if not jobs:
        raise UnsupportedBatch("No source routes selected")
    tree = ET.parse(snapshot)
    root = tree.getroot()
    routes = root.find("Routes")
    by_name = {route.get("name"): route for route in routes}
    if len(by_name) != len(routes):
        raise UnsupportedBatch("Ambiguous route names")
    selected = {route for _, route, _ in jobs}
    if len(selected) != len(jobs):
        raise UnsupportedBatch("Duplicate source routes")
    if len(jobs) > 1:
        if len(root.find("VCAManager")) or len(root.find("RouteGroups")):
            raise UnsupportedBatch("VCA/group coupling requires isolated export")
        for route in routes:
            for processor in route.findall(".//Processor"):
                if processor.get("type") in {"send", "intsend", "return", "intreturn", "port"}:
                    raise UnsupportedBatch("Send/return coupling requires isolated export")
                if processor.get("type") == "sidechain" and processor.findall(".//Connection"):
                    raise UnsupportedBatch("Connected sidechain requires isolated export")
    configurations = []
    for label, source, _ in jobs:
        if source not in by_name:
            raise UnsupportedBatch("Missing source route: " + source)
        route = by_name[source]
        if "MasterOut" in route.find("PresentationInfo").get("flags", ""):
            raise UnsupportedBatch("A scored stem cannot tap the master")
        if len(jobs) > 1 and route.findall("IO[@direction='Input']/Port[@type='audio']/Connection"):
            raise UnsupportedBatch("Audio input coupling requires isolated export")
        ports = route.findall("IO[@direction='Output']/Port[@type='audio']")
        if len(ports) not in (1, 2):
            raise UnsupportedBatch("Instrument output must be mono or stereo")
        config = ET.Element("ExportChannelConfiguration", split="0", channels=str(len(ports)))
        for channel, port in enumerate(ports, 1):
            node = ET.SubElement(config, "Channel", number=str(channel))
            ET.SubElement(node, "Port", name=port.get("name"))
        configurations.append((label, config))
    location = next(
        node
        for node in root.findall("Locations/Location")
        if "IsSessionRange" in node.get("flags", "")
    )
    location.set("start", "0")
    if not full:
        end = max(
            part["start"] + part["duration"] for _, _, record in jobs for part in record["passages"]
        )
        location.set("end", str(round(end * int(root.get("sample-rate")))))
    output_snapshot = snapshot.parent / (evidence.name + ".ardour")
    if output_snapshot.exists():
        raise UnsupportedBatch("Evaluator snapshot already exists")
    evidence.mkdir(parents=True, exist_ok=False)
    tree.write(output_snapshot, encoding="utf-8", xml_declaration=True)
    manifest = []
    for label, config in configurations:
        folder = evidence / label
        folder.mkdir()
        path = evidence / (label + ".xml")
        ET.ElementTree(config).write(path, encoding="utf-8", xml_declaration=True)
        manifest.append(f"{path}\t{folder}\n")
    (evidence / "outputs.tsv").write_text("".join(manifest))
    (evidence / "routes.tsv").write_text("".join(route + "\n" for _, route, _ in jobs))
    return output_snapshot


def render_batch(snapshot, jobs, evidence, environment, helper, *, full=False):
    try:
        prepared = prepare_batch(snapshot, jobs, evidence, full=full)
    except UnsupportedBatch as exc:
        return {}, {"route": "serial_fallback", "reason": str(exc), "tap_point": TAP_POINT}
    try:
        with (evidence / "prepare.log").open("wb") as log:
            subprocess.run(
                [
                    "xvfb-run",
                    "-a",
                    "/usr/lib/ardour6/luasession",
                    str(Path(__file__).with_name("prepare_batch.lua")),
                    str(prepared.parent),
                    prepared.stem,
                    str(evidence / "routes.tsv"),
                ],
                env=environment,
                stdout=log,
                stderr=log,
                check=True,
                timeout=180,
            )
        verify_solo_graph(prepared, evidence)
        native_environment = dict(environment)
        native_environment["LD_PRELOAD"] = ":".join(
            filter(None, (str(helper), environment.get("LD_PRELOAD")))
        )
        native_environment["ARDOUR_BATCH_MANIFEST"] = str(evidence / "outputs.tsv")
        with (evidence / "export.log").open("wb") as log:
            subprocess.run(
                [
                    "xvfb-run",
                    "-a",
                    "/usr/bin/ardour6-export",
                    "-b",
                    "float",
                    "-o",
                    str(evidence / "Batch.wav"),
                    str(prepared.parent),
                    prepared.stem,
                ],
                env=native_environment,
                stdout=log,
                stderr=log,
                check=True,
                timeout=1200,
            )
        outputs = {}
        for label, _, _ in jobs:
            paths = list((evidence / label).glob("*.wav"))
            if len(paths) != 1:
                raise RuntimeError("Missing or ambiguous native batch output")
            outputs[label] = paths[0]
        return outputs, {"route": "native_batch", "snapshot": str(prepared), "tap_point": TAP_POINT}
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        (evidence / "failure.json").write_text(json.dumps({"error": str(exc)}, indent=2) + "\n")
        return {}, {"route": "serial_fallback", "reason": str(exc), "tap_point": TAP_POINT}
