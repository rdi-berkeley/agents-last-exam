"""Produce evaluator-owned native and complete-delivery evidence on Linux."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from audio_delivery import check_delivery
from project_check import compare_project, midi_events, note_events, source_path
from rebind import rebind
from native_audio import render_native_stems


def copy_project(source: Path, destination: Path):
    source = source.resolve()
    destination = destination.resolve()
    libraries = {}

    def copy_file(original, target):
        original = Path(original)
        if original.suffix.casefold() in {".sf2", ".sf3"}:
            stat = original.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in libraries:
                os.link(libraries[identity], target)
                return str(target)
            shutil.copy2(original, target)
            libraries[identity] = target
            return str(target)
        return shutil.copy2(original, target)

    shutil.copytree(source, destination, symlinks=True, copy_function=copy_file)
    for original in source.rglob("*"):
        if not original.is_symlink():
            continue
        target = original.resolve(strict=True)
        copied = destination / original.relative_to(source)
        if target.is_relative_to(source):
            if original.readlink().is_absolute():
                copied.unlink()
                copied.symlink_to(
                    os.path.relpath(destination / target.relative_to(source), copied.parent),
                    target_is_directory=target.is_dir(),
                )
        else:
            copied.unlink()
            if target.is_dir():
                shutil.copytree(target, copied, copy_function=copy_file)
            else:
                copy_file(original, copied)


def native_reopen(candidate: Path, evidence: Path) -> dict:
    if not shutil.which("xvfb-run") or not Path("/usr/lib/ardour6/luasession").is_file():
        raise RuntimeError("Ardour 6 native evaluator runtime is unavailable")
    project = evidence / "native-project"
    copy_project(candidate.parent, project)
    snapshot = project / candidate.name
    rebind(snapshot)
    expected = Counter()
    tree = ET.parse(snapshot).getroot()
    sources = {source.get("id"): source for source in tree.findall("Sources/Source")}
    bypassed_equalizers = set()
    for route in tree.findall("Routes/Route"):
        for processor in route.findall("Processor"):
            bands = processor.findall("OriginalCubasePlugin/Equalizer/Band")
            if bands and all(band.find("Enabled").get("value") == "false" for band in bands):
                bypassed_equalizers.add((route.get("name"), processor.get("name")))
        playlist = tree.find(f'Playlists/Playlist[@id="{route.get("midi-playlist")}"]')
        if playlist is None:
            continue
        for region in playlist.findall("Region"):
            notes = note_events(midi_events(source_path(project, sources[region.get("source-0")])))
            for fields, count in notes.items():
                expected[(route.get("name"), region.get("name"), *fields)] += count
    environment = dict(os.environ)
    environment.update(
        {
            "ARDOUR_CONFIG_PATH": "/etc/ardour6:/usr/share/ardour6",
            "ARDOUR_DLL_PATH": "/usr/lib/ardour6",
            "ARDOUR_DATA_PATH": "/usr/share/ardour6",
            "LD_LIBRARY_PATH": "/usr/lib/ardour6",
            "XDG_CONFIG_HOME": str(evidence / "config"),
            "XDG_CACHE_HOME": str(evidence / "cache"),
        }
    )
    logs = []
    for index, name in enumerate((snapshot.stem, "EvaluatorReopen")):
        target = evidence / f"native-{index}.tsv"
        command = [
            "xvfb-run",
            "-a",
            "/usr/lib/ardour6/luasession",
            str(Path(__file__).with_name("native_readback.lua")),
            str(project),
            name,
            str(target),
            "EvaluatorReopen" if index == 0 else "read-only",
        ]
        try:
            completed = subprocess.run(command, env=environment, capture_output=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Native evaluator exceeded resource/time budget") from exc
        (evidence / f"native-{index}.log").write_bytes(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(
                f"Ardour native readback did not complete (exit {completed.returncode}); inspect retained log"
            )
        actual = Counter()
        unavailable = []
        for line in target.read_text().splitlines():
            kind, route, region, *fields = line.split("\t")
            if kind == "note":
                actual[(route, region, *map(int, fields))] += 1
            elif kind == "unavailable" and (route, region) not in bypassed_equalizers:
                unavailable.append([route, region])
        logs.append(
            {
                "notes": actual.total(),
                "missing": (expected - actual).total(),
                "extra": (actual - expected).total(),
                "unavailable": unavailable,
            }
        )
    return {
        "passed": all(
            not entry["missing"] and not entry["extra"] and not entry["unavailable"]
            for entry in logs
        ),
        "readbacks": logs,
        "snapshot": str(project / "EvaluatorReopen.ardour"),
    }


def evaluate_local(output, baseline, references, evidence, *, native=True):
    evidence.mkdir(parents=True, exist_ok=True)
    project_path = output / "migrated_project" / "migrated_project.ardour"
    inventory = output / "available_plugins.txt"
    required = [project_path, output / "overview.png", inventory]
    missing = [str(path.relative_to(output)) for path in required if not path.is_file()]
    inventory_valid = inventory.is_file() and bool(inventory.read_text(errors="replace").strip())
    if inventory.is_file() and not inventory_valid:
        missing.append("available_plugins.txt (empty)")
    delivery = check_delivery(output, references, evidence / "passages")
    result = {
        "missing_files": missing,
        "inventory_present": inventory.is_file(),
        "inventory_nonempty": inventory_valid,
        "delivery": delivery,
        "gate_passed": False,
    }
    if not project_path.is_file():
        return result
    result["project"] = compare_project(baseline, project_path)
    if result["project"].get("unsupported"):
        raise RuntimeError(
            "Active source automation needs an equivalent-adapter verifier; not a candidate zero"
        )
    if native and result["project"]["passed"]:
        result["native"] = native_reopen(project_path, evidence)
        result["native"]["state"] = compare_project(
            project_path, Path(result["native"]["snapshot"])
        )
    result["gate_passed"] = bool(
        not missing
        and result["project"]["passed"]
        and not result["project"]["unresolved_active_plugins"]
        and delivery["mix"]["valid"]
        and delivery["valid_stems"]
        and native
        and result.get("native", {}).get("passed")
        and result["native"]["state"]["passed"]
    )
    result["native_check_executed"] = native
    if result["gate_passed"]:
        render_native_stems(Path(result["native"]["snapshot"]), delivery, evidence / "native-audio")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output", "baseline", "references", "evidence"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        report = evaluate_local(args.output, args.baseline, args.references, args.evidence)
    except Exception as exc:
        report = {"infrastructure_error": f"{type(exc).__name__}: {exc}"}
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / "assessment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("delivery", "project", "native")
            }
        )
    )
