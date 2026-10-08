"""Package verified original-input conversions, without replacement instruments or references."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET


SOURCES = {
    "celeste_symphonic_suite": "migration-ardour-r05/native/Celeste/R05Clean2.ardour",
    "eora": "music-r07/final-native/eora/R07Clean2.ardour",
    "hollow_knight_symphonic_suite": "music-r07/silent-native/hollow_knight_symphonic_suite/R07Clean2.ardour",
    "twilight_princess_credits": "music-r07/final-native/twilight_princess_credits/R07Clean2.ardour",
    "undertale_medley": "music-r07/undertale-native-b/undertale_medley/R07Clean2.ardour",
}


def build(private_root: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    registry = {}
    for variant, relative in SOURCES.items():
        snapshot = private_root / relative
        project = snapshot.parent
        session = ET.parse(snapshot).getroot()
        manifest = json.loads((project / "conversion-manifest.json").read_text())
        processors = session.findall("Routes/Route/Processor")
        if any(entry.get("type") in ("lv2", "luaproc", "ladspa") for entry in processors):
            raise ValueError(f"Unexpected replacement in source-only bundle: {variant}")
        files = [(snapshot, "project.ardour")]
        for name in ("conversion-manifest.json", "native-initializers.tsv"):
            files.append((project / name, name))
        for directory in ("interchange", "source-state"):
            files.extend(
                (path, str(path.relative_to(project)))
                for path in sorted((project / directory).rglob("*"))
                if path.is_file() and (directory == "source-state" or path.suffix == ".mid")
            )
        for plugin in manifest["plugins"]:
            actual = hashlib.sha256((project / plugin["state_file"]).read_bytes()).hexdigest()
            if actual != plugin["sha256"]:
                raise ValueError(f"Original plugin hash mismatch: {variant}")
        archive_path = destination / f"{variant}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for source, name in files:
                archive.add(source, arcname=f"project/{name}", recursive=False)
        registry[variant] = {
            "archive": archive_path.name,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "source_archive_sha256": manifest["source_archive_sha256"],
            "notes": manifest["expected_note_count"],
            "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files},
            "source_only": True,
        }
    (destination / "sources.json").write_text(json.dumps(registry, indent=2) + "\n")
    return registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent / "assets")
    args = parser.parse_args()
    print(json.dumps(build(args.private_root, args.destination), indent=2))
