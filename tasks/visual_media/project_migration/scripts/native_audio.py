"""Re-render submitted native instruments; never trust submitted audio as project playback."""

import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from audio_delivery import extract_passage, inspect_audio, stem_name
from native_batch import TAP_POINT, render_batch


def render_native_stems(snapshot: Path, delivery: dict, evidence: Path, *, full=False):
    evidence.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(snapshot)
    routes = {}
    for route in tree.findall("Routes/Route"):
        routes.setdefault(route.get("name").strip().casefold(), []).append(route.get("name"))
    jobs = []
    for index, record in enumerate(delivery["stems"]):
        if not record["valid"] or not record["reference_audio"]["audible"]:
            continue
        matches = routes.get(stem_name(Path(record["reference"])), [])
        if len(matches) != 1:
            raise RuntimeError(
                f"Native output route needs verified source mapping: {record['reference']}"
            )
        jobs.append((f"EvaluatorSolo{index:03d}", matches[0], record))
    job_path = evidence / "solos.tsv"
    job_path.write_text("".join(f"{name}\t{route}\n" for name, route, _ in jobs))
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
    helper = environment.get("ARDOUR_BATCH_HELPER")
    if not helper or not Path(helper).is_file():
        raise RuntimeError("Instrument-output export requires the provisioned ARDOUR_BATCH_HELPER")
    batch_size = int(environment.get("ARDOUR_BATCH_SIZE", "2"))
    if batch_size < 1:
        raise RuntimeError("Evaluator ARDOUR_BATCH_SIZE must be positive")
    receipts = []

    def retain_job(job, rendered, prepared):
        name, route, record = job
        info = inspect_audio(rendered)
        if not info["finite"] or not info["frames"]:
            raise RuntimeError("Native export produced invalid audio")
        digest = hashlib.sha256()
        with rendered.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        info["sha256"] = digest.hexdigest()
        for index, passage in enumerate(record["passages"]):
            target = evidence / f"{name}-{index}-native.wav"
            extract_passage(rendered, target, passage["start"], passage["duration"])
            passage["native_path"] = str(target)
        record["native_audio"] = info
        receipts.append(
            {
                "route": route,
                "snapshot": prepared,
                "render": str(rendered),
                "audio": info,
                "tap_point": TAP_POINT,
            }
        )
        (evidence / "renders.json").write_text(json.dumps(receipts, indent=2) + "\n")
        if not full:
            rendered.unlink()

    batch_receipts = []
    for index in range(0, len(jobs), batch_size):
        group = jobs[index : index + batch_size]
        outputs, receipt = render_batch(
            snapshot,
            group,
            evidence / f"EvaluatorBatch{index:03d}",
            environment,
            helper,
            full=full,
        )
        batch_receipts.append(receipt)
        (evidence / "batch.json").write_text(json.dumps(batch_receipts, indent=2) + "\n")
        if outputs:
            for job in group:
                retain_job(job, outputs[job[0]], receipt["snapshot"])
            continue
        if len(group) == 1:
            raise RuntimeError("Instrument-output native export failed: " + receipt["reason"])
        for job in group:
            outputs, receipt = render_batch(
                snapshot,
                [job],
                evidence / job[0],
                environment,
                helper,
                full=full,
            )
            batch_receipts.append(receipt)
            (evidence / "batch.json").write_text(json.dumps(batch_receipts, indent=2) + "\n")
            if not outputs:
                raise RuntimeError("Instrument-output native export failed: " + receipt["reason"])
            retain_job(job, outputs[job[0]], receipt["snapshot"])
    delivery["native_rendered_stems"] = len(receipts)
    return receipts
