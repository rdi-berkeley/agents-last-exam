"""Linux setup and evaluator-owned native execution for the supported model interface."""

import asyncio
import base64
import hashlib
import json
import logging
import shlex
import time
import uuid
from pathlib import Path

from .delivery_scoring import (
    assess_presentation,
    compare_monitors,
    normalize_monitor_table,
    score_delivery,
)
from .remote_commands import run_command
from .verification import CASES, REQUIRED_IMAGES, EvaluationUnavailableError, verify_source_bundle


ROOT = Path(__file__).parent
NATIVE_SCRIPTS = (
    "native_input_adapter.py",
    "native_family_runner.py",
    "native_member_sections.py",
    "native_model_contract.py",
    "native_result_adapter.py",
    "native_source_audit.py",
    "native_replay_case.py",
    "native_replay_family.py",
    "native_parallel.py",
    "native_worker.py",
    "native_verify_delivery.py",
)
logger = logging.getLogger(__name__)


async def upload_files(session, root, files):
    for name, content in files.items():
        path = str(Path(root) / name)
        for offset in range(0, max(1, len(content)), 32768):
            payload = base64.b64encode(content[offset : offset + 32768]).decode()
            mode = "wb" if offset == 0 else "ab"
            script = (
                "import base64; from pathlib import Path; "
                f"path=Path({path!r}); path.parent.mkdir(parents=True,exist_ok=True); "
                f"path.open({mode!r}).write(base64.b64decode({payload!r}))"
            )
            result = await run_command(
                session, shlex.join(["python3", "-c", script]), timeout=30, check=False
            )
            if result.get("return_code", 1):
                raise EvaluationUnavailableError(
                    f"Native staging failed for {name}: {result.get('stderr')}"
                )


async def setup_native(metadata, session):
    if session is None or not {"input_dir", "software_dir", "remote_output_dir"}.issubset(metadata):
        raise EvaluationUnavailableError("Native setup needs task paths; not a solver score")
    manifest = json.loads((ROOT / "assets/source_manifest.json").read_text())["sha256"]
    files = {
        name: await session.read_bytes(str(Path(metadata["input_dir"]) / name)) for name in manifest
    }
    verify_source_bundle(files)
    probe = (
        "import importlib.metadata; import KratosMultiphysics; "
        "import KratosMultiphysics.GeoMechanicsApplication; "
        "import KratosMultiphysics.StructuralMechanicsApplication; "
        "import KratosMultiphysics.LinearSolversApplication; "
        "assert importlib.metadata.version('KratosMultiphysics') == '10.4.3'"
    )
    result = await run_command(
        session,
        shlex.join(["env", "OMP_NUM_THREADS=1", "/opt/ale-kratos/bin/python", "-c", probe]),
        timeout=30,
        check=False,
    )
    if result.get("return_code", 1):
        raise EvaluationUnavailableError(
            "Installed Kratos10.4.3 runtime unavailable; see native_interface.md installation command"
        )
    await upload_files(
        session,
        metadata["input_dir"],
        {
            name: (ROOT / "assets" / name).read_bytes()
            for name in (
                "benchmark.json",
                "model_definition.md",
                "delivery_schema.md",
                "native_interface.md",
            )
        },
    )
    await upload_files(
        session,
        metadata["software_dir"],
        {
            **{
                f"scripts/{name}": (ROOT / "scripts" / name).read_bytes() for name in NATIVE_SCRIPTS
            },
            **{
                f"assets/{name}": (ROOT / "assets" / name).read_bytes()
                for name in ("benchmark.json", "source_manifest.json")
            },
        },
    )
    await run_command(
        session, shlex.join(["mkdir", "-p", metadata["remote_output_dir"]]), timeout=20
    )


async def replay_submission(metadata, session, review, *, deadline=None):
    evidence = Path(review["evidence_directory"])
    if metadata.get("_evaluator_retained_replay"):
        return await read_retained_replay(metadata["_evaluator_retained_replay"], session, review)
    snapshot = evidence.parent
    identifier = uuid.uuid4().hex[:16]
    remote = str(Path(metadata["remote_output_dir"]).parent / ".support-evaluation" / identifier)
    files = {f"scripts/{name}": (ROOT / "scripts" / name).read_bytes() for name in NATIVE_SCRIPTS}
    files.update(
        {
            f"assets/{name}": (ROOT / "assets" / name).read_bytes()
            for name in ("benchmark.json", "source_manifest.json")
        }
    )
    files.update(
        {
            f"family/{name}": (snapshot / "family" / name).read_bytes()
            for name in ("family.json", "model.mdpa", "Materials.json")
        }
    )
    files["review.json"] = (evidence / "native-review.json").read_bytes()
    await upload_files(session, remote, files)
    cpus = tuple(metadata.get("native_cpus", (1, 2)))
    if (
        not 1 <= len(cpus) <= 2
        or len(set(cpus)) != len(cpus)
        or any(cpu not in (0, 1, 2, 3) for cpu in cpus)
    ):
        raise EvaluationUnavailableError("Invalid native evaluator CPU allocation")
    budget = int(metadata.get("native_replay_seconds", 6600))
    if deadline is not None:
        budget = min(budget, int(deadline - time.monotonic()) - 60)
    if not 120 <= budget <= 6600:
        raise EvaluationUnavailableError(
            "Native replay budget must leave room in the normal evaluation window"
        )
    unit = "support-evaluate-" + identifier
    command = [
        "sudo",
        "-n",
        "systemd-run",
        "--quiet",
        "--unit=" + unit,
        "--uid=user",
        "-p",
        "AllowedCPUs=" + " ".join(map(str, cpus)),
        "-p",
        f"CPUQuota={100 * len(cpus)}%",
        "-p",
        f"MemoryMax={2048 * len(cpus) + 1024}M",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "LimitCORE=0",
        "-p",
        f"RuntimeMaxSec={budget + 30}",
        "-p",
        "Delegate=yes",
        "-p",
        "RemainAfterExit=yes",
        "-p",
        f"StandardOutput=append:{remote}/launcher.log",
        "-p",
        f"StandardError=append:{remote}/launcher.log",
        "taskset",
        "--cpu-list",
        ",".join(map(str, cpus)),
        "env",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "/opt/ale-kratos/bin/python",
        "-u",
        remote + "/scripts/native_replay_family.py",
        remote + "/family",
        remote + "/replay",
        remote + "/review.json",
        "--total-seconds",
        str(budget),
        "--cpus",
        *map(str, cpus),
    ]
    launched = await run_command(session, shlex.join(command), timeout=20, check=False)
    receipt = {
        "unit": unit,
        "remote": remote,
        "command": command,
        "launch": launched,
        "resources": {
            "cpus": cpus,
            "worker_memory_mib": 2048,
            "aggregate_memory_mib": 2048 * len(cpus) + 1024,
            "runtime_budget_s": budget,
        },
        "script_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        },
    }
    (evidence / "native-launch.json").write_text(json.dumps(receipt, indent=2))
    if launched.get("return_code", 1):
        raise EvaluationUnavailableError(
            "Independent native replay could not launch; see native-launch.json"
        )
    started = time.monotonic()
    while True:
        status = await run_command(
            session,
            shlex.join(
                [
                    "systemctl",
                    "show",
                    unit,
                    "-p",
                    "MainPID,SubState,ExecMainStatus,Result,MemoryPeak",
                ]
            ),
            timeout=20,
            check=False,
        )
        fields = dict(
            line.split("=", 1) for line in status.get("stdout", "").splitlines() if "=" in line
        )
        receipt["status"] = fields
        receipt["observed_elapsed_s"] = time.monotonic() - started
        (evidence / "native-live.json").write_text(json.dumps(receipt, indent=2))
        if fields.get("MainPID") == "0":
            break
        if time.monotonic() - started > budget + 60:
            raise EvaluationUnavailableError(
                f"Native evaluator supervisor observation expired; retained unit {unit}"
            )
        logger.info("Support native replay %s: %s", unit, fields)
        await asyncio.sleep(45)
    try:
        replay = json.loads(await session.read_file(remote + "/replay/family-replay.json"))
    except Exception as error:
        raise EvaluationUnavailableError(
            f"Native replay has no terminal receipt: {unit}"
        ) from error
    (evidence / "family-replay.json").write_text(json.dumps(replay, indent=2))
    if fields.get("ExecMainStatus") != "0" or not replay.get("completed"):
        raise EvaluationUnavailableError(
            f"Native replay incomplete: {replay.get('error')}; unit {unit}; checkpoints retained"
        )
    observations = {}
    for case in CASES:
        path = replay["cases"][case]["directory"] + "/independent_monitors.json"
        observations[case] = json.loads(await session.read_file(path))
    (evidence / "native-observations.json").write_text(json.dumps(observations, indent=2))
    return {"remote": remote, "replay": replay, "observations": observations}


async def read_retained_replay(descriptor_path, session, review):
    """Reuse evaluator-owned results selected by private operator metadata."""
    evidence = Path(review["evidence_directory"])
    descriptor = json.loads(Path(descriptor_path).read_text())
    current_review = json.loads((evidence / "native-review.json").read_text())
    if (
        descriptor.get("format") != "support-evaluator-retained-replay-1"
        or not review.get("replay_eligible")
        or descriptor.get("input_hashes") != current_review["input_hashes"]
    ):
        raise EvaluationUnavailableError(
            "Retained evaluator result concerns a different reviewed model"
        )
    remote = descriptor["remote"]
    replay_bytes = await session.read_bytes(remote + "/replay/family-replay.json")
    if hashlib.sha256(replay_bytes).hexdigest() != descriptor["replay_sha256"]:
        raise EvaluationUnavailableError("Retained evaluator replay receipt changed")
    replay = json.loads(replay_bytes)
    if (
        not replay.get("completed")
        or replay.get("input_hashes") != current_review["input_hashes"]
        or set(replay.get("cases", {})) != set(CASES)
    ):
        raise EvaluationUnavailableError(
            "Retained evaluator replay lacks five complete matching cases"
        )
    retained_review = json.loads(await session.read_file(remote + "/review.json"))
    if any(
        retained_review.get(key) != current_review[key]
        for key in ("input_hashes", "monitors", "wall_range_z_m")
    ):
        raise EvaluationUnavailableError(
            "Retained native observation definition differs from source review"
        )
    expected_hashes = {}
    for case in CASES:
        item = replay["cases"][case]
        terminal_path = item["directory"] + "/terminal.json"
        terminal_bytes = await session.read_bytes(terminal_path)
        if hashlib.sha256(terminal_bytes).hexdigest() != item["terminal_sha256"]:
            raise EvaluationUnavailableError("Retained native terminal changed: " + case)
        terminal = json.loads(terminal_bytes)
        sections = (
            terminal.get("source_fact_audit", {}).get("audit", {}).get("support_sections", {})
        )
        if sections.get("version") != 1 or not sections.get("passed"):
            raise EvaluationUnavailableError(
                "Retained native history lacks role-specific section audit: " + case
            )
        if (
            terminal.get("case") != case
            or terminal.get("native_exit_code") != 0
            or terminal.get("input_hashes") != current_review["input_hashes"]
            or not terminal.get("source_fact_audit", {}).get("passed")
            or not terminal.get("extraction", {}).get("native_replay_and_reader_completed")
        ):
            raise EvaluationUnavailableError(
                "Retained native case lacks audited completion: " + case
            )
        expected_hashes.update(
            {
                item["directory"] + "/" + name: digest
                for name, digest in terminal["extraction"]["native_hashes"].items()
            }
        )
        expected_hashes[item["directory"] + "/independent_monitors.json"] = descriptor[
            "observation_sha256"
        ][case]
    probe = """import hashlib,json,sys
from pathlib import Path
for name,digest in json.loads(sys.argv[1]).items():
 path=Path(name)
 if not path.is_absolute() or path.is_symlink() or not path.is_file():raise ValueError('Missing retained native field: '+name)
 digest_state=hashlib.sha256()
 with path.open('rb') as stream:
  while block:=stream.read(1048576):digest_state.update(block)
 actual=digest_state.hexdigest()
 if actual!=digest:raise ValueError('Retained native field changed: '+name)
print(json.dumps({'all_retained_field_hashes_verified':True}))
"""
    checked = await run_command(
        session,
        shlex.join(["python3", "-c", probe, json.dumps(expected_hashes)]),
        timeout=60,
        check=False,
    )
    if checked.get("return_code", 1):
        raise EvaluationUnavailableError(
            "Retained native fields changed: " + checked.get("stderr", "")
        )
    observations = {}
    for case in CASES:
        content = await session.read_bytes(
            replay["cases"][case]["directory"] + "/independent_monitors.json"
        )
        if hashlib.sha256(content).hexdigest() != descriptor["observation_sha256"][case]:
            raise EvaluationUnavailableError("Retained observations changed during reading")
        observations[case] = json.loads(content)
    record = {
        "descriptor_sha256": hashlib.sha256(Path(descriptor_path).read_bytes()).hexdigest(),
        "replay_sha256": descriptor["replay_sha256"],
        "input_hashes": current_review["input_hashes"],
        "retained_field_hashes": expected_hashes,
        "native_histories_replayed": False,
        "prior_cumulative_native_compute_s": replay.get("cumulative_native_compute_s"),
        "prior_aggregate_resources": replay.get("resources"),
        "fresh_evaluation_runtime_validated": False,
    }
    (evidence / "retained-replay-verification.json").write_text(json.dumps(record, indent=2))
    (evidence / "family-replay.json").write_bytes(replay_bytes)
    (evidence / "native-observations.json").write_text(json.dumps(observations, indent=2))
    return {"remote": remote, "replay": replay, "observations": observations}


async def evaluate_delivery(metadata, session, review, replay):
    output = Path(metadata["remote_output_dir"])
    evidence = Path(review["evidence_directory"])
    names = [
        "answer.json",
        "case_summary.csv",
        "support_position_engineering_memo.md",
        *REQUIRED_IMAGES,
    ]
    files = {}
    for name in names:
        path = str(output / name)
        if not await session.file_exists(path):
            return {"score": 0.0, "reason": "Missing original deliverable: " + name}
        try:
            files[name] = await session.read_bytes(path)
        except Exception as error:
            raise EvaluationUnavailableError(
                f"Could not read existing deliverable: {name}"
            ) from error
    remote = replay["remote"]
    native = await run_command(
        session,
        shlex.join(
            [
                "sudo",
                "-n",
                "systemd-run",
                "--quiet",
                "--wait",
                "--pipe",
                "--unit=support-delivery-" + uuid.uuid4().hex[:16],
                "--uid=user",
                "-p",
                "AllowedCPUs=" + str(int(metadata.get("native_cpu", 2))),
                "-p",
                "CPUQuota=100%",
                "-p",
                "MemoryMax=3G",
                "-p",
                "MemorySwapMax=0",
                "-p",
                "RuntimeMaxSec=170",
                "-p",
                "LimitCORE=0",
                "taskset",
                "--cpu-list",
                str(int(metadata.get("native_cpu", 2))),
                "env",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "/opt/ale-kratos/bin/python",
                remote + "/scripts/native_verify_delivery.py",
                str(output / "model_family_support_position"),
                remote + "/replay/family-replay.json",
                remote + "/review.json",
                remote + "/native-delivery.json",
            ]
        ),
        timeout=180,
        check=False,
    )
    if native.get("return_code", 1):
        raise EvaluationUnavailableError(
            "Native delivered-result reader could not finish: " + native.get("stderr", "")
        )
    delivered = json.loads(await session.read_file(remote + "/native-delivery.json"))
    (evidence / "native-delivery.json").write_text(json.dumps(delivered, indent=2))
    checks = {
        "native_results_agree": True,
        "raw_monitors_agree": True,
        "required_files_present": True,
        "findings": [],
    }
    for case in CASES:
        item = delivered["cases"][case]
        try:
            if not item.get("native_result_read"):
                raise ValueError(item.get("error", "Missing native result"))
            compare_monitors(item["observations"], replay["observations"][case])
        except (ValueError, KeyError) as error:
            checks["native_results_agree"] = False
            checks["findings"].append({"case": case, "native_error": str(error)})
        table = None
        for directory in (
            output / "model_family_support_position/results" / case,
            output / "model_family_support_position" / case,
        ):
            for name in ("monitoring.csv", "monitoring.tsv", "independent_monitors.json"):
                path = str(directory / name)
                if not await session.file_exists(path):
                    continue
                content = await session.read_file(path)
                try:
                    table = normalize_monitor_table(content, Path(name).suffix)
                    break
                except (ValueError, KeyError):
                    continue
            if table is not None:
                break
        try:
            if table is None:
                checks["required_files_present"] = False
                raise ValueError("Missing supported raw monitoring table")
            compare_monitors(table, replay["observations"][case])
        except (ValueError, KeyError) as error:
            checks["raw_monitors_agree"] = False
            checks["findings"].append({"case": case, "raw_error": str(error)})
    magnitudes = {
        case: {name: row[name] for name in ("max_settlement_mm", "max_disp_mm")}
        for case, row in replay["observations"].items()
    }
    caveats = review.get("engineering_caveats", [])
    presentation = assess_presentation(files, magnitudes)
    checks.update({name: presentation[name] for name in ("images_fraction", "memo_fraction")})
    record = score_delivery(
        files["answer.json"].decode(), files["case_summary.csv"].decode(), magnitudes, checks
    )
    record.update(
        delivery_checks=checks,
        presentation_checks=presentation,
        native_replay_directory=remote,
        engineering_caveats=caveats,
    )
    (evidence / "score.json").write_text(json.dumps(record, indent=2))
    return record
