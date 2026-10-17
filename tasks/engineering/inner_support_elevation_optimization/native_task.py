"""Linux setup and evaluator-owned native execution for the supported model interface."""

import asyncio
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
from .remote_commands import run_command, upload_files
from .scripts.native_resources import (
    COORDINATOR_MEMORY_MIB,
    REPLAY_SECONDS,
    REPLAY_WORKERS,
    WORKER_MEMORY_MIB,
    ReplayBudgetExceeded,
)
from .verification import CASES, REQUIRED_IMAGES, EvaluationUnavailableError, verify_source_bundle


ROOT = Path(__file__).parent
NATIVE_SCRIPTS = (
    "native_input_adapter.py",
    "native_family_runner.py",
    "native_member_sections.py",
    "native_model_contract.py",
    "native_result_adapter.py",
    "native_result_reader.py",
    "native_source_audit.py",
    "native_replay_case.py",
    "native_replay_family.py",
    "native_parallel.py",
    "native_resources.py",
    "native_worker.py",
    "native_verify_delivery.py",
)
logger = logging.getLogger(__name__)
PUBLIC_NATIVE_SCRIPTS = {
    "native_input_adapter.py": "native_input_adapter.py",
    "native_family_runner.py": "native_family_runner.py",
    "native_member_sections.py": "native_member_sections.py",
    "native_result_adapter.py": "native_result_reader.py",
}


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
                f"scripts/{name}": (ROOT / "scripts" / source).read_bytes()
                for name, source in PUBLIC_NATIVE_SCRIPTS.items()
            },
            **{
                f"assets/{name}": (ROOT / "assets" / name).read_bytes()
                for name in ("benchmark.json", "source_manifest.json")
            },
        },
    )
    private_scripts = [
        str(Path(metadata["software_dir"]) / "scripts" / name)
        for name in NATIVE_SCRIPTS
        if name not in PUBLIC_NATIVE_SCRIPTS
    ]
    remove_private = (
        "from pathlib import Path; "
        f"[Path(name).unlink(missing_ok=True) for name in {private_scripts!r}]"
    )
    await run_command(session, shlex.join(["python3", "-c", remove_private]), timeout=20)
    await run_command(
        session, shlex.join(["mkdir", "-p", metadata["remote_output_dir"]]), timeout=20
    )


async def replay_submission(metadata, session, review, *, deadline=None):
    evidence = Path(review["evidence_directory"])
    if metadata.get("_evaluator_retained_replay"):
        return await read_retained_replay(metadata["_evaluator_retained_replay"], session, review)
    identifier = uuid.uuid4().hex[:16]
    remote = str(evidence.parent)
    files = {f"scripts/{name}": (ROOT / "scripts" / name).read_bytes() for name in NATIVE_SCRIPTS}
    files.update(
        {
            f"assets/{name}": (ROOT / "assets" / name).read_bytes()
            for name in ("benchmark.json", "source_manifest.json")
        }
    )
    await upload_files(session, remote, files)
    allocation = await run_command(
        session,
        shlex.join(
            [
                "python3",
                remote + "/scripts/native_resources.py",
                json.dumps(metadata.get("native_cpus")),
            ]
        ),
        timeout=20,
    )
    resources = json.loads(allocation["stdout"])
    cpus = tuple(resources["cpus"])
    budget = int(metadata.get("native_replay_seconds", REPLAY_SECONDS))
    if len(cpus) != REPLAY_WORKERS:
        raise EvaluationUnavailableError("Native replay needs the five published worker slots")
    if not 120 <= budget <= REPLAY_SECONDS:
        raise EvaluationUnavailableError(
            "Native replay budget must leave room in the normal evaluation window"
        )
    if deadline is not None and deadline - time.monotonic() < budget + 60:
        raise EvaluationUnavailableError("Insufficient time to grant the full native replay budget")
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
        f"MemoryMax={WORKER_MEMORY_MIB * len(cpus) + COORDINATOR_MEMORY_MIB}M",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=continue",
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
            **resources,
            "cpus": cpus,
            "worker_memory_mib": WORKER_MEMORY_MIB,
            "coordinator_memory_mib": COORDINATOR_MEMORY_MIB,
            "aggregate_memory_mib": WORKER_MEMORY_MIB * len(cpus) + COORDINATOR_MEMORY_MIB,
            "runtime_budget_s": budget,
        },
        "script_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        },
    }
    await upload_files(session, evidence, {"native-launch.json": json.dumps(receipt, indent=2).encode()})
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
                    "MainPID,SubState,ExecMainCode,ExecMainStatus,Result,MemoryPeak,ControlGroup",
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
        if fields.get("ControlGroup"):
            probe = """import json,sys
from pathlib import Path
root=Path('/sys/fs/cgroup')/sys.argv[1].lstrip('/')
result={}
for group in [root, *root.glob('*')]:
 if not group.is_dir():continue
 values={}
 for name in ('memory.events','memory.peak','memory.current','memory.max','cpu.stat'):
  path=group/name
  if path.exists():values[name]=path.read_text().strip()
 if values:result[group.name]=values
print(json.dumps(result))
"""
            try:
                usage = await run_command(
                    session,
                    shlex.join(["python3", "-c", probe, fields["ControlGroup"]]),
                    timeout=15,
                    check=False,
                )
                if usage.get("return_code", 1) == 0:
                    receipt["observed_cgroups"] = json.loads(usage["stdout"])
            except (OSError, ValueError, TimeoutError) as error:
                receipt["resource_probe_error"] = str(error)
        await upload_files(session, evidence, {"native-live.json": json.dumps(receipt, indent=2).encode()})
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
    await upload_files(session, evidence, {"family-replay.json": json.dumps(replay, indent=2).encode()})
    exhausted = replay.get("exhausted_resource")
    confirmed_limit = (
        exhausted == "wall_time" and replay.get("elapsed_s", 0) >= REPLAY_SECONDS
    ) or (
        exhausted == "worker_memory"
        and any(
            row.get("oom_kills", 0) > 0
            and row.get("memory_limit_bytes") == WORKER_MEMORY_MIB * 1024**2
            for row in replay.get("attempts", [])
        )
    )
    if (
        not replay.get("completed")
        and replay.get("failure_kind") == "resource_exhausted"
        and budget == REPLAY_SECONDS
        and replay.get("runtime_budget_s") == REPLAY_SECONDS
        and confirmed_limit
        and fields.get("Result") in ("exit-code", "success")
        and fields.get("ExecMainCode") == "1"
    ):
        raise ReplayBudgetExceeded(exhausted, replay["error"])
    if fields.get("ExecMainStatus") != "0" or not replay.get("completed"):
        raise EvaluationUnavailableError(
            f"Native replay incomplete: {replay.get('error')}; unit {unit}; checkpoints retained"
        )
    observations = {}
    for case in CASES:
        path = replay["cases"][case]["directory"] + "/independent_monitors.json"
        observations[case] = json.loads(await session.read_file(path))
    await upload_files(session, evidence, {"native-observations.json": json.dumps(observations, indent=2).encode()})
    return {"remote": remote, "replay": replay, "observations": observations}


async def read_retained_replay(descriptor_path, session, review):
    """Reuse evaluator-owned results selected by private operator metadata."""
    evidence = Path(review["evidence_directory"])
    descriptor = json.loads(Path(descriptor_path).read_text())
    current_review = json.loads(await session.read_file(str(evidence / "native-review.json")))
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
    await upload_files(session, evidence, {
        "retained-replay-verification.json": json.dumps(record, indent=2).encode(),
        "family-replay.json": replay_bytes,
        "native-observations.json": json.dumps(observations, indent=2).encode(),
    })
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
    await upload_files(session, evidence, {"native-delivery.json": json.dumps(delivered, indent=2).encode()})
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
    await upload_files(session, evidence, {"score.json": json.dumps(record, indent=2).encode()})
    return record
