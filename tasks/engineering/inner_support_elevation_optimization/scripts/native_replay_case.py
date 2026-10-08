"""Run one frozen model independently and read evaluator-owned native fields."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from native_input_adapter import read_family, read_json
from native_result_adapter import extract_monitors, read_native_result, read_stage1_baseline


def verify_review(family_path, review):
    family, _ = read_family(family_path)
    if review.get("input_hashes") != family["input_hashes"]:
        raise ValueError("Frozen native inputs differ from the evaluator's reviewed bytes")
    if review.get("monitors") != family["monitors"]:
        raise ValueError("Physical monitoring registration differs from reviewed input")
    if not review.get("source_registration_reviewed"):
        raise ValueError("Source registration must be reviewed before response extraction")
    if review.get("wall_range_z_m") != [-26.5, 7.0]:
        raise ValueError("This adapter requires full source outer-wall top/toe coverage")
    return family


def extract(output, review, *, expected_case=None):
    output = Path(output)
    receipt = read_json(output / "receipt.json")
    if expected_case is not None and receipt.get("case") != expected_case:
        raise ValueError("Native result belongs to a different support position")
    if not receipt.get("requested_scope_completed") or not receipt.get(
        "stage_9_native_solve_completed"
    ):
        raise ValueError("Full native Stage9 solve has not completed")
    if receipt["input_hashes"] != review["input_hashes"]:
        raise ValueError("Native execution input differs from frozen review")
    names = [
        "native_model.mdpa",
        "stage_1_k0_equilibrium_hold.npz",
        "crown_geometry.json",
        "native_nodal_fields.tsv",
    ]
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names}
    native = read_native_result(
        output / names[0],
        expected_sha256=hashes[names[0]],
        coordinates_path=output / names[3],
        coordinates_sha256=hashes[names[3]],
        precise_translations=True,
    )
    baseline = read_stage1_baseline(
        output / names[1], expected_sha256=hashes[names[1]], soil_nodes=native.soil_nodes
    )
    crowns = [row for row in read_json(output / names[2]) if row["system"] == "outer"]
    observations = extract_monitors(
        native,
        baseline,
        review["monitors"],
        outer_wall_property_ids={receipt["wall_properties"]["stage_2_outer_wall"]},
        wall_range=review["wall_range_z_m"],
        crown_geometry=crowns,
    )
    (output / "independent_monitors.json").write_text(json.dumps(observations, indent=2))
    return {
        "case": receipt["case"],
        "native_hashes": hashes,
        "station_count": len(observations["stations"]),
        "profile_rows": len(observations["profiles"]),
        "max_settlement_mm": observations["max_settlement_mm"],
        "max_disp_mm": observations["max_disp_mm"],
        "native_replay_and_reader_completed": True,
        "full_task_acceptance": False,
    }


def read_saved_case(family, output, review_path, position):
    output = Path(output)
    source_terminal = output / "terminal.json"
    terminal = read_json(source_terminal)
    review = read_json(review_path)
    verify_review(family, review)
    if terminal.get("native_exit_code") != 0:
        raise ValueError("Saved-field reading requires a successful native process")
    if terminal.get("input_hashes") != review["input_hashes"]:
        raise ValueError("Saved execution differs from the frozen source review")
    record = {
        "case": f"pos_{position}m",
        "source_terminal_sha256": hashlib.sha256(source_terminal.read_bytes()).hexdigest(),
        "review_sha256": hashlib.sha256(Path(review_path).read_bytes()).hexdigest(),
        "native_rerun": False,
        "full_task_acceptance": False,
    }
    try:
        record["extraction"] = extract(output, review, expected_case=record["case"])
        record["reader_exit_code"] = 0
    except Exception as error:
        record["reader_error"] = f"{type(error).__name__}: {error}"
        record["reader_exit_code"] = 1
        raise
    finally:
        (output / "reader-terminal.json").write_text(json.dumps(record, indent=2))
    return record


def prepare_output(output, review, restart, resume_in_place):
    output = Path(output)
    if not resume_in_place:
        output.mkdir(parents=True, exist_ok=False)
        return None
    expected = output / "restart/latest"
    if restart is None or Path(restart).resolve() != expected.resolve():
        raise ValueError("In-place continuation requires this result's committed latest checkpoint")
    metadata = read_json(expected.with_suffix(".json"))
    if metadata["input_hashes"] != review["input_hashes"]:
        raise ValueError("In-place checkpoint differs from frozen native inputs")
    with expected.with_suffix(".rest").open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
        if digest.hexdigest() != metadata["native_state_sha256"]:
            raise ValueError("In-place checkpoint hash mismatch")
    names = ("receipt.json", "native_stdout.log", "terminal.json")
    if any((output / ("before-continuation-" + name)).exists() for name in names):
        raise ValueError("This result already has an in-place continuation archive")
    for name in names:
        source = output / name
        if source.exists():
            source.rename(output / ("before-continuation-" + name))
    return {
        "checkpoint_label": metadata["label"],
        "native_state_sha256": metadata["native_state_sha256"],
        "previous_committed_elapsed_s": metadata["receipt"]["elapsed_s"],
    }


def main(
    family,
    output,
    review_path,
    position,
    runtime,
    max_iterations=80,
    reference=None,
    restart=None,
    resume_in_place=False,
):
    if not 1 <= max_iterations <= 1000:
        raise ValueError("Native iteration budget must be within 1..1000")
    review = read_json(review_path)
    verify_review(family, review)
    output = Path(output)
    continuation = prepare_output(output, review, restart, resume_in_place)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("native_family_runner.py")),
        str(family),
        str(output),
        "--until",
        "full",
        "--position",
        str(position),
        "--linear-backend",
        "amgcl-gmres",
        "--max-iterations",
        str(max_iterations),
        "--write-checkpoints",
    ]
    if reference:
        command.extend(["--equivalence-reference", str(reference)])
    if restart:
        metadata = read_json(Path(restart).with_suffix(".json"))
        if metadata["input_hashes"] != review["input_hashes"]:
            raise ValueError("Restart inputs differ from the frozen source review")
        command.extend(["--restart", str(restart)])
    started = time.monotonic()
    terminal = {
        "case": f"pos_{position}m",
        "command": command,
        "review_sha256": hashlib.sha256(Path(review_path).read_bytes()).hexdigest(),
        "input_hashes": review["input_hashes"],
        "full_task_acceptance": False,
        "in_place_continuation": continuation,
    }
    with (output / "native_stdout.log").open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            returncode = process.wait(timeout=runtime)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            returncode = 124
    terminal.update(native_exit_code=returncode, elapsed_s=time.monotonic() - started)
    (output / "terminal.json").write_text(json.dumps(terminal, indent=2))
    if returncode:
        return returncode
    try:
        from native_source_audit import main as audit_source

        terminal["source_fact_audit"] = audit_source(
            family, output / "restart/latest", review_path, output / "source-audit"
        )
    except Exception as error:
        terminal["source_audit_error"] = f"{type(error).__name__}: {error}"
        (output / "terminal.json").write_text(json.dumps(terminal, indent=2))
        raise
    try:
        terminal["extraction"] = extract(output, review, expected_case=f"pos_{position}m")
    except Exception as error:
        terminal["reader_error"] = f"{type(error).__name__}: {error}"
        (output / "terminal.json").write_text(json.dumps(terminal, indent=2))
        raise
    (output / "terminal.json").write_text(json.dumps(terminal, indent=2))
    print(json.dumps(terminal), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    parser.add_argument("output")
    parser.add_argument("review")
    parser.add_argument("--position", type=int, choices=range(5), required=True)
    parser.add_argument("--runtime", type=int, default=3600)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--equivalence-reference")
    parser.add_argument("--restart")
    parser.add_argument("--resume-in-place", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if not 60 <= args.runtime <= 7200:
        raise ValueError("Native child runtime must be within 60..7200 seconds")
    if args.extract_only:
        print(
            json.dumps(read_saved_case(args.family, args.output, args.review, args.position)),
            flush=True,
        )
    else:
        raise SystemExit(
            main(
                args.family,
                args.output,
                args.review,
                args.position,
                args.runtime,
                args.max_iterations,
                args.equivalence_reference,
                args.restart,
                args.resume_in_place,
            )
        )
