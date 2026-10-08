"""Independently replay one submitted family with a shared native Stage7 prefix."""

import argparse
import json
import math
import os
import time
from pathlib import Path

from native_input_adapter import read_json
from native_replay_case import verify_review
from native_parallel import NativeWorkers, file_digest, run_cases


def verify_common_stage7(family, checkpoint, review, output):
    from native_family_runner import FamilyRunner
    from native_source_audit import audit

    runner = FamilyRunner.from_checkpoint(family, output, checkpoint)
    metadata = read_json(Path(checkpoint).with_suffix(".json"))
    if not metadata["label"].startswith("stage_7_excavate_"):
        raise ValueError("Five alternatives require the completed common Stage7 state")
    if runner.family["input_hashes"] != review["input_hashes"]:
        raise ValueError("Common native state belongs to a different submitted model")
    if set(runner.wall_nodes) != {"stage_2_outer_wall"}:
        raise ValueError("Common prefix already contains an inner retaining system")
    if any(int(row["label"].split("_")[1]) >= 8 for row in runner.receipt["stages"]):
        raise ValueError("Common state contains case-dependent construction history")
    supports = runner.receipt["support_installations"]
    if len(supports) != 4 or any(row["system"] != "outer" for row in supports):
        raise ValueError("Common prefix must contain exactly four outer support levels")
    if abs(runner.water_elevation + 10.6) > 1e-8:
        raise ValueError("Common prefix does not reach the source Stage7 water milestone")
    outer_events = [row for row in runner.receipt["excavation_events"] if row["system"] == "outer"]
    if not outer_events or abs(outer_events[-1]["bottom_m"] + 10.1) > 1e-8:
        raise ValueError("Common prefix does not complete outer excavation")
    record = {
        "native_state_sha256": metadata["native_state_sha256"],
        "input_hashes": runner.family["input_hashes"],
        "source_fact_audit": audit(runner),
        "common_stage7_ready": True,
        "full_task_acceptance": False,
    }
    (Path(output) / "common-stage7-audit.json").write_text(json.dumps(record, indent=2))
    return record


def replay_family(
    family,
    output,
    review_path,
    *,
    total_seconds=6600,
    max_iterations=400,
    cpus=(1, 2),
    workers=None,
    case_script=None,
    common_checkpoint=None,
    completed_pos0=None,
    prior_ledger=None,
    initial_checkpoint=None,
):
    if not 120 <= total_seconds <= 6600:
        raise ValueError("Family runtime must be within120..6600 seconds")
    if (common_checkpoint is not None or initial_checkpoint is not None) and prior_ledger is None:
        raise ValueError("A retained common prefix requires its attempted-time ledger")
    if initial_checkpoint is not None and (
        common_checkpoint is not None or completed_pos0 is not None
    ):
        raise ValueError("Initial-state recovery cannot also reuse a later common state or case")
    prior_attempts = read_json(prior_ledger) if prior_ledger else []
    if not isinstance(prior_attempts, list) or any(
        not isinstance(row, dict)
        or isinstance(row.get("elapsed_s"), bool)
        or not isinstance(row.get("elapsed_s"), (int, float))
        or not math.isfinite(row["elapsed_s"])
        or row["elapsed_s"] < 0
        for row in prior_attempts
    ):
        raise ValueError("Prior attempts need finite nonnegative elapsed seconds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    review = read_json(review_path)
    verify_review(family, review)
    if initial_checkpoint is not None:
        initial = read_json(Path(initial_checkpoint).with_suffix(".json"))
        if (
            initial["input_hashes"] != review["input_hashes"]
            or initial["receipt"].get("support_installations")
            or int(initial["label"].split("_")[1]) > 2
            or file_digest(Path(initial_checkpoint).with_suffix(".rest"))
            != initial["native_state_sha256"]
        ):
            raise ValueError("Initial checkpoint must precede support installation in this family")
    started = time.monotonic()
    record = {
        "input_hashes": review["input_hashes"],
        "cases": {},
        "attempts": [],
        "completed": False,
        "full_task_acceptance": False,
        "started_monotonic": started,
        "runtime_budget_s": total_seconds,
        "prior_attempts": prior_attempts,
    }
    workers = workers or NativeWorkers(cpus)
    try:
        deadline = started + total_seconds
        if completed_pos0 is not None:
            if common_checkpoint is None or prior_ledger is None:
                raise ValueError(
                    "Reusing a completed native case needs common state and attempted-time ledger"
                )
            completed = Path(completed_pos0)
            terminal = read_json(completed / "terminal.json")
            if (
                terminal.get("native_exit_code") != 0
                or terminal.get("case") != "pos_0m"
                or terminal.get("input_hashes") != review["input_hashes"]
                or not terminal.get("source_fact_audit", {}).get("passed")
                or terminal.get("source_fact_audit", {})
                .get("audit", {})
                .get("support_sections", {})
                .get("version")
                != 1
                or not terminal.get("extraction", {}).get("native_replay_and_reader_completed")
            ):
                raise ValueError(
                    "Existing pos0 must be an audited native result of this same family"
                )
            for name, digest in terminal["extraction"]["native_hashes"].items():
                if file_digest(completed / name) != digest:
                    raise ValueError("Previously completed native fields changed")
            record["cases"]["pos_0m"] = {
                "directory": str(completed),
                "terminal_sha256": file_digest(completed / "terminal.json"),
                "extraction": terminal["extraction"],
                "reused_evaluator_owned_result": True,
            }
        else:
            run_cases(
                family,
                output,
                review_path,
                [0],
                initial_checkpoint or common_checkpoint,
                record,
                workers,
                deadline,
                max_iterations,
                case_script=case_script,
            )
        common = Path(common_checkpoint) if common_checkpoint else None
        if common is None:
            for attempt in reversed(record["attempts"]):
                candidate = Path(attempt["directory"]) / "restart/branch"
                if candidate.with_suffix(".json").exists() and read_json(
                    candidate.with_suffix(".json")
                )["label"].startswith("stage_7_excavate_"):
                    common = candidate
                    break
        if common is None:
            raise ValueError("Completed case has no common native Stage7 checkpoint")
        record["common_stage7"] = verify_common_stage7(
            family, common, review, output / "common-stage7-audit"
        )
        frozen = output / "common-stage7" / "state"
        frozen.parent.mkdir()
        os.link(common.with_suffix(".rest"), frozen.with_suffix(".rest"))
        frozen.with_suffix(".json").write_bytes(common.with_suffix(".json").read_bytes())
        immutable = {
            str(frozen.with_suffix(extension)): file_digest(frozen.with_suffix(extension))
            for extension in (".rest", ".json")
        }
        if (
            immutable[str(frozen.with_suffix(".rest"))]
            != record["common_stage7"]["native_state_sha256"]
        ):
            raise ValueError("Common state changed while freezing the branch source")
        record["immutable_common_checkpoint"] = {"path": str(frozen), "hashes": immutable}
        run_cases(
            family,
            output,
            review_path,
            [1, 2, 3, 4],
            frozen,
            record,
            workers,
            deadline,
            max_iterations,
            case_script=case_script,
            immutable=immutable,
        )
        record["completed"] = True
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["elapsed_s"] = time.monotonic() - started
        record["native_compute_s"] = sum(row["elapsed_s"] for row in record["attempts"])
        record["prior_native_compute_s"] = sum(row["elapsed_s"] for row in record["prior_attempts"])
        record["cumulative_native_compute_s"] = (
            record["native_compute_s"] + record["prior_native_compute_s"]
        )
        record["worker_cpu_s"] = sum(row["cpu_s"] for row in record["attempts"])
        record["resources"] = workers.aggregate_usage()
        (output / "family-replay.json").write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    parser.add_argument("output")
    parser.add_argument("review")
    parser.add_argument("--total-seconds", type=int, default=6600)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--cpus", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--common-checkpoint")
    parser.add_argument("--completed-pos0")
    parser.add_argument("--prior-ledger")
    parser.add_argument("--initial-checkpoint")
    arguments = parser.parse_args()
    print(
        json.dumps(
            replay_family(
                arguments.family,
                arguments.output,
                arguments.review,
                total_seconds=arguments.total_seconds,
                max_iterations=arguments.max_iterations,
                cpus=arguments.cpus,
                common_checkpoint=arguments.common_checkpoint,
                completed_pos0=arguments.completed_pos0,
                prior_ledger=arguments.prior_ledger,
                initial_checkpoint=arguments.initial_checkpoint,
            )
        ),
        flush=True,
    )
