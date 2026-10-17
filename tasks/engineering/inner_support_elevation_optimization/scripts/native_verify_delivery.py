"""Read submitted native output against evaluator-created baseline and monitoring."""

import argparse
import hashlib
import json
from pathlib import Path

from native_input_adapter import read_json
from native_result_adapter import extract_monitors, read_native_result, read_stage1_baseline


def read_delivery(submission, replay_path, review_path, output):
    replay = read_json(replay_path)
    review = read_json(review_path)
    if not replay.get("completed") or replay["input_hashes"] != review["input_hashes"]:
        raise ValueError("Delivery reading requires complete independently reviewed replay")
    record = {"cases": {}, "input_hashes": replay["input_hashes"]}
    for case, verified in replay["cases"].items():
        trusted = Path(verified["directory"])
        terminal_path = trusted / "terminal.json"
        if hashlib.sha256(terminal_path.read_bytes()).hexdigest() != verified["terminal_sha256"]:
            raise ValueError("Evaluator replay terminal changed")
        receipt = read_json(trusted / "receipt.json")
        delivered = Path(submission) / "results" / case
        if not delivered.is_dir():
            delivered = Path(submission) / case
        item = {"native_result_read": False}
        try:
            paths = {
                name: delivered / name for name in ("native_model.mdpa", "native_nodal_fields.tsv")
            }
            hashes = {
                name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
            }
            native = read_native_result(
                paths["native_model.mdpa"],
                expected_sha256=hashes["native_model.mdpa"],
                coordinates_path=paths["native_nodal_fields.tsv"],
                coordinates_sha256=hashes["native_nodal_fields.tsv"],
                precise_translations=True,
            )
            baseline_path = trusted / "stage_1_k0_equilibrium_hold.npz"
            baseline = read_stage1_baseline(
                baseline_path,
                expected_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                soil_nodes=native.soil_nodes,
            )
            observations = extract_monitors(
                native,
                baseline,
                review["monitors"],
                outer_wall_property_ids={receipt["wall_properties"]["stage_2_outer_wall"]},
                wall_range=review["wall_range_z_m"],
                crown_geometry=[
                    row
                    for row in read_json(trusted / "crown_geometry.json")
                    if row["system"] == "outer"
                ],
            )
            item.update(native_result_read=True, observations=observations, input_hashes=hashes)
        except (ValueError, OSError, KeyError) as error:
            item["error"] = f"{type(error).__name__}: {error}"
        record["cases"][case] = item
    Path(output).write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("replay")
    parser.add_argument("review")
    parser.add_argument("output")
    arguments = parser.parse_args()
    read_delivery(arguments.submission, arguments.replay, arguments.review, arguments.output)
