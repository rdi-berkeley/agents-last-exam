from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.transport_safety.abm_hangzhou_metro.scripts import public_model as model
from tasks.transport_safety.abm_hangzhou_metro.scripts import score_outputs as scorer


def write_input(root, trips, *, capacity=2, entry_limit=550, extra_line=False):
    for directory in ("data", "gis", "network_config"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    params = {
        "operation_hours": {"start": "04:00", "end": "04:30"},
        "default_headway_minutes": 5,
        "peak_periods": [],
        "train_capacity": capacity,
        "metro_speed_kmh": 70,
        "station_stop_time_minutes": 1,
        "transfer_time_minutes": 3,
        "station_entry_flow_limit_per_minute": entry_limit,
        "block_length_meters": 100,
        "simulation_step_minutes": 1,
        "route_choice": {
            "k_shortest_paths": 15,
            "logit_epsilon": -10,
            "logit_beta_transfer_times": -2.346038,
            "logit_beta_travel_time": -0.186402,
        },
    }
    (root / "network_config/operation_parameters.json").write_text(json.dumps(params))
    points = {"A": [0.0, 0.0], "B": [0.005, 0.0], "C": [0.01, 0.0], "D": [0.005, 0.015]}
    definitions = [("main_f", "main", ["A", "B", "C"]), ("main_r", "main", ["C", "B", "A"])]
    if extra_line:
        definitions += [("branch_f", "branch", ["B", "D"]), ("branch_r", "branch", ["D", "B"])]
        if extra_line != "branch":
            definitions += [("alt_f", "alt", ["A", "D", "C"]), ("alt_r", "alt", ["C", "D", "A"])]
    lines, stations, sequence = [], [], []
    for service, physical, names in definitions:
        direction = names[0] + "-" + names[-1]
        full_name = physical + "(" + direction + ")"
        lines.append(
            {
                "properties": {"linename": full_name},
                "geometry": {"type": "LineString", "coordinates": [points[name] for name in names]},
            }
        )
        for index, station in enumerate(names, 1):
            stations.append(
                {
                    "properties": {"linename": full_name, "stationnames": station},
                    "geometry": {"type": "Point", "coordinates": points[station]},
                }
            )
            sequence.append([service, physical, direction, index, station])
    (root / "gis/hangzhou_lines.json").write_text(json.dumps({"features": lines}))
    (root / "gis/hangzhou_stations.json").write_text(json.dumps({"features": stations}))
    with (root / "network_config/station_sequence.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line_id", "line_name_cn", "direction", "station_order", "station_name"])
        writer.writerows(sequence)
    with (root / "data/afc_hangzhou.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["card_id", "o_station", "d_station", "start_time", "end_time"])
        writer.writerows(trips)
    (root / "simulation_contract.md").write_bytes(scorer.PUBLIC_CONTRACT.read_bytes())
    return root


def write_bundle(root, trips, ends, transfers=None, seed=42):
    root.mkdir(exist_ok=True)
    transfers = transfers or [0] * len(trips)
    actual, predicted = [], []
    with (root / "passenger_records.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(scorer.OUTPUT_COLUMNS)
        for trip, end, count in zip(trips, ends, transfers, strict=True):
            if end is None:
                continue
            observed = (trip[4] - trip[3]) % 1440
            writer.writerow([*trip[:4], end, count, trip[4], end - trip[3], observed])
            actual.append(observed)
            predicted.append(end - trip[3])
    residual = [pred - real for pred, real in zip(predicted, actual, strict=True)]
    mean = sum(actual) / len(actual)
    squared = sum(value**2 for value in residual)
    total = sum((value - mean) ** 2 for value in actual)
    report = {
        "R2": 1 - squared / total if total else 0.0,
        "RMSE": math.sqrt(squared / len(actual)),
        "Total passengers": len(actual),
        "std(sim-real)": math.sqrt(
            sum((value - sum(residual) / len(residual)) ** 2 for value in residual) / len(residual)
        ),
    }
    (root / "validation_report.txt").write_text(
        "\n".join(f"{key} = {value}" for key, value in report.items())
    )
    (root / "simulation_manifest.json").write_text(
        json.dumps({"contract_version": model.CONTRACT_VERSION, "seed": seed})
    )
    return root


@pytest.fixture
def queue_case(tmp_path):
    trips = [
        ("first", "A", "C", 240, 249),
        ("second", "A", "C", 240, 251),
        ("third", "A", "C", 240, 254),
        ("downstream", "B", "C", 246, 255),
    ]
    inputs = write_input(tmp_path / "input", trips)
    output = write_bundle(tmp_path / "output", trips, [249, 249, 254, 254])
    return inputs, output


def test_hand_computed_capacity_and_fifo(queue_case):
    inputs, output = queue_case
    result = model.simulate(inputs, 42)
    assert result.ends == [249, 249, 254, 254]
    assert result.diagnostics["peak_train_load"] == 2
    assert result.diagnostics["boardings"] == result.diagnostics["alightings"] == 4
    assert scorer.score_output_bundle(
        input_dir=inputs, output_dir=output, reference_dir=Path("/nonexistent/not-an-oracle")
    ).passed


def test_entry_limit_is_not_train_capacity(tmp_path):
    trips = [(str(index), "A", "C", 244, 255 + index) for index in range(6)]
    inputs = write_input(tmp_path / "input", trips, capacity=10, entry_limit=1)
    assert model.simulate(inputs, 0).ends == [249, 254, 254, 254, 254, 254]


def test_transfer_walk_queue_priority_and_capacity(tmp_path):
    trips = [("transfer", "A", "D", 240, 254), ("local", "B", "D", 248, 258)]
    inputs = write_input(tmp_path / "input", trips, capacity=1, extra_line="branch")
    result = model.simulate(inputs, 42)
    assert result.ends == [253, 258]
    assert result.transfers == [1, 0]
    assert result.diagnostics["boardings"] == result.diagnostics["alightings"] == 3


def test_no_hidden_anticoopy_distance_gate(tmp_path):
    trips = [("one", "A", "C", 240, 249), ("two", "A", "B", 241, 247)]
    inputs = write_input(tmp_path / "input", trips)
    output = write_bundle(tmp_path / "output", trips, [249, 247])
    assert scorer.score_output_bundle(input_dir=inputs, output_dir=output).passed


def test_labels_do_not_influence_simulation(tmp_path):
    trips = [("one", "A", "C", 240, 249), ("two", "B", "C", 242, 253)]
    inputs = write_input(tmp_path / "input", trips)
    before = model.simulate(inputs, 9)
    write_input(inputs, [(*trip[:4], trip[4] + 100) for trip in trips])
    after = model.simulate(inputs, 9)
    assert before.ends == after.ends
    assert before.transfers == after.transfers
    assert before.diagnostics == after.diagnostics


def test_distinct_seeds_are_valid_not_one_solution(tmp_path):
    trips = [(str(index), "A", "C", 240, 250 + index) for index in range(20)]
    inputs = write_input(tmp_path / "input", trips, capacity=50, extra_line=True)
    results = [model.simulate(inputs, seed) for seed in (0, 1)]
    assert results[0].ends != results[1].ends or results[0].transfers != results[1].transfers
    for seed, result in enumerate(results):
        output = write_bundle(tmp_path / f"seed{seed}", trips, result.ends, result.transfers, seed)
        assert scorer.score_output_bundle(input_dir=inputs, output_dir=output).passed


def test_hash_mnl_not_argmax():
    options = (model.Route((("first", 0, 1),), 3), model.Route((("other", 0, 1),), 7))
    params = {
        "route_choice": {
            "logit_beta_transfer_times": -2.346038,
            "logit_beta_travel_time": -0.186402,
        }
    }
    first_probability = 1 / (1 + math.exp(-0.186402 * 4))
    chosen = set()
    for index in range(100):
        uniform = (
            int(
                hashlib.sha256(f"hangzhou-discrete-event-v2:42:{index}".encode()).hexdigest()[:16],
                16,
            )
            / 2**64
        )
        expected = options[0 if uniform < first_probability else 1]
        assert model.choose_route(options, params, 42, index) == expected
        chosen.add(expected)
    assert len(chosen) == 2


def test_fractional_headway_phase_and_extended_day():
    params = {
        "operation_hours": {"start": "23:55", "end": "24:12"},
        "default_headway_minutes": 5,
        "station_stop_time_minutes": 1,
        "block_length_meters": 100,
        "metro_speed_kmh": 70,
        "peak_periods": [{"start": "24:00", "end": "24:08", "headway_minutes": 2.5}],
    }
    assert model.departures(params) == [1435, 1440, 1443, 1445, 1448, 1450]


def test_same_station_and_unserved_conservation(tmp_path):
    trips = [
        ("same", "A", "A", 239, 242),
        ("too_late", "A", "C", 270, 280),
        ("outside", "A", "C", 271, 281),
    ]
    inputs = write_input(tmp_path / "input", trips)
    result = model.simulate(inputs, 42)
    assert result.ends == [242, None, None]
    assert result.diagnostics["queued_at_close"] == 1
    assert result.diagnostics["not_admitted"] == 1
    assert (
        result.diagnostics["demand"]
        == result.diagnostics["completed"] + result.diagnostics["unserved"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("end_time_simulate", "248"),
        ("duration_simulation", "8"),
        ("transfer_times", "1"),
        ("end_time_real", "999"),
        ("duration_real", "999"),
        ("duration_simulation", "NaN"),
        ("duration_simulation", "1.5"),
    ],
)
def test_false_outputs_rejected(queue_case, field, value):
    inputs, output = queue_case
    path = output / "passenger_records.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = value
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scorer.OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert not scorer.score_output_bundle(input_dir=inputs, output_dir=output).passed


@pytest.mark.parametrize("change", ["omit", "duplicate", "unknown", "labels", "offset"])
def test_conservation_and_label_shortcuts_rejected(queue_case, change):
    inputs, output = queue_case
    path = output / "passenger_records.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if change == "omit":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "unknown":
        rows[0]["card_id"] = "not-an-afc-trip"
    else:
        for row in rows:
            delta = 0 if change == "labels" else 4
            row["end_time_simulate"] = str(int(row["end_time_real"]) + delta)
            row["duration_simulation"] = str(int(row["duration_real"]) + delta)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scorer.OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert not scorer.score_output_bundle(input_dir=inputs, output_dir=output).passed


@pytest.mark.parametrize("metric", scorer.REPORT_KEYS)
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "99999"])
def test_report_must_be_finite_and_truthful(queue_case, metric, bad):
    inputs, output = queue_case
    path = output / "validation_report.txt"
    lines = path.read_text().splitlines()
    path.write_text(
        "\n".join(f"{metric} = {bad}" if line.startswith(metric + " =") else line for line in lines)
    )
    assert not scorer.score_output_bundle(input_dir=inputs, output_dir=output).passed


@pytest.mark.parametrize("seed", [True, -1, 2**64, 0.5, "42", None])
def test_manifest_seed_type(queue_case, seed):
    inputs, output = queue_case
    (output / "simulation_manifest.json").write_text(
        json.dumps({"contract_version": model.CONTRACT_VERSION, "seed": seed})
    )
    assert not scorer.score_output_bundle(input_dir=inputs, output_dir=output).passed


@pytest.mark.parametrize(
    "missing", ["simulation_contract.md", "gis/hangzhou_lines.json", "data/afc_hangzhou.csv"]
)
def test_missing_evaluator_data_is_not_solver_failure(queue_case, missing):
    inputs, output = queue_case
    (inputs / missing).unlink()
    with pytest.raises(RuntimeError, match="evaluator"):
        scorer.score_output_bundle(input_dir=inputs, output_dir=output)


def test_missing_manifest_has_explicit_failure(queue_case):
    inputs, output = queue_case
    (output / "simulation_manifest.json").unlink()
    result = scorer.score_output_bundle(input_dir=inputs, output_dir=output)
    assert result.details["missing"] == "simulation_manifest.json"


def test_main_import_does_not_capture_unrelated_scorer():
    script = """
import sys, types
unrelated = types.ModuleType('score_outputs')
sys.modules['score_outputs'] = unrelated
before = list(sys.path)
from tasks.transport_safety.abm_hangzhou_metro import main
from tasks.transport_safety.abm_hangzhou_metro.scripts import score_outputs
assert main.score_output_bundle is score_outputs.score_output_bundle
assert sys.modules['score_outputs'] is unrelated
assert sys.path == before
assert 'simulation_contract_file' in main.config.to_metadata()
assert 'candidate_manifest' in main.config.to_metadata()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "missing_candidate", ["candidate_csv", "candidate_report", "candidate_manifest"]
)
@pytest.mark.parametrize("contract_state", ["current", "stale", "empty", "missing", "read_error"])
def test_loader_checks_contract_before_missing_output(missing_candidate, contract_state):
    from ale_run.tasks.loader import TaskLoader

    task_dir = Path(__file__).resolve().parents[2] / "tasks/transport_safety/abm_hangzhou_metro"
    loader = TaskLoader(str(task_dir))
    task_cfg = loader.build_task_cfg()
    evaluate = loader.get_evaluate_fn()
    assert evaluate.__module__ == "_task_transport_safety_abm_hangzhou_metro"
    metadata = task_cfg.metadata
    evaluator_keys = (
        "afc_csv",
        "lines_geojson",
        "stations_geojson",
        "station_sequence_csv",
        "operation_parameters_json",
        "simulation_contract_file",
    )
    candidate_keys = ("candidate_csv", "candidate_report", "candidate_manifest")
    contents = {metadata[key]: b"unused" for key in evaluator_keys + candidate_keys}
    contents.pop(metadata[missing_candidate])
    contract_path = metadata["simulation_contract_file"]
    contents[contract_path] = scorer.PUBLIC_CONTRACT.read_bytes()
    if contract_state == "stale":
        contents[contract_path] = b"old public specification"
    elif contract_state == "empty":
        contents[contract_path] = b""
    elif contract_state == "missing":
        contents.pop(contract_path)
    checks, reads = [], []

    class Session:
        async def file_exists(self, path):
            checks.append(path)
            return path in contents

        async def read_bytes(self, path):
            reads.append(path)
            assert path == contract_path
            if contract_state == "read_error":
                raise OSError("contract transport failed")
            return contents[path]

    if contract_state == "current":
        assert asyncio.run(evaluate(task_cfg, Session())) == [0.0]
        assert reads == [contract_path]
        assert metadata[missing_candidate] in checks
    else:
        if contract_state == "read_error":
            error, message = OSError, "contract transport failed"
        elif contract_state == "missing":
            error, message = RuntimeError, "evaluator-controlled simulation_contract.md missing"
        else:
            error, message = RuntimeError, "contract mismatch"
        with pytest.raises(error, match=message):
            asyncio.run(evaluate(task_cfg, Session()))
        assert not set(checks).intersection(metadata[key] for key in candidate_keys)
        assert reads == ([] if contract_state == "missing" else [contract_path])


@pytest.mark.parametrize("filesystem", ["ext4", "tmpfs"])
def test_evaluate_stages_public_inputs_without_reference(queue_case, monkeypatch, filesystem):
    from ale_run.tasks.loader import TaskLoader

    task_dir = Path(__file__).resolve().parents[2] / "tasks/transport_safety/abm_hangzhou_metro"
    loader = TaskLoader(str(task_dir))
    task_cfg = loader.build_task_cfg()
    main = loader._load_module()
    evaluate = loader.get_evaluate_fn()

    inputs, output = queue_case
    metadata = {
        "afc_csv": inputs / "data/afc_hangzhou.csv",
        "lines_geojson": inputs / "gis/hangzhou_lines.json",
        "stations_geojson": inputs / "gis/hangzhou_stations.json",
        "station_sequence_csv": inputs / "network_config/station_sequence.csv",
        "operation_parameters_json": inputs / "network_config/operation_parameters.json",
        "simulation_contract_file": inputs / "simulation_contract.md",
        "candidate_csv": output / "passenger_records.csv",
        "candidate_report": output / "validation_report.txt",
        "candidate_manifest": output / "simulation_manifest.json",
    }
    task_cfg.metadata.update(metadata)
    reads = []

    class Session:
        async def file_exists(self, path):
            return path.is_file()

        async def read_bytes(self, path):
            reads.append(path)
            return path.read_bytes()

    monkeypatch.setenv("ALE_EVALUATOR_SCRATCH_DIR", str(inputs.parent / "scratch"))
    monkeypatch.setattr(
        main.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=filesystem)
    )
    if filesystem == "tmpfs":
        with pytest.raises(RuntimeError, match="disk-backed"):
            asyncio.run(evaluate(task_cfg, Session()))
        assert reads == [metadata["simulation_contract_file"]]
    else:
        assert asyncio.run(evaluate(task_cfg, Session())) == [1.0]
        assert set(reads) == set(metadata.values())
        assert reads.count(metadata["simulation_contract_file"]) == 1


def test_schema_allows_row_permutations_and_integer_numeric_spelling(queue_case):
    inputs, output = queue_case
    path = output / "passenger_records.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name in scorer.OUTPUT_COLUMNS[3:]:
            row[name] += ".0"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scorer.OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(reversed(rows))
    assert scorer.score_output_bundle(input_dir=inputs, output_dir=output).passed


def test_parallel_services_are_distinct_route_options(tmp_path):
    inputs = write_input(tmp_path / "input", [("trip", "A", "B", 240, 247)])
    sequence_path = inputs / "network_config/station_sequence.csv"
    with sequence_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header, rows = reader.fieldnames, list(reader)
    duplicated = [dict(row, line_id="parallel_" + row["line_id"]) for row in rows]
    with sequence_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows + duplicated)
    params, _, graph = model.load_network(inputs)
    routes = model.route_options(graph, "A", "B", params)
    assert len(routes) == 2
    assert {route.legs[0][0] for route in routes} == {"main_f", "parallel_main_f"}


def test_geometry_uses_polyline_not_station_chord():
    coordinates = [[0, 0], [0, 0.01], [0.01, 0.01], [0.01, 0]]
    distance = model.chainage([0.01, 0], coordinates)
    assert distance > 2.99 * model.haversine_meters([0, 0], [0.01, 0])


def test_unsafe_headway_is_evaluator_failure(tmp_path):
    inputs = write_input(tmp_path / "input", [("trip", "A", "B", 240, 247)])
    path = inputs / "network_config/operation_parameters.json"
    params = json.loads(path.read_text())
    params["default_headway_minutes"] = 1
    path.write_text(json.dumps(params))
    output = write_bundle(tmp_path / "output", [("trip", "A", "B", 240, 247)], [247])
    with pytest.raises(RuntimeError, match="clearance"):
        scorer.score_output_bundle(input_dir=inputs, output_dir=output)


@pytest.mark.parametrize(
    "metric,bad", [("RMSE", "-0.001"), ("std(sim-real)", "-0.001"), ("R2", "1.0001")]
)
def test_report_physical_bounds_even_inside_metric_tolerance(tmp_path, metric, bad):
    trips = [("one", "A", "C", 240, 249), ("two", "A", "B", 241, 247)]
    inputs = write_input(tmp_path / "input", trips)
    output = write_bundle(tmp_path / "output", trips, [249, 247])
    report = output / "validation_report.txt"
    report.write_text(
        "\n".join(
            f"{metric} = {bad}" if line.startswith(metric + " =") else line
            for line in report.read_text().splitlines()
        )
    )
    assert not scorer.score_output_bundle(input_dir=inputs, output_dir=output).passed


def test_wrong_public_spec_is_evaluator_error(queue_case):
    inputs, output = queue_case
    (inputs / "simulation_contract.md").write_text("old public specification")
    with pytest.raises(RuntimeError, match="contract mismatch"):
        scorer.score_output_bundle(input_dir=inputs, output_dir=output)
