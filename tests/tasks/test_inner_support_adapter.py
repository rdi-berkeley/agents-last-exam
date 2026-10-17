import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "tasks/engineering/inner_support_elevation_optimization/scripts"
)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_family(tmp_path):
    load_script("build_native_family").build(tmp_path, column=True)
    return tmp_path


def test_native_preparation_contains_fourteen_real_layers_and_positive_tetrahedra(native_family):
    family, benchmark = load_script("native_input_adapter").read_family(native_family)
    assert [layer["id"] for layer in family["layers"]] == [row[0] for row in benchmark["soils"]]
    assert set(family["input_hashes"]) == {"model.mdpa", "Materials.json", "family.json"}
    mesh = native_family.joinpath("model.mdpa").read_text()
    node_text = mesh.split("Begin Nodes\n")[1].split("End Nodes")[0]
    nodes = {
        int(cells[0]): np.array(list(map(float, cells[1:])))
        for cells in (line.split() for line in node_text.splitlines())
    }
    elements = mesh.split("Begin Elements UPwSmallStrainElement3D4N\n")[1].split("End Elements")[0]
    volumes = []
    used = set()
    for line in elements.splitlines():
        _, property_id, *identifiers = map(int, line.split())
        coordinates = np.array([nodes[identifier] for identifier in identifiers])
        volumes.append(np.linalg.det((coordinates[1:] - coordinates[0]).T) / 6)
        used.add(property_id)
    assert min(volumes) > 0
    assert sum(volumes) == pytest.approx(4 * sum(row[1] for row in benchmark["soils"]))
    assert used == set(range(1, 15))


def test_deeper_domain_requires_explicit_bottom_rock_continuation_and_preserves_source_layers(
    native_family,
):
    adapter = load_script("native_input_adapter")
    path = native_family / "family.json"
    family = json.loads(path.read_text())
    original_layers = json.loads(json.dumps(family["layers"]))
    family["domain"][4] = -53.5
    path.write_text(json.dumps(family))
    with pytest.raises(ValueError, match="explicit supported extension"):
        adapter.read_family(native_family)
    family["engineering_choices"]["deep_rock_continuation"] = {
        "source_layer_id": "4-2",
        "bottom_m": -53.5,
        "rationale": "Same-property rock continuation for a private bottom-boundary diagnostic",
    }
    path.write_text(json.dumps(family))
    accepted, _ = adapter.read_family(native_family)
    native_layers = adapter.effective_layers(accepted)
    assert accepted["layers"] == original_layers
    assert native_layers[:-1] == original_layers[:-1]
    assert native_layers[-1] == {**original_layers[-1], "bottom": -53.5}
    family["engineering_choices"]["deep_rock_continuation"]["source_layer_id"] = "4-1"
    path.write_text(json.dumps(family))
    with pytest.raises(ValueError, match="deepest source rock"):
        adapter.read_family(native_family)


def test_extended_native_mesh_assigns_deep_cells_to_unchanged_bottom_rock(tmp_path):
    builder = load_script("build_native_family")
    source = tmp_path / "source"
    builder.build(source, column=True)
    family = json.loads((source / "family.json").read_text())
    geometry = {
        "outer_polygon": family["outer_polygon"],
        "inner_polygon": family["inner_polygon"],
        "monitors": {},
        "extra_levels_m": [-53.5, -42.62],
        "engineering_choices": {
            "deep_rock_continuation": {
                "source_layer_id": "4-2",
                "bottom_m": -53.5,
                "rationale": "Diagnostic extension",
            }
        },
    }
    destination = tmp_path / "deeper"
    builder.build(destination, column=True, geometry=geometry)
    adapter = load_script("native_input_adapter")
    deeper, _ = adapter.read_family(destination)
    assert deeper["layers"] == family["layers"]
    assert (destination / "Materials.json").read_bytes() == (source / "Materials.json").read_bytes()
    mesh = (destination / "model.mdpa").read_text()
    nodes = {
        int(cells[0]): float(cells[3])
        for cells in map(
            str.split, mesh.split("Begin Nodes\n")[1].split("End Nodes")[0].splitlines()
        )
    }
    deep_elements = []
    for line in (
        mesh.split("Begin Elements UPwSmallStrainElement3D4N\n")[1]
        .split("End Elements")[0]
        .splitlines()
    ):
        _, property_id, *connectivity = map(int, line.split())
        if max(nodes[node] for node in connectivity) <= family["layers"][-1]["bottom"]:
            deep_elements.append(property_id)
    assert deep_elements and set(deep_elements) == {14}


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda family: family.update(processes=[{"python_module": "untrusted"}]),
            "Unsupported family keys",
        ),
        (lambda family: family["layers"].pop(), "fourteen"),
        (lambda family: family["layers"][0].update(bottom=0), "thickness"),
        (lambda family: family["source_hashes"].update(extra="forged"), "provenance"),
        (lambda family: family.update(units="mm-N-s"), "units"),
        (
            lambda family: family["engineering_choices"].update(
                solver_controls={"residual_relative_tolerance": 0}
            ),
            "tolerance",
        ),
        (
            lambda family: family["engineering_choices"].update(
                solver_controls={"python_module": "candidate"}
            ),
            "tolerance",
        ),
    ],
)
def test_invalid_family_is_rejected_before_native_import(native_family, mutation, match):
    path = native_family / "family.json"
    family = json.loads(path.read_text())
    mutation(family)
    path.write_text(json.dumps(family))
    with pytest.raises(ValueError, match=match):
        load_script("native_input_adapter").read_family(native_family)


def test_executable_constitutive_name_is_not_imported(native_family):
    path = native_family / "Materials.json"
    materials = json.loads(path.read_text())
    materials["properties"][0]["Material"]["constitutive_law"]["name"] = "candidate.module"
    path.write_text(json.dumps(materials))
    with pytest.raises(ValueError, match="GeoMohrCoulombLaw3D"):
        load_script("native_input_adapter").read_family(native_family)


def test_duplicate_property_bindings_are_rejected(native_family):
    path = native_family / "Materials.json"
    materials = json.loads(path.read_text())
    materials["properties"][1]["properties_id"] = materials["properties"][0]["properties_id"]
    path.write_text(json.dumps(materials))
    with pytest.raises(ValueError, match="Repeated native"):
        load_script("native_input_adapter").read_family(native_family)


def test_mesh_escape_is_rejected(native_family, tmp_path):
    mesh = native_family / "model.mdpa"
    target = tmp_path / "external.mdpa"
    mesh.rename(target)
    mesh.symlink_to(target)
    with pytest.raises(ValueError, match="Unsafe"):
        load_script("native_input_adapter").read_family(native_family)


def test_duplicate_and_nonfinite_native_json_fail(tmp_path):
    adapter = load_script("native_input_adapter")
    path = tmp_path / "bad.json"
    path.write_text('{"value":1,"value":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        adapter.read_json(path)
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        adapter.read_json(path)


def test_staged_trial_preserves_source_soil_and_declares_choices(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "staged"
    load_script("build_native_family").build(source)
    result = load_script("prepare_staged_trial").prepare(source, destination)
    for name in ("model.mdpa", "Materials.json"):
        assert source.joinpath(name).read_bytes() == destination.joinpath(name).read_bytes()
    original = json.loads(source.joinpath("family.json").read_text())
    family, _ = load_script("native_input_adapter").read_family(destination)
    for name in (
        "layers",
        "outer_polygon",
        "inner_polygon",
        "outer_wall_faces",
        "inner_wall_faces",
        "source_hashes",
    ):
        assert original[name] == family[name]
    assert result["soil_mesh_unchanged"]
    assert family["braces"][0]["elevations_m"] == [6.5, 0.7, -3.3, -6.6]
    assert family["braces"][1]["elevations_m"] == [-10.1]
    assert (
        "explicit trial engineering choice" in family["engineering_choices"]["inner_brace_choice"]
    )
    for definition in family["braces"]:
        adjacency = {}
        for first, second in definition["segments_xy_m"]:
            first, second = tuple(first), tuple(second)
            assert first != second
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
        visited = {next(iter(adjacency))}
        pending = list(visited)
        while pending:
            for node in adjacency[pending.pop()] - visited:
                visited.add(node)
                pending.append(node)
        assert visited == set(adjacency)


def test_brace_crossings_are_native_shared_joints():
    segments = load_script("prepare_staged_trial").split_segments(
        [
            ([0, 0], [4, 4]),
            ([0, 4], [4, 0]),
        ]
    )
    assert len(segments) == 4
    assert all([2.0, 2.0] in segment for segment in segments)


@pytest.mark.parametrize("angle", [0, 0.47, 1.2])
def test_collinear_brace_traces_share_joints_and_count_each_span_once(angle):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([53.0, -27.0])
    source = [([0, 0], [4, 0]), ([3, 0], [1, 0]), ([2, -1], [2, 1]), ([4, 0], [6, 0])]
    transformed = [[(rotation @ point + shift).tolist() for point in edge] for edge in source]
    segments = load_script("prepare_staged_trial").split_segments(transformed)
    assert len(segments) == 7
    total_length = sum(np.linalg.norm(np.subtract(second, first)) for first, second in segments)
    assert total_length == pytest.approx(8, abs=1e-6)
    center = np.round(rotation @ [2, 0] + shift, 7).tolist()
    assert sum(center in edge for edge in segments) == 4


def test_parallel_separate_members_and_zero_length_trace_do_not_merge():
    split = load_script("prepare_staged_trial").split_segments
    source = [([0, 0], [4, 0]), ([0, 1], [4, 1]), ([2, 2], [2, 2])]
    assert split(source) == [[[0.0, 0.0], [4.0, 0.0]], [[0.0, 1.0], [4.0, 1.0]]]


def test_native_backend_allowlist_uses_measured_scalar_ilut_configuration():
    settings = load_script("native_input_adapter").native_linear_settings
    assert settings("sparse-lu") == {"solver_type": "LinearSolversApplication.sparse_lu"}
    alternative = settings("amgcl-gmres")
    assert alternative["solver_type"] == "amgcl"
    assert alternative["tolerance"] == 1e-10
    assert alternative["smoother_type"] == "ilut"
    assert alternative["scaling"]
    assert alternative["block_size"] == 1
    assert alternative["use_block_matrices_if_possible"] is False
    with pytest.raises(ValueError, match="Unsupported installed"):
        settings("candidate.module")


@pytest.mark.parametrize(
    "field,change", [("displacement_m", 1e-4), ("effective_stress_kPa", 1.0), ("node_ids", 1)]
)
def test_backend_equivalence_rejects_changed_native_fields(monkeypatch, field, change):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    compare = load_script("native_family_runner").compare_native_fields
    reference = {
        "displacement_m": np.array([[0.01, 0, -0.02]]),
        "effective_stress_kPa": np.array([[-100.0, -50.0, -80.0]]),
        "node_ids": np.array([17]),
    }
    actual = {name: values.copy() for name, values in reference.items()}
    assert compare(actual, reference)["passed"]
    actual[field].flat[0] += change
    assert not compare(actual, reference)["passed"]


def test_backend_displacement_budget_is_below_task_response_accuracy(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    compare = load_script("native_family_runner").compare_native_fields
    reference = {"displacement_m": np.zeros((1, 3))}
    result = compare({"displacement_m": np.full((1, 3), 5e-7)}, reference)
    assert result["passed"]
    assert result["fields"]["displacement_m"]["absolute_tolerance"] == pytest.approx(0.0003 / 300)
    assert not compare({"displacement_m": np.full((1, 3), 5e-6)}, reference)["passed"]
