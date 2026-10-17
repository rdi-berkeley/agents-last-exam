import ast
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tasks.engineering.inner_support_elevation_optimization import native_task
from tasks.engineering.inner_support_elevation_optimization.scripts import (
    native_result_adapter as private,
    native_result_reader as reader,
)


@pytest.fixture
def native_capture(tmp_path):
    coordinates = np.array([[0.123456789, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]])
    translations = np.array([[0.0123456789, -0.00123456789, -0.0987654321]] * 4)
    lines = ["Begin Properties 1", "YOUNG_MODULUS 1000", "End Properties", "Begin Nodes"]
    for identifier, point in enumerate(coordinates, 1):
        lines.append(f"{identifier} " + " ".join(format(value, ".6g") for value in point))
    lines.extend(
        [
            "End Nodes",
            "Begin Elements UPwSmallStrainElement3D4N",
            "1 1 1 2 3 4",
            "End Elements",
        ]
    )
    for index, component in enumerate(reader.COMPONENTS):
        lines.append(f"Begin NodalData {component}")
        lines.extend(
            f"{identifier} 0 {row[index]:.6g}" for identifier, row in enumerate(translations, 1)
        )
        lines.append("End NodalData")
    lines.extend(
        [
            "Begin Constraints LinearMasterSlaveConstraint DISPLACEMENT_X DISPLACEMENT_X",
            "71 9 [1] 2 1",
            "End Constraints",
            "Begin NodalData ROTATION_Z",
            "2 0 0.004",
            "End NodalData",
            "Begin Constraints LinearMasterSlaveConstraint ROTATION_Z ROTATION_Z ROTATION_Z",
            "72 -0.003 [0.25,0.75] 2 1 3",
            "End Constraints",
            "Begin Constraints LinearMasterSlaveConstraint DISPLACEMENT_Y DISPLACEMENT_Y ROTATION_Z",
            "73 0.006 [1,-0.5] 2 1 3",
            "End Constraints",
            "Begin SubModelPart Active",
            "Begin SubModelPartElements",
            "1",
            "End SubModelPartElements",
            "Begin SubModelPartConstraints",
            "71",
            "72",
            "73",
            "End SubModelPartConstraints",
            "End SubModelPart",
            "Begin SubModelPart Soil",
            "Begin SubModelPartNodes",
            "1",
            "2",
            "3",
            "4",
            "End SubModelPartNodes",
            "End SubModelPart",
        ]
    )
    mdpa = tmp_path / "native_model.mdpa"
    mdpa.write_text("\n".join(lines) + "\n")
    capture = tmp_path / "native_nodal_fields.tsv"
    capture.write_text(
        "node\tx0_m\ty0_m\tz0_m\tux_m\tuy_m\tuz_m\n"
        + "".join(
            "\t".join(map(str, [identifier, *point, *values])) + "\n"
            for identifier, (point, values) in enumerate(zip(coordinates, translations), 1)
        )
    )
    baseline = tmp_path / "stage1.npz"
    np.savez(baseline, node_ids=np.arange(1, 5), displacement_m=np.full((4, 3), 0.02))
    return mdpa, capture, baseline, coordinates, translations


@pytest.fixture
async def public_payload(tmp_path, monkeypatch):
    software = tmp_path / "software"
    scripts = software / "scripts"
    scripts.mkdir(parents=True)
    for name in native_task.NATIVE_SCRIPTS:
        (scripts / name).write_bytes((native_task.ROOT / "scripts" / name).read_bytes())
    (scripts / "agent_model.py").write_text("solver_work = True\n")
    uploaded = {}
    verified = []

    class Session:
        async def read_bytes(self, path):
            return Path(path).name.encode()

    async def upload(session, root, files):
        uploaded[root] = files
        for name, content in files.items():
            destination = Path(root) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

    async def command(session, command, **kwargs):
        arguments = shlex.split(command)
        if arguments[0] != "env":
            subprocess.run(arguments, check=True, capture_output=True)
        return {"return_code": 0}

    monkeypatch.setattr(native_task, "upload_files", upload)
    monkeypatch.setattr(native_task, "run_command", command)
    monkeypatch.setattr(native_task, "verify_source_bundle", verified.append)
    await native_task.setup_native(
        {
            "input_dir": str(tmp_path / "input"),
            "software_dir": str(software),
            "remote_output_dir": str(tmp_path / "output"),
        },
        Session(),
    )
    assert len(verified) == 1
    assert set(uploaded[str(software)]) == {
        *(f"scripts/{name}" for name in native_task.PUBLIC_NATIVE_SCRIPTS),
        "assets/benchmark.json",
        "assets/source_manifest.json",
    }
    return scripts


async def test_public_setup_replaces_old_extractor_and_imports_without_private_code(public_payload):
    assert {path.name for path in public_payload.iterdir()} == {
        "native_input_adapter.py",
        "native_family_runner.py",
        "native_member_sections.py",
        "native_result_adapter.py",
        "agent_model.py",
    }
    assert (public_payload / "agent_model.py").read_text() == "solver_work = True\n"
    assert (public_payload / "native_result_adapter.py").read_bytes() == Path(
        reader.__file__
    ).read_bytes()
    code = """
import importlib, inspect, json, sys
sys.path.insert(0, sys.argv[1])
for name in ('native_input_adapter', 'native_family_runner', 'native_member_sections'):
    importlib.import_module(name)
import native_result_adapter as reader
print(json.dumps(sorted(name for name, value in vars(reader).items()
                        if inspect.isfunction(value) and value.__module__ == reader.__name__)))
assert not any(name in sys.modules for name in (
    'native_model_contract', 'native_source_audit', 'native_replay_case',
    'native_replay_family', 'native_verify_delivery', 'native_result_reader'))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(public_payload)],
        cwd=public_payload,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == sorted(
        [
            "trusted_bytes",
            "read_native_result",
            "read_native_coordinates",
            "read_native_translations",
            "read_stage1_baseline",
        ]
    )


async def test_public_reader_keeps_raw_offsets_baseline_and_fields_without_writing_answers(
    public_payload, native_capture, monkeypatch
):
    specification = importlib.util.spec_from_file_location(
        "support_public_reader_test", public_payload / "native_result_adapter.py"
    )
    public = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, specification.name, public)
    specification.loader.exec_module(public)
    mdpa, capture, baseline, coordinates, translations = native_capture
    before = {path.name: path.read_bytes() for path in mdpa.parent.iterdir() if path.is_file()}
    result = public.read_native_result(
        mdpa,
        expected_sha256=hashlib.sha256(mdpa.read_bytes()).hexdigest(),
        coordinates_path=capture,
        coordinates_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        precise_translations=True,
    )
    stage1 = public.read_stage1_baseline(
        baseline,
        expected_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        soil_nodes=result.soil_nodes,
    )
    np.testing.assert_array_equal(list(result.nodes.values()), coordinates)
    np.testing.assert_array_equal(
        [result.displacement(node) for node in result.nodes], translations
    )
    np.testing.assert_array_equal(list(stage1.values()), np.full((4, 3), 0.02))
    assert result.constraints[2, "DISPLACEMENT_X"] == (71, 9.0, (1.0,), (1,))
    assert result.constraints[2, "DISPLACEMENT_Y"] == (73, 0.006, (1.0, -0.5), (1, 3))
    assert result.rotation_constraints[2, "ROTATION_Z"] == (72, -0.003, (0.25, 0.75), (1, 3))
    assert result.constraint_variables == {
        (2, "DISPLACEMENT_X"): ("DISPLACEMENT_X",),
        (2, "DISPLACEMENT_Y"): ("DISPLACEMENT_Y", "ROTATION_Z"),
        (2, "ROTATION_Z"): ("ROTATION_Z", "ROTATION_Z"),
    }
    assert result.rotations == {"ROTATION_X": {}, "ROTATION_Y": {}, "ROTATION_Z": {2: 0.004}}
    assert result.elements == {1: ("UPwSmallStrainElement3D4N", 1, (1, 2, 3, 4))}
    assert result.active == {1}
    assert result.active_constraints == {71, 72, 73}
    assert result.properties == {1: {"YOUNG_MODULUS": 1000.0}}
    assert set(vars(result)) == {
        "nodes",
        "elements",
        "fields",
        "constraints",
        "rotation_constraints",
        "constraint_variables",
        "active",
        "active_constraints",
        "soil_nodes",
        "rotations",
        "properties",
    }
    for name in (
        "extract_monitors",
        "check_reported_monitors",
        "wall_vertex",
        "beam_physical_point",
        "GEOMETRY_ATOL_M",
        "FIELD_ATOL_M",
        "FIELD_RTOL",
        "reference_answer",
        "score_delivery",
    ):
        assert not hasattr(public, name)
    assert {
        path.name: path.read_bytes() for path in mdpa.parent.iterdir() if path.is_file()
    } == before


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda text: text.replace("DISPLACEMENT_Z", "UNAVAILABLE_Z"),
            "Missing native displacement",
        ),
        (lambda text: text.replace("Begin Nodes\n", "Begin Nodes\n1 0 0 0\n"), "Duplicate"),
        (lambda text: text.replace("0.0123457", "nan"), "nonfinite"),
    ],
)
def test_public_reader_rejects_incomplete_duplicate_or_nonfinite_fields(
    native_capture, mutation, match
):
    mdpa = native_capture[0]
    mdpa.write_text(mutation(mdpa.read_text()))
    with pytest.raises(ValueError, match=match):
        reader.read_native_result(
            mdpa, expected_sha256=hashlib.sha256(mdpa.read_bytes()).hexdigest()
        )


def test_public_reader_requires_matching_digests_and_does_not_invent_precision(native_capture):
    mdpa, capture, *_ = native_capture
    with pytest.raises(ValueError, match="digest mismatch"):
        reader.read_native_result(mdpa, expected_sha256="0" * 64)
    digest = hashlib.sha256(mdpa.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Precise translations require"):
        reader.read_native_result(mdpa, expected_sha256=digest, precise_translations=True)
    with pytest.raises(ValueError, match="digest mismatch"):
        reader.read_native_result(mdpa, expected_sha256=digest, coordinates_path=capture)


def test_private_extraction_reuses_reader_and_private_payload_is_dependency_complete(tmp_path):
    for name in ("NativeResult", "read_native_result", "read_stage1_baseline", "trusted_bytes"):
        assert getattr(private, name) is getattr(reader, name)
    assert callable(private.extract_monitors)
    for name in native_task.NATIVE_SCRIPTS:
        (tmp_path / name).write_bytes((native_task.ROOT / "scripts" / name).read_bytes())
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import native_result_adapter, native_result_reader, native_replay_case, native_verify_delivery
assert native_result_adapter.read_native_result is native_result_reader.read_native_result
assert native_replay_case.extract_monitors is native_result_adapter.extract_monitors
assert native_verify_delivery.extract_monitors is native_result_adapter.extract_monitors
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path)], check=True, capture_output=True
    )


def test_reader_has_no_task_specific_answer_or_grading_dependencies():
    tree = ast.parse(Path(reader.__file__).read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports == {"csv", "hashlib", "io", "re", "dataclasses", "pathlib", "numpy"}
