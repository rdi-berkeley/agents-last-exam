import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "tasks/life_sciences/idp_ensemble_scoring/scripts"
COLUMNS = ["Method", "Total", "CS", "JC", "NOE/PRE"]
RESTORED_SHA256 = "db941838f4587e8bfeddfaad062c8e6d6e094c6632b844207451736eba2a1e80"
ORIGINAL_SHA256 = "b5b0c4167d7cee57fed673b6be962257a0cf9e0e856291c080a1b1dac719bd19"
RELATIVE_PDB = Path("Ensembles/Model4/asyn/150_validated.pdb")


@pytest.fixture
def rows():
    return [
        {
            "Method": f"Model{index}",
            "Total": str(index / 10),
            "CS": "0.1234",
            "JC": "0.2345",
            "NOE/PRE": "0.4567",
        }
        for index in range(1, 6)
    ]


def csv_text(rows, columns=COLUMNS, **kwargs):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, **kwargs)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def grade(tmp_path, reference, submission):
    reference_path = tmp_path / "reference.csv"
    output_path = tmp_path / "Final_Output.csv"
    reference_path.write_text(reference)
    output_path.write_text(submission)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "verify_output.py"),
            "--reference-file",
            str(reference_path),
            "--output-file",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "representation", ["quoted", "scientific", "bom", "reordered", "whitespace"]
)
def test_equivalent_csv(tmp_path, rows, representation):
    reference = csv_text(rows)
    columns = COLUMNS
    options = {}
    if representation == "quoted":
        options["quoting"] = csv.QUOTE_ALL
    elif representation == "scientific":
        for row in rows:
            for column in COLUMNS[1:]:
                row[column] = f"{float(row[column]):.10e}"
    elif representation == "reordered":
        rows.reverse()
        columns = list(reversed(COLUMNS))
    elif representation == "whitespace":
        for row in rows:
            for column in COLUMNS:
                row[column] = " " + row[column] + " "
    submission = csv_text(rows, columns, **options)
    if representation == "bom":
        submission = "\ufeff" + submission
    assert grade(tmp_path, reference, submission) == {"score": 1.0, "passed": True, "reasons": []}


@pytest.mark.parametrize("model", range(5))
@pytest.mark.parametrize("column", COLUMNS[1:])
def test_each_wrong_cell_loses_credit(tmp_path, rows, model, column):
    reference = csv_text(rows)
    rows[model][column] = "0.98"
    result = grade(tmp_path, reference, csv_text(rows))
    assert result["score"] == 0.95
    assert not result["passed"]


@pytest.mark.parametrize(
    "value", ["NaN", "nan", "+inf", "-Infinity", "1.00001", "-0.00001", "", "true", "1/2", "1e9999"]
)
def test_invalid_numeric_values_are_hard_gates(tmp_path, rows, value):
    reference = csv_text(rows)
    rows[0]["CS"] = value
    result = grade(tmp_path, reference, csv_text(rows))
    assert result["score"] == 0
    assert result["reasons"]


@pytest.mark.parametrize(
    "malformation",
    [
        "duplicate_model",
        "extra_model",
        "missing_model",
        "duplicate_column",
        "missing_column",
        "extra_column",
        "forbidden_column",
        "short_row",
        "long_row",
        "unterminated_quote",
        "empty",
    ],
)
def test_malformed_csv_is_rejected_without_crashing(tmp_path, rows, malformation):
    reference = csv_text(rows)
    columns = COLUMNS
    if malformation == "duplicate_model":
        rows.append(rows[0].copy())
    elif malformation == "extra_model":
        rows.append(dict(rows[0], Method="Model6"))
    elif malformation == "missing_model":
        rows.pop()
    elif malformation == "duplicate_column":
        columns = COLUMNS + ["CS"]
    elif malformation == "missing_column":
        columns = COLUMNS[:-1]
        rows = [{key: value for key, value in row.items() if key in columns} for row in rows]
    elif malformation in ("extra_column", "forbidden_column"):
        extra = "SAXS" if malformation == "forbidden_column" else "Unspecified"
        columns = COLUMNS + [extra]
        rows = [dict(row, **{extra: "0.5"}) for row in rows]
    submission = csv_text(rows, columns)
    if malformation == "short_row":
        submission = submission.replace("Model1,0.1,0.1234,0.2345,0.4567", "Model1,0.1")
    elif malformation == "long_row":
        submission = submission.replace("Model1,0.1", "Model1,0.1,0.1")
    elif malformation == "unterminated_quote":
        submission += '"unterminated'
    elif malformation == "empty":
        submission = ""
    result = grade(tmp_path, reference, submission)
    assert result["score"] == 0
    assert not result["passed"]


def test_two_decimal_rule_is_not_widened(tmp_path, rows):
    reference = csv_text(rows)
    rows[0]["CS"] = "0.12499"
    assert grade(tmp_path, reference, csv_text(rows))["score"] == 1
    rows[0]["CS"] = "0.12501"
    assert grade(tmp_path, reference, csv_text(rows))["score"] == 0.95


@pytest.fixture
def audit_module():
    pytest.importorskip("Bio")
    spec = importlib.util.spec_from_file_location("idp_audit_inputs", SCRIPTS / "audit_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def restored_input():
    value = os.environ.get("IDP_AUDIT_OVERLAY_ROOT")
    if not value:
        pytest.skip("set IDP_AUDIT_OVERLAY_ROOT for authentic artifact regression")
    return Path(value) / RELATIVE_PDB


def test_authentic_recovery_integrity(audit_module, restored_input):
    assert hashlib.sha256(restored_input.read_bytes()).hexdigest() == RESTORED_SHA256
    signature = audit_module.structure_signature(restored_input)
    assert len(signature) == 140
    assert sum(len(residue[2]) for residue in signature) == 2016


@pytest.mark.parametrize("count", [199, 201])
def test_incomplete_or_oversized_pool_is_reported(tmp_path, audit_module, restored_input, count):
    (tmp_path / "info.py").write_text(
        "PROTEINS_TO_TEST = {'CS': ['asyn'], 'JC': [], 'NOE': [], 'PRE': []}\n"
    )
    pool = tmp_path / "Ensembles/Model1/asyn"
    pool.mkdir(parents=True)
    for index in range(count):
        (pool / f"{index}.pdb").symlink_to(restored_input)
    report = audit_module.audit(tmp_path)
    assert [
        "Ensembles/Model1/asyn",
        f"expected 200 conformers, found {count}",
    ] in report["issues"]


@pytest.mark.parametrize(
    "damage", ["html", "empty", "truncated", "nan", "missing_backbone", "duplicate_atom"]
)
def test_structural_corruption_rejected(tmp_path, audit_module, restored_input, damage):
    lines = restored_input.read_text().splitlines(keepends=True)
    atom_index = next(index for index, line in enumerate(lines) if line.startswith("ATOM  "))
    if damage == "html":
        value = "<!DOCTYPE html><html>download warning</html>"
    elif damage == "empty":
        value = "END\n"
    elif damage == "truncated":
        value = "".join(lines[: atom_index + 1])
    elif damage == "nan":
        lines[atom_index] = lines[atom_index][:30] + "     nan" + lines[atom_index][38:]
        value = "".join(lines)
    elif damage == "missing_backbone":
        del lines[atom_index]
        value = "".join(lines)
    else:
        lines.insert(atom_index, lines[atom_index])
        value = "".join(lines)
    path = tmp_path / "bad.pdb"
    path.write_text(value)
    with pytest.raises(Exception):
        audit_module.structure_signature(path)


def test_original_html_is_preserved_and_rejected(audit_module):
    value = os.environ.get("IDP_AUDIT_ORIGINAL_ROOT")
    if not value:
        pytest.skip("set IDP_AUDIT_ORIGINAL_ROOT for original artifact regression")
    path = Path(value) / RELATIVE_PDB
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_SHA256
    with pytest.raises(ValueError, match="nonempty"):
        audit_module.structure_signature(path)


def test_full_inventory_evidence():
    value = os.environ.get("IDP_AUDIT_INVENTORY_REPORT")
    if not value:
        pytest.skip("set IDP_AUDIT_INVENTORY_REPORT after running the full structural audit")
    report = json.loads(Path(value).read_text())
    assert report["ensembles"] == 95
    assert report["conformers"] == 19000
    assert len(report["proteins"]) == 19
    assert report["issues"] == []


@pytest.fixture
def runtime_evidence():
    value = os.environ.get("IDP_AUDIT_RUNTIME_EVIDENCE")
    if not value:
        pytest.skip("set IDP_AUDIT_RUNTIME_EVIDENCE for actual guest pipeline evidence")
    return Path(value)


@pytest.fixture
def actual_input():
    value = os.environ.get("IDP_AUDIT_INPUT_ROOT")
    if not value:
        pytest.skip("set IDP_AUDIT_INPUT_ROOT for actual data regressions")
    return Path(value)


def test_actual_reference_and_equivalent_format(tmp_path, actual_input):
    reference = (actual_input.parent / "reference/Expected_Final_Output.csv").read_text()
    rows = list(csv.DictReader(io.StringIO(reference)))
    for row in rows:
        for column in COLUMNS[1:]:
            row[column] = f"{float(row[column]):.12e}"
    submission = csv_text(list(reversed(rows)), quoting=csv.QUOTE_ALL)
    assert grade(tmp_path, reference, submission)["score"] == 1


def test_actual_restored_pinned_inference(runtime_evidence, actual_input):
    root = runtime_evidence / "restored-cli"
    runtime = json.loads((root / "runtime.json").read_text())
    assert runtime["return_code"] == 0
    assert runtime["pdb_sha256"] == RESTORED_SHA256
    assert runtime["packages"]["scikit-learn"] == "0.22"
    assert runtime["packages"]["biopython"] == "1.74"
    assert runtime["packages"]["numpy"] == "1.19.0"
    assert runtime["packages"]["pandas"] == "1.1.0"
    manifest = json.loads((runtime_evidence.parent / "runtime-source-hashes.json").read_text())
    assert manifest["host"] == manifest["guest"]
    assert manifest["differences"] == {}
    assert sum(key.endswith(".sav") for key in manifest["host"]) == 18
    with (root / "shifts.csv").open() as stream:
        shifts = {int(row["RESNUM"]): row for row in csv.DictReader(stream)}
    assert len(shifts) == 140
    with (actual_input / "Experimental_Data/asyn/asyn_cs_exp.txt").open() as stream:
        experiment = list(csv.DictReader(stream))
    assert len(experiment) == 518
    for row in experiment:
        assert math.isfinite(float(shifts[int(row["resnum"])][row["atomname"] + "_X"]))


def test_actual_html_fails_same_pinned_inference(runtime_evidence):
    root = runtime_evidence / "damaged-cli"
    runtime = json.loads((root / "runtime.json").read_text())
    assert runtime["return_code"] != 0
    assert runtime["pdb_sha256"] == ORIGINAL_SHA256
    assert not (root / "shifts.csv").exists()
    assert "0 sample(s)" in (root / "inference.log").read_text()


def test_historical_structural_scores_disagree_with_legacy_reference(runtime_evidence):
    value = os.environ.get("IDP_AUDIT_LEGACY_REFERENCE")
    if not value:
        pytest.skip("set IDP_AUDIT_LEGACY_REFERENCE to the preserved legacy CSV")
    reference_path = Path(value)
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == (
        "32d8f5dfc6e70d2f9c141d7d465d0fe5c63900f5a173d8a42842fe65189acb0a"
    )
    with reference_path.open() as stream:
        reference = {row["Method"]: row for row in csv.DictReader(stream)}
    with (runtime_evidence / "structural/normalized.csv").open() as stream:
        calculated = list(csv.DictReader(stream))
    assert json.loads((runtime_evidence / "structural/failures.json").read_text()) == []
    assert len(list((runtime_evidence / "structural").glob("*.npy"))) == 55
    mismatches = {
        (row["Method"], column)
        for row in calculated
        for column in ("JC", "NOE/PRE")
        if round(float(row[column]), 2) != round(float(reference[row["Method"]][column]), 2)
    }
    assert mismatches == {
        ("Model1", "JC"),
        ("Model4", "JC"),
        *((f"Model{index}", "NOE/PRE") for index in range(1, 6)),
    }
