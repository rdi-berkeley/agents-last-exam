import hashlib
import json
from pathlib import Path

import pytest

from tasks.life_sciences.amber_minimization_script_prep_instance_1.scripts import (
    verify_submission as minimization,
    workflow_parser,
)
from tasks.life_sciences.amber_three_stage_mmgbsa_workflow_instance_1.scripts import (
    verify_submission as mmgbsa,
)


RELEASE = Path(__file__).resolve().parents[3] / "release-staging/ale-task-fixes-20260913"


@pytest.fixture
def archived_bundle(request):
    task, solver = request.param
    checks = RELEASE / "validation-checks"
    evidence_path = checks / (
        "amber-min-r02-evidence.json" if task == "min" else "amber-three-r02-final-audit.json"
    )
    if not evidence_path.exists():
        pytest.skip("Amber r02 validation artifacts are not staged")
    evidence = json.loads(evidence_path.read_text())
    if task == "min":
        run = evidence["runs"][solver]
        selected, expected = run["selected"], run["output_files"]
    else:
        selected = evidence["selection"][
            "kimi_after" if solver == "final_kimi" else "codex_after_repair"
        ]
        expected = evidence["artifacts"]
        if solver == "astra":
            prepared = json.loads((checks / "amber-three-r02-audit.json").read_text())
            expected = prepared["artifact_hashes"]["astra_actual"]
    run_path = Path(selected["run_json"])
    assert hashlib.sha256(run_path.read_bytes()).hexdigest() == selected["sha256"]
    output = run_path.parent / "output"
    payloads = {name: (output / name).read_bytes() for name in expected}
    assert {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in payloads.items()
    } == expected
    bundle = {name: data.decode() for name, data in payloads.items()}
    yield task, solver, output, expected, bundle
    assert {name: (output / name).read_bytes() for name in expected} == payloads
    assert hashlib.sha256(run_path.read_bytes()).hexdigest() == selected["sha256"]


@pytest.mark.parametrize(
    "archived_bundle",
    [("min", "final_kimi"), ("min", "astra"), ("mmgbsa", "final_kimi"), ("mmgbsa", "astra")],
    indirect=True,
)
def test_actual_r02_artifacts(archived_bundle, monkeypatch):
    task, solver, output, hashes, bundle = archived_bundle
    native_run = workflow_parser.subprocess.run
    syntax_checks = []

    def syntax_only(argv, **kwargs):
        assert argv == ["/bin/bash", "--noprofile", "--norc", "-n"]
        assert kwargs["timeout"] == 5
        syntax_checks.append(argv)
        return native_run(argv, **kwargs)

    monkeypatch.setattr(workflow_parser.subprocess, "run", syntax_only)
    if task == "min":
        result = minimization.evaluate_output_bundle(bundle, present_files=sorted(bundle))
    else:
        reference = (
            RELEASE
            / "data-v1.1/files/life_sciences"
            / "amber_three_stage_mmgbsa_workflow_instance_1/base/reference/FINAL_RESULTS_MMGBSA.dat"
        ).read_text()
        result = mmgbsa.evaluate_output_bundle(
            bundle, present_files=sorted(bundle), hidden_reference_text=reference
        )
        assert result["delta_total"] == (-116.3475 if solver == "final_kimi" else -116.3459)
        assert result["hidden_delta_total"] == -116.3296
    print(
        json.dumps(
            {
                "task": task,
                "solver": solver,
                "output": str(output),
                "artifacts": hashes,
                "result": result,
                "bash_syntax_checks": len(syntax_checks),
                "submitted_commands_executed": False,
            },
            sort_keys=True,
        )
    )
    assert result["passed"], result
    assert result["score"] == 1.0


@pytest.mark.parametrize("archived_bundle", [("min", "final_kimi")], indirect=True)
@pytest.mark.parametrize(
    "filename,old,new,reason",
    [
        ("leap.in", "set default", "default", "unsupported LEaP command"),
        ("leap.in", "mbondi3", "mbondi2", "incorrect unit"),
        ("submit_min.sh", " || ", " && ", "tleap must build"),
        ("submit_min.sh", "    mv ", "    echo mv ", "products do not reach"),
        ("submit_min.sh", "tleap -s -f", "echo tleap -s -f", "conditional tleap build"),
    ],
)
def test_actual_minimization_negative_controls(archived_bundle, filename, old, new, reason):
    _, _, _, _, original = archived_bundle
    assert old in original[filename]
    bundle = original | {filename: original[filename].replace(old, new)}
    result = minimization.evaluate_output_bundle(bundle)
    print(json.dumps({"control": new, "file": filename, "result": result}, sort_keys=True))
    assert not result["passed"], result
    assert any(reason in item for item in result["reasons"]), result


@pytest.mark.parametrize("archived_bundle", [("mmgbsa", "final_kimi")], indirect=True)
@pytest.mark.parametrize(
    "filename,old,new,reason",
    [
        (
            "submit_mmgbsa.sh",
            '"${RUN_AMBERTOOLS}" MMPBSA.py -O',
            ': > "${TRAJECTORY}"\n"${RUN_AMBERTOOLS}" MMPBSA.py -O',
            "unmodified staged input",
        ),
        (
            "submit_mmgbsa.sh",
            '"${RUN_AMBERTOOLS}" MMPBSA.py -O',
            'rm /input/prod.mdcrd\n"${RUN_AMBERTOOLS}" MMPBSA.py -O',
            "unmodified staged input",
        ),
        (
            "submit_mmgbsa.sh",
            "receptor_mask=:1-299",
            "receptor_mask=:300-964",
            "incorrect MMGBSA receptor_mask",
        ),
        (
            "FINAL_RESULTS_MMGBSA.dat",
            "ligand_mask=:300-964",
            "ligand_mask=:301-964",
            "incorrect MMGBSA ligand_mask",
        ),
        ("submit_mmgbsa.sh", "igb=8,", "igb=5,", "MMGBSA igb must equal 8"),
        ("FINAL_RESULTS_MMGBSA.dat", "endframe=250", "endframe=249", "all 250 staged frames"),
    ],
)
def test_actual_mmgbsa_negative_controls(archived_bundle, filename, old, new, reason):
    _, _, _, _, original = archived_bundle
    assert old in original[filename]
    bundle = original | {filename: original[filename].replace(old, new)}
    result = mmgbsa.evaluate_output_bundle(bundle)
    print(json.dumps({"control": new, "file": filename, "result": result}, sort_keys=True))
    assert not result["passed"], result
    assert result["delta_total"] == -116.3475
    assert any(reason in item for item in result["reasons"]), result
