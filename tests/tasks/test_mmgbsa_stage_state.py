import hashlib
from pathlib import Path

import pytest

from tasks.life_sciences.amber_three_stage_mmgbsa_workflow_instance_1.scripts import (
    workflow_checks as checks,
)
from tasks.life_sciences.amber_three_stage_mmgbsa_workflow_instance_1.scripts.verify_submission import (
    evaluate_output_bundle,
)


NAME = "GLN_phb2_parl_pgam5_model_0"
HEADER = f"""#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
NAME={NAME}
ROOT=/script
PARAMS="$ROOT/params"
"""
BUILD = """mkdir -p "$PARAMS"
if [ ! -s "$PARAMS/$NAME.prmtop" ] || [ ! -s "$PARAMS/$NAME.inpcrd" ]; then
cat > build.in <<EOF
source leaprc.protein.ff14SB
set default PBRadii mbondi3
complex = loadpdb ../input/complex_structure.pdb
saveamberparm complex $PARAMS/$NAME.prmtop $PARAMS/$NAME.inpcrd
quit
EOF
tleap -f build.in
fi
test -s "$PARAMS/$NAME.prmtop"
test -s "$PARAMS/$NAME.inpcrd"
ante-MMPBSA.py -p "$PARAMS/$NAME.prmtop" -r "$PARAMS/${NAME}_A.prmtop" -l "$PARAMS/${NAME}_BC.prmtop" -m :1-299 --radii mbondi3
"""
CONTROL = """&general
 startframe=1, endframe=250, interval=1, entropy=0,
/
&gb
 igb=8, saltcon=0.150,
/
"""
MMGBSA = (
    HEADER
    + """test -s "$PARAMS/$NAME.prmtop"
test -s "$PARAMS/${NAME}_A.prmtop"
test -s "$PARAMS/${NAME}_BC.prmtop"
test -s ../input/prod.mdcrd
cat > score.in <<'EOF'
"""
    + CONTROL
    + """EOF
MMPBSA.py -O -i score.in -o FINAL_RESULTS_MMGBSA.dat -cp "$PARAMS/$NAME.prmtop" -rp "$PARAMS/${NAME}_A.prmtop" -lp "$PARAMS/${NAME}_BC.prmtop" -y ../input/prod.mdcrd
"""
)
REPORT = (
    "\n".join("|" + line for line in CONTROL.splitlines())
    + f"""
|Complex topology file: {NAME}.prmtop
|Receptor topology file: {NAME}_A.prmtop
|Ligand topology file: {NAME}_BC.prmtop
|Receptor mask: ":1-299"
|Ligand mask: ":300-964"
|Calculations performed using 250 complex frames.
GENERALIZED BORN:
DELTA G gas 1300.0000 60.0000 3.7947
DELTA G solv -1420.0000 59.0000 3.7315
DELTA TOTAL -120.0000 18.0000 1.1384
"""
)


def md_stage(label, mode, coordinates, restart, *, weights=""):
    length = "maxcyc=4000" if mode == 1 else "nstlim=20000"
    return f"""cat > {label}.in <<'EOF'
{label}
&cntrl
 imin={mode}, ntb=0, igb=8, {length}, dt=0.002,
/
{weights}EOF
pmemd.cuda -O -i {label}.in -p "$PARAMS/$NAME.prmtop" -c {coordinates} -o {label}.out -r {restart}
"""


def bundle_for_modes(modes=(1, 1, 0, 0)):
    stages = []
    coordinates = '"$PARAMS/$NAME.inpcrd"'
    for index, mode in enumerate(modes):
        label = f"stage{index}"
        restart = f"{label}.rst"
        stages.append(md_stage(label, mode, coordinates, restart))
        coordinates = restart
    return {
        "submit_min.sh": HEADER + BUILD + "".join(stages),
        "submit_prod.sh": HEADER
        + f'test -s "$PARAMS/$NAME.prmtop"\ntest -s {coordinates}\n'
        + md_stage("prod", 0, coordinates, "prod.rst").replace(
            "-r prod.rst", "-r prod.rst -x prod.mdcrd"
        ),
        "submit_mmgbsa.sh": MMGBSA,
        "FINAL_RESULTS_MMGBSA.dat": REPORT,
    }


@pytest.fixture
def bundle():
    return bundle_for_modes()


@pytest.mark.parametrize("property_name", ["PBRadii", "PBradii"])
def test_native_radii_property_spellings(bundle, property_name):
    bundle["submit_min.sh"] = bundle["submit_min.sh"].replace("PBRadii", property_name)
    result = evaluate_output_bundle(bundle)
    assert result["passed"], result


@pytest.mark.parametrize(
    "setting",
    [
        "set default PBradii mbondi2",
        "set default PBradii MBONDI3",
        "set default PBradii",
        "set default PBradii mbondi3 extra",
        "set default PBradiii mbondi3",
        "# set default PBradii mbondi3",
        "set default PBradii mbondi3\nset default PBRadii mbondi2",
        "set default PBRadii mbondi3\nset default PBradii mbondi2",
    ],
)
def test_radii_spelling_fix_preserves_required_values(bundle, setting):
    bundle["submit_min.sh"] = bundle["submit_min.sh"].replace("set default PBRadii mbondi3", setting)
    result = evaluate_output_bundle(bundle)
    assert not result["passed"], result


@pytest.mark.parametrize("modes", [(1, 0), (1, 1, 0), (1, 1, 1, 0, 0), (1, 0, 0, 0)])
def test_multiple_minimizations_before_preparation(modes):
    result = evaluate_output_bundle(bundle_for_modes(modes))
    assert result["passed"], result


@pytest.mark.parametrize("modes", [(0, 0), (1, 1), (1, 0, 1), (1, 1, 0, 1), (5, 0)])
def test_invalid_preparation_phases(modes):
    result = evaluate_output_bundle(bundle_for_modes(modes))
    assert not result["passed"], result


@pytest.mark.parametrize("guard", ["test -s", "test -f", "test -e"])
@pytest.mark.parametrize(
    "trajectory",
    [
        "../input/prod.mdcrd",
        "//input/prod.mdcrd",
        "/script/input/prod.mdcrd",
        "/media/user/data/agenthle/life_sciences/amber_three_stage_mmgbsa_workflow_instance_1/base/input/prod.mdcrd",
        "prod.mdcrd",
    ],
)
def test_staged_trajectory_guards_have_provenance(bundle, guard, trajectory):
    bundle["submit_mmgbsa.sh"] = MMGBSA.replace("../input/prod.mdcrd", trajectory).replace(
        "test -s", guard
    )
    result = evaluate_output_bundle(bundle)
    assert result["passed"], result


def test_script_parent_input_resolution_retains_staged_provenance(bundle):
    setup = """SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
BASE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
INPUT_DIR="${BASE_DIR}/input"
"""
    bundle["submit_mmgbsa.sh"] = MMGBSA.replace(
        "set -euo pipefail\n", "set -euo pipefail\n" + setup
    ).replace("../input/prod.mdcrd", '"$INPUT_DIR/prod.mdcrd"')
    result = evaluate_output_bundle(bundle)
    assert result["passed"], result
    shells, errors, _ = checks._resolve_stages(
        tuple(bundle[name] for name in ("submit_min.sh", "submit_prod.sh", "submit_mmgbsa.sh"))
    )
    assert not errors
    command = next(
        command for command in shells["submit_mmgbsa.sh"].commands if command.argv[0] == "MMPBSA.py"
    )
    trajectory = checks.path(checks.options(command.argv)["-y"], command.cwd)
    assert trajectory == "/input/prod.mdcrd"
    assert command.products[trajectory] == checks.STAGED_TRAJECTORY


@pytest.mark.parametrize("stage", ["submit_min.sh", "submit_prod.sh", "submit_mmgbsa.sh"])
@pytest.mark.parametrize("damaged,consumed", [("/input", "//input"), ("//input", "/input")])
@pytest.mark.parametrize("damage", ["rm", ": >", "echo fake >", "cp missing.nc"])
def test_equivalent_root_paths_share_damage_provenance(bundle, stage, damaged, consumed, damage):
    bundle["submit_mmgbsa.sh"] = MMGBSA.replace("../input/prod.mdcrd", consumed + "/prod.mdcrd")
    bundle[stage] = bundle[stage].replace(
        "set -euo pipefail\n", f"set -euo pipefail\n{damage} {damaged}/prod.mdcrd\n"
    )
    result = evaluate_output_bundle(bundle)
    assert not result["passed"], result


@pytest.mark.parametrize("transport", ["cp", "mv"])
def test_double_root_staged_trajectory_transport(bundle, transport):
    bundle["submit_prod.sh"] += f"mkdir -p samples\n{transport} //input/prod.mdcrd samples/\n"
    bundle["submit_mmgbsa.sh"] = MMGBSA.replace("../input/prod.mdcrd", "samples/prod.mdcrd")
    result = evaluate_output_bundle(bundle)
    assert result["passed"], result


@pytest.mark.parametrize("transport", ["cp", "mv"])
@pytest.mark.parametrize("stage", ["submit_min.sh", "submit_prod.sh", "submit_mmgbsa.sh"])
def test_staged_trajectory_copy_move_between_stages(bundle, transport, stage):
    transfer = f"mkdir -p samples\n{transport} ../input/prod.mdcrd samples/\n"
    bundle[stage] = bundle[stage].replace("set -euo pipefail\n", "set -euo pipefail\n" + transfer)
    bundle["submit_mmgbsa.sh"] = (
        bundle["submit_mmgbsa.sh"]
        .replace("test -s ../input/prod.mdcrd", "test -s samples/prod.mdcrd")
        .replace("-y ../input/prod.mdcrd", "-y samples/prod.mdcrd")
    )
    result = evaluate_output_bundle(bundle)
    assert result["passed"], result


@pytest.mark.parametrize("stage", ["submit_min.sh", "submit_prod.sh", "submit_mmgbsa.sh"])
@pytest.mark.parametrize(
    "damage",
    [
        "rm ../input/prod.mdcrd",
        ": > ../input/prod.mdcrd",
        "echo fake > ../input/prod.mdcrd",
        "cp missing.nc ../input/prod.mdcrd",
    ],
)
def test_staged_input_damage_is_not_restaged(bundle, stage, damage):
    bundle[stage] = bundle[stage].replace(
        "set -euo pipefail\n", "set -euo pipefail\n" + damage + "\n"
    )
    result = evaluate_output_bundle(bundle)
    assert not result["passed"], result


@pytest.mark.parametrize("damage", ["rm", "empty", "wrong_kind"])
@pytest.mark.parametrize("artifact", [f"params/{NAME}.prmtop", "stage3.rst"])
def test_prior_stage_products_must_survive_until_consumed(bundle, damage, artifact):
    command = {
        "rm": f"rm {artifact}",
        "empty": f": > {artifact}",
        "wrong_kind": f"cp ../input/prod.mdcrd {artifact}",
    }[damage]
    bundle["submit_min.sh"] += command + "\n"
    result = evaluate_output_bundle(bundle)
    assert not result["passed"], result


@pytest.mark.parametrize("artifact", [f"params/{NAME}.prmtop", "stage3.rst"])
def test_missing_products_fail_even_without_existence_guard(bundle, artifact):
    bundle["submit_min.sh"] += f"rm {artifact}\n"
    bundle["submit_prod.sh"] = (
        "\n".join(
            line for line in bundle["submit_prod.sh"].splitlines() if not line.startswith("test ")
        )
        + "\n"
    )
    result = evaluate_output_bundle(bundle)
    assert not result["passed"], result


def test_same_basename_in_different_directory_is_not_restart_handoff(bundle):
    bundle["submit_prod.sh"] = bundle["submit_prod.sh"].replace(
        "-c stage3.rst", "-c wrong/stage3.rst"
    )
    assert not evaluate_output_bundle(bundle)["passed"]


def test_production_cannot_replace_explicit_staged_trajectory(bundle):
    bundle["submit_prod.sh"] = bundle["submit_prod.sh"].replace(
        "-x prod.mdcrd", "-x ../input/prod.mdcrd"
    )
    result = evaluate_output_bundle(bundle)
    assert not result["passed"], result
    assert any("unmodified staged input" in reason for reason in result["reasons"])


def test_production_trajectory_copy_is_not_staged_provenance(bundle):
    bundle["submit_prod.sh"] += "cp prod.mdcrd production_copy.nc\n"
    bundle["submit_mmgbsa.sh"] = MMGBSA.replace("../input/prod.mdcrd", "production_copy.nc")
    assert not evaluate_output_bundle(bundle)["passed"]


@pytest.mark.parametrize("change", ["maxcyc=0", "maxcyc=-1", "maxcyc=1.5", "maxcyc='4000'"])
def test_each_minimization_length_is_validated(bundle, change):
    marker = "stage1\n&cntrl\n imin=1, ntb=0, igb=8, maxcyc=4000"
    assert marker in bundle["submit_min.sh"]
    bundle["submit_min.sh"] = bundle["submit_min.sh"].replace(
        marker, marker.replace("maxcyc=4000", change)
    )
    assert not evaluate_output_bundle(bundle)["passed"]


def test_production_cannot_be_minimization(bundle):
    bundle["submit_prod.sh"] = bundle["submit_prod.sh"].replace("imin=0", "imin=1")
    assert not evaluate_output_bundle(bundle)["passed"]


def test_native_heating_weight_records(bundle):
    weights = (
        "&wt TYPE='TEMP0', istep1=0, istep2=20000, value1=0., value2=300., /\n&wt TYPE='END', /\n"
    )
    original = md_stage("stage2", 0, "stage1.rst", "stage2.rst")
    heated = md_stage("stage2", 0, "stage1.rst", "stage2.rst", weights=weights).replace(
        "imin=0", "nmropt=1, imin=0"
    )
    bundle["submit_min.sh"] = bundle["submit_min.sh"].replace(original, heated)
    result = evaluate_output_bundle(bundle)
    assert result["passed"], result


def test_report_reuses_stage_resolution(bundle, monkeypatch):
    checks._resolve_stages.cache_clear()
    original = checks._StageShell
    constructions = []

    def counted(*args, **kwargs):
        constructions.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(checks, "_StageShell", counted)
    result = evaluate_output_bundle(bundle)
    assert result["passed"], result
    assert len(constructions) == 3


@pytest.mark.parametrize("label", ["Complex", "Receptor", "Ligand"])
def test_report_headers_checked_after_staged_input_guard(bundle, label):
    bundle["FINAL_RESULTS_MMGBSA.dat"] = REPORT.replace(
        f"|{label} topology file: ", f"|{label} topology file: wrong_"
    )
    result = evaluate_output_bundle(bundle)
    assert not result["passed"], result
    assert any("topology differs from command input" in reason for reason in result["reasons"])


def test_cache_does_not_reuse_trace_after_script_change(bundle):
    assert evaluate_output_bundle(bundle)["passed"]
    bundle["submit_mmgbsa.sh"] = MMGBSA.replace("igb=8", "igb=5")
    assert not evaluate_output_bundle(bundle)["passed"]


def test_unchanged_fresh_archive():
    repo = Path(__file__).resolve().parents[2]
    run = (
        repo.parent
        / "task-fairness-gpt6-20260907/runs/life_sciences__amber_three_stage_mmgbsa_workflow_instance_1/20260907_081702"
    )
    matches = list(run.rglob("eval_result.json"))
    if not matches:
        pytest.skip("fresh archived solver artifact is not staged")
    assert len(matches) == 1
    output = matches[0].parent / "output"
    expected = {
        "submit_min.sh": "57fc05ce4b85b4c8a0daf97a7f1eab1dbc67b283da3456120a70ac5c9d75eb21",
        "submit_prod.sh": "ef0cdef321039647886fcf0afb5d01e979007af551615a8ac9f92f59114e5b7f",
        "submit_mmgbsa.sh": "ee47f12878a446065d01287a530dfcdafd3c15c60f8d34165173a9b94f24fb10",
        "FINAL_RESULTS_MMGBSA.dat": "dd966a0a48e66016298eb859914bb09422da69311f26a0142920e2e79555e604",
    }
    payloads = {name: (output / name).read_bytes() for name in expected}
    assert {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()} == expected
    result = evaluate_output_bundle({name: data.decode() for name, data in payloads.items()})
    assert result["passed"], result
    assert result["delta_total"] == -116.3302
