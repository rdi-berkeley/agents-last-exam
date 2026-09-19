from pathlib import Path

import pytest

from tasks.life_sciences.amber_minimization_script_prep_instance_1.scripts import (
    verify_submission as minimization,
)
from tasks.life_sciences.amber_minimization_script_prep_instance_1.scripts.workflow_parser import (
    ContractError,
    Shell,
    namelists,
    options,
    path,
)
from tasks.life_sciences.amber_three_stage_mmgbsa_workflow_instance_1.scripts import (
    verify_submission as mmgbsa,
)


MIN_BASE = "GLN_phb2_lc3_aurka_model_0"
MIN_LEAP = f"""source leaprc.protein.ff14SB
set default PBRadii mbondi3
protein = loadPdb "complex_structure.pdb"
saveAmberParm protein {MIN_BASE}.prmtop {MIN_BASE}.inpcrd
savePdb protein {MIN_BASE}_fixed.pdb
quit
"""
MIN_MDIN = """Relaxation
&cntrl
 imin=1, ntb=0, cut=999.0, igb=8,
 maxcyc=5000, ncyc=2500, ntpr=100, ntxo=2,
 saltcon=0.15, intdiel=1.0, extdiel=78.5,
/
"""
MIN_SCRIPT = f'''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=7:00:00
ROOT="$(cd "$(dirname "$0")" && pwd)"
NAME="{MIN_BASE}"
PARAMS="$ROOT/params"
module load amber/22 cuda/11.6.2
mkdir -p "$PARAMS"
if [ ! -f "$PARAMS/$NAME.prmtop" ] || [ ! -f "$PARAMS/$NAME.inpcrd" ]; then
  tleap -f "$ROOT/leap.in"
  mv "$ROOT/$NAME.prmtop" "$PARAMS/$NAME.prmtop"
  mv "$ROOT/$NAME.inpcrd" "$PARAMS/$NAME.inpcrd"
fi
"$AMBERHOME/bin/pmemd.cuda" -O -i "$ROOT/step2_implicit.mini.mdin" -p "$PARAMS/$NAME.prmtop" -c "$PARAMS/$NAME.inpcrd" -ref "$PARAMS/$NAME.inpcrd" -o min.out -r min.rst
'''


@pytest.fixture
def min_bundle():
    return {"leap.in": MIN_LEAP, "step2_implicit.mini.mdin": MIN_MDIN, "submit_min.sh": MIN_SCRIPT}


@pytest.mark.parametrize("cut", ["999", "999.000", "+9.99e2", "9.99D+02"])
def test_minimization_numeric_equivalence(min_bundle, cut):
    min_bundle["step2_implicit.mini.mdin"] = MIN_MDIN.replace("999.0", cut)
    result = minimization.evaluate_output_bundle(min_bundle)
    assert result["passed"], result


@pytest.mark.parametrize(
    "change",
    [
        lambda text: (
            text.replace("&cntrl", "&CNTRL").replace("igb", "IGB").replace("/\n", "&END\n")
        ),
        lambda text: text.replace(", ", "\n "),
        lambda text: text.replace("cut=999.0", "! cut=12\n cut=999"),
    ],
)
def test_minimization_namelist_language(min_bundle, change):
    min_bundle["step2_implicit.mini.mdin"] = change(MIN_MDIN)
    result = minimization.evaluate_output_bundle(min_bundle)
    assert result["passed"], result


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("loadPdb", "LOADPDB").replace("saveAmberParm", "saveamberparm"),
        lambda text: (
            text.replace("protein =", "ProteinUnit =")
            .replace("Parm protein", "Parm ProteinUnit")
            .replace("Pdb protein", "Pdb ProteinUnit")
        ),
        lambda text: text.replace(f"{MIN_BASE}.prmtop", f'"./{MIN_BASE}.prmtop"'),
        lambda text: "# loadpdb wrong.pdb\n" + text,
    ],
)
def test_minimization_leap_equivalence(min_bundle, change):
    min_bundle["leap.in"] = change(MIN_LEAP)
    result = minimization.evaluate_output_bundle(min_bundle)
    assert result["passed"], result


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace('"$AMBERHOME/bin/pmemd.cuda"', '"pmemd.cuda"'),
        lambda text: text.replace("NAME", "SYSTEM_ID").replace("$PARAMS", "${PARAMS}"),
        lambda text: text.replace("-O -i", "-i").replace("-r min.rst", "-r min.rst -O"),
        lambda text: text.replace('-p "$PARAMS/$NAME.prmtop"', '-p "$PARAMS/./$NAME.prmtop"'),
        lambda text: text.replace(
            '"$AMBERHOME/bin/pmemd.cuda"', 'command "$AMBERHOME/bin/pmemd.cuda"'
        ),
        lambda text: (
            text.replace('"$AMBERHOME/bin/pmemd.cuda"', 'run_min() { "$AMBERHOME/bin/pmemd.cuda"')
            + "}\nrun_min\n"
        ),
    ],
)
def test_minimization_shell_equivalence(min_bundle, change):
    min_bundle["submit_min.sh"] = change(MIN_SCRIPT)
    result = minimization.evaluate_output_bundle(min_bundle)
    assert result["passed"], result


@pytest.mark.parametrize(
    "invocation",
    ["tleap -s", '"$AMBERHOME/bin/tleap" -s', "run_ambertools.sh tleap -s"],
)
def test_minimization_tleap_standalone_s(min_bundle, invocation):
    min_bundle["submit_min.sh"] = MIN_SCRIPT.replace("tleap -f", invocation + " -f")
    result = minimization.evaluate_output_bundle(min_bundle)
    assert result["passed"], result


def test_tleap_s_option_arity_is_tool_specific():
    assert options(["/amber/bin/tleap", "-f", "leap.in", "-s"]) == {
        "-f": "leap.in",
        "-s": "",
    }
    assert options(["ante-MMPBSA.py", "-s", ":WAT,HOH", "-m", ":1-299"]) == {
        "-s": ":WAT,HOH",
        "-m": ":1-299",
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["tleap", "-s", "-f"],
        ["tleap", "-s", "unexpected", "-f", "leap.in"],
        ["tleap", "-s", "-s", "-f", "leap.in"],
        ["ante-MMPBSA.py", "-s", "-m", ":1-299"],
        ["pmemd.cuda", "-s", "-O"],
    ],
)
def test_tleap_flag_fix_preserves_invalid_option_checks(argv):
    with pytest.raises(ContractError):
        options(argv)


@pytest.mark.parametrize("prefix", ["/", "//", "///"])
def test_linux_root_path_spelling(prefix):
    assert path(prefix + "input/./prod.mdcrd") == "/input/prod.mdcrd"
    assert path("../input/prod.mdcrd", prefix + "script") == "/input/prod.mdcrd"
    assert path(prefix + "other/prod.mdcrd") != "/input/prod.mdcrd"


@pytest.mark.parametrize(
    "old,new",
    [
        ("cut=999.0", "cut=998.9"),
        ("igb=8", "igb=5"),
        ("imin=1", "imin=0"),
        ("ntb=0", "ntb=1"),
        ("maxcyc=5000", "maxcyc=1999"),
        ("ncyc=2500", "ncyc=5000"),
        ("saltcon=0.15", "saltcon=NaN"),
        ("extdiel=78.5", "extdiel='oops'"),
        ("intdiel=1.0", "intdiel=1e999"),
        ("cut=999.0", "cut='999.0'"),
        ("&cntrl", "&gb"),
        ("/\n", ""),
        ("igb=8", "igb=8, igb=5"),
        ("cut=999.0", "! cut=999.0"),
        ("ntpr=100", "ntpr=garbage"),
        ("ntxo=2", "ntxo=7"),
    ],
)
def test_minimization_rejects_bad_namelists(min_bundle, old, new):
    min_bundle["step2_implicit.mini.mdin"] = MIN_MDIN.replace(old, new)
    assert not minimization.evaluate_output_bundle(min_bundle)["passed"]


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("saveAmberParm protein", "saveAmberParm Protein"),
        lambda text: text.replace("complex_structure.pdb", "Complex_structure.pdb"),
        lambda text: text.replace("source leaprc", "# source leaprc"),
        lambda text: text.replace("mbondi3", "mbondi2"),
        lambda text: "quit\n" + text,
        lambda text: text.replace("saveAmberParm", "protein = loadPdb wrong.pdb\nsaveAmberParm"),
        lambda text: text.replace("savePdb protein", "savePdb other"),
    ],
)
def test_minimization_rejects_bad_leap(min_bundle, change):
    min_bundle["leap.in"] = change(MIN_LEAP)
    assert not minimization.evaluate_output_bundle(min_bundle)["passed"]


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace(
            '"$AMBERHOME/bin/pmemd.cuda"', 'echo "$AMBERHOME/bin/pmemd.cuda"'
        ),
        lambda text: text.replace('"$AMBERHOME/bin/pmemd.cuda"', '# "$AMBERHOME/bin/pmemd.cuda"'),
        lambda text: text.replace(
            '"$AMBERHOME/bin/pmemd.cuda"', 'false && "$AMBERHOME/bin/pmemd.cuda"'
        ),
        lambda text: (
            text.replace(
                '"$AMBERHOME/bin/pmemd.cuda"', 'if false; then "$AMBERHOME/bin/pmemd.cuda"'
            )
            + "fi\n"
        ),
        lambda text: (
            text.replace(
                '"$AMBERHOME/bin/pmemd.cuda"', 'never_called() { "$AMBERHOME/bin/pmemd.cuda"'
            )
            + "}\n"
        ),
        lambda text: text.replace(
            '"$AMBERHOME/bin/pmemd.cuda"', 'exit 0\n"$AMBERHOME/bin/pmemd.cuda"'
        ),
        lambda text: text.replace('-p "$PARAMS/$NAME.prmtop"', '-p "$PARAMS/wrong.prmtop"'),
        lambda text: text.replace('-c "$PARAMS/$NAME.inpcrd"', '-c "$ROOT/$NAME.inpcrd"'),
        lambda text: text.replace('-ref "$PARAMS/$NAME.inpcrd"', "-ref wrong.inpcrd"),
        lambda text: text.replace('tleap -f "$ROOT/leap.in"', 'echo tleap -f "$ROOT/leap.in"'),
        lambda text: text.replace("if [ ! -f", "if [ -f"),
        lambda text: text.replace(" || ", " && "),
        lambda text: text.replace("  mv", "  echo mv"),
        lambda text: text.replace("module load", "echo module load"),
        lambda text: text + text.splitlines()[-1] + "\n",
        lambda text: text.replace("fi\n", ""),
        lambda text: text.replace("#SBATCH --mem=16GB\n", "") + "#SBATCH --mem=16GB\n",
        lambda text: text.replace('"$AMBERHOME/bin/pmemd.cuda"', "'${AMBERHOME}/bin/pmemd.cuda'"),
    ],
)
def test_minimization_rejects_bad_shell(min_bundle, change):
    min_bundle["submit_min.sh"] = change(MIN_SCRIPT)
    assert not minimization.evaluate_output_bundle(min_bundle)["passed"]


def test_minimization_file_set(min_bundle):
    assert not minimization.evaluate_output_bundle(min_bundle, present_files=[])["passed"]
    min_bundle["extra.txt"] = "extra"
    assert not minimization.evaluate_output_bundle(min_bundle)["passed"]


def test_minimization_local_reference():
    reference = (
        Path(__file__).resolve().parents[2]
        / "task-data-hf/extracted/life_sciences/amber_minimization_script_prep_instance_1/base/reference"
    )
    if not reference.exists():
        pytest.skip("local task data not staged")
    bundle = {name: (reference / name).read_text() for name in minimization.REQUIRED_FILES}
    result = minimization.evaluate_output_bundle(bundle)
    assert result["passed"], result


@pytest.mark.parametrize(
    "existing,expected",
    [({}, 1), ({"top": "x"}, 1), ({"coord": "x"}, 1), ({"top": "x", "coord": "x"}, 0)],
)
@pytest.mark.parametrize(
    "condition",
    [
        "[ ! -f top ] || [ ! -f coord ]",
        "[[ ! -s top ||\n ! -s coord ]]",
        "! test -e top || ! test -e coord",
    ],
)
def test_shell_file_existence_paths(existing, expected, condition):
    shell = Shell(f"if {condition}; then tleap -f leap.in; else echo existing; fi\n", existing)
    assert sum(command.argv[0] == "tleap" for command in shell.commands) == expected


@pytest.mark.parametrize(
    "script,expected",
    [
        (
            'run() { pmemd.cuda "$@"; return 0; echo unreachable; }; run -O; echo after',
            [["pmemd.cuda", "-O"], ["echo", "after"]],
        ),
        ("run() { exit 0; }; run; pmemd.cuda -O", []),
        ("set -e; false; pmemd.cuda -O", []),
        ("set -e; if false; then echo bad; else pmemd.cuda -O; fi", [["pmemd.cuda", "-O"]]),
        ("set -e; false || pmemd.cuda -O", [["pmemd.cuda", "-O"]]),
        (
            'NAME=old; run() { local NAME=new; echo "$NAME"; }; run; echo "$NAME"',
            [["echo", "new"], ["echo", "old"]],
        ),
        ('NAME=old; run() { NAME=new; }; run; echo "$NAME"', [["echo", "new"]]),
    ],
)
def test_shell_function_and_branch_semantics(script, expected):
    assert [command.argv for command in Shell(script).commands] == expected


MM_BASE = "GLN_phb2_parl_pgam5_model_0"
MM_HEADER = f'''#!/bin/bash
#SBATCH --nodes=1
BASE="{MM_BASE}"
PARAMS="/script/params"
'''
MM_MIN = (
    MM_HEADER
    + """mkdir -p "$PARAMS"
pmemd.cuda -O -i mini.mdin -p "$PARAMS/$BASE.prmtop" -c "$PARAMS/$BASE.inpcrd" -o min.out -r min.rst
pmemd.cuda -O -i equil.mdin -p "$PARAMS/$BASE.prmtop" -c min.rst -o equil.out -r equil.rst
"""
)
MM_PROD = (
    MM_HEADER
    + """pmemd.cuda -O -i production.mdin -p "$PARAMS/$BASE.prmtop" -c equil.rst -o prod.out -r prod.rst -x prod.mdcrd
"""
)
MM_INPUT = """&general
 startframe=1, endframe=999999, interval=1,
 verbose=1, keep_files=0,
/
&gb
 igb=8, saltcon=0.150,
/
"""
MM_SCRIPT = (
    MM_HEADER
    + """cat > calculation.in <<'CONTROL'
"""
    + MM_INPUT
    + """CONTROL
MMPBSA.py -O -i calculation.in -o FINAL_RESULTS_MMGBSA.dat -cp "$PARAMS/$BASE.prmtop" -rp "$PARAMS/${BASE}_A.prmtop" -lp "$PARAMS/${BASE}_BC.prmtop" -y ../input/prod.mdcrd
"""
)
MM_RESULT = (
    "\n".join("|" + line for line in MM_INPUT.splitlines())
    + f"""
|Complex topology file: {MM_BASE}.prmtop
|Receptor topology file: {MM_BASE}_A.prmtop
|Ligand topology file: {MM_BASE}_BC.prmtop
|Receptor mask: ":1-299"
|Ligand mask: ":300-964"
|Calculations performed using 250.0 complex frames.
GENERALIZED BORN:
DELTA G gas 1378.4092 68.4709 4.3305
DELTA G solv -1494.7388 60.6153 3.8336
DELTA TOTAL -116.3296 18.9461 1.1983
"""
)


@pytest.fixture
def mm_bundle():
    return {
        "submit_min.sh": MM_MIN,
        "submit_prod.sh": MM_PROD,
        "submit_mmgbsa.sh": MM_SCRIPT,
        "FINAL_RESULTS_MMGBSA.dat": MM_RESULT,
    }


@pytest.mark.parametrize(
    "old,new",
    [
        ("igb=8", "IGB = 8"),
        ("saltcon=0.150", "SALTCON=1.50e-1"),
        ("endframe=999999", "endframe=250"),
        ("startframe=1, ", ""),
        ("endframe=999999, ", ""),
        ("startframe", "star"),
        ("&gb", "&GB"),
        (" igb=8", " # full-line comment\n igb=8"),
    ],
)
def test_mmgbsa_namelist_equivalence(mm_bundle, old, new):
    mm_bundle["submit_mmgbsa.sh"] = MM_SCRIPT.replace(old, new)
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"], result


@pytest.mark.parametrize("target", ["script", "report", "both"])
@pytest.mark.parametrize(
    "masks",
    [
        "receptor_mask=:1-299, ligand_mask=:300-964,",
        "RECEPTOR_MASK=:1-299, LIGAND_MASK=:300-964,",
        "rece=:1-299, liga=:300-964,",
    ],
)
def test_mmgbsa_native_unquoted_masks(mm_bundle, target, masks):
    control = MM_INPUT.replace("&general\n", "&general\n " + masks + "\n")
    if target in {"script", "both"}:
        mm_bundle["submit_mmgbsa.sh"] = MM_SCRIPT.replace(MM_INPUT, control)
    if target in {"report", "both"}:
        mm_bundle["FINAL_RESULTS_MMGBSA.dat"] = MM_RESULT.replace(
            "|&general\n", "|&general\n| " + masks + "\n"
        )
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"], result
    assert result["delta_total"] == -116.3296


@pytest.mark.parametrize("target", ["script", "report"])
@pytest.mark.parametrize(
    "masks",
    [
        "receptor_mask=:300-964, ligand_mask=:1-299,",
        "receptor_mask=:1-298, ligand_mask=:300-964,",
        "receptor_mask=:1-299, ligand_mask=:300-965,",
        "receptor_mask=:1-299 ligand_mask=:300-964,",
        "receptor_mask=:1-299, receptor_mask=:1-299,",
        "receptor_mask=:1-299, unknown_mask=:300-964,",
        "receptor_mask=:1-299 # inline comment,",
    ],
)
def test_unquoted_masks_preserve_semantic_and_syntax_checks(mm_bundle, target, masks):
    if target == "script":
        mm_bundle["submit_mmgbsa.sh"] = MM_SCRIPT.replace(
            "&general\n", "&general\n " + masks + "\n"
        )
    else:
        mm_bundle["FINAL_RESULTS_MMGBSA.dat"] = MM_RESULT.replace(
            "|&general\n", "|&general\n| " + masks + "\n"
        )
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert not result["passed"], result


def test_unquoted_mmpbsa_masks_do_not_relax_md_namelists():
    with pytest.raises(ContractError):
        namelists("&cntrl restraintmask=:1-299, /\n")


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("MMPBSA.py -O", '"/amber/bin/MMPBSA.py" -O'),
        lambda text: text.replace(
            "MMPBSA.py -O", '"/script/software/run_ambertools.sh" MMPBSA.py -O'
        ),
        lambda text: text.replace("MMPBSA.py -O", 'calculate() { MMPBSA.py "$@"; }\ncalculate -O'),
        lambda text: text.replace("PARAMS", "TOPOLOGY_DIR").replace("BASE", "SYSTEM"),
        lambda text: text.replace(
            "cat > calculation.in <<'CONTROL'", 'cat <<"CONTROL" > ./calculation.in'
        ),
        lambda text: text.replace("-i calculation.in", "-i ./calculation.in"),
    ],
)
def test_mmgbsa_shell_equivalence(mm_bundle, change):
    mm_bundle["submit_mmgbsa.sh"] = change(MM_SCRIPT)
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"], result


@pytest.mark.parametrize(
    "mask_option,mask",
    [("-m", ":1-299"), ("-n", ":300-964"), ("--receptor-mask", "::A"), ("--ligand-mask", "::B,C")],
)
def test_mmgbsa_arbitrary_topology_names_with_provenance(mm_bundle, mask_option, mask):
    split = f'ante-MMPBSA.py -p "$PARAMS/$BASE.prmtop" -r receptor.parm7 -l partner.parm7 {mask_option} "{mask}" --radii mbondi3\n'
    mm_bundle["submit_mmgbsa.sh"] = (
        MM_SCRIPT.replace("MMPBSA.py -O", split + "MMPBSA.py -O")
        .replace('"$PARAMS/${BASE}_A.prmtop"', "receptor.parm7")
        .replace('"$PARAMS/${BASE}_BC.prmtop"', "partner.parm7")
    )
    mm_bundle["FINAL_RESULTS_MMGBSA.dat"] = MM_RESULT.replace(
        MM_BASE + "_A.prmtop", "receptor.parm7"
    ).replace(MM_BASE + "_BC.prmtop", "partner.parm7")
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"], result


def test_mmgbsa_cpptraj_topology_provenance(mm_bundle):
    split = f"""cpptraj <<'CPP'
parm /script/params/{MM_BASE}.prmtop
parmstrip :300-964
parmwrite out receptor.parm7
parm /script/params/{MM_BASE}.prmtop
parmstrip :1-299
parmwrite out ligand.parm7
CPP
"""
    mm_bundle["submit_mmgbsa.sh"] = (
        MM_SCRIPT.replace("MMPBSA.py -O", split + "MMPBSA.py -O")
        .replace('"$PARAMS/${BASE}_A.prmtop"', "receptor.parm7")
        .replace('"$PARAMS/${BASE}_BC.prmtop"', "ligand.parm7")
    )
    mm_bundle["FINAL_RESULTS_MMGBSA.dat"] = MM_RESULT.replace(
        MM_BASE + "_A.prmtop", "receptor.parm7"
    ).replace(MM_BASE + "_BC.prmtop", "ligand.parm7")
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"], result


@pytest.mark.parametrize(
    "old,new",
    [
        ("igb=8", "igb=5"),
        ("igb=8", "igb=7"),
        ("igb=8", "igb=8, molsurf=1"),
        ("igb=8", "igb=8, ifqnt=1"),
        ("igb=8", "igb=8, surften=0.005"),
        ("saltcon=0.150", "saltcon=0.0"),
        ("saltcon=0.150", "saltcon='0.150'"),
        ("saltcon=0.150", "saltcon=1.5D-1"),
        ("saltcon=0.150", "saltcon=NaN"),
        ("saltcon=0.150", "saltcon=Infinity"),
        ("igb=8", "igb=8, IGB=5"),
        ("&gb", "&pb"),
        ("interval=1", "interval=10"),
        ("endframe=999999", "endframe=249"),
        ("startframe=1", "startframe=2"),
        ("startframe=1", "startframe=1, entropy=1"),
        ("igb=8, saltcon", "igb=8 saltcon"),
        ("igb=8", "# igb=8"),
        ("igb=8", "igb=8 # inline comment"),
    ],
)
def test_mmgbsa_rejects_wrong_or_malformed_physics(mm_bundle, old, new):
    mm_bundle["submit_mmgbsa.sh"] = MM_SCRIPT.replace(old, new)
    assert not mmgbsa.evaluate_output_bundle(mm_bundle)["passed"]


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("MMPBSA.py -O", "echo MMPBSA.py -O"),
        lambda text: text.replace("MMPBSA.py -O", "false && MMPBSA.py -O"),
        lambda text: text.replace("MMPBSA.py -O", "if false; then MMPBSA.py -O") + "fi\n",
        lambda text: text.replace("MMPBSA.py -O", "unused() { MMPBSA.py -O") + "}\n",
        lambda text: text.replace("MMPBSA.py -O", "exit 0\nMMPBSA.py -O"),
        lambda text: text.replace("-i calculation.in", "-i unused.in"),
        lambda text: text.replace("cat > calculation.in", "cat > unused.in"),
        lambda text: text.replace(
            "MMPBSA.py -O", "printf '%s\\n' 'not a namelist' > calculation.in\nMMPBSA.py -O"
        ),
        lambda text: text.replace("-y ../input/prod.mdcrd", "-y ../input/wrong.mdcrd"),
        lambda text: text.replace(
            '-rp "$PARAMS/${BASE}_A.prmtop"', '-rp "$PARAMS/${BASE}_BC.prmtop"'
        ),
        lambda text: text.replace('-lp "$PARAMS/${BASE}_BC.prmtop"', '-lp "$PARAMS/$BASE.prmtop"'),
        lambda text: text.replace(
            "MMPBSA.py -O",
            'ante-MMPBSA.py -p "$PARAMS/$BASE.prmtop" -r "$PARAMS/${BASE}_A.prmtop" -l "$PARAMS/${BASE}_BC.prmtop" -m :300-964\nMMPBSA.py -O',
        ),
        lambda text: text + text.splitlines()[-1] + "\n",
    ],
)
def test_mmgbsa_rejects_deceptive_commands_and_role_swaps(mm_bundle, change):
    mm_bundle["submit_mmgbsa.sh"] = change(MM_SCRIPT)
    assert not mmgbsa.evaluate_output_bundle(mm_bundle)["passed"]


@pytest.mark.parametrize(
    "filename,old,new",
    [
        ("submit_min.sh", "-c min.rst", "-c unrelated.rst"),
        ("submit_min.sh", "-i mini.mdin", "-i wrong.prmtop"),
        ("submit_min.sh", "pmemd.cuda", "echo pmemd.cuda"),
        ("submit_prod.sh", "-c equil.rst", "-c min.rst"),
        ("submit_prod.sh", "-x prod.mdcrd", "-x wrong.mdcrd"),
        ("submit_prod.sh", '-p "$PARAMS/$BASE.prmtop"', '-p "$PARAMS/${BASE}_A.prmtop"'),
    ],
)
def test_mmgbsa_rejects_broken_md_handoffs(mm_bundle, filename, old, new):
    mm_bundle[filename] = mm_bundle[filename].replace(old, new)
    assert not mmgbsa.evaluate_output_bundle(mm_bundle)["passed"]


@pytest.mark.parametrize(
    "old,new",
    [
        ("-116.3296", "-9.9e1"),
        ("-116.3296", "-116.3296junk"),
        ("-116.3296", "NaN"),
        ("-116.3296", "-110"),
        ("250.0 complex", "25.0 complex"),
        ("igb=8", "igb=5"),
        (":1-299", ":300-964"),
        (MM_BASE + "_A.prmtop", "wrong.prmtop"),
        ("GENERALIZED BORN:", "POISSON BOLTZMANN:"),
    ],
)
def test_mmgbsa_rejects_incorrect_results(mm_bundle, old, new):
    mm_bundle["FINAL_RESULTS_MMGBSA.dat"] = MM_RESULT.replace(old, new)
    assert not mmgbsa.evaluate_output_bundle(mm_bundle)["passed"]


def test_mmgbsa_result_numeric_equivalence(mm_bundle):
    mm_bundle["FINAL_RESULTS_MMGBSA.dat"] = MM_RESULT.replace("-116.3296", "-1.163296e2")
    assert mmgbsa.evaluate_output_bundle(mm_bundle)["passed"]


def test_mmgbsa_duplicate_result_and_extra_files(mm_bundle):
    bundle = dict(mm_bundle)
    bundle["FINAL_RESULTS_MMGBSA.dat"] += "DELTA TOTAL -116.3296 18.9461 1.1983\n"
    assert not mmgbsa.evaluate_output_bundle(bundle)["passed"]
    mm_bundle["extra.txt"] = "extra"
    assert not mmgbsa.evaluate_output_bundle(mm_bundle)["passed"]


def test_mmgbsa_local_reference():
    reference = (
        Path(__file__).resolve().parents[2]
        / "task-data-hf/extracted/life_sciences/amber_three_stage_mmgbsa_workflow_instance_1/base/reference"
    )
    if not reference.exists():
        pytest.skip("local task data not staged")
    bundle = {name: (reference / name).read_text() for name in mmgbsa.REQUIRED_FILES}
    result = mmgbsa.evaluate_output_bundle(bundle)
    assert result["passed"], result


@pytest.mark.parametrize(
    "task",
    ["amber_minimization_script_prep_instance_1", "amber_three_stage_mmgbsa_workflow_instance_1"],
)
def test_generated_prompts_disclose_supported_shell_contract(task):
    import importlib
    import json

    from tasks.life_sciences.amber_minimization_script_prep_instance_1.scripts.workflow_parser import (
        SHELL_CONTRACT,
    )

    module = importlib.import_module(f"tasks.life_sciences.{task}.main")
    prompt = module.config.task_description
    card = json.loads(Path(module.__file__).with_name("task_card.json").read_text())
    assert SHELL_CONTRACT in prompt
    assert SHELL_CONTRACT in card["taskPrompt"]
    assert "not executed" in prompt
    assert "Do not use loops" in prompt
    if "mmgbsa" in task:
        for text in (prompt, card["taskPrompt"]):
            for required in (
                "igb=8",
                "saltcon=0.150",
                "250",
                "molsurf=0",
                "surften=0.0072",
                "entropy=0",
                "ff14SB",
                "mbondi3",
            ):
                assert required in text


@pytest.mark.parametrize(
    "insertion",
    [
        "echo fake > calculation.in\n",
        'echo fake > "$PARAMS/${BASE}_A.prmtop"\n',
        'cp "$PARAMS/${BASE}_BC.prmtop" "$PARAMS/${BASE}_A.prmtop"\n',
        'rm "$PARAMS/${BASE}_A.prmtop"\n',
    ],
)
def test_mmgbsa_rejects_corrupted_artifacts(mm_bundle, insertion):
    mm_bundle["submit_mmgbsa.sh"] = MM_SCRIPT.replace("MMPBSA.py -O", insertion + "MMPBSA.py -O")
    assert not mmgbsa.evaluate_output_bundle(mm_bundle)["passed"]


@pytest.mark.parametrize(
    "insertion",
    [
        'rm "$PARAMS/$NAME.prmtop"\n',
        "set -e\nfalse\n",
        "stop() { exit 0; }; stop\n",
    ],
)
def test_minimization_rejects_deleted_or_unreachable_inputs(min_bundle, insertion):
    min_bundle["submit_min.sh"] = MIN_SCRIPT.replace(
        '"$AMBERHOME/bin/pmemd.cuda"', insertion + '"$AMBERHOME/bin/pmemd.cuda"'
    )
    assert not minimization.evaluate_output_bundle(min_bundle)["passed"]


@pytest.mark.parametrize(
    "old,new",
    [
        ("--gres=gpu:1", "--gres=gpu:1,gpu:2"),
        ("--nodes=1", "--nodes=1-2"),
        ("--nodes=1", "--nodes=1\n#SBATCH --nodes=2"),
        ("--cpus-per-task=2", "--cpus-per-task=20"),
        ("#!/bin/bash", "#!/bin/bashfake"),
    ],
)
def test_minimization_rejects_incorrect_scheduler_shape(min_bundle, old, new):
    min_bundle["submit_min.sh"] = MIN_SCRIPT.replace(old, new)
    assert not minimization.evaluate_output_bundle(min_bundle)["passed"]


@pytest.mark.parametrize("igb,passed", [(8, True), (0, False)])
def test_mmgbsa_validates_generated_md_controls(mm_bundle, igb, passed):
    controls = f"""cat > mini.mdin <<'EOF'
Minimize
&cntrl
imin=1, ntb=0, igb={igb}, maxcyc=5000,
/
EOF
cat > equil.mdin <<'EOF'
Equilibrate
&cntrl
imin=0, ntb=0, igb=8, nstlim=5000,
/
EOF
"""
    mm_bundle["submit_min.sh"] = MM_MIN.replace("mkdir -p", controls + "mkdir -p")
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"] == passed, result


@pytest.mark.parametrize("radii,passed", [("mbondi3", True), ("mbondi2", False)])
def test_mmgbsa_validates_generated_leap_build(mm_bundle, radii, passed):
    build = f"""cat > build.leap <<EOF
source leaprc.protein.ff14SB
set default PBRadii {radii}
complex = loadPdb complex_structure.pdb
saveAmberParm complex $PARAMS/$BASE.prmtop $PARAMS/$BASE.inpcrd
quit
EOF
tleap -f build.leap
"""
    mm_bundle["submit_min.sh"] = MM_MIN.replace("mkdir -p", build + "mkdir -p")
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"] == passed, result


@pytest.mark.parametrize(
    "source,passed", [("../input/prod.mdcrd", True), ("new_simulation/prod.mdcrd", False)]
)
def test_mmgbsa_trajectory_copy_provenance(mm_bundle, source, passed):
    mm_bundle["submit_mmgbsa.sh"] = MM_SCRIPT.replace(
        "MMPBSA.py -O", f"cp {source} samples.nc\nMMPBSA.py -O"
    ).replace("-y ../input/prod.mdcrd", "-y samples.nc")
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"] == passed, result


def test_mmgbsa_rejects_overwritten_staged_trajectory(mm_bundle):
    mm_bundle["submit_mmgbsa.sh"] = MM_SCRIPT.replace(
        "MMPBSA.py -O", "echo fake > ../input/prod.mdcrd\nMMPBSA.py -O"
    )
    assert not mmgbsa.evaluate_output_bundle(mm_bundle)["passed"]


GPT6_MIN_SCRIPT = """#!/bin/bash
#SBATCH --job-name=GLN_phb2_lc3_aurka_model_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

module load amber/22
module load cuda/11.6.2

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$SCRIPT_DIR" && pwd)
# SLURM executes a spool copy; retain the original deployment location.
if [ ! -f "$SCRIPT_DIR/leap.in" ]; then
    SCRIPT_DIR="/media/user/data/agenthle/life_sciences/amber_minimization_script_prep_instance_1/base/output"
fi
INPUT_DIR="$SCRIPT_DIR/../input"
PARAMS_DIR="$SCRIPT_DIR/params"
BASENAME="GLN_phb2_lc3_aurka_model_0"
TOP="$PARAMS_DIR/$BASENAME.prmtop"
COORDS="$PARAMS_DIR/$BASENAME.inpcrd"

mkdir -p "$PARAMS_DIR"
if [ ! -s "$TOP" ] || [ ! -s "$COORDS" ]; then
    cp "$INPUT_DIR/complex_structure.pdb" "$PARAMS_DIR/complex_structure.pdb"
    cd "$PARAMS_DIR"
    "$AMBERHOME/bin/tleap" -f "$SCRIPT_DIR/leap.in"
fi

# Do not launch minimization if LEaP failed to produce usable build files.
test -s "$TOP"
test -s "$COORDS"
cd "$SCRIPT_DIR"
"$AMBERHOME/bin/pmemd.cuda" -O \\
    -i "$SCRIPT_DIR/step2_implicit.mini.mdin" \\
    -p "$TOP" \\
    -c "$COORDS" \\
    -ref "$COORDS" \\
    -o "$SCRIPT_DIR/min.out" \\
    -r "$SCRIPT_DIR/min.rst"
"""
GPT6_MIN_LEAP = """source leaprc.protein.ff14SB
set default PBRadii mbondi3

complex = loadpdb complex_structure.pdb
check complex
saveamberparm complex GLN_phb2_lc3_aurka_model_0.prmtop GLN_phb2_lc3_aurka_model_0.inpcrd
savepdb complex GLN_phb2_lc3_aurka_model_0_fixed.pdb
quit
"""
GPT6_MIN_MDIN = """GLN_phb2_lc3_aurka_model_0 implicit-solvent minimization
&cntrl
  imin=1,
  maxcyc=5000,
  ncyc=2500,
  ntmin=1,
  drms=0.001,
  ntx=1,
  irest=0,
  ntb=0,
  cut=999.0,
  igb=8,
  saltcon=0.15,
  intdiel=1.0,
  extdiel=78.5,
  ntc=2,
  ntf=2,
  ntr=0,
  ntpr=100,
  ntwx=0,
  ntxo=2,
/
"""


@pytest.fixture
def gpt6_min_bundle():
    return {
        "submit_min.sh": GPT6_MIN_SCRIPT,
        "leap.in": GPT6_MIN_LEAP,
        "step2_implicit.mini.mdin": GPT6_MIN_MDIN,
    }


def test_gpt6_minimization_exact_20260907_artifact(gpt6_min_bundle):
    import hashlib

    expected = {
        "leap.in": "0993015cd9bae2049ca50467ead068f2e5b1f2afc72966a6f92e3469ea23aaa3",
        "step2_implicit.mini.mdin": "7e88f550c1b03a37257a75d345fc10257759517f66de1130b5ab3f63184871c5",
        "submit_min.sh": "3a0559ce1f03a65dab44693f5641c3dfcc8d5d13c7ba22fde1b0a9d6a6b221ff",
    }
    assert {
        name: hashlib.sha256(text.encode()).hexdigest() for name, text in gpt6_min_bundle.items()
    } == expected
    result = minimization.evaluate_output_bundle(gpt6_min_bundle)
    assert result["passed"], result


@pytest.mark.parametrize(
    "topology,coordinates",
    [
        (None, None),
        ("top", None),
        (None, "coord"),
        ("top", "coord"),
        ("", "coord"),
        ("top", ""),
        ("", ""),
    ],
)
@pytest.mark.parametrize("guard", ["test -s", "test -f", "test -e"])
def test_gpt6_minimization_build_and_skip_paths(topology, coordinates, guard):
    existing = {"leap.in": GPT6_MIN_LEAP}
    for extension, content in (("prmtop", topology), ("inpcrd", coordinates)):
        if content is not None:
            existing[f"params/{MIN_BASE}.{extension}"] = content
    shell = Shell(GPT6_MIN_SCRIPT.replace("test -s", guard), existing)
    assert not shell.stopped
    builds = [command for command in shell.commands if command.argv[0].endswith("/tleap")]
    assert len(builds) == (0 if topology and coordinates else 1)
    runs = [command for command in shell.commands if command.argv[0].endswith("/pmemd.cuda")]
    assert len(runs) == 1


@pytest.mark.parametrize(
    "insertion",
    [
        'rm "$TOP"\n',
        'rm "$COORDS"\n',
        ': > "$TOP"\n',
        'echo fake > "$TOP"\n',
        'cp "$COORDS" "$TOP"\n',
        'cp "$TOP" "$COORDS"\n',
        "false\n",
    ],
)
def test_gpt6_minimization_rejects_invalid_postbuild_artifacts(gpt6_min_bundle, insertion):
    gpt6_min_bundle["submit_min.sh"] = GPT6_MIN_SCRIPT.replace(
        'test -s "$TOP"', insertion + 'test -s "$TOP"'
    )
    result = minimization.evaluate_output_bundle(gpt6_min_bundle)
    assert not result["passed"], result


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace(
            '[ ! -s "$TOP" ] || [ ! -s "$COORDS" ]', '[ ! -s "$TOP" ] && [ ! -s "$COORDS" ]'
        ),
        lambda text: text.replace('[ ! -s "$TOP" ] || [ ! -s "$COORDS" ]', '[ ! -s "$TOP" ]'),
        lambda text: text.replace('"$AMBERHOME/bin/tleap" -f', 'echo "$AMBERHOME/bin/tleap" -f'),
        lambda text: text.replace(
            '"$AMBERHOME/bin/tleap" -f', 'false && "$AMBERHOME/bin/tleap" -f'
        ),
        lambda text: text.replace(
            'test -s "$TOP"',
            'if [ -s "$TOP" ] && [ -s "$COORDS" ]; then exit 0; fi\ntest -s "$TOP"',
        ),
        lambda text: text.replace(
            "fi\n\n# Do not launch", "else\n    exit 0\nfi\n\n# Do not launch"
        ),
        lambda text: text.replace(
            '-f "$SCRIPT_DIR/leap.in"\nfi', '-f "$SCRIPT_DIR/missing.in"\nfi'
        ),
    ],
)
def test_gpt6_minimization_rejects_broken_guard_scenarios(gpt6_min_bundle, change):
    gpt6_min_bundle["submit_min.sh"] = change(GPT6_MIN_SCRIPT)
    result = minimization.evaluate_output_bundle(gpt6_min_bundle)
    assert not result["passed"], result


@pytest.mark.parametrize("transport", ["cp", "mv"])
@pytest.mark.parametrize("directory_target", [False, True])
def test_minimization_postbuild_checks_after_file_transport(
    min_bundle, transport, directory_target
):
    script = MIN_SCRIPT.replace("ROOT=", "set -e\nROOT=", 1).replace("  mv ", f"  {transport} ")
    if directory_target:
        script = script.replace(
            f'{transport} "$ROOT/$NAME.prmtop" "$PARAMS/$NAME.prmtop"',
            f'{transport} "$ROOT/$NAME.prmtop" "$PARAMS/"',
        ).replace(
            f'{transport} "$ROOT/$NAME.inpcrd" "$PARAMS/$NAME.inpcrd"',
            f'{transport} "$ROOT/$NAME.inpcrd" "$PARAMS/"',
        )
    min_bundle["submit_min.sh"] = script.replace(
        "fi\n", 'fi\ntest -s "$PARAMS/$NAME.prmtop"\ntest -s "$PARAMS/$NAME.inpcrd"\n'
    )
    result = minimization.evaluate_output_bundle(min_bundle)
    assert result["passed"], result


@pytest.mark.parametrize("damage", ["", "rm min.rst;", ": > min.rst;", "false;"])
def test_mmgbsa_post_md_restart_guard(mm_bundle, damage):
    mm_bundle["submit_min.sh"] = MM_MIN.replace("mkdir -p", "set -e\nmkdir -p").replace(
        "pmemd.cuda -O -i equil.mdin", f"{damage}\ntest -s min.rst\npmemd.cuda -O -i equil.mdin"
    )
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"] == (not damage), result


@pytest.mark.parametrize("damage", ["", "rm receptor.parm7;", ": > receptor.parm7;", "false;"])
@pytest.mark.parametrize("builder", ["ante", "cpptraj"])
def test_mmgbsa_post_split_topology_guards(mm_bundle, builder, damage):
    if builder == "ante":
        split = 'ante-MMPBSA.py -p "$PARAMS/$BASE.prmtop" -r receptor.parm7 -l ligand.parm7 -m :1-299 --radii mbondi3\n'
    else:
        split = f"""cpptraj <<'CPP'
parm /script/params/{MM_BASE}.prmtop
parmstrip :300-964
parmwrite out receptor.parm7
parm /script/params/{MM_BASE}.prmtop
parmstrip :1-299
parmwrite out ligand.parm7
CPP
"""
    mm_bundle["submit_mmgbsa.sh"] = (
        MM_SCRIPT.replace(
            "MMPBSA.py -O",
            f"set -e\n{split}{damage}\ntest -s receptor.parm7\ntest -s ligand.parm7\nMMPBSA.py -O",
        )
        .replace('"$PARAMS/${BASE}_A.prmtop"', "receptor.parm7")
        .replace('"$PARAMS/${BASE}_BC.prmtop"', "ligand.parm7")
    )
    mm_bundle["FINAL_RESULTS_MMGBSA.dat"] = MM_RESULT.replace(
        MM_BASE + "_A.prmtop", "receptor.parm7"
    ).replace(MM_BASE + "_BC.prmtop", "ligand.parm7")
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"] == (not damage), result


def test_mmgbsa_post_leap_product_guards(mm_bundle):
    build = """cat > build.leap <<EOF
source leaprc.protein.ff14SB
set default PBRadii mbondi3
complex = loadpdb complex_structure.pdb
saveamberparm complex $PARAMS/$BASE.prmtop $PARAMS/$BASE.inpcrd
quit
EOF
set -e
tleap -f build.leap
test -s "$PARAMS/$BASE.prmtop"
test -s "$PARAMS/$BASE.inpcrd"
"""
    mm_bundle["submit_min.sh"] = MM_MIN.replace(
        "pmemd.cuda -O -i mini.mdin", build + "pmemd.cuda -O -i mini.mdin"
    )
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"], result


GPT6_MM_HEATING = """50 ps restrained heating from 0 to 300 K
&cntrl
 imin=0, irest=0, ntx=1, nstlim=25000, dt=0.002,
 ntb=0, igb=8, saltcon=0.150, cut=999.0,
 ntc=2, ntf=2, ntt=3, gamma_ln=2.0, ig=20260907,
 tempi=0.0, temp0=300.0,
 ntr=1, restraint_wt=1.0, restraintmask='@CA,C,N',
 nmropt=1, ntpr=500, ntwx=500, ntwr=5000, ioutfm=1,
/
&wt TYPE='TEMP0', istep1=0, istep2=25000, value1=0.0, value2=300.0, /
&wt TYPE='END', /
"""


def test_gpt6_mmgbsa_heating_weight_records():
    groups = namelists(GPT6_MM_HEATING)
    assert groups["cntrl"]["igb"] == 8
    assert groups["cntrl"]["nmropt"] == 1
    assert groups["wt"] == [
        {"type": "TEMP0", "istep1": 0, "istep2": 25000, "value1": 0, "value2": 300},
        {"type": "END"},
    ]


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("&wt", "&WT").replace("TYPE", "type"),
        lambda text: text.replace("'TEMP0'", '"TEMP0"').replace("'END'", '"END"'),
        lambda text: text.replace("value2=300.0", "VALUE2=3.00D+02"),
        lambda text: text.replace(", ", "\n "),
        lambda text: text.replace("&wt TYPE", "! &wt TYPE='not a record' /\n&wt TYPE"),
        lambda text: text.replace("/\n", "&end\n"),
        lambda text: text.replace("&", "$").replace("/\n", "$end\n"),
    ],
)
def test_weight_record_semantic_equivalence(change):
    assert namelists(change(GPT6_MM_HEATING)) == namelists(GPT6_MM_HEATING)


def test_native_weight_sequence_preserves_order_defaults_and_overlaps():
    text = """&cntrl imin=0, nmropt=1, /
&wt type='TEMP0', istep1=0, istep2=1000, value1=10., value2=1200., /
&wt type='TEMP0', istep1=1001, istep2=3000, value1=1200., value2=1200., /
&wt type='TAUTP', istep1=0, istep2=3000, value1=0.2, value2=0.2, /
&wt type='REST', istep1=0, istep2=3000, value1=0.1, value2=1., /
&wt type='REST', istep1=0, istep2=0, value1=-1., /
&wt type='END' /
LISTOUT=POUT
DISANG=RST.f
"""
    weights = namelists(text)["wt"]
    assert [record["type"] for record in weights] == [
        "TEMP0",
        "TEMP0",
        "TAUTP",
        "REST",
        "REST",
        "END",
    ]
    assert [record.get("value1") for record in weights] == [10, 1200, 0.2, 0.1, -1, None]
    assert weights[-2]["istep2"] == 0
    assert "value2" not in weights[-2]


def test_end_only_weight_section_is_valid():
    assert namelists("&cntrl nmropt=1, /\n&wt type='END' /\n")["wt"] == [{"type": "END"}]


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("&wt TYPE='END', /\n", ""),
        lambda text: text + "&wt TYPE='TEMP0', value1=100., /\n",
        lambda text: text + "&wt TYPE='END', /\n",
        lambda text: text.replace("TYPE='TEMP0',", ""),
        lambda text: text.replace("TYPE='TEMP0'", "TYPE=300"),
        lambda text: text.replace("TYPE='TEMP0'", "TYPE=''"),
        lambda text: text.replace("TYPE='TEMP0'", "TYPE='   '"),
        lambda text: text.replace("TYPE='TEMP0'", "TYPE='TEMP0', type='REST'"),
        lambda text: text.replace("value2=300.0", "value2=300.0, VALUE2=200.0"),
        lambda text: text.replace("value2=300.0", "value2=garbage"),
        lambda text: text.replace("value2=300.0, /", "value2=300.0,"),
        lambda text: text.replace("&wt TYPE='END', /", "&wt TYPE='END',"),
        lambda text: text.replace("TYPE='END'", "TYPE='end'"),
    ],
)
def test_malformed_weight_sections_rejected(change):
    with pytest.raises(ContractError):
        namelists(change(GPT6_MM_HEATING))


@pytest.mark.parametrize("group", ["cntrl", "gb", "general", "decomp"])
@pytest.mark.parametrize("mmpbsa", [False, True])
def test_repeated_weight_support_does_not_allow_duplicate_singletons(group, mmpbsa):
    text = f"&{group} value=1, /\n&{group.upper()} value=1, /\n"
    with pytest.raises(ContractError, match="duplicate"):
        namelists(text, mmpbsa=mmpbsa)


def test_mmpbsa_mode_does_not_accept_md_weight_records():
    with pytest.raises(ContractError, match="MMPBSA"):
        namelists(MM_INPUT + "&wt type='TEMP0', /\n&wt type='END', /\n", mmpbsa=True)


@pytest.mark.parametrize("malformed", [False, True])
def test_mmgbsa_weighted_equilibration_control_is_checked(mm_bundle, malformed):
    heating = GPT6_MM_HEATING
    if malformed:
        heating = heating.replace("&wt TYPE='END', /\n", "")
    controls = "cat > mini.mdin <<'EOF'\n" + MIN_MDIN + "EOF\n"
    controls += "cat > equil.mdin <<'EOF'\n" + heating + "EOF\n"
    mm_bundle["submit_min.sh"] = MM_MIN.replace("mkdir -p", controls + "mkdir -p")
    result = mmgbsa.evaluate_output_bundle(mm_bundle)
    assert result["passed"] is not malformed, result


def test_actual_archived_gpt6_heating_control_unchanged():
    import hashlib

    root = Path(__file__).resolve().parents[3] / "task-fairness-gpt6-20260907/runs"
    candidates = list(
        root.glob(
            "life_sciences__amber_three_stage_mmgbsa_workflow_instance_1/20260907_081702/logs/**/output/submit_min.sh"
        )
    )
    if not candidates:
        pytest.skip("fresh GPT6 historical artifact not staged")
    assert len(candidates) == 1
    script = candidates[0].read_text()
    assert (
        hashlib.sha256(script.encode()).hexdigest()
        == "57fc05ce4b85b4c8a0daf97a7f1eab1dbc67b283da3456120a70ac5c9d75eb21"
    )
    controls = [text for path, text in Shell(script).files.items() if path.endswith("_heat.in")]
    assert controls == [GPT6_MM_HEATING]
    assert namelists(controls[0])["wt"][-1] == {"type": "END"}
