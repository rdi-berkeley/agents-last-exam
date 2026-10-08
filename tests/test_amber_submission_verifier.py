import unittest

from tasks.life_sciences.amber_minimization_script_prep_instance_1.scripts import verify_submission

VERIFIER = vars(verify_submission)
BASE = VERIFIER["SYSTEM_BASENAME"]
HEADER = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
module load amber/22
module load cuda/11.6.2
BASE="GLN_phb2_lc3_aurka_model_0"
PARAMS_DIR="${SCRIPT_DIR}/params"
PRMTOP="${PARAMS_DIR}/${BASE}.prmtop"
INPCRD="${PARAMS_DIR}/${BASE}.inpcrd"
if [ ! -f "$PRMTOP" ]; then
    tleap -f leap.in
fi
"""
COMMAND = '"${AMBERHOME}/bin/pmemd.cuda" -O -i step2_implicit.mini.mdin -p "$PRMTOP" -c "$INPCRD" -ref "$INPCRD" -o min.out -r min.rst'


class AmberSubmissionVerifierTests(unittest.TestCase):
    def test_quoted_executable_and_indirect_variables(self):
        self.assertEqual(VERIFIER["_check_submit"](HEADER + COMMAND), [])

    def test_literal_paths_and_line_continuations(self):
        command = COMMAND.replace('"${AMBERHOME}/bin/pmemd.cuda"', "pmemd.cuda")
        command = command.replace('"$PRMTOP"', f"params/{BASE}.prmtop")
        command = command.replace('"$INPCRD"', f"params/{BASE}.inpcrd")
        command = command.replace(" -", " \\\n  -")
        self.assertEqual(VERIFIER["_check_submit"](HEADER + command), [])

    def test_quoted_directory_with_spaces(self):
        header = HEADER.replace("${SCRIPT_DIR}/params", "/work dir/params")
        self.assertEqual(VERIFIER["_check_submit"](header + COMMAND), [])

    def test_recursive_assignment_chain(self):
        header = HEADER + 'TOPO="$PRMTOP"\nCOORD="$INPCRD"\n'
        command = COMMAND.replace("$PRMTOP", "${TOPO}").replace("$INPCRD", "$COORD")
        self.assertEqual(VERIFIER["_check_submit"](header + command), [])

    def test_wrong_file_is_still_rejected(self):
        header = HEADER.replace('PRMTOP="${PARAMS_DIR}/${BASE}.prmtop"', 'PRMTOP="wrong.prmtop"')
        self.assertIn("missing -p wiring", VERIFIER["_check_submit"](header + COMMAND))

    def test_undefined_and_late_variables_are_rejected(self):
        header = HEADER.replace('PRMTOP="${PARAMS_DIR}/${BASE}.prmtop"\n', "")
        self.assertIn("missing -p wiring", VERIFIER["_check_submit"](header + COMMAND))
        self.assertIn(
            "missing -p wiring",
            VERIFIER["_check_submit"](header + COMMAND + '\nPRMTOP="${PARAMS_DIR}/${BASE}.prmtop"'),
        )

    def test_single_quotes_do_not_expand_variables(self):
        command = COMMAND.replace('"$PRMTOP"', "'$PRMTOP'")
        self.assertIn("missing -p wiring", VERIFIER["_check_submit"](HEADER + command))
        header = HEADER.replace(
            'PRMTOP="${PARAMS_DIR}/${BASE}.prmtop"', "PRMTOP='${PARAMS_DIR}/${BASE}.prmtop'"
        )
        self.assertIn("missing -p wiring", VERIFIER["_check_submit"](header + COMMAND))

    def test_comments_and_echo_are_not_invocations(self):
        for fake in ["# " + COMMAND, "echo '" + COMMAND + "'"]:
            with self.subTest(fake=fake):
                self.assertIn("missing pmemd.cuda -O", VERIFIER["_check_submit"](HEADER + fake))

    def test_missing_and_duplicate_options(self):
        for command in [COMMAND.replace('-ref "$INPCRD"', ""), COMMAND + ' -ref "$INPCRD"']:
            with self.subTest(command=command):
                self.assertIn("missing -ref wiring", VERIFIER["_check_submit"](HEADER + command))

    def test_duplicate_commands(self):
        self.assertIn(
            "must invoke pmemd.cuda exactly once",
            VERIFIER["_check_submit"](HEADER + COMMAND + "\n" + COMMAND),
        )

    def test_command_substitution_is_not_evaluated(self):
        header = HEADER + 'PRMTOP="$(printf ' + BASE + '.prmtop)"\n'
        self.assertIn("missing -p wiring", VERIFIER["_check_submit"](header + COMMAND))

    def test_real_workflow_errors_remain(self):
        leap = f"source leaprc.protein.ff14SB\ndefault PBRadii mbondi3\ncomplex = loadpdb complex_structure.pdb\nsaveamberparm complex {BASE}.prmtop {BASE}.inpcrd\nsavepdb complex {BASE}_fixed.pdb\nquit\n"
        self.assertEqual(VERIFIER["_check_leap"](leap), ["missing mbondi3 radii"])
        self.assertIn(
            "missing mbondi3 radii",
            VERIFIER["_check_leap"](
                leap.replace("default PBRadii mbondi3", "set default PBRadii mbondi2")
            ),
        )

    def test_inactive_or_text_only_commands_are_rejected(self):
        for text in [
            "if false; then\n" + COMMAND + "\nfi\n",
            "cat <<'DOC'\n" + COMMAND + "\nDOC\n",
            "cat <<'DOC' > notes.txt\n" + COMMAND + "\nDOC\n",
            "never_called() {\n" + COMMAND + "\n}\n",
            "exit 0\n" + COMMAND,
            "set -n\n" + COMMAND,
            "false && " + COMMAND,
            "echo 'documentation\n" + COMMAND + "\n'\n",
        ]:
            with self.subTest(text=text):
                self.assertIn("missing pmemd.cuda -O", VERIFIER["_check_submit"](HEADER + text))


if __name__ == "__main__":
    unittest.main()


import pytest


@pytest.mark.parametrize(
    "wrapper",
    [
        "if true; then {command}; fi",
        "if false; then false; else {command}; fi",
        "if false; then false; elif true; then {command}; fi",
        "run_min() {{ {command}; }}; run_min",
        "run_min() {{\n{command}\nreturn 0\n}}\nrun_min",
        "true && {command}",
        "false || {command}",
        "{{ {command}; }}",
    ],
)
def test_provably_executed_control_flow_is_accepted(wrapper):
    assert VERIFIER["_check_submit"](HEADER + wrapper.format(command=COMMAND)) == []


@pytest.mark.parametrize(
    "wrapper",
    [
        "if false; then {command}; fi",
        "f() {{ {command}; }}",
        "f() {{ {command}; }}; f; f",
        "if unknown_test; then {command}; fi",
        "f() {{ f; }}; f; {command}",
        "false && {command}",
        "true || {command}",
        "f() {{ return; {command}; }}; f",
        "f() {{ exit 0; }}; f; {command}",
    ],
)
def test_dead_ambiguous_or_multiple_calls_are_rejected(wrapper):
    assert VERIFIER["_check_submit"](HEADER + wrapper.format(command=COMMAND))
