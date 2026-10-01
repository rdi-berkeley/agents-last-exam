import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from tasks.engineering.openroad_sky130_ibex_pnr_signoff.scripts import verify_submission as verifier


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT.parent / "task-fairness-gpt6-20260907"
RUN = AUDIT / "runs/engineering__openroad_sky130_ibex_pnr_signoff/20260908_163956"
STARTER = (
    ROOT
    / "task-data-hf/extracted/engineering/openroad_sky130_ibex_pnr_signoff"
    / "base/input/starter_project"
)


@pytest.mark.parametrize(
    "wrapper",
    [
        "{}",
        "`{}`",
        "``{}``",
        "**{}**",
        "*{}*",
        "__{}__",
        "_{}_",
        "'{}'",
        '"{}"',
        "\u201c{}\u201d",
        "**`{}`**",
    ],
)
@pytest.mark.parametrize("arrow", ["\u2192", "->", "=>", "\u2192\ufe0f"])
@pytest.mark.parametrize("sentence_period", ["", "."])
def test_change_representation(wrapper, arrow, sentence_period):
    line = "- **Change**: " + ": ".join(
        [
            wrapper.format("CORE_UTILIZATION"),
            wrapper.format("70") + " " + arrow + " " + wrapper.format("45"),
        ]
    )
    assert verifier._journal_records_diff(line + sentence_period, "CORE_UTILIZATION", "70", "45")


@pytest.mark.parametrize(
    "before,after", [("70.0", "+45"), ("7e1", "4.50e1"), ("070", "45.000"), (".70e2", "450e-1")]
)
@pytest.mark.parametrize("sentence_period", ["", "."])
def test_exact_numeric_equivalence(before, after, sentence_period):
    assert verifier._journal_records_diff(
        f"- Change: `CORE_UTILIZATION`: {before} -> {after}{sentence_period}",
        "CORE_UTILIZATION",
        "70",
        "45",
    )


@pytest.mark.parametrize(
    "line",
    [
        "CORE_UTILIZATION: 45 -> 70",
        "OTHER_CORE_UTILIZATION: 70 -> 45",
        "CORE_UTILIZATION_OTHER: 70 -> 45",
        "COREUTILIZATION: 70 -> 45",
        "core_utilization: 70 -> 45",
        "70 -> 45",
        "CORE_UTILIZATION: 70 -> 450",
        "CORE_UTILIZATION: 170 -> 45",
        "CORE_UTILIZATION: 70 -> 45.01",
        "CORE_UTILIZATION: 70 -> 45ns",
        "CORE_UTILIZATION: 70 -> 45 extra",
        "CORE_UTILIZATION: 70 -> 45 -> 70",
        "CORE_UTILIZATION: 70 <- 45",
        "CORE_UTILIZATION: 70 <-> 45",
        "CORE_UTILIZATION: 70 ->",
        "CORE_UTILIZATION: -> 45",
        "CORE_UTILIZATION: NaN -> 45",
        "CORE_UTILIZATION: 70 -> Infinity",
        "CORE_UTILIZATION: 70 -> `45`junk",
        "OTHER: CORE_UTILIZATION: 70 -> 45",
    ],
)
@pytest.mark.parametrize("sentence_period", ["", "."])
def test_wrong_changes_do_not_match(line, sentence_period):
    assert not verifier._journal_records_diff(
        line + sentence_period, "CORE_UTILIZATION", "70", "45"
    )


@pytest.mark.parametrize(
    "line,before,after",
    [
        ("PARAM: 0.05 -> 0.15. \t", "0.05", "0.15"),
        ("PARAM: met4 -> met5.", "met4", "met5"),
        ("PARAM: old.v -> new.v.", "old.v", "new.v"),
        ("PARAM: old.v -> new.v.", "old.v", "new.v."),
        ("PARAM: `old.v.` -> `new.v.`.", "old.v.", "new.v."),
        ("PARAM: `*sv*` -> `*v*`.", "*sv*", "*v*"),
    ],
)
def test_sentence_period_preserves_complete_values(line, before, after):
    assert verifier._journal_records_diff(line, "PARAM", before, after)


@pytest.mark.parametrize(
    "line,before,after",
    [
        ("PARAM: 0.05. -> 0.15.", "0.05", "0.15"),
        ("PARAM: 0.05 -> 0.151.", "0.05", "0.15"),
        ("PARAM: 0.05 -> 0.15.extra.", "0.05", "0.15"),
        ("PARAM: 0.05 -> 0.15. More prose.", "0.05", "0.15"),
        ("PARAM: 0.05 -> `0.15`junk.", "0.05", "0.15"),
        ("PARAM: 0.05 -> `0.15`..", "0.05", "0.15"),
        ("PARAM: met4 -> met50.", "met4", "met5"),
        ("PARAM: met4 -> MET5.", "met4", "met5"),
        ("PARAM: met4 -> met5.extra.", "met4", "met5"),
        ("PARAM: met4 -> met5..", "met4", "met5"),
        ("PARAM: met4 -> `met5.`.", "met4", "met5"),
        ("PARAM: old.v -> \"new.v.\".", "old.v", "new.v"),
        ("PARAM: old.v -> **`new.v.`**.", "old.v", "new.v"),
        ("PARAM: old.v -> new.v.", "old.v", "new.v.extra"),
        ("PARAM: file.v -> .", "file.v", None),
        ("PARAM: file.v -> unset.extra.", "file.v", None),
        ("PARAM: file.v -> `unset.`.", "file.v", None),
        ("PARAM: -noabc -> empty (default Kogge-Stone).", "-noabc", ""),
    ],
)
def test_sentence_period_does_not_relax_value_matching(line, before, after):
    assert not verifier._journal_records_diff(line, "PARAM", before, after)


@pytest.mark.parametrize("after", ["met50", "MET5", "met5_extra", "met5/met6"])
def test_make_values_are_complete_and_case_sensitive(after):
    assert not verifier._journal_records_diff(
        f"MAX_ROUTING_LAYER: met4 -> {after}", "MAX_ROUTING_LAYER", "met4", "met5"
    )


@pytest.mark.parametrize("sentence_period", ["", "."])
def test_make_expression_preserves_wildcards_and_identifiers(sentence_period):
    before = "$(sort $(wildcard $(DESIGN_HOME)/src/*.sv))"
    after = "$(sort $(wildcard $(DESIGN_HOME)/rtl/*.sv))"
    line = f"- **Change**: `VERILOG_FILES`: `{before}` -> `{after}`{sentence_period}"
    assert verifier._journal_records_diff(line, "VERILOG_FILES", before, after)
    assert not verifier._journal_records_diff(
        line.replace("*.sv", ".sv"), "VERILOG_FILES", before, after
    )
    assert not verifier._journal_records_diff(
        line.replace("DESIGN_HOME", "design_home"), "VERILOG_FILES", before, after
    )
    assert verifier._journal_records_diff("FILES: `*.sv` -> `*.v`", "FILES", "*.sv", "*.v")
    assert not verifier._journal_records_diff("FILES: sv -> v", "FILES", "*sv*", "*v*")


@pytest.mark.parametrize("wrapper", ["`{}`", '"{}"', "'{}'", "**`{}`**"])
def test_literal_wildcard_wrapped_values(wrapper):
    line = f"FILES: {wrapper.format('*sv*')} -> {wrapper.format('*v*')}"
    assert verifier._journal_records_diff(line, "FILES", "*sv*", "*v*")


@pytest.mark.parametrize("wrapper", ["`{}`", "**`{}`**", "__`{}`__"])
def test_code_span_preserves_identifier_underscores(wrapper):
    line = f"{wrapper.format('_PARAM_')}: `70` -> `45`"
    assert verifier._journal_records_diff(line, "_PARAM_", "70", "45")
    assert not verifier._journal_records_diff(line, "PARAM", "70", "45")


@pytest.mark.parametrize(
    "line",
    [
        "- **Change:** **CORE_UTILIZATION:** `70` -> `45`",
        "- __Change:__ __CORE_UTILIZATION:__ `70` -> `45`",
        "- *Change:* *CORE_UTILIZATION:* `70` -> `45`",
        "- **Change:** `CORE_UTILIZATION`: `70` -> `45`",
    ],
)
def test_label_punctuation_inside_emphasis(line):
    assert verifier._journal_records_diff(line, "CORE_UTILIZATION", "70", "45")
    assert not verifier._journal_records_diff(line, "UTILIZATION", "70", "45")
    assert not verifier._journal_records_diff(line, "CORE_UTILIZATION", "45", "70")


@pytest.mark.parametrize("after", ["", " ", "**", "__", "1e99999999999999999999999999"])
def test_empty_or_unrepresentable_values_are_not_changes(after):
    assert not verifier._journal_records_diff(f"PARAM: 1 -> {after}", "PARAM", "1", None)
    assert not verifier._journal_records_diff(f"PARAM: 1 -> {after}", "PARAM", "1", "45")


@pytest.mark.parametrize("unset", ["unset", "None", "(empty)", "-", '""', "`(unset)`"])
@pytest.mark.parametrize("sentence_period", ["", "."])
def test_explicit_unset(unset, sentence_period):
    assert verifier._journal_records_diff(
        f"ADDER_MAP_FILE: `file.v` -> {unset}{sentence_period}", "ADDER_MAP_FILE", "file.v", None
    )
    assert verifier._journal_records_diff(
        f"ADDER_MAP_FILE: {unset} -> `file.v`{sentence_period}", "ADDER_MAP_FILE", None, "file.v"
    )


@pytest.mark.parametrize("before", ["(unset; default 0)", "unset (default 0)"])
def test_annotated_unset_values_are_accepted(before):
    assert verifier._journal_records_diff(
        f"OPENROAD_HIERARCHICAL: {before} -> 1.",
        "OPENROAD_HIERARCHICAL",
        None,
        "1",
    )


def test_unset_annotations_do_not_accept_arbitrary_suffixes():
    assert not verifier._journal_records_diff(
        "OPENROAD_HIERARCHICAL: unset.extra -> 1",
        "OPENROAD_HIERARCHICAL",
        None,
        "1",
    )


def test_whitespace_is_not_value_content():
    assert verifier._journal_records_diff(
        "  - Change: `SYNTH_ARGS` : `-abc   -def` \t->  `-ghi  -jkl`  ",
        "SYNTH_ARGS",
        "-abc -def",
        "-ghi -jkl",
    )


def make_gate_fixture(tmp_path, heading, matched=1, expected=1, sentence_period=""):
    starter = tmp_path / "starter"
    config = starter / "flow/designs/sky130hd/ibex/config.mk"
    config.parent.mkdir(parents=True)
    config.write_text("".join(f"export PARAM_{index} = 70\n" for index in range(expected)))
    output = tmp_path / "output"
    logs = output / verifier.PASS_LOG_DIR_REL
    logs.mkdir(parents=True)
    (logs / "config.mk.pass0").write_text(
        "".join(f"export PARAM_{index} = 45\n" for index in range(expected))
    )
    (logs / "pass0.stamp").write_text("completed\n")
    (output / "JOURNAL.md").write_text(
        heading
        + "\n- **Observation**: report metric 1\n"
        + "".join(
            f"- **Change**: `PARAM_{index}`: `70` -> `45`{sentence_period}\n"
            for index in range(matched)
        )
    )
    return output, starter


@pytest.mark.parametrize(
    "heading",
    [
        "## Pass 1",
        "# Pass 1",
        "### pass 1",
        "  #### PASS\t1  ####",
        "## **Pass 1**",
        "## `Pass 1`",
        "## Pass 1: tuning",
    ],
)
def test_pass_heading_equivalence(tmp_path, heading):
    output, starter = make_gate_fixture(tmp_path, heading)
    gate = verifier.gate_g11_journal(output, starter)
    assert (gate.score, gate.passed) == (10, True)


@pytest.mark.parametrize(
    "heading", ["## Pass 10", "## Pass 1extra", "## Bypass 1", "## Pass 2", "## Summary"]
)
def test_wrong_pass_heading(tmp_path, heading):
    output, starter = make_gate_fixture(tmp_path, heading)
    assert verifier.gate_g11_journal(output, starter).score == 0


@pytest.mark.parametrize(
    "matched,score,passed",
    [(10, 10, True), (9, 10, True), (8, 6, False), (7, 6, False), (6, 0, False), (0, 0, False)],
)
@pytest.mark.parametrize("sentence_period", ["", "."])
def test_fidelity_thresholds_unchanged(tmp_path, matched, score, passed, sentence_period):
    output, starter = make_gate_fixture(tmp_path, "## Pass 1", matched, 10, sentence_period)
    gate = verifier.gate_g11_journal(output, starter)
    assert (gate.weight, gate.score, gate.passed) == (10, score, passed)
    assert f"fidelity = {matched}/10" in gate.detail
    assert verifier.gate_g10_pass_budget(output).score == 5


@pytest.mark.parametrize("suffix", ["## Summary", "## Pass 2", "# Other section"])
@pytest.mark.parametrize("sentence_period", ["", "."])
def test_change_in_wrong_section_is_not_reused(tmp_path, suffix, sentence_period):
    output, starter = make_gate_fixture(tmp_path, "## Pass 1", 0)
    journal = output / "JOURNAL.md"
    journal.write_text(journal.read_text() + f"{suffix}\nPARAM_0: 70 -> 45{sentence_period}\n")
    assert verifier.gate_g11_journal(output, starter).score == 0


@pytest.mark.parametrize("fence", ["```", "~~~"])
@pytest.mark.parametrize("sentence_period", ["", "."])
def test_code_examples_are_not_pass_changes(tmp_path, fence, sentence_period):
    output, starter = make_gate_fixture(tmp_path, "## Pass 1", 0)
    journal = output / "JOURNAL.md"
    journal.write_text(
        journal.read_text()
        + f"{fence}markdown\n## Pass 1\nPARAM_0: 70 -> 45{sentence_period}\n{fence}\n"
    )
    assert verifier.gate_g11_journal(output, starter).score == 0


@pytest.mark.parametrize(
    "quoted",
    [
        "> PARAM_0: 70 -> 45.",
        "> - **Change**: `PARAM_0`: `70` -> `45`.",
        "> ## Pass 1\n> PARAM_0: 70 -> 45.",
        "\"PARAM_0: 70 -> 45.\"",
    ],
)
def test_quoted_examples_are_not_pass_changes(tmp_path, quoted):
    output, starter = make_gate_fixture(tmp_path, "## Pass 1", 0)
    journal = output / "JOURNAL.md"
    journal.write_text(journal.read_text() + quoted + "\n")
    assert verifier.gate_g11_journal(output, starter).score == 0


def test_standalone_standard_library_import(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", verifier.__file__, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--submission-dir" in result.stdout


def test_unchanged_real_journal_and_run():
    outputs = list(RUN.glob("logs/*/codex/*/*/v0/*/output"))
    if not outputs or not STARTER.is_dir():
        pytest.skip("Preserved local OpenROAD run and canonical starter required")
    assert len(outputs) == 1
    output = outputs[0]
    paths = [
        output / "JOURNAL.md",
        output / "config.mk",
        output.parent / "run.json",
        output.parent / "eval_result.json",
    ]
    paths.extend(sorted((output / verifier.PASS_LOG_DIR_REL).glob("*")))
    paths = [path for path in paths if path.is_file()]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    journal = (output / "JOURNAL.md").read_text()
    assert verifier._journal_records_diff(journal, "CORE_UTILIZATION", "70", "45")
    gate = verifier.gate_g11_journal(output, STARTER)
    assert (gate.score, gate.passed) == (6, False)
    assert "fidelity = 8/9" in gate.detail
    assert "pass 2: journal missing ADDER_MAP_FILE: None -> " in gate.detail
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before
