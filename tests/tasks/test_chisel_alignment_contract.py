"""Static scorer regressions and opt-in compilation on the retained local VM."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import re
import shlex
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.engineering.chisel_verilog_alignment_seq_1 import main


REPO = Path(__file__).resolve().parents[2]
TASK = "engineering/chisel_verilog_alignment_seq_1/base"
DATA = REPO / "task-data-hf/extracted" / TASK
INPUT_HASHES = {
    "TASK.md": "1d1a56a674e807ba238e560342da003866880ce0d2872f3184364a5cbb355b92",
    "src/main/scala/Fifo.scala": "802279c134a01b7268fd8801b35adcc29c1768771559ad505e04d91c012f6bd6",
    "generated/DoubleBufferFifo.fir": "8c41d97b34ed30a4213ac3a502c77a3b56c4dbc146fb0c58b6debd2e255a3dc9",
    "generated/optimized.sv": "c152be7ed6366ca6aab50d13e658ec5f331aadceda2e138d49f5bcd141eac688",
}
SOURCES = ("Fifo.scala:26:27", "Fifo.scala:30:22", "Fifo.scala:46:20", "Fifo.scala:59:51")


@pytest.fixture
def answer():
    return {
        "target_signal": "stateReg == 2'h2",
        "target_location_in_optimized_sv": "21.23-39",
        "chisel_sources": list(SOURCES),
    }


@pytest.mark.parametrize("sources", list(itertools.permutations(SOURCES)))
def test_equivalent_source_set_order_receives_full_credit(answer, sources):
    assert main._score({**answer, "chisel_sources": list(sources)}, answer) == 1.0


@pytest.mark.parametrize("copies", [1, 2, 10])
def test_duplicates_are_deduplicated_as_publicly_specified(answer, copies):
    assert (
        main._score({**answer, "chisel_sources": list(reversed(SOURCES)) * copies}, answer) == 1.0
    )


@pytest.mark.parametrize("omitted", SOURCES)
def test_each_independently_supported_location_is_required(answer, omitted):
    observed = {**answer, "chisel_sources": [source for source in SOURCES if source != omitted]}
    assert main._score(observed, answer) == 0.2


@pytest.mark.parametrize(
    "extra",
    [
        "Fifo.scala:27:22",
        "Fifo.scala:51:29",
        "Fifo.scala:52:19",
        "Fifo.scala:45:29",
        "Fifo.scala:999:1",
    ],
)
def test_unrelated_annotation_or_fabricated_location_loses_source_credit(answer, extra):
    assert main._score({**answer, "chisel_sources": [*SOURCES, extra]}, answer) == 0.2


def test_old_incomplete_gold_and_optimized_comment_are_not_correct(answer):
    for sources in (
        ["Fifo.scala:46:20", "Fifo.scala:59:51"],
        [
            "Fifo.scala:26:27",
            "Fifo.scala:27:22",
            "Fifo.scala:30:22",
            "Fifo.scala:46:20",
            "Fifo.scala:51:29",
            "Fifo.scala:52:19",
        ],
        [],
    ):
        assert main._score({**answer, "chisel_sources": sources}, answer) == 0.2


@pytest.mark.parametrize("malformed", [None, [], 1, True, "answer", 1.5])
def test_nonobject_json_receives_zero(answer, malformed):
    assert main._score(malformed, answer) == 0.0


@pytest.mark.parametrize(
    "sources",
    [
        None,
        {},
        "Fifo.scala:26:27",
        [None],
        [26],
        [[]],
        [{}],
        ["Fifo.scala:26:27\n"],
        ["Fifo.scala:26:27\x00"],
        ["Fifo.scala:26:27 extra"],
        ["Fifo.scala:NaN:27"],
        ["Fifo.scala:26:Infinity"],
        ["Fifo.scala:-1:27"],
        ["Other.scala:26:27"],
        ["src/main/scala/Fifo.scala:26:27"],
    ],
)
def test_malformed_sources_fail_schema_gate(answer, sources):
    assert main._score({**answer, "chisel_sources": sources}, answer) == 0.0


@pytest.mark.parametrize(
    "key", ["target_signal", "target_location_in_optimized_sv", "chisel_sources"]
)
def test_missing_required_fields_receive_zero(answer, key):
    observed = dict(answer)
    del observed[key]
    assert main._score(observed, answer) == 0.0


@pytest.mark.parametrize(
    "target", ["stateReg == 2'h1", "stateReg == 2'h2 & io_deq_ready", "2'h2 == stateReg", None]
)
def test_literal_target_identifier_is_still_required_for_full_credit(answer, target):
    assert main._score({**answer, "target_signal": target}, answer) == 0.9


@pytest.mark.parametrize("payload", [b"{", b"null", b"[]", b"1", b"true", b"\xff"])
def test_evaluate_rejects_malformed_or_nonobject_output(answer, payload):
    class Session:
        async def read_bytes(self, path):
            return payload if path == "output" else json.dumps(answer).encode()

    config = SimpleNamespace(
        metadata={"output_file": "output", "reference_file": "reference", "variant_name": "base"}
    )
    assert asyncio.run(main.evaluate(config, Session())) == [0.0]


def test_correct_reference_and_unchanged_visible_inputs(answer):
    if not DATA.exists():
        pytest.skip("Staged task data not present")
    assert json.loads((DATA / "reference/answer.json").read_text()) == answer
    for relative, expected in INPUT_HASHES.items():
        assert (
            hashlib.sha256((DATA / "input/workdir" / relative).read_bytes()).hexdigest() == expected
        )


def guest_command(command):
    port = int(os.environ["CHISEL_DIAGNOSTIC_PORT"])
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/cmd",
        data=json.dumps(
            {"command": "run_command", "params": {"command": command, "timeout": 120}}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=130) as response:
        events = [
            json.loads(line[6:])
            for line in response.read().decode().splitlines()
            if line.startswith("data: ")
        ]
    assert events, "No response from retained diagnostic VM"
    result = events[-1]
    assert result.get("return_code") == 0, result
    return result["stdout"]


def source_locations(annotation):
    assert "src/main/scala/Fifo.scala:" in annotation
    locations = set()
    for line, columns in re.findall(r":(\d+):(\d+|\{[\d,]+\})", annotation):
        for column in columns.strip("{}").split(","):
            locations.add(f"Fifo.scala:{line}:{column}")
    return locations


@pytest.mark.skipif(
    not os.environ.get("CHISEL_DIAGNOSTIC_PORT"), reason="Explicit retained local VM required"
)
def test_independent_fixed_compiler_and_sat_reference():
    remote = guest_command("mktemp -d /tmp/chisel-audit/contract-XXXXXXXX").strip()
    assert re.fullmatch(r"/tmp/chisel-audit/contract-[A-Za-z0-9]+", remote)
    staged = f"/media/user/data/agenthle/{TASK}"
    guest_command(f"cp -a {staged}/input/workdir {remote}/workdir")
    for relative, expected in INPUT_HASHES.items():
        assert guest_command(f"sha256sum {remote}/workdir/{relative}").split()[0] == expected

    firtool = "/opt/firtool-1.138.0/bin/firtool"
    assert "CIRCT firtool-1.138.0" in guest_command(f"{firtool} --version")
    yosys = f"bash {staged}/software/yosys"
    version = guest_command(f"{yosys} -V").strip()
    compile_command = (
        f"{firtool} --disable-opt --preserve-values=all "
        "--lowering-options=locationInfoStyle=wrapInAtSquareBracket "
        f"{remote}/workdir/generated/DoubleBufferFifo.fir -o {remote}/unoptimized.sv"
    )
    guest_command(compile_command)
    emission = guest_command(f"cat {remote}/unoptimized.sv")
    assert (
        hashlib.sha256(emission.encode()).hexdigest()
        == "e6c9047619d73b93b54734818806284fb7e8394ce503f0308cf1e8fbd9f318ae"
    )
    setup = (
        f"read_verilog -sv -DSYNTHESIS {remote}/unoptimized.sv; "
        f"hierarchy -top DoubleBuffer; proc; write_json {remote}/netlist.json"
    )
    guest_command(f"{yosys} -Q -T -p {shlex.quote(setup)}")
    netlist = json.loads(guest_command(f"cat {remote}/netlist.json"))
    nets = netlist["modules"]["DoubleBuffer"]["netnames"]
    module_lines = (
        emission.split("module DoubleBuffer(", 1)[1].split("endmodule", 1)[0].splitlines()
    )
    declarations = {}
    for line in module_lines:
        match = re.match(r"\s*wire\s+(?:\[\d+:\d+\]\s+)?(\w+)\b", line)
        if match:
            name = match.group(1)
            assert name in nets
            declarations[name] = line
    assert declarations
    results = {}
    sources = set()
    for name, declaration in declarations.items():
        proofs = [
            f"sat -seq 1 -set stateReg {state} -prove {name} {int(state == 2)}"
            for state in range(4)
        ]
        script = f"read_json {remote}/netlist.json; " + "; ".join(proofs)
        log = guest_command(f"{yosys} -Q -T -l {remote}/proof-{name}.log -p {shlex.quote(script)}")
        outcomes = re.findall(
            r"SAT proof finished - (no model found: SUCCESS!|model found: FAIL!)", log
        )
        assert len(outcomes) == 4, log
        passed = [outcome.endswith("SUCCESS!") for outcome in outcomes]
        results[name] = passed
        if all(passed):
            sources.update(source_locations(declaration))
        else:
            assert not all(passed[:3]), f"Reachable-only equivalence needs investigation: {name}"

    optimized_setup = (
        f"read_verilog -sv -DSYNTHESIS {remote}/workdir/generated/optimized.sv; "
        "hierarchy -top DoubleBuffer; proc; "
        "sat -seq 1 -set stateReg 2 -set io_deq_ready 0 -prove _GEN_2 1"
    )
    negative_log = guest_command(
        f"{yosys} -Q -T -l {remote}/whole-target-negative.log -p {shlex.quote(optimized_setup)}"
    )
    assert "SAT proof finished - model found: FAIL!" in negative_log
    print(
        json.dumps(
            {
                "remote": remote,
                "compile_command": compile_command,
                "yosys": version,
                "wire_proofs_for_states_0_1_2_3": results,
                "derived_sources": sorted(sources),
            },
            indent=2,
        )
    )
    reference = json.loads((DATA / "reference/answer.json").read_text())
    assert sources == set(reference["chisel_sources"])
    assert not sources.intersection({"Fifo.scala:27:22", "Fifo.scala:51:29", "Fifo.scala:52:19"})
    assert main._score({**reference, "chisel_sources": sorted(sources)}, reference) == 1.0
    for extra in ("Fifo.scala:27:22", "Fifo.scala:51:29", "Fifo.scala:52:19"):
        assert (
            main._score({**reference, "chisel_sources": sorted(sources | {extra})}, reference)
            == 0.2
        )


def test_annotation_expansion_keeps_same_line_multiple_columns():
    assert source_locations("@[src/main/scala/Fifo.scala:26:27, :45:{15,29}, :59:51]") == {
        "Fifo.scala:26:27",
        "Fifo.scala:45:15",
        "Fifo.scala:45:29",
        "Fifo.scala:59:51",
    }
