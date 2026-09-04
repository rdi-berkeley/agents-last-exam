"""AgentHLE task: computing_math/gen1_battle_engine_reconstruction.

Reconstruct a Generation I battle engine, bit-exactly, from behaviour alone.

The agent is given a labelled corpus of battle scenarios and the two machine
generated format documents the engine emits about itself. It is given no rules:
no damage formula, no critical-hit rule, no random-number consumption order, no
turn order. It must produce a program that reproduces held-out scenarios exactly.

The reference is `pkmn/engine` (MIT) built in cartridge-accurate mode at a pinned
commit. Nothing about the reference reaches the VM: the corpus is data, and the
held-out transcripts are compared host side.

All task data is written by ``start()``. Nothing is baked into an image and
nothing is fetched at run time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).resolve().parent
DATA = TASK_DIR / "data"
SCRIPTS = TASK_DIR / "scripts"

DOMAIN_NAME = "computing_math"
TASK_NAME = "gen1_battle_engine_reconstruction"
VARIANT_NAME = "base"

EVAL_DIR = "/tmp/agenthle_eval/gen1_battle_engine_reconstruction"
PER_SCENARIO_TIMEOUT_S = 60.0
RUNNER_TIMEOUT_S = 9000.0

_HOLDOUT = json.loads((DATA / "holdout" / "scenarios.json").read_text(encoding="utf-8"))
_EXPECTED = json.loads((DATA / "holdout" / "expected.json").read_text(encoding="utf-8"))

INPUT_FILES: dict[str, Path] = {
    "corpus/scenarios.json": DATA / "visible" / "scenarios.json",
    "corpus/expected.json": DATA / "visible" / "expected.json",
    "docs/protocol.json": DATA / "protocol.json",
    "docs/layout.json": DATA / "layout.json",
}


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    REQUIRES_TASK_DATA: bool = False

    @property
    def submission_path(self) -> str:
        return f"{self.remote_output_dir}/engine.py"

    @property
    def task_description(self) -> str:
        return """\
Reconstruct a battle engine from its observed behaviour.

Under `{input}` is a corpus of battles from a deterministic Generation I battle \
engine. You are given no specification: no damage formula, no critical-hit rule, \
no random-number consumption order, no turn order. The corpus is the only \
statement of the rules that exists.

## What you are given

- `corpus/scenarios.json`: 638 scenarios. Each is self-contained and carries \
the roll tape it used as hex, the exact starting battle state, the move each side \
selected, and an update cap. The engine consumes the tape bytes in order as its \
only source of randomness; nothing is hidden in them.
- `corpus/expected.json`: for each of those scenarios, the protocol log the \
engine emitted on every update, and the final battle state.
- `docs/protocol.json`: the protocol vocabulary. Names every event the engine \
can emit. It does not say when any of them happen.
- `docs/layout.json`: the byte layout of the battle state. Field offsets and \
sizes. It does not say what changes any of them.

Each side has exactly one Pokemon, so there are no switches to reason about.

## The interface

Your engine is invoked once per scenario as

    python3 engine.py <scenario.json>

where the scenario file has the same shape as an entry in `scenarios.json`. It \
must write to stdout, in order, one line per update:

    u<n> <p1 choice type> <p1 choice data> <p2 choice type> <p2 choice data> <protocol log hex>

and then a final line:

    state <final battle state hex>

## Deliverable

Write your engine to `{submission}`. A single self-contained Python 3 file, \
system interpreter and standard library only.

## How this is graded

Your engine is run against held-out scenarios you have not seen. Every mechanic \
in the held-out set is demonstrated somewhere in the corpus you were given; what \
differs is the roll tape. Comparison is exact, byte for byte, on both the \
protocol log and the final state. There is no tolerance.

Two numbers are reported: the fraction of mechanics reproduced, and whether \
every held-out scenario reproduced exactly.

Do not modify anything under `input/`. Do not rely on internet access.
"""

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({"submission_path": self.submission_path, "eval_dir": EVAL_DIR})
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    description = (cfg.task_description
                   .replace("{input}", cfg.input_dir)
                   .replace("{submission}", cfg.submission_path))
    return [cb.Task(
        description=description,
        metadata=cfg.to_metadata(),
        computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
    )]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Write the visible corpus and format docs; confirm nothing hidden leaked."""
    await _setup(task_cfg, session)
    meta = task_cfg.metadata
    input_dir, out_dir, ref_dir = (meta["input_dir"], meta["remote_output_dir"],
                                   meta["reference_dir"])

    await session.run_command(f"rm -rf {input_dir!r} {out_dir!r} {ref_dir!r}", check=False)
    await session.run_command(
        f"mkdir -p {input_dir!r}/corpus {input_dir!r}/docs {out_dir!r}", check=True)
    for rel, src in INPUT_FILES.items():
        await session.write_file(f"{input_dir}/{rel}", src.read_text(encoding="utf-8"))

    # Reference correctly hidden: no held-out scenario id and no held-out
    # transcript exists anywhere on this machine while the agent works.
    probe = min(_EXPECTED)
    leak = await session.run_command(
        f"grep -rl --binary-files=without-match {probe!r} {input_dir!r} 2>/dev/null; "
        f"ls {ref_dir!r} 2>/dev/null", check=False)
    if (leak.get("stdout") or "").strip():
        raise RuntimeError(f"held-out data leaked onto the VM: {leak['stdout'][:400]}")
    logger.info("[gen1] staged %d files; %d held-out scenarios kept host side",
                len(INPUT_FILES), len(_EXPECTED))


def _holdout_files() -> dict[str, str]:
    """Scenario files for the graded set.

    Emitted verbatim. A graded scenario must have exactly the shape of a visible
    one: rewriting a field here would mean grading against a format the agent
    never saw, which is the harness lying rather than the agent failing.
    """
    return {f"{sc['id']}.json": json.dumps(sc) for sc in _HOLDOUT}


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Run the submission over the held-out scenarios; score host side."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen1_grade", SCRIPTS / "grade.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load grader from {SCRIPTS / 'grade.py'}")
    grade = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grade)

    meta = task_cfg.metadata
    submission = meta["submission_path"]
    if not await session.file_exists(submission):
        logger.info("[gen1] no submission at %s", submission)
        return [0.0]

    scen_dir = f"{EVAL_DIR}/scenarios"
    await session.run_command(f"rm -rf {EVAL_DIR!r}", check=False)
    await session.run_command(f"mkdir -p {scen_dir!r}", check=True)
    for name, text in _holdout_files().items():
        await session.write_file(f"{scen_dir}/{name}", text)
    await session.write_file(f"{EVAL_DIR}/eval_runner.py",
                             (SCRIPTS / "eval_runner.py").read_text(encoding="utf-8"))

    result = await session.run_command(
        f"python3 {EVAL_DIR}/eval_runner.py {submission!r} {scen_dir!r} "
        f"{PER_SCENARIO_TIMEOUT_S}",
        timeout=RUNNER_TIMEOUT_S, check=False)
    stdout = (result.get("stdout") or "").strip()
    if result.get("return_code") != 0 or not stdout:
        logger.error("[gen1] eval runner failed: rc=%s stderr=%s",
                     result.get("return_code"), (result.get("stderr") or "")[:600])
        return [0.0]
    try:
        results = json.loads(stdout)
    except json.JSONDecodeError as exc:
        logger.error("[gen1] eval runner output unparseable: %s", exc)
        return [0.0]

    report = grade.score(results, _EXPECTED)
    logger.info("[gen1] exact=%d/%d mechanics=%.3f full_pass=%.0f broken=%s",
                report["scenarios_exact"], report["scenarios_total"],
                report["mechanics"], report["full_pass"], report["broken"][:8])
    return [float(report["mechanics"])]
