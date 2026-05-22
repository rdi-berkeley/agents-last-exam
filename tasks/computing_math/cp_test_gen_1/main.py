"""Competitive programming test generator task: Binary String Copying.

The agent writes gen.cpp, a C++ test case generator that accepts a seed and
prints adversarial test cases. Evaluation compiles the generator alongside 10
hidden submissions, runs 50 seeds, and scores based on how many submissions
receive the expected verdict.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
EVAL_TMP_DIR = "/tmp/agenthle_eval/cp_test_gen_1"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def _parse_verdicts(text: str) -> dict:
    verdicts = {}
    for line in text.strip().split("\n"):
        if ":" not in line:
            continue
        name, verdict_str = line.split(":", 1)
        name = name.strip()
        verdict_set = {v.strip().upper() for v in verdict_str.strip().split("/")}
        verdicts[name] = verdict_set
    return verdicts


def _compute_score(verdicts: dict) -> float:
    if "j" not in verdicts or verdicts["j"] != {"AC"}:
        logger.warning("Gate failed: j verdict is %s", verdicts.get("j"))
        return 0.0

    points = 0

    if "i" in verdicts and verdicts["i"] == {"AC"}:
        points += 1

    for sub in ["a", "c", "e", "f", "h"]:
        if sub in verdicts and ("WA" in verdicts[sub] or "TLE" in verdicts[sub]):
            points += 1

    for sub in ["b", "d"]:
        if sub in verdicts and ("TLE" in verdicts[sub] or "WA" in verdicts[sub]):
            points += 1

    if "g" in verdicts:
        if "WA" in verdicts["g"]:
            points += 1
            if "AC" in verdicts["g"]:
                points += 1

    logger.info("Score breakdown: %d/10 points", points)
    return points / 10.0


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "computing_math"
    TASK_NAME: str = "cp_test_gen_1"
    VARIANT_NAME: str = "default"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/gen.cpp"

    @property
    def task_description(self) -> str:
        return f"""\
You are tasked with writing gen.cpp, a C++ program that prints one complete \
valid test case to stdout per invocation and accepts a single integer seed \
as argv[1] (e.g. ./gen 1, ./gen 2, ./gen 50) to vary output across invocations.

The problem statement is provided in Binary String Copying Problem Statement.pdf. \
The 10 submission IDs are listed in Binary String Copying Inputs.xlsx: they are \
each a single letter, proceeding alphabetically from top to bottom (row 2 is a, \
row 3 is b, etc). Without knowledge of the submission implementations, the \
generator must be designed by reasoning purely from the problem structure about \
what kinds of inputs stress different algorithmic approaches:

To trigger TLE on slow solutions, generate worst-case inputs where the sums of \
n and m hit their limits (sum of n = sum of m = 200000) using a single large \
test case with m operations whose [l, r] ranges cover wide spans of a string \
with many alternating 0/1 segments, forcing per-query string reconstruction or \
hashing of long substrings.

To trigger WA on incorrect solutions, generate inputs that exercise boundary \
conditions: operations with l = r (a no-op sort that produces the original \
string and must be deduplicated against any other no-op operation), strings that \
are already sorted everywhere except at narrow boundary windows so distinct \
(l, r) pairs collapse to the same sorted result, all-0 and all-1 strings where \
every operation is a no-op, and ranges whose sorted result equals the original \
because the substring is already sorted — all cases where naive comparison or \
careless hashing diverges from the correct deduplicated count.

To produce both AC and WA verdicts from the same submission across 50 seeds, \
the 50 generated test cases must include a mix of small inputs (n, m <= 100) \
and maximum-size inputs so that a randomized or hash-based solution \
coincidentally agrees on small inputs but collides on large adversarial ones.

gen.cpp must be compiled and verified locally with g++ before submission.

## Input Files
Located at: `{self.input_dir}`
- `Binary String Copying Problem Statement.pdf` — full problem statement
- `Binary String Copying Inputs.xlsx` — submission IDs

## Output
Save your final generator as: `{self.remote_output_dir}/gen.cpp`

## Compilation
Use: `g++ -O2 -std=c++17 gen.cpp -o gen`

## Constraints
- Each test case must be valid input for the Binary String Copying problem
- Sum of n across all test cases in one invocation must not exceed 200000
- Sum of m across all test cases in one invocation must not exceed 200000
- Each query must satisfy 1 <= l <= r <= n

## Evaluation
Your generator will be called as `./gen $seed` for seeds 1 through 50 to \
produce 50 test cases. Submission j serves as the reference solution. Each of \
the remaining 9 submissions will be run against all 50 test cases with a \
2-second time limit and 256MB memory limit per execution. The verdict for each \
submission is the union of all outcomes (AC, TLE, WA, RE) across the 50 test \
cases, joined by "/" in the order AC/TLE/WA/RE.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "output_file": self.output_file,
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_file = meta["output_file"]
    ref_dir = meta["reference_dir"]

    if not await session.exists(output_file):
        logger.error("gen.cpp not found at %s", output_file)
        return [0.0]

    wrapper_script = _read_script("run_judge_wrapper.sh")
    await session.makedirs(EVAL_TMP_DIR)
    await session.write_file(f"{EVAL_TMP_DIR}/run_judge_wrapper.sh", wrapper_script)

    result = await session.run_command(
        f'bash "{EVAL_TMP_DIR}/run_judge_wrapper.sh" '
        f'"{meta["remote_output_dir"]}" "{ref_dir}" "{EVAL_TMP_DIR}"',
        check=False,
    )

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    rc = result.get("return_code", 1)

    if rc != 0:
        logger.error("Judge wrapper failed (rc=%d): %s", rc, stderr or stdout)
        return [0.0]

    try:
        verdicts_text = await session.read_file(f"{EVAL_TMP_DIR}/verdicts.txt")
    except Exception as exc:
        logger.error("Could not read verdicts.txt: %s", exc)
        return [0.0]

    logger.info("Raw verdicts:\n%s", verdicts_text)
    verdicts = _parse_verdicts(verdicts_text)
    score = _compute_score(verdicts)
    logger.info("Final score: %.2f (%d/10)", score, int(score * 10))
    return [score]
