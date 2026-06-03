"""Chisel-to-Verilog source-location alignment benchmark.

The agent receives a Chisel hardware design and its firtool-optimized
SystemVerilog, then must identify the exact Chisel source locations that a
specific sub-expression semantically corresponds to. Scoring is deterministic
JSON set-equality comparison — no LLM judge needed.
"""

import json
import logging
import re
from dataclasses import dataclass

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "chisel_verilog_alignment_seq_1"


@dataclass
class ChiselAlignmentConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "base"

    @property
    def workdir(self) -> str:
        return f"{self.input_dir}/workdir"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/answer.json"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/answer.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are a hardware design engineer specializing in Chisel and SystemVerilog.

## Your Task

Given a Chisel hardware design (Fifo.scala) and its firtool-optimized SystemVerilog output, identify the exact set of Chisel source locations (Fifo.scala:line:col) that a specific sub-expression of the optimized Verilog semantically corresponds to.

The optimized Verilog's @[...] source-location comment is both noisy (lists unrelated locations) and incomplete (misses real locations), so answering requires genuine analysis — not comment-reading.

## Working Directory

All materials are under: `{self.workdir}/`

- `TASK.md` — detailed instructions and rules
- `src/main/scala/Fifo.scala` — original Chisel source (DoubleBuffer FIFO, 87 lines)
- `generated/optimized.sv` — firtool emission (target: line 21, sub-expression `stateReg == 2'h2`)
- `generated/DoubleBufferFifo.fir` — FIRRTL intermediate (you may regenerate alternative emissions)

## Tools Available

Use the canonical task-local CLI entry points under `{self.software_dir}/`:

- `{self.software_dir}/yosys` — Yosys 0.33+
- `{self.software_dir}/firtool` — CIRCT firtool 1.138.0
- `{self.software_dir}/sbt` — sbt 1.9.9 with Chisel 6.5.0 dependencies prewarmed
- `{self.software_dir}/python` — Python 3.10
- `{self.software_dir}/jq` — jq 1.x

The VM has no internet access during solve time.

## Rules

1. Do NOT edit any file under `src/`, `generated/`, or `TASK.md`. Writes are only allowed under the task output directory.
2. Focus on the sub-expression `stateReg == 2'h2` at line 21, columns 23-39 of optimized.sv.
3. Approach is unconstrained (formal verification, simulation, semantic analysis — whatever works).

## Output

Write a single file: `{self.remote_output_dir}/answer.json`

```json
{{
  "target_signal": "stateReg == 2'h2",
  "target_location_in_optimized_sv": "21.23-39",
  "chisel_sources": ["Fifo.scala:<L1>:<C1>", ...]
}}
```
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "workdir": self.workdir,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
            }
        )
        return metadata


config = ChiselAlignmentConfig()


@cb.tasks_config(split="train")
def load():
    cfg = ChiselAlignmentConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_file = meta["output_file"]
    reference_file = meta["reference_file"]

    try:
        agent_bytes = await session.read_bytes(output_file)
    except Exception as exc:
        logger.error("agent output missing: %s", exc)
        return [0.0]

    try:
        ref_bytes = await session.read_bytes(reference_file)
    except Exception as exc:
        logger.error("reference file missing: %s", exc)
        return [0.0]

    try:
        agent_data = json.loads(agent_bytes)
        ref_data = json.loads(ref_bytes)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("JSON parse error: %s", exc)
        return [0.0]

    score = _score(agent_data, ref_data)
    logger.info("score=%.4f for %s", score, meta["variant_name"])
    return [score]


def _score(agent: dict, reference: dict) -> float:
    score = 0.0

    if not isinstance(agent.get("chisel_sources"), list):
        return 0.0
    sources = agent["chisel_sources"]
    pattern = re.compile(r"^Fifo\.scala:\d+:\d+$")
    if not all(isinstance(s, str) and pattern.match(s) for s in sources):
        return 0.0

    required_fields = {"target_signal", "target_location_in_optimized_sv", "chisel_sources"}
    if not required_fields.issubset(agent.keys()):
        return 0.0

    score += 0.10

    if agent.get("target_signal") == "stateReg == 2'h2":
        score += 0.10

    agent_set = set(agent["chisel_sources"])
    ref_set = set(reference["chisel_sources"])
    if agent_set == ref_set:
        score += 0.80

    return score


if __name__ == "__main__":
    for task in load():
        print(task.description)
