"""physical_sciences/gillespie_gene_regulatory_network -- Linux task."""

import json
import logging
import asyncio
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "physical_sciences"
TASK_NAME = "gillespie_gene_regulatory_network"
VARIANT_NAME = "base"

SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


@dataclass
class GillespieGeneRegulatoryNetworkConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def problem_spec_file(self) -> str:
        return f"{self.input_dir}/problem_spec.md"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def tier1_output_file(self) -> str:
        return f"{self.remote_output_dir}/tier1_results.json"

    @property
    def tier2_output_file(self) -> str:
        return f"{self.remote_output_dir}/tier2_results.json"

    @property
    def tier3_output_file(self) -> str:
        return f"{self.remote_output_dir}/tier3_results.json"

    @property
    def solver_output_file(self) -> str:
        return f"{self.remote_output_dir}/gillespie_solver.py"

    @property
    def task_description(self) -> str:
        return f"""\
You are implementing a NumPy-only stochastic simulation workflow for a
three-gene mutual-inhibition regulatory network. The full scientific
specification is in `{self.problem_spec_file}`.

## Your Task
1. Read `{self.problem_spec_file}` end to end. It defines:
   - Tier 1: exact Gillespie SSA for a birth-death process.
   - Tier 2: exact Gillespie SSA for a three-gene mutual-inhibition network.
   - Tier 3: bifurcation scan plus tau-leaping comparison.
   - Required JSON schemas, seeds, event counts, parameters, and runtime budget.

2. Set up the Python runtime with:
   ```
   uv sync --frozen --project {self.runtime_env_dir}
   ```
   Then run your code with `{self.runtime_env_dir}/.venv/bin/python` or activate
   that environment. It provides NumPy only; use the standard library for JSON,
   time, and math utilities.

3. Implement the simulations from scratch in Python/NumPy:
   - Use `numpy.random.default_rng(seed)` for all random number generation.
   - Use exact SSA for Tier 1 and Tier 2.
   - Implement tau-leaping for Tier 3 with the leap condition and non-negative
     population handling described in the spec.
   - Do not use SciPy, GillesPy2, BioSimulator, StochPy, COPASI, StochKit, or
     any dedicated stochastic simulation package.

4. Save all final artifacts under `{self.remote_output_dir}`:
   - `{self.tier1_output_file}`
   - `{self.tier2_output_file}`
   - `{self.tier3_output_file}`
   - `{self.solver_output_file}`

   `gillespie_solver.py` should be your reusable implementation, not a notebook
   transcript, so the simulation can be reproduced from source.

## Output Contract
- Follow the JSON schemas in `problem_spec.md` for all three result files.
- Probabilities and basin fractions must be finite values in `[0, 1]` and should
  sum to one where the spec defines a partition.
- Use the exact Tier 3 alpha grid and comparison points from the spec.
- Overwrite stale files in `{self.remote_output_dir}` if they already exist.

## Boundaries
- Do not modify anything under `{self.input_dir}`.
- Write only under `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> Dict[str, Any]:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "task_id": f"{DOMAIN_NAME}/{TASK_NAME}",
                "problem_spec_file": self.problem_spec_file,
                "runtime_env_dir": self.runtime_env_dir,
                "tier1_output_file": self.tier1_output_file,
                "tier2_output_file": self.tier2_output_file,
                "tier3_output_file": self.tier3_output_file,
                "solver_output_file": self.solver_output_file,
            }
        )
        return metadata


config = GillespieGeneRegulatoryNetworkConfig()


@cb.tasks_config(split="train")
def load():
    cfg = GillespieGeneRegulatoryNetworkConfig()
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
    tag = meta["variant_name"]

    required = [
        meta["reference_dir"],
        f'{meta["reference_dir"]}/tier1_results.json',
        f'{meta["reference_dir"]}/tier2_results.json',
    ]
    missing = [path for path in required if not (await session.file_exists(path) or await session.directory_exists(path))]
    if missing:
        logger.error("[%s] reference assets missing on VM: %s", tag, missing)
        return [0.0]

    with tempfile.TemporaryDirectory(prefix="agenthle_gillespie_eval_") as tmp:
        tmp_path = Path(tmp)
        local_out = tmp_path / "output"
        local_ref = tmp_path / "reference"
        local_out.mkdir()
        local_ref.mkdir()

        remote_to_local = [
            (f'{meta["remote_output_dir"]}/tier1_results.json', local_out / "tier1_results.json"),
            (f'{meta["remote_output_dir"]}/tier2_results.json', local_out / "tier2_results.json"),
            (f'{meta["remote_output_dir"]}/tier3_results.json', local_out / "tier3_results.json"),
            (f'{meta["remote_output_dir"]}/gillespie_solver.py', local_out / "gillespie_solver.py"),
            (f'{meta["reference_dir"]}/tier1_results.json', local_ref / "tier1_results.json"),
            (f'{meta["reference_dir"]}/tier2_results.json', local_ref / "tier2_results.json"),
        ]
        try:
            for remote_path, local_path in remote_to_local:
                local_path.write_text(await session.read_file(remote_path), encoding="utf-8")
        except Exception as exc:
            logger.error("[%s] failed to retrieve evaluation files: %s", tag, exc)
            return [0.0]

        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                str(SCRIPTS_DIR / "score_outputs.py"),
                "--output-dir",
                str(local_out),
                "--reference-dir",
                str(local_ref),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        stdout = result.stdout
        stderr = result.stderr
        rc = result.returncode

        if stderr:
            logger.info("[%s] local verifier stderr: %s", tag, stderr.strip()[:2000])
        if rc != 0:
            logger.error(
                "[%s] local verifier failed rc=%s stdout=%s stderr=%s", tag, rc, stdout, stderr
            )
            return [0.0]

    payload: Optional[Dict[str, Any]] = None
    for line in reversed(stdout.strip().splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "score" in candidate:
            payload = candidate
            break

    if payload is None:
        logger.error("[%s] could not parse verifier JSON; stdout=%s", tag, stdout[:1000])
        return [0.0]

    try:
        score = float(payload["score"])
    except (TypeError, ValueError):
        logger.error("[%s] verifier score was not numeric: %r", tag, payload.get("score"))
        return [0.0]
    return [max(0.0, min(1.0, score))]
