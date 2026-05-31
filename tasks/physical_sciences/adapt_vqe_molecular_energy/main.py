"""physical_sciences/adapt_vqe_molecular_energy -- AgentHLE Linux task."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "physical_sciences"
TASK_NAME = "adapt_vqe_molecular_energy"
VARIANT_NAME = "base"

SCRIPTS_DIR = Path(__file__).parent / "scripts"
EVAL_TMP_SUBDIR = "_eval_tmp"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


@dataclass
class AdaptVqeMolecularEnergyConfig(LinuxTaskConfig):
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
    def output_results_file(self) -> str:
        return f"{self.output_dir}/results.json"

    @property
    def eval_dir(self) -> str:
        # Task-local evaluator scratch root. Not part of the canonical staged
        # tree; used here as the parent of eval_tmp_dir for runtime-only state.
        return f"{self.task_dir}/eval_data"

    @property
    def eval_tmp_dir(self) -> str:
        return f"{self.eval_dir}/{EVAL_TMP_SUBDIR}"

    @property
    def task_description(self) -> str:
        return f"""\
You are solving a molecular ground-state energy benchmark using only NumPy and
SciPy. The visible inputs are a task statement and three Jordan-Wigner
Hamiltonians:

- `{self.input_dir}/h2_hamiltonian.json`
- `{self.input_dir}/lih_hamiltonian.json`
- `{self.input_dir}/beh2_hamiltonian.json`

Read `{self.problem_spec_file}` first. It describes the VQE / ADAPT-VQE task,
the tier structure, the visible output schema, and the library constraints.

## Your Task
1. Set up the provided runtime environment:
   ```
   uv sync --frozen --project {self.runtime_env_dir}
   ```
   Then use `{self.runtime_env_dir}/.venv/bin/python` (or activate the venv).

2. Implement dense-matrix VQE / ADAPT-VQE logic using only NumPy and SciPy.
   You may use `scipy.linalg.expm` and `scipy.optimize.minimize`.

3. Use the visible Hamiltonian JSON files as the source of truth for:
   - molecule identity
   - qubit count
   - electron count
   - active-space note
   - Pauli-string Hamiltonian terms

4. Produce exactly one file under `{self.output_dir}`:
   - `results.json`

5. `results.json` must follow the schema in `{self.problem_spec_file}` and
   include:
   - Tier 1 (`H2`): `molecule`, `energy_ha`, `method`, `n_parameters`
   - Tier 2 / 3: `molecule`, `energy_ha`, `method`, `adapt_iterations`,
     `n_parameters`, `operator_sequence`

## Constraints
- Allowed libraries: NumPy and SciPy only.
- Banned: Qiskit, Cirq, PennyLane, OpenFermion, tequila, or any other
  quantum-computing library.
- Write only under `{self.output_dir}`.
- Do not modify anything under `{self.input_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{DOMAIN_NAME}/{TASK_NAME}",
                "problem_spec_file": self.problem_spec_file,
                "runtime_env_dir": self.runtime_env_dir,
                "output_results_file": self.output_results_file,
                "eval_tmp_dir": self.eval_tmp_dir,
            }
        )
        return metadata


config = AdaptVqeMolecularEnergyConfig()


@cb.tasks_config(split="train")
def load():
    cfg = AdaptVqeMolecularEnergyConfig()
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
    reference_results = f'{meta["reference_dir"]}/results.json'
    if not (await session.file_exists(reference_results) or await session.directory_exists(reference_results)):
        logger.error("reference results missing on VM: %s", reference_results)
        return [0.0]

    await session.interface.create_dir(meta["eval_tmp_dir"])
    verifier_path = f'{meta["eval_tmp_dir"]}/score_outputs.py'
    await session.write_file(verifier_path, _read_script("score_outputs.py"))

    command = (
        f'cd {meta["eval_tmp_dir"]} && '
        f'python "{verifier_path}" '
        f'--output-dir "{meta["output_dir"]}" '
        f'--reference-file "{reference_results}"'
    )
    result = await session.run_command(command, check=False)
    stdout = result.get("stdout", "") if isinstance(result, dict) else ""
    stderr = result.get("stderr", "") if isinstance(result, dict) else ""
    rc = result.get("return_code", 1) if isinstance(result, dict) else 1

    if stderr:
        logger.info("verifier stderr: %s", stderr.strip()[:2000])

    if rc != 0:
        logger.error("verifier failed rc=%s stdout=%s stderr=%s", rc, stdout[:1000], stderr[:1000])
        return [0.0]

    payload: Optional[dict[str, Any]] = None
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    if payload is None:
        logger.error(
            "could not parse verifier JSON; stdout=%s stderr=%s", stdout[:2000], stderr[:1000]
        )
        return [0.0]

    score = payload.get("score", 0.0)
    try:
        return [float(score)]
    except (TypeError, ValueError):
        logger.error("invalid score payload: %r", payload)
        return [0.0]
