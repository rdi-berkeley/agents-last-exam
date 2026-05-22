"""Task definition for engineering/mpc_control_building_v1."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "mpc_control_building_v1"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


class MPCControlBuildingConfig(LinuxTaskConfig):
    def __init__(self, *, REMOTE_OUTPUT_DIR: str | None = None) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=REMOTE_OUTPUT_DIR or os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def task_description(self) -> str:
        return f"""\
You are a building-controls engineer developing model predictive control (MPC)
for a single-family house cooling system.

## Input

Use the task input directory `{self.input_dir}`. It contains:

- `SFH.idf`: the EnergyPlus 22.1.0 building model.
- `Denver_current_TMY.epw`: the weather file.
- `task_spec.json`: deterministic controller, tariff, reporting, and output-contract details.
- `runtime_env/pyproject.toml`: Python dependencies useful for MPC, RC-model fitting, and reporting.

You must use the provided IDF and EPW. Do not substitute a different building,
weather year, timestep, or tariff.

EnergyPlus 22.1.0 is pre-installed on the VM. See
`{self.software_dir}/README.md` for the absolute binary path and a
sample invocation. Use the `-x` flag to run ExpandObjects automatically
(the IDF contains HVACTemplate objects).

## Task

Build and evaluate an MPC controller using EnergyPlus as the virtual testbed.

1. Run a deterministic baseline on-off HVAC controller from July 1 through
   July 28 at 15-minute intervals. Use occupied cooling setpoint 24 C, 2 C
   unoccupied setback, 0.5 C deadband, flow states 0.0/0.1/0.3 kg/s, supply
   air temperature 13 C, COP 3.0, and the occupancy schedule randomized with
   seed 142857.
2. Train a 3R2C thermal model from the simulated data.
3. Build two MPC policies: one for energy-cost saving and one for demand
   response / on-peak load reduction.
4. Deploy both MPC policies against EnergyPlus or an equivalent closed-loop
   EnergyPlus-generated co-simulation trace.
5. Report July 28 performance: cooling energy, electric energy, tariff cost,
   peak cooling load, peak-period average load, comfort degree-hours, load
   shifting behavior, and RC-model calibration quality.

The time-of-use tariff is:

- 0.06 USD/kWh during 00:00-11:59 and 20:00-23:59.
- 0.12 USD/kWh during 12:00-16:59.
- 0.25 USD/kWh during 17:00-19:59.

## Required Output

Write these artifacts under `{self.remote_output_dir}`:

1. `baseline_data.csv`: 15-minute baseline time series for July 1-July 28.
2. `mpc_energy_saving_data.csv`: 15-minute closed-loop time series for the
   energy-saving MPC.
3. `mpc_demand_response_data.csv`: 15-minute closed-loop time series for the
   demand-response MPC.
4. `mpc_actions_energy_saving.csv`: July 28 MPC actions for energy saving.
5. `mpc_actions_demand_response.csv`: July 28 MPC actions for demand response.
6. `rc_log_energy_saving.json`: fitted RC-model diagnostics.
7. `rc_log_demand_response.json`: fitted RC-model diagnostics.
8. `metrics_comparison.csv`: rows named `baseline`, `mpc_energy_saving`, and
   `mpc_demand_response`, with columns `cooling_kwh`, `elec_kwh`, `cost_usd`,
   `peak_load_kw`, `peak_hour_avg_kw`, and `discomfort_dh`.
9. `results_summary.json`: concise narrative and machine-readable summary of
   baseline, MPC, savings, and RC calibration.

The data CSVs must include at least these columns: `hour`, `cooling_w`,
`t_zone`, and `setpoint`. They should also include outdoor temperature, solar,
occupancy, and controller/action fields where available.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "expected_output_files": [
                    "baseline_data.csv",
                    "mpc_energy_saving_data.csv",
                    "mpc_demand_response_data.csv",
                    "mpc_actions_energy_saving.csv",
                    "mpc_actions_demand_response.csv",
                    "rc_log_energy_saving.json",
                    "rc_log_demand_response.json",
                    "metrics_comparison.csv",
                    "results_summary.json",
                ]
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    cfg = MPCControlBuildingConfig()
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
    await session.run_command(f'mkdir -p "{EVAL_TMP_DIR}"')
    await session.write_file(
        f"{EVAL_TMP_DIR}/verify_outputs.py",
        _read_script("verify_outputs.py"),
    )
    result = await session.run_command(
        f'UV_CACHE_DIR=/tmp/uv-cache uv run --with pandas --with numpy '
        f'python "{EVAL_TMP_DIR}/verify_outputs.py" '
        f'--output-dir "{meta["remote_output_dir"]}" '
        f'--input-dir "{meta["input_dir"]}" '
        f'--reference-dir "{meta["reference_dir"]}"'
    )
    stdout = result.get("output", result.get("stdout", ""))
    try:
        data = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        logger.error("verify_outputs.py did not produce JSON: %s", stdout)
        return [0.0]
    logger.info("MPC control building eval result: %s", data)
    return [float(data.get("score", 0.0))]
