"""Climate prediction benchmark — physical_sciences/climate_prediction.

The agent builds a climate-emulation pipeline from a staged CMIP6-derived Zarr
store and predicts the held-out SSP245 test window.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "physical_sciences"
TASK_NAME = "climate_prediction"
VARIANT_NAME = "base"

EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"
SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


@dataclass
class ClimatePredictionConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/processed/test_predictions.npy"

    @property
    def csv_file(self) -> str:
        return f"{self.remote_output_dir}/submissions/kaggle_submission.csv"

    @property
    def task_description(self) -> str:
        return f"""\
You are a climate scientist building an end-to-end climate-emulation pipeline on Linux.

## Your Task

Build a climate-emulation pipeline from a staged CMIP6-derived Zarr store and predict \
the held-out SSP245 test window.

1. Bootstrap the Python runtime: `bash {self.software_dir}/bootstrap_runtime.sh`
2. Open the staged Zarr store at `{self.input_dir}/data.zarr/` with cftime-aware decoding.
3. Build training tensors from scenarios `ssp126`, `ssp370`, and `ssp585` across all months.
4. Build test inputs from the last 120 months of `ssp245`.
5. Use `member_id = 0` for target tensors.
6. Broadcast scalar forcings (`CO2`, `CH4`) to the `(48, 72)` spatial grid.
7. Rename `latitude`/`longitude` dimensions to `y`/`x` for spatial inputs.
8. Fit normalization on the training split only.
9. Train any benchmark-safe open-source model or baseline.
10. Write the required output files under `{self.remote_output_dir}/` exactly as described \
by `{self.input_dir}/output_contract.json`.

## Input Files

- Zarr store: `{self.input_dir}/data.zarr/`
- Metadata: `{self.input_dir}/metadata.json`
- Output contract: `{self.input_dir}/output_contract.json`
- Starter guide: `{self.input_dir}/Starter.md`
- Runtime manifest: `{self.input_dir}/runtime_env/`

## Runtime

- Bootstrap deps: `bash {self.software_dir}/bootstrap_runtime.sh`
- Run Python with deps: `{self.software_dir}/python_with_task_deps.sh your_script.py`

## Output

Save all required files under `{self.remote_output_dir}/`:
- `processed/train_inputs.npy`, `processed/train_outputs.npy`, `processed/test_inputs.npy`
- `processed/metadata.json`, `processed/test_predictions.npy`
- `submissions/kaggle_submission.csv`

The visible `data.zarr` intentionally masks the held-out `ssp245` test-window target \
labels (`tas`, `pr`) as NaN. Use the training split to learn a predictor.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "output_file": self.output_file,
                "csv_file": self.csv_file,
            }
        )
        return metadata


config = ClimatePredictionConfig()


@cb.tasks_config(split="train")
def load():
    cfg = ClimatePredictionConfig()
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
    task_dir = meta["task_dir"]
    input_dir = meta["input_dir"]
    reference_dir = meta["reference_dir"]
    remote_output_dir = meta["remote_output_dir"]

    await session.run_command(
        f'cd "{task_dir}" && bash software/bootstrap_runtime.sh',
        check=False,
    )

    await session.interface.create_dir(EVAL_TMP_DIR)
    await session.write_file(
        f"{EVAL_TMP_DIR}/verify_climate.py",
        _read_script("verify_climate.py"),
    )

    result = await session.run_command(
        f'cd "{task_dir}" && bash software/python_with_task_deps.sh '
        f'"{EVAL_TMP_DIR}/verify_climate.py" '
        f'--output-dir "{remote_output_dir}" '
        f'--reference-dir "{reference_dir}" '
        f'--input-dir "{input_dir}"',
        check=False,
    )

    stdout = result.get("stdout", "").strip()
    stderr = result.get("stderr", "").strip()

    if stderr:
        logger.info("verifier stderr:\n%s", stderr[-2000:])

    if result.get("return_code", 1) != 0:
        logger.error(
            "verifier failed (rc=%s): %s",
            result.get("return_code"),
            stderr[-500:],
        )
        return [0.0]

    try:
        last_line = stdout.strip().splitlines()[-1]
        data = json.loads(last_line)
        score = float(data["score"])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
        logger.error("failed to parse verifier output: %s | stdout=%s", e, stdout[-500:])
        return [0.0]

    logger.info(
        "score=%.4f | tas_rmse=%.4f pr_rmse=%.4f skill=%.4f",
        score,
        data.get("tas_rmse", -1),
        data.get("pr_rmse", -1),
        data.get("skill", -1),
    )
    return [score]


if __name__ == "__main__":
    for task in load():
        print(task.description)
