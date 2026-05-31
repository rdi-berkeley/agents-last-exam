"""limited_angle_ct_dps_reconstruction — AgentHLE task."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

# cua_bench may exec_module task files without pre-registering them in
# sys.modules; dataclass string annotation handling expects that entry.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_PATH = SCRIPT_DIR / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from score_outputs import score_reconstruction_bytes

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "health_medicine"
    TASK_NAME: str = "limited_angle_ct_dps_reconstruction"
    VARIANT_NAME: str = "base"

    @property
    def output_npy(self) -> str:
        return f"{self.output_dir}/reconstruction.npy"

    @property
    def reference_npy(self) -> str:
        return f"{self.reference_dir}/reference.npy"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Task Directory
`{self.task_dir}`

## Visible Inputs
- Limited-angle sinogram: `{self.input_dir}/sinogram.npy`
- Geometry spec: `{self.input_dir}/geometry.json`
- Diffusion model card: `{self.input_dir}/model_card.md`
- Pretrained checkpoint: `{self.input_dir}/ddpm_CT.pth`
- Detailed instructions: `{self.input_dir}/task_instructions.md`
- Optional runtime manifest: `{self.input_dir}/runtime_env/pyproject.toml` and `{self.input_dir}/runtime_env/uv.lock`

## Helper Scripts
- Runtime bootstrap: `{self.software_dir}/bootstrap_runtime.sh`
- Runtime Python entry point: `{self.software_dir}/python_with_runtime.sh`

## Your Task
Reconstruct a 512x512 attenuation image from the staged limited-angle fan-beam CT sinogram using Diffusion Posterior Sampling (DPS) or a compatible DPS-style method.

The intended workflow is:
1. Read `{self.input_dir}/task_instructions.md`, `model_card.md`, and `geometry.json`.
2. Build the task-local runtime once with `{self.software_dir}/bootstrap_runtime.sh`.
3. Use `{self.software_dir}/python_with_runtime.sh` for your Python runs after the bootstrap succeeds.
4. Load the limited-angle sinogram and configure the LEAP fan-beam geometry from `geometry.json`.
5. Load the DDPM checkpoint according to the architecture and scheduler in `model_card.md`.
6. Run a DPS-style reconstruction that combines the CT forward model with the diffusion prior.
7. Save only the final attenuation-domain reconstruction to:
   `{self.output_dir}/reconstruction.npy`

## Output Contract
- The file must be NumPy `.npy` data readable by `numpy.load`.
- The array must have shape `(512, 512)`.
- The array must be numeric and correspond to attenuation values in the original image domain.

## Evaluation Contract
Your reconstruction is judged by central-crop image quality.
The binary pass condition is:
- `PSNR >= 32.0 dB`
- `SSIM >= 0.90`
- both computed with `data_range = 0.04`

## Constraints
- Do not modify files under `{self.input_dir}`.
- Write your final answer only under `{self.output_dir}`.
- Do not use the internet or any external model weights beyond the staged checkpoint.

## Visibility Note
The benchmark harness exposes only the staged `input/`, `software/`, and the designated writable output directory during solve time. Other evaluator-controlled staged assets may exist in the canonical admin package but are not part of the solve-time surface.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "output_npy": self.output_npy,
                "reference_npy": self.reference_npy,
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
    output_npy = meta["output_npy"]
    reference_npy = meta["reference_npy"]
    try:
        if not (await session.file_exists(output_npy) or await session.directory_exists(output_npy)):
            logger.warning("Missing output file: %s", output_npy)
            return [0.0]
        output_bytes = await session.read_bytes(output_npy)
        reference_bytes = await session.read_bytes(reference_npy)
        score, details = score_reconstruction_bytes(output_bytes, reference_bytes)
        logger.info("evaluation details: %s", details)
        return [score]
    except Exception as exc:
        logger.error("Evaluation failure: %s", exc)
        return [0.0]
