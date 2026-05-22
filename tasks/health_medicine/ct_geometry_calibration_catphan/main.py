"""ct_geometry_calibration_catphan — AgentHLE Task

Calibrate CT geometry parameters (SAD, SDD, detector offset) for a Catphan
phantom scan.  The agent iteratively adjusts parameters in a LEAP-based FBP
reconstruction script and compares the output against a reference image until
SSIM >= 0.95 and unnormalized MSE <= 4E-6.
"""

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity as ssim

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
SSIM_THRESHOLD = 0.95
MSE_THRESHOLD = 4e-6


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "health_medicine"
    TASK_NAME: str = "ct_geometry_calibration_catphan"
    VARIANT_NAME: str = "instance_1"

    @property
    def output_npy(self) -> str:
        return f"{self.remote_output_dir}/reconstructed_calibrated.npy"

    @property
    def output_json(self) -> str:
        return f"{self.remote_output_dir}/geometry_calibrated.json"

    @property
    def reference_npy(self) -> str:
        return f"{self.reference_dir}/reference_image.npy"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Your Task

Calibrate the CT geometry parameters for a Catphan phantom reconstruction.

The input directory contains:
- `sinogram.npy` — measured sinogram (fan-beam CT, float32)
- `reference_image.npy` — ground-truth reconstructed image (512x512, float32)
- `reconstruct.py` — FBP reconstruction script using LEAP/leaptorch with nominal (slightly wrong) geometry parameters (SAD=800, SDD=1200, offset=0)

Your goal is to find the correct geometry parameters (SAD, SDD, detector offset,
and any other relevant geometric parameters) so that the reconstructed image
matches the reference image.

## Steps

1. Inspect `reconstruct.py` to understand the current geometry parameters and
   the LEAP-based FBP reconstruction pipeline.
2. Run the initial reconstruction with nominal parameters and compare against
   `reference_image.npy` to establish a baseline error.
3. Iteratively adjust the geometry parameters, run reconstructions, and compare
   outputs to `reference_image.npy` using SSIM and MSE metrics.
4. Continue optimization (e.g., using scipy.optimize.minimize or
   differential_evolution) until the reconstructed image achieves
   SSIM >= 0.95 and unnormalized MSE <= 4e-6 relative to `reference_image.npy`.
5. Save the final calibrated parameters to
   `{self.remote_output_dir}/geometry_calibrated.json`
   and the final reconstructed image to
   `{self.remote_output_dir}/reconstructed_calibrated.npy`.

## Input Files
- Located at: `{self.input_dir}`

## Output
- `{self.remote_output_dir}/geometry_calibrated.json` — calibrated parameters
- `{self.remote_output_dir}/reconstructed_calibrated.npy` — final reconstruction

## Software
- Python 3.10 + LEAP 1.26 (leaptorch), NumPy, SciPy, scikit-image, PyTorch (CPU), imageio.
- Use `software/python` (the canonical entry point for this task) to invoke
  Python; it resolves to `/usr/bin/python` on the VM.
  Example: `software/python input/reconstruct.py input/sinogram.npy`

## Important Constraints
- Do not modify files under `{self.input_dir}`
- SSIM is computed via `skimage.metrics.structural_similarity`
- Unnormalized MSE = mean((ref - recon)**2)
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "output_npy": self.output_npy,
                "output_json": self.output_json,
                "reference_npy": self.reference_npy,
            }
        )
        return metadata


config = TaskConfig()


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score the agent output locally.

    1. Check that both output files exist on the VM.
    2. Download the reconstructed .npy and the reference .npy.
    3. Compute SSIM and unnormalized MSE.
    4. Return 1.0 if SSIM >= 0.95 AND MSE <= 4E-6, else 0.0.
    """
    meta = task_cfg.metadata
    output_npy = meta["output_npy"]
    output_json = meta["output_json"]
    reference_npy = meta["reference_npy"]

    if not await session.exists(output_npy):
        logger.warning("Output npy not found: %s", output_npy)
        return [0.0]
    if not await session.exists(output_json):
        logger.warning("Output json not found: %s", output_json)
        return [0.0]

    if not await session.exists(reference_npy):
        raise RuntimeError(
            f"evaluator-controlled reference missing: {reference_npy}"
        )

    ref_bytes = await session.read_bytes(reference_npy)
    out_bytes = await session.read_bytes(output_npy)

    if not ref_bytes or not out_bytes:
        logger.warning("One or both npy files are empty")
        return [0.0]

    ref_arr = np.load(io.BytesIO(ref_bytes))
    out_arr = np.load(io.BytesIO(out_bytes))

    ssim_val = ssim(ref_arr, out_arr, data_range=ref_arr.max() - ref_arr.min())
    mse_val = float(np.mean((ref_arr.astype(np.float64) - out_arr.astype(np.float64)) ** 2))

    logger.info(
        "SSIM=%.6f  MSE=%.2e  (thresholds: SSIM>=%.2f, MSE<=%.1e)",
        ssim_val,
        mse_val,
        SSIM_THRESHOLD,
        MSE_THRESHOLD,
    )

    if ssim_val >= SSIM_THRESHOLD and mse_val <= MSE_THRESHOLD:
        return [1.0]
    return [0.0]
