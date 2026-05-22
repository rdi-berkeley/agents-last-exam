"""AgentHLE task: Sentinel-2 NDVI raster and zonal statistics."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
REMOTE_EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\ndvi_zonal_statistics_d02"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: float | None = None,
    check: bool = False,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command)


def _parse_json_stdout(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("verifier returned empty stdout")
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return json.loads(text)


class NDVIZonalStatisticsConfig(GeneralTaskConfig):
    VARIANT_NAME: str = "base"

    def __init__(self) -> None:
        super().__init__(
            DOMAIN_NAME="agriculture_env",
            TASK_NAME="ndvi_zonal_statistics_d02",
            VARIANT_NAME="base",
            OS_TYPE="windows",
            REMOTE_ROOT_DIR=os.environ.get("REMOTE_ROOT_DIR", r"E:\agenthle"),
        )

    @property
    def task_dir(self) -> str:
        return rf"{self.REMOTE_ROOT_DIR}\{self.DOMAIN_NAME}\{self.TASK_NAME}\{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def output_test_pos_dir(self) -> str:
        return rf"{self.task_dir}\output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return rf"{self.task_dir}\output_test_neg"

    @property
    def software_dir(self) -> str:
        return rf"{self.task_dir}\software"

    @property
    def python_geo(self) -> str:
        return rf"{self.software_dir}\python_geo.bat"

    @property
    def evaluator_python(self) -> str:
        return rf"{self.software_dir}\evaluator_venv\Scripts\python.exe"

    @property
    def runtime_pyproject(self) -> str:
        return rf"{self.input_dir}\runtime_env\pyproject.toml"

    @property
    def shapefile_path(self) -> str:
        return rf"{self.input_dir}\rotation_count.shp"

    @property
    def raster_path(self) -> str:
        return rf"{self.input_dir}\s2_l2a_30m_clip.tif"

    @property
    def ndvi_output(self) -> str:
        return rf"{self.remote_output_dir}\ndvi.tif"

    @property
    def csv_output(self) -> str:
        return rf"{self.remote_output_dir}\polygon_ndvi_stats.csv"

    @property
    def task_description(self) -> str:
        return f"""You are given a polygon shapefile of agricultural units and a 6-band Sentinel-2 GeoTIFF.

Task directory:
`{self.task_dir}`

Inputs:
- `{self.shapefile_path}`
  - associated `.dbf`, `.shx`, `.prj`, and `.cpg` files are present in the same directory
  - each feature has `id_lcp`, `rot_count`, and polygon geometry
- `{self.raster_path}`

Band numbering is 1-based:
- band 1 = B2
- band 2 = B3
- band 3 = B4 (Red)
- band 4 = B8 (NIR)
- band 5 = B11
- band 6 = B12

The shapefile and raster use different coordinate reference systems.

Optional Python runtime:
- Use `{self.python_geo}` if you want the benchmark-provisioned geospatial Python stack.
- The package manifest is visible at `{self.runtime_pyproject}`.

Required outputs under `{self.remote_output_dir}`:
- `ndvi.tif`
- `polygon_ndvi_stats.csv`

Required logic:
1. Load the input shapefile. Use `id_lcp` as the polygon identifier and `rot_count` exactly as provided. Do not recompute or modify crop rotation counts.
2. Load the input Sentinel-2 raster.
3. Compute NDVI using `(B8 - B4) / (B8 + B4)`, where B4 is raster band 3 and B8 is raster band 4.
4. Use the raster values exactly as provided. Do not apply reflectance scaling, offsets, normalization, clipping, cloud masking, or any other masking rule.
5. A pixel is invalid only if band 3 is non-finite, band 4 is non-finite, or `(B8 + B4) == 0`.
6. Write `ndvi.tif` as a single-band `float32` GeoTIFF with the same width, height, CRS, transform, extent, and pixel alignment as the input raster.
7. Store invalid pixels in `ndvi.tif` as `NaN` values in the raster array.
8. Do not reproject or resample the raster.
9. Reproject polygon geometries to the raster CRS before computing zonal statistics. Do not dissolve, explode, simplify, buffer, repair, or otherwise modify the input polygon geometries.
10. Use the pixel-center rule (`all_touched=False`) for polygon inclusion.
11. For every polygon, compute `valid_px`, `mean_ndvi`, and `median_ndvi` from finite NDVI pixels.
12. Include every polygon exactly once. If a polygon has no finite NDVI pixels, write `valid_px` as `0`, `mean_ndvi` as `NA`, and `median_ndvi` as `NA`.
13. Write `polygon_ndvi_stats.csv` with exactly these columns, in order: `id_lcp`, `rot_count`, `valid_px`, `mean_ndvi`, `median_ndvi`.
14. Sort the CSV by `id_lcp` in ascending lexicographic order. Treat `id_lcp` as a string identifier.
15. Compute statistics from the final `float32` NDVI raster values, then round `mean_ndvi` and `median_ndvi` to 6 decimal places only when writing the CSV.

Rules:
- Use only the provided local files.
- Do not modify input files.
- Do not change required output filenames.
- Do not ask for confirmation; execute directly.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_name": self.VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "python_geo": self.python_geo,
                "evaluator_python": self.evaluator_python,
                "runtime_pyproject": self.runtime_pyproject,
                "shapefile_path": self.shapefile_path,
                "raster_path": self.raster_path,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "ndvi_output": self.ndvi_output,
                "csv_output": self.csv_output,
                "canonical_gcs_root": (
                    "gs://ale-data-all/agriculture_env/ndvi_zonal_statistics_d02/base/"
                ),
            }
        )
        return metadata


config = NDVIZonalStatisticsConfig()


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
    reference_files = [
        rf"{meta['reference_dir']}\ndvi.tif",
        rf"{meta['reference_dir']}\polygon_ndvi_stats.csv",
    ]
    for path in reference_files:
        if not await session.exists(path):
            raise RuntimeError(
                f"evaluator-controlled reference missing: {path}"
            )
    if not await session.exists(meta["evaluator_python"]):
        raise RuntimeError(
            f"evaluator Python interpreter missing: {meta['evaluator_python']}"
        )

    await session.makedirs(REMOTE_EVAL_TMP_DIR)
    verify_script_path = rf"{REMOTE_EVAL_TMP_DIR}\verify_outputs.py"
    await session.write_file(verify_script_path, _read_script("verify_outputs.py"))

    command_args = [
        meta["evaluator_python"],
        verify_script_path,
        "--pred-dir",
        meta["remote_output_dir"],
        "--gt-dir",
        meta["reference_dir"],
    ]
    result = await _run_command(
        session,
        subprocess.list2cmdline(command_args),
        timeout=900.0,
        check=False,
    )

    if result.get("return_code", 0) != 0:
        logger.error("Verifier exited non-zero: %s", (result.get("stderr", "") or "")[:1000])
        return [0.0]

    payload = _parse_json_stdout(result.get("stdout", ""))
    logger.info("NDVI zonal verifier=%s", json.dumps(payload))
    return [float(payload.get("score", 0.0))]
