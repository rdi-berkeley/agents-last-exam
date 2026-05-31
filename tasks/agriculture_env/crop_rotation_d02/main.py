"""AgentHLE task: crop rotation audit over French agricultural parcels."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

# cua_bench loads task modules via exec_module without always pre-registering
# them in sys.modules; dataclass needs this for string annotation handling.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_config import GeneralTaskConfig  # noqa: E402
from tasks.common_setup import BaseTaskSetup  # noqa: E402

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
REMOTE_EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\crop_rotation_d02"

DOMAIN_NAME = "agriculture_env"
TASK_NAME = "crop_rotation_d02"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "OUTPUT_SUBDIR must be one of: " + ", ".join(sorted(ALLOWED_OUTPUT_DIRS))
        )
    return normalized


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def _parse_json_stdout(raw: str) -> dict[str, object]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("verifier returned empty stdout")
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return json.loads(text)


@dataclass
class CropRotationAuditConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "windows"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def input_gpkg(self) -> str:
        return rf"{self.input_dir}\seq1524_d02.gpkg"

    @property
    def task_prompt_file(self) -> str:
        return rf"{self.input_dir}\task_prompt.md"

    @property
    def citation_file(self) -> str:
        return rf"{self.input_dir}\citation.txt"

    @property
    def runtime_env_dir(self) -> str:
        return rf"{self.input_dir}\runtime_env"

    @property
    def runtime_pyproject(self) -> str:
        return rf"{self.runtime_env_dir}\pyproject.toml"

    @property
    def runtime_lockfile(self) -> str:
        return rf"{self.runtime_env_dir}\uv.lock"

    @property
    def runtime_requirements(self) -> str:
        return rf"{self.runtime_env_dir}\requirements.txt"

    @property
    def software_readme(self) -> str:
        return rf"{self.software_dir}\README.txt"

    @property
    def python_geo(self) -> str:
        return rf"{self.software_dir}\python_geo.bat"

    @property
    def output_dir(self) -> str:
        output_dir_name = _canonical_output_dir_name(self.OUTPUT_SUBDIR)
        return rf"{self.task_dir}\{output_dir_name}"

    @property
    def eligible_output_file(self) -> str:
        return rf"{self.output_dir}\eligible_units.gpkg"

    @property
    def flagged_output_file(self) -> str:
        return rf"{self.output_dir}\flagged_units.gpkg"

    @property
    def reference_eligible_file(self) -> str:
        return rf"{self.reference_dir}\eligible_units.gpkg"

    @property
    def reference_flagged_file(self) -> str:
        return rf"{self.reference_dir}\flagged_units.gpkg"

    @property
    def task_description(self) -> str:
        return f"""You are auditing crop rotation eligibility for stable agricultural parcels in France.

Task directory:
- `{self.task_dir}`

Visible inputs:
- Parcel GeoPackage: `{self.input_gpkg}`
- Task brief: `{self.task_prompt_file}`
- Citation note: `{self.citation_file}`
- Optional Python runtime manifest:
  - `{self.runtime_pyproject}`
  - `{self.runtime_lockfile}`
  - `{self.runtime_requirements}`
- Software note: `{self.software_readme}`

Optional Python runtime:
- Use `{self.python_geo}` if you want the benchmark-provisioned geospatial Python stack (pinned `numpy`, `pandas`, `geopandas`, `shapely`, `pyogrio`, `fiona`, `pyproj`).
- The package manifest is visible at `{self.runtime_pyproject}` if you prefer to build your own environment.

Your task:
1. Read `{self.task_prompt_file}` and follow it exactly.
2. Load the input layer `seq1524_d02` from `{self.input_gpkg}`.
3. Produce these two GeoPackages under `{self.output_dir}`:
   - `eligible_units.gpkg`
   - `flagged_units.gpkg`
4. Use exact layer names:
   - `eligible_units`
   - `flagged_units`

Rules:
- Do not modify files under `input/`.
- Preserve the input geometry and original columns exactly as required by the task brief.
- Keep the output CRS as EPSG:2154.
- Write final outputs only under `{self.output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "input_gpkg": self.input_gpkg,
                "task_prompt_file": self.task_prompt_file,
                "citation_file": self.citation_file,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lockfile": self.runtime_lockfile,
                "runtime_requirements": self.runtime_requirements,
                "software_readme": self.software_readme,
                "python_geo": self.python_geo,
                "eligible_output_file": self.eligible_output_file,
                "flagged_output_file": self.flagged_output_file,
                "reference_eligible_file": self.reference_eligible_file,
                "reference_flagged_file": self.reference_flagged_file,
                "output_dir": self.output_dir,
                "output_dir_name": _canonical_output_dir_name(self.OUTPUT_SUBDIR),
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = CropRotationAuditConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    required_paths = [
        meta["eligible_output_file"],
        meta["flagged_output_file"],
        meta["reference_eligible_file"],
        meta["reference_flagged_file"],
    ]
    missing = [path for path in required_paths if not (await session.file_exists(path) or await session.directory_exists(path))]
    if missing:
        logger.error("Missing output or hidden reference for evaluation: %s", missing)
        return [0.0]

    try:
        await session.interface.create_dir(REMOTE_EVAL_TMP_DIR)
        verify_script_path = rf"{REMOTE_EVAL_TMP_DIR}\score_outputs.py"
        await session.write_file(verify_script_path, _read_script("score_outputs.py"))
        command = subprocess.list2cmdline(
            [
                "python",
                verify_script_path,
                "--pred-dir",
                meta["output_dir"],
                "--gt-dir",
                meta["reference_dir"],
            ]
        )
        result = await session.run_command(command, check=False)
        report = _parse_json_stdout(result.get("stdout", ""))
        if result.get("return_code", 0) != 0:
            logger.error(
                "crop_rotation_d02 verifier exited non-zero: stdout=%s stderr=%s",
                (result.get("stdout", "") or "")[:1000],
                (result.get("stderr", "") or "")[:1000],
            )
            return [0.0]
    except Exception as exc:
        logger.exception("crop_rotation_d02 evaluation failed: %s", exc)
        return [0.0]

    logger.info("crop_rotation_d02 verifier=%s", json.dumps(report, sort_keys=True))
    return [float(report.get("total_score", 0.0)) / 100.0]


if __name__ == "__main__":
    for task in load():
        print(task.description)
