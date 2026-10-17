"""physical_sciences/hst_acs_wfc_visit_reduction -- Linux task."""

from __future__ import annotations

import json
import logging
import math
import shlex
import asyncio
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup  # noqa: E402
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402
from tasks.physical_sciences.hst_acs_wfc_visit_reduction.scripts.score_outputs import (  # noqa: E402
    REQUIRED_VISIT_FILES,
)

_setup = BaseTaskSetup()


logger = logging.getLogger(__name__)

DOMAIN_NAME = "physical_sciences"
TASK_NAME = "hst_acs_wfc_visit_reduction"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).parent / "scripts"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"
STATIC_OUTPUT_DIRS = {"output_test_pos", "output_test_neg"}


class CandidateReductionFailure(RuntimeError):
    pass


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


@dataclass
class HstAcsWfcVisitReductionConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def prompt_file(self) -> str:
        return f"{self.input_dir}/TASK_PROMPT.md"

    @property
    def visible_visit_dir(self) -> str:
        return f"{self.input_dir}/acs_visit_f606w_lockman"

    @property
    def starter_script(self) -> str:
        return f"{self.input_dir}/starter_project/reduce_visit.py"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_with_task_deps.sh"

    @property
    def candidate_script(self) -> str:
        return f"{self.remote_output_dir}/reduce_visit.py"

    @property
    def hidden_input_dir(self) -> str:
        return f"{self.reference_dir}/hidden_input"

    @property
    def reference_outputs_dir(self) -> str:
        return f"{self.reference_dir}/reference_outputs"

    @property
    def task_description(self) -> str:
        return f"""\
You are reducing synthetic HST ACS/WFC visit data on a Linux VM. Read the full
instructions in `{self.prompt_file}` and use the visible visit data under:
`{self.visible_visit_dir}`.

## Goal
Implement a reusable ACS/WFC visit-reduction script at:
`{self.candidate_script}`

Your script must accept:
`python reduce_visit.py --input <visit_root_or_parent_input_dir> --output <output_dir>`

For each visit in the input, write a folder directly under the supplied output
directory, for example `<output_dir>/<visit_id>/`, containing:
- `drizzled_image.csv`
- `source_catalog.csv`
- `alignment_solution.csv`
- `photometry_qc.json`
- `reduction_report.md`

## Alignment Shift Convention
Write `alignment_solution.csv` with columns exactly
`exposure_id,dx_pix,dy_pix,rms_pix,matched_sources`.
The recommended convention is detector displacement from the Gaia-anchored
mosaic: `x_mosaic = x_detector - dx_pix`, `y_mosaic = y_detector - dy_pix`.
Applied registration shifts with the opposite sign are also accepted:
`x_mosaic = x_detector + dx_pix`, `y_mosaic = y_detector + dy_pix`.
Use one convention for both axes and every exposure within a visit, and
state it in `reduction_report.md`. These are equivalent translation
representations, not permission to change individual signs, swap axes or
exposure identities, or leave the mosaic unregistered. Shifts are in pixels
at the unchanged detector/output scale, relative to the anchored mosaic,
not merely relative to the first exposure.

## Photometry QC Contract
Use a circular aperture radius of 3.0 output pixels and drizzle pixfrac 0.8.
Keep the final pixel scale at the visit calibration context's `pixel_scale_arcsec`.
These parameters must be used in the reduction, not just recorded in the QC file.

Write `photometry_qc.json` as an object with the following exact keys. Additional
diagnostics are allowed. Use JSON numbers for measurements/counts, not unit-bearing
strings; measurements must be finite.

| Key | Value and units |
| --- | --- |
| `visit_id` | String: input visit folder name. |
| `filter` | String: visit filter from the association table or FITS header. |
| `num_sources` | Integer: number of rows in the output source catalog. |
| `background_median_e_s` | Number: estimated median background count rate, electrons/s per output pixel. |
| `cosmic_ray_pixels_masked` | Integer: total masked pixels with DQ bit 4096 set, summed over exposures. |
| `hot_pixels_masked` | Integer: total masked pixels with DQ bit 16 set, summed over exposures. |
| `aperture_radius_pix` | Number: 3.0 output pixels. |
| `pixfrac` | Number: 0.8, the dimensionless drizzle drop fraction. |
| `final_scale_arcsec_per_pix` | Number: final pixel scale in arcsec/pixel from the calibration context. |
| `astrometric_rms_pix` | Number: RMS astrometric residual against the matched Gaia-like anchors, in pixels. |

Count each flagged pixel in each exposure, not unique mosaic coordinates. Test
DQ flags as bits; a pixel with both bits set contributes to both counts.
In `reduction_report.md`, describe the actual calacs-style calibration,
AstroDrizzle-style alignment/coaddition, astrometric RMS, and cosmic-ray masking.
Case, hyphens, and whitespace in these report phrases may vary; the required
methods and scientific measurements do not change.

The starter implementation is at `{self.starter_script}`. The runtime manifest is
under `{self.runtime_env_dir}`, and the helper wrapper is:
`{self.python_wrapper}`

Do not modify files under `{self.input_dir}`. Write your final implementation and
any scratch artifacts only under `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{DOMAIN_NAME}/{TASK_NAME}",
                "prompt_file": self.prompt_file,
                "visible_visit_dir": self.visible_visit_dir,
                "starter_script": self.starter_script,
                "runtime_env_dir": self.runtime_env_dir,
                "python_wrapper": self.python_wrapper,
                "candidate_script": self.candidate_script,
                "hidden_input_dir": self.hidden_input_dir,
                "reference_outputs_dir": self.reference_outputs_dir,
                "required_visit_files": sorted(REQUIRED_VISIT_FILES),
            }
        )
        return metadata


config = HstAcsWfcVisitReductionConfig()


@cb.tasks_config(split="train")
def load():
    cfg = HstAcsWfcVisitReductionConfig()
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


def _write_tree_from_mapping(root: Path, files: dict[str, bytes]) -> None:
    for rel_path, payload in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


async def _pull_tree(
    session: cb.DesktopSession, remote_root: str, *, allow_missing: bool = False
) -> dict[str, bytes]:
    if not await session.directory_exists(remote_root):
        if allow_missing:
            return {}
        raise RuntimeError(f"missing evaluator directory: {remote_root}")
    result = await session.run_command(
        f"find {_shell_quote(remote_root)} -type f -printf '%P\\n'",
        check=False,
    )
    if result.get("return_code") != 0:
        raise RuntimeError(f"could not list remote tree {remote_root}: {result}")
    files: dict[str, bytes] = {}
    for rel_path in result.get("stdout", "").splitlines():
        clean = rel_path.strip()
        if clean:
            files[clean] = await session.read_bytes(f"{remote_root}/{clean}")
    return files


async def _run_candidate(session: cb.DesktopSession, meta: dict[str, Any]) -> str:
    candidate = meta["candidate_script"]
    if not await session.file_exists(candidate):
        raise CandidateReductionFailure(f"missing required candidate script: {candidate}")
    if not await session.file_exists(meta["python_wrapper"]):
        raise RuntimeError(f"missing evaluator runtime wrapper: {meta['python_wrapper']}")
    for key in ("visible_visit_dir", "hidden_input_dir"):
        if not await session.directory_exists(meta[key]):
            raise RuntimeError(f"missing evaluator input: {meta[key]}")

    run_dir = f"{EVAL_TMP_DIR}/candidate_run"
    input_dir = f"{run_dir}/combined_input"
    generated_dir = f"{run_dir}/generated_output"
    command = "\n".join(
        [
            "set -euo pipefail",
            f"rm -rf {_shell_quote(run_dir)}",
            f"mkdir -p {_shell_quote(input_dir)} {_shell_quote(generated_dir)}",
            f"cp -a {_shell_quote(meta['visible_visit_dir'])} {_shell_quote(input_dir)}/",
            f"cp -a {_shell_quote(meta['hidden_input_dir'])}/* {_shell_quote(input_dir)}/",
        ]
    )
    result = await session.run_command(f"bash -lc {_shell_quote(command)}", check=False)
    if result.get("return_code") != 0:
        raise RuntimeError(f"could not prepare candidate evaluation inputs: {result}")
    command = (
        f"{_shell_quote(meta['python_wrapper'])} {_shell_quote(candidate)} "
        f"--input {_shell_quote(input_dir)} --output {_shell_quote(generated_dir)}"
    )
    result = await session.run_command(f"bash -lc {_shell_quote(command)}", check=False)
    if type(result.get("return_code")) is not int:
        raise RuntimeError(f"candidate execution has no valid exit code: {result}")
    if result["return_code"] != 0:
        raise CandidateReductionFailure(
            "candidate reduction failed: "
            + str(result.get("stdout", ""))[-1000:]
            + str(result.get("stderr", ""))[-1000:]
        )
    return generated_dir


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_dir_name = Path(meta["remote_output_dir"]).name
    reference_files = await _pull_tree(session, meta["reference_outputs_dir"])
    if not reference_files:
        raise RuntimeError("evaluator reference outputs are empty")
    try:
        if output_dir_name in STATIC_OUTPUT_DIRS:
            output_root = meta["remote_output_dir"]
        else:
            output_root = await _run_candidate(session, meta)
    except CandidateReductionFailure as exc:
        logger.info("candidate reduction failed: %s", exc)
        return [0.0]
    output_files = await _pull_tree(session, output_root, allow_missing=True)

    with tempfile.TemporaryDirectory(prefix="agenthle_hst_acs_eval_") as tmp:
        tmp_path = Path(tmp)
        local_output = tmp_path / "output"
        local_reference = tmp_path / "reference_outputs"
        _write_tree_from_mapping(local_output, output_files)
        _write_tree_from_mapping(local_reference, reference_files)

        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                str(SCRIPTS_DIR / "score_outputs.py"),
                "--output-dir",
                str(local_output),
                "--reference-dir",
                str(local_reference),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    if result.stderr:
        logger.info("local scorer stderr: %s", result.stderr.strip()[:2000])
    if result.returncode != 0:
        raise RuntimeError(
            f"HST scorer failed with exit {result.returncode}: {result.stderr[-2000:]}"
        )
    payload: dict[str, Any] | None = None
    try:
        candidate = json.loads(result.stdout)
        if isinstance(candidate, dict) and "score" in candidate:
            payload = candidate
    except json.JSONDecodeError:
        for line in reversed(result.stdout.strip().splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "score" in candidate:
                payload = candidate
                break
    if payload is None:
        raise RuntimeError(f"could not parse HST scorer JSON: {result.stdout[:2000]}")
    score = payload["score"]
    if type(score) not in (int, float) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"invalid HST scorer score: {score!r}")
    logger.info("evaluation result: %s", json.dumps(payload, sort_keys=True))
    return [float(score)]


if __name__ == "__main__":
    for task in load():
        print(task.description)
