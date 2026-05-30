"""Underwater pathtracer effect task.

The agent receives a starter C++ pathtracer that renders a plain Cornell-box
spheres scene plus two images: a no-underwater baseline render and a visual
target reference. The evaluator rebuilds the modified renderer, runs a fixed
canonical render command, and uses a local vision-language judge to score the
generated PNG against hidden visual features/reference imagery.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from io import BytesIO

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

from tasks.utils.evaluation import llm_vision_json_judge, resolve_llm_judge_model

logger = logging.getLogger(__name__)

DOMAIN_NAME = "visual_media"
TASK_NAME = "underwater_pathtracer_effect"
VARIANT_NAME = "base"

REQUIRED_INPUT_FILES = (
    "baseline_no_underwater.png",
    "target_reference.png",
    "task_prompt.md",
)
REQUIRED_SOFTWARE_FILES = (
    "pathtracer/CMakeLists.txt",
    "pathtracer/render.sh",
)
REQUIRED_OUTPUT_FILES = (
    "final.png",
    "run_manifest.json",
)
PROBE_OUTPUT = "final_probe.png"
PROBE_SAMPLES = 2
SSIM_LOWER_BOUND = 0.55
PIXEL_DIFF_THRESHOLD = 0.002
EVALUATOR_ONLY_DIR_NAMES = (
    "reference",
    "output_test_pos",
    "output_test_neg",
)

EVAL_MODEL = resolve_llm_judge_model(
    env_var="UNDERWATER_PATH_TRACER_JUDGE_MODEL",
    default=os.environ.get("LLM_JUDGE_MODEL", "gpt-5.4"),
)

JUDGE_PROMPT = """\
You are evaluating a C++ pathtracer rendering task.

Images are provided in this exact order:
1. Baseline render from the starter renderer with no underwater effect.
2. Public target reference shown to the agent.
3. Hidden evaluator reference for the desired underwater appearance.
4. Candidate render produced by rerunning the agent's modified renderer.

Judge whether image 4 is a convincing renderer-produced underwater version of
the Cornell-box spheres scene, using images 1-3 as context. Ignore Monte Carlo
noise unless it prevents recognizing the effect. Do not require pixel-perfect
matching, but do require the same composition and meaningful underwater light
transport rather than a generic blue tint.

Return a JSON object with this exact shape:
{
  "render_validity": 0.0,
  "surface_refraction_reflection": 0.0,
  "chromatic_attenuation": 0.0,
  "volumetric_scattering": 0.0,
  "light_transport": 0.0,
  "scene_fidelity": 0.0,
  "overall_match": 0.0,
  "summary": "short explanation",
  "major_errors": ["short strings"]
}

Rubric:
- render_validity: candidate is nonblank, coherent, and looks like a pathtraced
  Cornell-box render rather than a corrupt image.
- surface_refraction_reflection: visible waterline/surface plus specular,
  reflective, or refractive behavior near the water boundary. Objects below
  the water surface should appear distorted or shifted.
- chromatic_attenuation: depth-dependent color shift where red attenuates faster
  than blue/green. A uniform blue tint scores low; true chromatic absorption
  with visible color gradient by depth scores high.
- volumetric_scattering: visible underwater haze, fog, or volume scattering.
  Light shafts, god-ray-like effects, or bright scattering volume near the
  light source are strong positive indicators.
- light_transport: plausible recursive light transport through the water
  surface. This includes caustic-like brightening, underwater shadow
  interaction, or multi-bounce illumination through the water-air interface.
- scene_fidelity: preserves the box, walls, floor, and spheres with comparable
  framing and object placement.
- overall_match: holistic similarity to the hidden reference and public target.

Use values between 0 and 1. If an item is absent, score it 0. If unsure, score
conservatively.
"""


def _q(value: str) -> str:
    return shlex.quote(str(value))


class UnderwaterPathtracerEffectConfig(LinuxTaskConfig):
    """Linux task config for the underwater pathtracer visual target task."""

    def __init__(self, *, REMOTE_OUTPUT_DIR: str | None = None) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=REMOTE_OUTPUT_DIR or "output",
        )

    @property
    def starter_root(self) -> str:
        return f"{self.software_dir}/pathtracer"

    @property
    def render_wrapper(self) -> str:
        return f"{self.starter_root}/render.sh"

    @property
    def baseline_image(self) -> str:
        return f"{self.input_dir}/baseline_no_underwater.png"

    @property
    def target_reference_image(self) -> str:
        return f"{self.input_dir}/target_reference.png"

    @property
    def public_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def hidden_reference_image(self) -> str:
        return f"{self.reference_dir}/hidden_reference.png"

    @property
    def canonical_gcs_root(self) -> str:
        return f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}"

    @property
    def output_image(self) -> str:
        return f"{self.remote_output_dir}/final.png"

    @property
    def output_manifest(self) -> str:
        return f"{self.remote_output_dir}/run_manifest.json"

    @property
    def scene_file(self) -> str:
        return f"{self.starter_root}/dae/sky/CBspheres_lambertian.dae"

    @property
    def eval_build_dir(self) -> str:
        return f"{self.starter_root}/build_agenthle_eval"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM. Your task is to modify a C++ pathtracer so a
plain Cornell-box spheres render becomes an underwater scene matching the
provided target reference.

## Task Directory
- `{self.task_dir}`

## Visible Inputs
- Baseline render with no underwater effect: `{self.baseline_image}`
- Target visual reference: `{self.target_reference_image}`
- Detailed task brief: `{self.public_prompt_file}`

## Starter Code
- C++ pathtracer source: `{self.starter_root}`
- Render wrapper: `{self.render_wrapper}`

## Your Task
1. Inspect the baseline image and target reference.
2. Study the starter code, especially:
   - `src/pathtracer/medium.h` (Medium class with TODO stubs)
   - `src/pathtracer/pathtracer.cpp` (volumetric path and indirect bounces)
   - `src/pathtracer/advanced_bsdf.cpp` (reflection/refraction BSDFs)
3. Implement a convincing underwater effect requiring:
   - Water surface with Fresnel reflection/refraction
   - Chromatic absorption (red fades faster than blue)
   - Participating medium with volumetric scattering
   - Recursive ray tracing through the water-air interface
   - Beer-Lambert depth-dependent attenuation
4. Run `{self.render_wrapper}` to build and preview your result.
5. Write the final artifacts under `{self.remote_output_dir}`:
   - `final.png`
   - `run_manifest.json`

## Constraints
- Do not modify files under `{self.input_dir}`.
- Work only from the visible `input/` images and the provided
  `software/pathtracer` source tree.
- The evaluator will rebuild and rerun your modified renderer with its own
  canonical CMake + pathtracer command. A hand-edited or copied PNG is not
  sufficient.
- Keep final deliverables inside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "starter_root": self.starter_root,
                "render_wrapper": self.render_wrapper,
                "baseline_image": self.baseline_image,
                "target_reference_image": self.target_reference_image,
                "public_prompt_file": self.public_prompt_file,
                "hidden_reference_image": self.hidden_reference_image,
                "canonical_gcs_root": self.canonical_gcs_root,
                "output_image": self.output_image,
                "output_manifest": self.output_manifest,
                "scene_file": self.scene_file,
                "eval_build_dir": self.eval_build_dir,
                "eval_model": EVAL_MODEL,
            }
        )
        return metadata


config = UnderwaterPathtracerEffectConfig(
    REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR")
)


@cb.tasks_config(split="train")
def load():
    """Register task variants."""
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: float | None = None,
    check: bool = False,
) -> dict:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        # Older/current remote CUA sessions do not accept a timeout kwarg.
        return await session.run_command(command, check=check)
    except Exception as exc:
        logger.warning("run_command raised %s: %s", type(exc).__name__, exc)
        return {"stdout": "", "stderr": str(exc), "return_code": -1}


def _clamp01(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _weighted_judge_score(payload: dict) -> float:
    weights = {
        "render_validity": 0.10,
        "surface_refraction_reflection": 0.15,
        "chromatic_attenuation": 0.15,
        "volumetric_scattering": 0.15,
        "light_transport": 0.15,
        "scene_fidelity": 0.10,
        "overall_match": 0.20,
    }
    return sum(weights[key] * _clamp01(payload.get(key)) for key in weights)


def _png_pixel_fingerprint(image_bytes: bytes) -> tuple[tuple[int, int], bytes] | None:
    """Return a decoded RGBA pixel fingerprint for PNG copy gates."""
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(BytesIO(image_bytes)) as image:
            decoded = image.convert("RGBA")
            return decoded.size, decoded.tobytes()
    except (OSError, UnidentifiedImageError, ValueError):
        return None

def _same_png_pixels(a: bytes, b: bytes) -> bool:
    a_pixels = _png_pixel_fingerprint(a)
    b_pixels = _png_pixel_fingerprint(b)
    return a_pixels is not None and a_pixels == b_pixels


def _compute_ssim_and_diff(a_bytes: bytes, b_bytes: bytes) -> tuple[float, float] | None:
    """Compute SSIM and mean absolute pixel difference between two PNGs.

    Returns (ssim, mean_abs_diff) or None if images cannot be decoded/compared.
    ssim is in [0, 1], mean_abs_diff is in [0, 1] (normalized to pixel range).
    """
    try:
        import numpy as np
        from PIL import Image, UnidentifiedImageError

        with Image.open(BytesIO(a_bytes)) as img_a, Image.open(BytesIO(b_bytes)) as img_b:
            arr_a = np.array(img_a.convert("RGB"), dtype=np.float64) / 255.0
            arr_b = np.array(img_b.convert("RGB"), dtype=np.float64) / 255.0

        if arr_a.shape != arr_b.shape:
            return None

        mean_abs_diff = float(np.mean(np.abs(arr_a - arr_b)))

        # SSIM with default constants (Wang et al. 2004)
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        mu_a = arr_a.mean(axis=(0, 1))
        mu_b = arr_b.mean(axis=(0, 1))
        sigma_a_sq = ((arr_a - mu_a) ** 2).mean(axis=(0, 1))
        sigma_b_sq = ((arr_b - mu_b) ** 2).mean(axis=(0, 1))
        sigma_ab = ((arr_a - mu_a) * (arr_b - mu_b)).mean(axis=(0, 1))

        numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        denominator = (mu_a**2 + mu_b**2 + C1) * (sigma_a_sq + sigma_b_sq + C2)
        ssim_per_channel = numerator / denominator
        ssim = float(ssim_per_channel.mean())

        return (ssim, mean_abs_diff)
    except (OSError, UnidentifiedImageError, ValueError, ImportError):
        return None


def _canonical_render_command(
    meta: dict, *, output_path: str | None = None, samples: int | None = None
) -> str:
    """Build and run the renderer with evaluator-owned command text.

    The agent may edit source/CMake/scene assets, but evaluation should not
    trust a mutable convenience wrapper or stale output files.

    The samples parameter overrides the number of camera rays per pixel (-s).
    When None, falls back to PT_SAMPLES env or default 32.
    """
    root = _q(meta["starter_root"])
    build = _q(meta["eval_build_dir"])
    actual_output = output_path or meta["output_image"]
    output = _q(actual_output)
    output_dir = _q(meta["remote_output_dir"])
    scene = _q(meta["scene_file"])
    manifest = _q(meta["output_manifest"])
    manifest_py = meta["output_manifest"]
    if samples is not None:
        samples_arg = str(samples)
    else:
        samples_arg = '"${PT_SAMPLES:-32}"'
    is_primary = output_path is None
    script = f"""set -euo pipefail
rm -f {output}
cmake -S {root} -B {build} -DCMAKE_BUILD_TYPE=Release
cmake --build {build} -j"$(nproc 2>/dev/null || echo 4)"
mkdir -p {output_dir}
{build}/pathtracer -s {samples_arg} -l "${{PT_LIGHT_SAMPLES:-8}}" -t "${{PT_THREADS:-8}}" -m "${{PT_DEPTH:-8}}" -r 768 576 -f {output} {scene}"""
    if is_primary:
        script += f"""
python - <<'PY'
import json
import time
from pathlib import Path
Path({manifest_py!r}).write_text(json.dumps({{
    "output": {actual_output!r},
    "renderer": "pathtracer",
    "scene": {meta["scene_file"]!r},
    "evaluation_rerendered": True,
    "samples": {samples if samples is not None else 32},
    "created_at": time.time(),
}}, indent=2))
PY"""
    return f"bash -lc {_q(script)}"


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Rebuild/rerender with sample-count probe, verify stochastic rendering, then LLM judge.

    Evaluation flow:
    1. Build once (cmake caches), render at full quality (-s 32) → final.png
    2. Render again at minimal quality (-s 2) → final_probe.png
    3. Verify final.png ≠ final_probe.png (proves code responds to sample count)
    4. Verify SSIM(final, probe) > threshold (proves same scene, just noisier)
    5. Run LLM vision judge on final.png for visual quality scoring
    """
    meta = task_cfg.metadata
    output_dir_name = meta["remote_output_dir"].rstrip("/").rsplit("/", 1)[-1]
    admin_fixture_mode = output_dir_name in {"output_test_pos", "output_test_neg"}

    probe_path = f'{meta["remote_output_dir"]}/{PROBE_OUTPUT}'

    if not admin_fixture_mode:
        # Render 1: full quality → final.png
        render_result = await _run_command(
            session,
            _canonical_render_command(meta, samples=32),
            timeout=1800.0,
            check=False,
        )
        if render_result.get("return_code", 1) != 0:
            logger.error("Render (s=32) failed: %s", render_result.get("stderr", "")[:800])
            return [0.0]

        # Render 2: minimal samples → final_probe.png (build cached, fast render)
        render_result2 = await _run_command(
            session,
            _canonical_render_command(meta, output_path=probe_path, samples=PROBE_SAMPLES),
            timeout=600.0,
            check=False,
        )
        if render_result2.get("return_code", 1) != 0:
            logger.error("Render (s=%d) failed: %s", PROBE_SAMPLES, render_result2.get("stderr", "")[:800])
            return [0.0]
    else:
        logger.info("Admin fixture mode: scoring pre-staged %s", meta["remote_output_dir"])

    for fname in REQUIRED_OUTPUT_FILES:
        path = f'{meta["remote_output_dir"]}/{fname}'
        if not await session.exists(path):
            logger.warning("Missing output file: %s", path)
            return [0.0]

    if not await session.exists(meta["hidden_reference_image"]):
        logger.error("Hidden reference missing: %s", meta["hidden_reference_image"])
        return [0.0]

    try:
        baseline_bytes = await session.read_bytes(meta["baseline_image"])
        public_target_bytes = await session.read_bytes(meta["target_reference_image"])
        hidden_reference_bytes = await session.read_bytes(meta["hidden_reference_image"])
        candidate_bytes = await session.read_bytes(meta["output_image"])
    except Exception as exc:
        logger.error("Failed to read evaluation images: %s", exc)
        return [0.0]

    if not candidate_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        logger.warning("Candidate output is not a PNG")
        return [0.0]

    # --- Copy gates ---
    if candidate_bytes == baseline_bytes or _same_png_pixels(candidate_bytes, baseline_bytes):
        logger.warning("Candidate output matches the no-underwater baseline")
        return [0.0]
    if candidate_bytes == public_target_bytes or _same_png_pixels(candidate_bytes, public_target_bytes):
        logger.warning("Candidate output matches the public target reference")
        return [0.0]
    if candidate_bytes == hidden_reference_bytes or _same_png_pixels(
        candidate_bytes, hidden_reference_bytes
    ):
        logger.warning("Candidate output matches the hidden reference exactly")
        return [0.0]

    # --- Sample-count probe: verify the renderer responds to -s parameter ---
    if not admin_fixture_mode:
        if not await session.exists(probe_path):
            logger.error("Probe render missing: %s", probe_path)
            return [0.0]

        try:
            probe_bytes = await session.read_bytes(probe_path)
        except Exception as exc:
            logger.error("Failed to read probe image: %s", exc)
            return [0.0]

        if _same_png_pixels(candidate_bytes, probe_bytes):
            logger.warning(
                "Sample-count probe FAILED: -s 32 and -s %d produced identical pixels. "
                "The renderer ignores the sample count — output is likely fabricated.",
                PROBE_SAMPLES,
            )
            return [0.0]

        ssim_result = _compute_ssim_and_diff(candidate_bytes, probe_bytes)
        if ssim_result is None:
            logger.error("Failed to compute SSIM between full and probe renders")
            return [0.0]

        ssim, mean_diff = ssim_result
        logger.info(
            "[%s] Sample-count probe: SSIM=%.4f, mean_abs_diff=%.4f (s=32 vs s=%d)",
            TASK_NAME, ssim, mean_diff, PROBE_SAMPLES,
        )

        if mean_diff < PIXEL_DIFF_THRESHOLD:
            logger.warning(
                "Sample-count probe FAILED: mean pixel difference %.6f < %.4f. "
                "Renders nearly identical despite 16x sample difference — not stochastic.",
                mean_diff, PIXEL_DIFF_THRESHOLD,
            )
            return [0.0]

        if ssim < SSIM_LOWER_BOUND:
            logger.warning(
                "Sample-count probe FAILED: SSIM=%.4f < %.2f. "
                "Renders diverge too much — not the same scene geometry.",
                ssim, SSIM_LOWER_BOUND,
            )
            return [0.0]

    # --- LLM vision judge ---
    try:
        judge_payload = await llm_vision_json_judge(
            prompt=JUDGE_PROMPT,
            image_bytes_list=[
                baseline_bytes,
                public_target_bytes,
                hidden_reference_bytes,
                candidate_bytes,
            ],
            model=meta.get("eval_model"),
            max_tokens=900,
            temperature=0,
        )
    except Exception as exc:
        logger.error("LLM image judge failed: %s", exc)
        return [0.0]

    score = _weighted_judge_score(judge_payload)
    logger.info(
        "[%s] LLM image judge score=%.4f payload=%s",
        TASK_NAME,
        score,
        json.dumps(judge_payload, ensure_ascii=False)[:2000],
    )
    return [float(score)]
